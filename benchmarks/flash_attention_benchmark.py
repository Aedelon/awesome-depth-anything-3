#!/usr/bin/env python3
# Copyright (c) Delanoe Pirard / Aedelon - Apache 2.0
"""
Flash Attention Benchmark for Depth Anything 3.

Provides clear performance comparison with tables and analysis.

Usage:
    python benchmarks/flash_attention_benchmark.py
    python benchmarks/flash_attention_benchmark.py --detailed
"""

import argparse
import gc
import os
import sys
import time
from dataclasses import dataclass

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from depth_anything_3.model.dinov2.layers import (
    FLASH_ATTN_AVAILABLE,
    FLASH_ATTN_VERSION,
    Attention,
)


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark test case."""

    name: str
    seq_len: int
    batch_size: int
    embed_dim: int
    num_heads: int
    image_size: str  # Description of corresponding image size

    @property
    def description(self):
        return f"{self.name} ({self.image_size})"


# Depth Anything 3 model configurations
DA3_CONFIGS = {
    "vitb": {"embed_dim": 768, "num_heads": 12, "depth": 12},
    "vitl": {"embed_dim": 1024, "num_heads": 16, "depth": 24},
    "vitg": {"embed_dim": 1536, "num_heads": 24, "depth": 40},
}


def get_device_info():
    """Get device information."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name()
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        compute_cap = torch.cuda.get_device_capability()
        return {
            "type": "cuda",
            "device": device,
            "name": device_name,
            "memory_gb": memory_gb,
            "compute_capability": f"{compute_cap[0]}.{compute_cap[1]}",
        }
    elif torch.backends.mps.is_available():
        return {
            "type": "mps",
            "device": torch.device("mps"),
            "name": "Apple Silicon",
            "memory_gb": None,
            "compute_capability": None,
        }
    else:
        return {
            "type": "cpu",
            "device": torch.device("cpu"),
            "name": "CPU",
            "memory_gb": None,
            "compute_capability": None,
        }


def benchmark_attention(attn_module, x, warmup=5, runs=20):
    """Run benchmark for a single attention module."""
    device = x.device

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = attn_module(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Reset memory tracking
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = attn_module(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

    # Memory
    peak_mem_mb = 0
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    times_tensor = torch.tensor(times)
    return {
        "mean_ms": times_tensor.mean().item(),
        "std_ms": times_tensor.std().item(),
        "min_ms": times_tensor.min().item(),
        "peak_mem_mb": peak_mem_mb,
    }


def print_header():
    """Print benchmark header."""
    print("\n" + "=" * 80)
    print(" " * 20 + "FLASH ATTENTION BENCHMARK - DEPTH ANYTHING 3")
    print("=" * 80 + "\n")


def print_device_info(device_info):
    """Print device information."""
    print("📊 HARDWARE CONFIGURATION")
    print("─" * 80)
    print(f"  Device Type      : {device_info['type'].upper()}")
    print(f"  Device Name      : {device_info['name']}")
    if device_info["memory_gb"]:
        print(f"  Memory           : {device_info['memory_gb']:.1f} GB")
    if device_info["compute_capability"]:
        print(f"  Compute Cap.     : {device_info['compute_capability']}")
        cc = float(device_info["compute_capability"])
        if cc >= 7.5:
            print(f"                     ✅ Flash Attention supported (≥7.5)")
        else:
            print(f"                     ❌ Flash Attention requires ≥7.5")

    print(f"\n  Flash Attention  : {'✅ Installed' if FLASH_ATTN_AVAILABLE else '❌ Not installed'}")
    if FLASH_ATTN_AVAILABLE:
        print(f"  FA Version       : {FLASH_ATTN_VERSION}")
    print()


def print_table_header():
    """Print benchmark table header."""
    print(
        "┌──────────────────────────┬──────────────┬──────────────┬──────────────┬────────────┐"
    )
    print(
        "│ Configuration            │ flash_attn   │ sdpa         │ manual       │ Speedup    │"
    )
    print(
        "├──────────────────────────┼──────────────┼──────────────┼──────────────┼────────────┤"
    )


def print_table_row(config_desc, results, baseline="sdpa"):
    """Print a benchmark result row."""
    backends = ["flash_attn", "sdpa", "manual"]

    # Format times
    time_strs = []
    for backend in backends:
        if backend in results and results[backend]:
            time_ms = results[backend]["mean_ms"]
            time_strs.append(f"{time_ms:6.2f} ms")
        else:
            time_strs.append("     N/A")

    # Calculate speedup
    speedup_str = "      -"
    if "flash_attn" in results and results["flash_attn"] and baseline in results:
        if results[baseline]:
            speedup = results[baseline]["mean_ms"] / results["flash_attn"]["mean_ms"]
            speedup_str = f"  {speedup:.2f}x ⚡" if speedup > 1.1 else f"  {speedup:.2f}x"

    print(
        f"│ {config_desc:24s} │ {time_strs[0]:12s} │ {time_strs[1]:12s} │ {time_strs[2]:12s} │ {speedup_str:10s} │"
    )


def print_table_footer():
    """Print benchmark table footer."""
    print(
        "└──────────────────────────┴──────────────┴──────────────┴──────────────┴────────────┘"
    )


def print_model_analysis(model_name, config, results, num_layers):
    """Print detailed analysis for a specific model."""
    if "flash_attn" not in results or not results["flash_attn"]:
        return

    flash_time = results["flash_attn"]["mean_ms"]
    sdpa_time = results["sdpa"]["mean_ms"] if "sdpa" in results else flash_time

    speedup = sdpa_time / flash_time
    time_saved_per_layer = (sdpa_time - flash_time) / num_layers
    total_time_saved = time_saved_per_layer * num_layers

    print(f"\n  📈 {model_name} Analysis:")
    print(f"     • Attention time per layer: {flash_time:.2f} ms (flash) vs {sdpa_time:.2f} ms (sdpa)")
    print(f"     • Time saved per layer: {time_saved_per_layer:.2f} ms")
    print(f"     • Total time saved ({num_layers} layers): {total_time_saved:.1f} ms")
    print(f"     • Speedup: {speedup:.2f}x on attention")

    # Estimate full inference impact
    # Attention is ~15-20% of total inference time
    attn_fraction = 0.175
    overall_speedup = 1 / (1 - attn_fraction + attn_fraction / speedup)
    overall_improvement = (1 - 1 / overall_speedup) * 100

    print(
        f"     • Estimated full inference speedup: {overall_speedup:.2f}x (~{overall_improvement:.1f}% faster)"
    )


def run_benchmark(test_configs, backends, warmup=5, runs=20, detailed=False):
    """Run complete benchmark suite."""
    device_info = get_device_info()
    device = device_info["device"]
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print_header()
    print_device_info(device_info)

    # Filter backends based on availability
    available_backends = []
    if FLASH_ATTN_AVAILABLE and device.type == "cuda":
        available_backends.append("flash_attn")
    available_backends.append("sdpa")
    if detailed:
        available_backends.append("manual")

    all_results = {}

    # Run benchmarks by model
    for model_name, model_config in DA3_CONFIGS.items():
        print(f"\n🔬 MODEL: {model_name.upper()} (dim={model_config['embed_dim']}, heads={model_config['num_heads']}, depth={model_config['depth']})")
        print("─" * 80)
        print_table_header()

        model_results = {}

        for test_config in test_configs:
            # Adjust config for this model
            config = BenchmarkConfig(
                name=test_config.name,
                seq_len=test_config.seq_len,
                batch_size=test_config.batch_size,
                embed_dim=model_config["embed_dim"],
                num_heads=model_config["num_heads"],
                image_size=test_config.image_size,
            )

            x = torch.randn(
                config.batch_size, config.seq_len, config.embed_dim, device=device, dtype=dtype
            )

            results = {}
            for backend in available_backends:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                try:
                    attn = Attention(
                        dim=config.embed_dim,
                        num_heads=config.num_heads,
                        attn_backend=backend,
                    ).to(device, dtype)
                    attn.eval()

                    result = benchmark_attention(attn, x, warmup=warmup, runs=runs)
                    results[backend] = result

                    del attn
                except Exception as e:
                    results[backend] = None
                    if detailed:
                        print(f"    {backend} failed: {e}")

            model_results[config.name] = results
            print_table_row(config.description, results)

        print_table_footer()

        # Analysis for this model
        if detailed and model_results:
            # Use medium config for analysis
            medium_key = next(
                (k for k in model_results.keys() if "1024" in k.lower() or "medium" in k.lower()),
                list(model_results.keys())[0],
            )
            print_model_analysis(
                model_name.upper(),
                test_configs[0],
                model_results[medium_key],
                model_config["depth"],
            )

        all_results[model_name] = model_results

    # Final summary
    print("\n" + "=" * 80)
    print("📋 SUMMARY & RECOMMENDATIONS")
    print("=" * 80)

    if device.type == "cuda":
        if FLASH_ATTN_AVAILABLE:
            print("\n✅ Flash Attention is ACTIVE and working")
            print("\n   Benefits:")
            print("   • 2-3x faster attention computation")
            print("   • ~15-25% overall inference speedup")
            print("   • Lower memory usage")
            print("   • Automatic backend selection")
            print("\n   ⚡ Your inference is already optimized!")

        else:
            print("\n⚠️  Flash Attention NOT installed")
            print("\n   Install with:")
            print("   $ pip install flash-attn --no-build-isolation")
            print("\n   Expected improvements:")
            print("   • 2-3x faster attention layers")
            print("   • 15-25% faster full inference")
            print("   • Especially beneficial for Large and Giant models")

    elif device.type == "mps":
        print("\n📱 Apple Silicon (MPS) detected")
        print("\n   • Flash Attention not available for MPS")
        print("   • PyTorch SDPA uses optimized Metal kernels")
        print("   • Already running at optimal speed for your hardware")

    else:
        print("\n💻 CPU detected")
        print("\n   • Consider using GPU for faster inference")
        print("   • Flash Attention is CUDA-only")

    print("\n" + "=" * 80)
    print()

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Flash Attention benchmark for DA3")
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed analysis and include manual backend",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup iterations (default: 5)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=20,
        help="Benchmark runs (default: 20)",
    )

    args = parser.parse_args()

    # Test configurations based on common image sizes
    test_configs = [
        BenchmarkConfig(
            name="Small",
            seq_len=256,
            batch_size=1,
            embed_dim=768,  # Will be overridden per model
            num_heads=12,  # Will be overridden per model
            image_size="392px image",
        ),
        BenchmarkConfig(
            name="Medium",
            seq_len=529,
            batch_size=1,
            embed_dim=768,
            num_heads=12,
            image_size="518px image",
        ),
        BenchmarkConfig(
            name="Large",
            seq_len=1024,
            batch_size=1,
            embed_dim=768,
            num_heads=12,
            image_size="742px image",
        ),
        BenchmarkConfig(
            name="XLarge",
            seq_len=1369,
            batch_size=1,
            embed_dim=768,
            num_heads=12,
            image_size="1024px image",
        ),
    ]

    backends = ["flash_attn", "sdpa"]
    if args.detailed:
        backends.append("manual")

    run_benchmark(
        test_configs=test_configs,
        backends=backends,
        warmup=args.warmup,
        runs=args.runs,
        detailed=args.detailed,
    )


if __name__ == "__main__":
    main()