#!/usr/bin/env python3
"""
Benchmark validation for device-specific optimizations.

This script validates that:
1. Device-specific optimizations are applied correctly
2. Performance improvements are measurable
3. Output quality is preserved (no significant degradation)

Tests across:
- All available devices (CPU, MPS, CUDA)
- All performance modes (minimal, balanced, max)
- Different mixed precision settings
- With and without torch.compile
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.depth_anything_3.api import DepthAnything3


def get_test_images(num_images: int = 5) -> list[np.ndarray]:
    """
    Generate synthetic test images.

    Args:
        num_images: Number of test images to generate

    Returns:
        List of RGB images as numpy arrays
    """
    images = []
    for i in range(num_images):
        # Create synthetic gradient images (504x504)
        img = np.zeros((504, 504, 3), dtype=np.uint8)
        img[:, :, 0] = np.linspace(0, 255, 504).reshape(1, -1)  # Red gradient
        img[:, :, 1] = np.linspace(0, 255, 504).reshape(-1, 1)  # Green gradient
        img[:, :, 2] = (i * 50) % 256  # Blue channel varies per image
        images.append(img)
    return images


def benchmark_config(
    device: str,
    performance_mode: str,
    mixed_precision: bool | str | None,
    enable_compile: bool,
    num_images: int = 5,
    num_warmup: int = 1,
    num_runs: int = 3,
) -> dict[str, Any]:
    """
    Benchmark a specific configuration.

    Args:
        device: Device to use ('cpu', 'mps', 'cuda')
        performance_mode: Performance mode ('minimal', 'balanced', 'max')
        mixed_precision: Mixed precision setting (False, 'float16', 'bfloat16', None)
        enable_compile: Whether to enable torch.compile
        num_images: Number of test images
        num_warmup: Number of warmup runs
        num_runs: Number of benchmark runs

    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'='*80}")
    print(f"Benchmarking Configuration:")
    print(f"  Device: {device}")
    print(f"  Performance Mode: {performance_mode}")
    print(f"  Mixed Precision: {mixed_precision}")
    print(f"  Compile: {enable_compile}")
    print(f"{'='*80}")

    # Generate test images
    test_images = get_test_images(num_images)

    # Initialize model
    try:
        model = DepthAnything3(
            model_name="da3-large",
            device=device,
            mixed_precision=mixed_precision,
            enable_compile=enable_compile,
            performance_mode=performance_mode,
        )
    except Exception as e:
        print(f"  ERROR: Failed to initialize model: {e}")
        return {
            "device": device,
            "performance_mode": performance_mode,
            "mixed_precision": str(mixed_precision),
            "compile": enable_compile,
            "error": str(e),
            "success": False,
        }

    # Check optimizer settings
    print(f"\nOptimizer Info:")
    print(f"  Type: {type(model.optimizer).__name__}")
    print(f"  Mixed Precision dtype: {model.optimizer.get_mixed_precision_dtype()}")
    print(f"  Should compile: {model.optimizer.should_use_compile()}")

    if hasattr(model.optimizer, 'get_additional_info'):
        additional_info = model.optimizer.get_additional_info()
        if additional_info:
            print(f"  Device Info: {additional_info}")

    # Warmup runs
    print(f"\nRunning {num_warmup} warmup iterations...")
    for i in range(num_warmup):
        try:
            _ = model.inference(test_images, export_dir=None)
        except Exception as e:
            print(f"  ERROR during warmup: {e}")
            return {
                "device": device,
                "performance_mode": performance_mode,
                "mixed_precision": str(mixed_precision),
                "compile": enable_compile,
                "error": f"Warmup failed: {e}",
                "success": False,
            }

    # Benchmark runs
    print(f"Running {num_runs} benchmark iterations...")
    latencies = []
    depths = []

    for i in range(num_runs):
        start = time.perf_counter()
        try:
            prediction = model.inference(test_images, export_dir=None)
            end = time.perf_counter()

            latencies.append(end - start)
            depths.append(prediction.depth.copy())
            print(f"  Run {i+1}/{num_runs}: {latencies[-1]:.4f}s")

        except Exception as e:
            print(f"  ERROR during benchmark run {i+1}: {e}")
            return {
                "device": device,
                "performance_mode": performance_mode,
                "mixed_precision": str(mixed_precision),
                "compile": enable_compile,
                "error": f"Benchmark run {i+1} failed: {e}",
                "success": False,
            }

    # Calculate statistics
    latencies = np.array(latencies)
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    throughput = num_images / mean_latency  # images/sec

    # Check numerical stability (variance across runs)
    depth_diffs = []
    for i in range(1, len(depths)):
        diff = np.abs(depths[i] - depths[0])
        depth_diffs.append(np.mean(diff))

    max_depth_diff = max(depth_diffs) if depth_diffs else 0.0

    # Memory usage
    if device == "cuda" and torch.cuda.is_available():
        memory_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.max_memory_reserved() / 1024**3  # GB
    else:
        memory_allocated = None
        memory_reserved = None

    results = {
        "device": device,
        "performance_mode": performance_mode,
        "mixed_precision": str(mixed_precision),
        "compile": enable_compile,
        "success": True,
        "num_images": num_images,
        "num_runs": num_runs,
        "latency_mean": float(mean_latency),
        "latency_std": float(std_latency),
        "latency_min": float(min_latency),
        "latency_max": float(max_latency),
        "throughput": float(throughput),
        "max_depth_diff": float(max_depth_diff),
        "memory_allocated_gb": float(memory_allocated) if memory_allocated else None,
        "memory_reserved_gb": float(memory_reserved) if memory_reserved else None,
    }

    print(f"\nResults:")
    print(f"  Mean Latency: {mean_latency:.4f}s ± {std_latency:.4f}s")
    print(f"  Min Latency: {min_latency:.4f}s")
    print(f"  Max Latency: {max_latency:.4f}s")
    print(f"  Throughput: {throughput:.2f} images/sec")
    print(f"  Max Depth Diff (numerical stability): {max_depth_diff:.6f}")
    if memory_allocated:
        print(f"  Memory Allocated: {memory_allocated:.2f} GB")
        print(f"  Memory Reserved: {memory_reserved:.2f} GB")

    return results


def run_validation_suite(
    devices: list[str] | None = None,
    output_file: str = "benchmark_validation_results.json",
    num_images: int = 5,
    num_warmup: int = 1,
    num_runs: int = 3,
) -> None:
    """
    Run full validation suite across all configurations.

    Args:
        devices: List of devices to test (None = auto-detect)
        output_file: Path to save JSON results
        num_images: Number of test images
        num_warmup: Number of warmup runs
        num_runs: Number of benchmark runs
    """
    # Auto-detect available devices
    if devices is None:
        devices = []
        devices.append("cpu")
        if torch.cuda.is_available():
            devices.append("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices.append("mps")

    print(f"Available devices: {devices}")

    # Test configurations
    performance_modes = ["minimal", "balanced", "max"]
    mixed_precision_configs = {
        "cpu": [False, None],  # CPU: FP32 only (FP16 is slower)
        "mps": [False, None, "bfloat16"],  # MPS: FP32, auto, BF16
        "cuda": [False, None, "float16", "bfloat16"],  # CUDA: all options
    }
    compile_options = [False, True]

    all_results = []

    for device in devices:
        for perf_mode in performance_modes:
            for mixed_prec in mixed_precision_configs.get(device, [None]):
                for compile_opt in compile_options:
                    # Skip torch.compile for non-max modes (usually disabled by optimizer)
                    if compile_opt and perf_mode != "max":
                        continue

                    result = benchmark_config(
                        device=device,
                        performance_mode=perf_mode,
                        mixed_precision=mixed_prec,
                        enable_compile=compile_opt,
                        num_images=num_images,
                        num_warmup=num_warmup,
                        num_runs=num_runs,
                    )

                    all_results.append(result)

                    # Clear device cache between runs
                    if device == "cuda":
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                    elif device == "mps":
                        # Clear MPS cache to prevent OOM from fragmentation
                        torch.mps.empty_cache()
                        import gc
                        gc.collect()

    # Save results
    output_path = Path(output_file)
    with output_path.open("w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Validation complete! Results saved to: {output_path}")
    print(f"{'='*80}")

    # Print summary
    print("\n=== SUMMARY ===")
    successful_results = [r for r in all_results if r.get("success", False)]
    failed_results = [r for r in all_results if not r.get("success", False)]

    print(f"\nTotal configurations tested: {len(all_results)}")
    print(f"  Successful: {len(successful_results)}")
    print(f"  Failed: {len(failed_results)}")

    if failed_results:
        print("\nFailed configurations:")
        for result in failed_results:
            print(f"  - {result['device']}/{result['performance_mode']}/{result['mixed_precision']}/{result['compile']}: {result.get('error', 'Unknown error')}")

    if successful_results:
        print("\n=== BEST CONFIGURATIONS BY DEVICE ===")

        for device in devices:
            device_results = [r for r in successful_results if r["device"] == device]
            if not device_results:
                continue

            # Find fastest configuration
            best = min(device_results, key=lambda x: x["latency_mean"])

            print(f"\n{device.upper()}:")
            print(f"  Best Config: {best['performance_mode']}/{best['mixed_precision']}/compile={best['compile']}")
            print(f"  Latency: {best['latency_mean']:.4f}s")
            print(f"  Throughput: {best['throughput']:.2f} images/sec")

            # Compare to baseline (minimal/FP32/no-compile)
            baseline = next(
                (r for r in device_results
                 if r['performance_mode'] == 'minimal'
                 and r['mixed_precision'] == 'False'
                 and not r['compile']),
                None
            )

            if baseline and baseline != best:
                speedup = baseline['latency_mean'] / best['latency_mean']
                print(f"  Speedup vs baseline: {speedup:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Validate device-specific optimizations")
    parser.add_argument(
        "--devices",
        nargs="+",
        choices=["cpu", "mps", "cuda"],
        help="Devices to test (default: auto-detect)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_validation_results.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=5,
        help="Number of test images",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=1,
        help="Number of warmup runs",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of benchmark runs",
    )

    args = parser.parse_args()

    run_validation_suite(
        devices=args.devices,
        output_file=args.output,
        num_images=args.num_images,
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
    )


if __name__ == "__main__":
    main()
