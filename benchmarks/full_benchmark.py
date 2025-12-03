#!/usr/bin/env python3
# Copyright (c) 2025 Delanoe Pirard / Aedelon - Apache 2.0
"""
Full Benchmark Suite for Depth Anything 3

Tests ALL optimization combinations for each device (CPU, MPS, CUDA).

Optimizations tested:
- Preprocessing: CPU (PIL) vs GPU (NVJPEG on CUDA)
- Attention: SDPA (Flash Attention) vs Manual

Usage:
    python benchmarks/full_benchmark.py              # Best device only
    python benchmarks/full_benchmark.py -d all       # All devices
    python benchmarks/full_benchmark.py -d cuda      # CUDA only
    python benchmarks/full_benchmark.py --quick      # Quick mode
"""

import argparse
import gc
import logging
import os
import shutil
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

# Suppress ALL logging before any imports
logging.disable(logging.CRITICAL)
os.environ["DA3_LOG_LEVEL"] = "ERROR"
warnings.filterwarnings("ignore")

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Suppress depth_anything_3 logger specifically
logging.getLogger("depth_anything_3").disabled = True
logging.getLogger("dinov2").disabled = True


# ============================================================================
# STYLES
# ============================================================================

class Style:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def colored(text, color, bold=False):
    prefix = Style.BOLD if bold else ""
    return f"{prefix}{color}{text}{Style.RESET}"


# ============================================================================
# UTILITIES
# ============================================================================

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def get_available_devices() -> List[torch.device]:
    """Get all available devices for benchmarking."""
    devices = [torch.device("cpu")]
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    return devices


def get_device_name(device: torch.device) -> str:
    """Get human-readable device name."""
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    elif device.type == "mps":
        return "Apple Silicon (MPS)"
    else:
        import platform
        return f"CPU ({platform.processor() or 'Unknown'})"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    mean_ms: float
    std_ms: float
    fps: float

    @classmethod
    def from_times(cls, times: List[float], batch_size: int = 1):
        mean_ms = np.mean(times)
        std_ms = np.std(times)
        fps = 1000 / mean_ms * batch_size
        return cls(mean_ms=mean_ms, std_ms=std_ms, fps=fps)


@dataclass
class OptimizationConfig:
    """Configuration for a specific optimization combination."""
    name: str
    preprocessing: str  # "cpu" or "gpu"
    attention: str      # "sdpa" or "manual"
    description: str

    @property
    def short_name(self) -> str:
        prep = "GPU" if self.preprocessing == "gpu" else "CPU"
        attn = "SDPA" if self.attention == "sdpa" else "Manual"
        return f"{prep}+{attn}"


# ============================================================================
# BENCHMARK FUNCTIONS
# ============================================================================

def get_optimization_configs(device: torch.device) -> List[OptimizationConfig]:
    """Get all valid optimization configurations for a device."""
    configs = []

    if device.type == "cuda":
        # CUDA: All 4 combinations
        configs = [
            OptimizationConfig("gpu_sdpa", "gpu", "sdpa", "GPU Decode (NVJPEG) + SDPA (Flash)"),
            OptimizationConfig("gpu_manual", "gpu", "manual", "GPU Decode (NVJPEG) + Manual Attn"),
            OptimizationConfig("cpu_sdpa", "cpu", "sdpa", "CPU Decode (PIL) + SDPA (Flash)"),
            OptimizationConfig("cpu_manual", "cpu", "manual", "CPU Decode (PIL) + Manual Attn"),
        ]
    elif device.type == "mps":
        # MPS: CPU preprocessing is better, 2 combinations
        configs = [
            OptimizationConfig("cpu_sdpa", "cpu", "sdpa", "CPU Decode (PIL) + SDPA"),
            OptimizationConfig("cpu_manual", "cpu", "manual", "CPU Decode (PIL) + Manual Attn"),
        ]
    else:
        # CPU: 2 combinations
        configs = [
            OptimizationConfig("cpu_sdpa", "cpu", "sdpa", "SDPA Attention"),
            OptimizationConfig("cpu_manual", "cpu", "manual", "Manual Attention"),
        ]

    return configs


def benchmark_preprocessing_detailed(device: torch.device, runs: int = 5) -> Dict:
    """Benchmark preprocessing in detail."""
    from depth_anything_3.utils.io.input_processor import InputProcessor
    from depth_anything_3.utils.io.gpu_input_processor import GPUInputProcessor

    results = {}
    temp_dir = "temp_bench_preproc"

    sizes = [
        ("720p", 1280, 720),
        ("1080p", 1920, 1080),
        ("4K", 3840, 2160),
    ]

    os.makedirs(temp_dir, exist_ok=True)

    try:
        cpu_proc = InputProcessor()
        gpu_proc = None
        if device.type == "cuda":
            gpu_proc = GPUInputProcessor(device=device)

        for name, w, h in sizes:
            results[name] = {}

            # Create test files
            files = []
            pil_imgs = []
            for i in range(4):
                img = Image.new("RGB", (w, h), color=(100 + i*10, 150, 200))
                fpath = f"{temp_dir}/{name}_{i}.jpg"
                img.save(fpath, quality=95)
                files.append(fpath)
                pil_imgs.append(img.copy())

            # CPU benchmark
            cleanup()
            for _ in range(2):
                cpu_proc(image=pil_imgs, process_res=518, num_workers=8)

            times = []
            for _ in range(runs):
                start = time.perf_counter()
                cpu_proc(image=pil_imgs, process_res=518, num_workers=8)
                times.append((time.perf_counter() - start) * 1000)
            results[name]["cpu"] = BenchmarkResult.from_times(times, batch_size=4)

            # GPU benchmark (NVJPEG for CUDA)
            if gpu_proc and gpu_proc.use_gpu:
                cleanup()
                for _ in range(2):
                    gpu_proc(image=files, process_res=518, num_workers=1)
                sync_device(device)

                times = []
                for _ in range(runs):
                    sync_device(device)
                    start = time.perf_counter()
                    gpu_proc(image=files, process_res=518, num_workers=1)
                    sync_device(device)
                    times.append((time.perf_counter() - start) * 1000)
                results[name]["gpu"] = BenchmarkResult.from_times(times, batch_size=4)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return results


def benchmark_attention_detailed(device: torch.device, runs: int = 10) -> Dict:
    """Benchmark attention backends in detail."""
    from depth_anything_3.model.dinov2.layers import Attention

    results = {}
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    configs = [
        ("ViT-S (518px)", 384, 6, 529),
        ("ViT-L (518px)", 1024, 16, 529),
        ("ViT-L (770px)", 1024, 16, 1156),
    ]

    for name, dim, heads, seq_len in configs:
        results[name] = {}
        x = torch.randn(1, seq_len, dim, device=device, dtype=dtype)

        for backend in ["sdpa", "manual"]:
            cleanup()
            attn = Attention(dim=dim, num_heads=heads, attn_backend=backend).to(device, dtype)
            attn.eval()

            # Warmup
            with torch.no_grad():
                for _ in range(3):
                    attn(x)
            sync_device(device)

            # Benchmark
            times = []
            with torch.no_grad():
                for _ in range(runs):
                    sync_device(device)
                    start = time.perf_counter()
                    attn(x)
                    sync_device(device)
                    times.append((time.perf_counter() - start) * 1000)

            results[name][backend] = BenchmarkResult.from_times(times)
            del attn

    return results


def benchmark_inference_matrix(
    device: torch.device,
    models: List[str],
    runs: int = 3,
) -> Dict:
    """Benchmark all optimization combinations for inference."""
    from depth_anything_3.api import DepthAnything3

    results = {}
    temp_dir = "temp_bench_infer"
    configs = get_optimization_configs(device)

    os.makedirs(temp_dir, exist_ok=True)

    # Create test images (720p)
    img_paths = []
    pil_imgs = []
    for i in range(4):
        img = Image.new("RGB", (1280, 720), color=(100 + i*20, 150, 200))
        path = f"{temp_dir}/test_{i}.jpg"
        img.save(path, quality=95)
        img_paths.append(path)
        pil_imgs.append(img.copy())

    try:
        for model_name in models:
            results[model_name] = {}

            for config in configs:
                cleanup()

                # Set attention backend
                os.environ["DA3_ATTENTION_BACKEND"] = config.attention

                # Load model fresh (to apply attention backend)
                model = DepthAnything3(
                    model_name=model_name,
                    device=device,
                    use_cache=False,
                )

                # Choose input based on preprocessing
                if config.preprocessing == "gpu" and device.type == "cuda":
                    test_input = img_paths[:1]  # File paths for NVJPEG
                else:
                    test_input = pil_imgs[:1]   # PIL for CPU preprocessing

                # Warmup
                for _ in range(3):
                    model.inference(test_input, process_res=518)
                sync_device(device)

                # Benchmark
                times = []
                for _ in range(runs):
                    sync_device(device)
                    start = time.perf_counter()
                    model.inference(test_input, process_res=518)
                    sync_device(device)
                    times.append((time.perf_counter() - start) * 1000)

                results[model_name][config.name] = {
                    "result": BenchmarkResult.from_times(times, batch_size=1),
                    "config": config,
                }

                del model
                cleanup()

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return results


# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def print_header(title: str):
    """Print section header."""
    print()
    print(colored("═" * 70, Style.CYAN))
    print(colored("║", Style.CYAN) + colored(f" {title}", Style.BOLD).center(77) + colored("║", Style.CYAN))
    print(colored("═" * 70, Style.CYAN))


def print_subheader(title: str):
    """Print subsection header."""
    print()
    print(colored(f"▶ {title}", Style.YELLOW, bold=True))
    print(colored("─" * 70, Style.DIM))


def format_speedup(speedup: float) -> str:
    """Format speedup with color."""
    if speedup >= 1.5:
        return colored(f"{speedup:.2f}x", Style.GREEN, bold=True)
    elif speedup >= 1.1:
        return colored(f"{speedup:.2f}x", Style.GREEN)
    elif speedup >= 0.95:
        return f"{speedup:.2f}x"
    else:
        return colored(f"{speedup:.2f}x", Style.RED)


def print_preprocessing_results(results: Dict, device: torch.device):
    """Print preprocessing benchmark results."""
    print_subheader("PREPROCESSING (4 images batch)")

    has_gpu = any("gpu" in r for r in results.values())

    if has_gpu:
        print(f"  {'Resolution':<12} {'CPU (PIL)':<14} {'GPU (NVJPEG)':<14} {'Speedup':<10}")
        print(f"  {'-'*50}")

        for name, data in results.items():
            cpu_ms = data["cpu"].mean_ms
            if "gpu" in data:
                gpu_ms = data["gpu"].mean_ms
                speedup = cpu_ms / gpu_ms
                print(f"  {name:<12} {cpu_ms:>8.1f} ms    {gpu_ms:>8.1f} ms    {format_speedup(speedup)}")
            else:
                print(f"  {name:<12} {cpu_ms:>8.1f} ms    {'N/A':<14}")
    else:
        print(f"  {'Resolution':<12} {'CPU (PIL)':<14}")
        print(f"  {'-'*30}")
        for name, data in results.items():
            cpu_ms = data["cpu"].mean_ms
            print(f"  {name:<12} {cpu_ms:>8.1f} ms")

    # Summary
    if has_gpu:
        speedups = []
        for data in results.values():
            if "gpu" in data:
                speedups.append(data["cpu"].mean_ms / data["gpu"].mean_ms)
        if speedups:
            avg = np.mean(speedups)
            print()
            print(f"  {colored('→', Style.GREEN)} GPU preprocessing avg {colored(f'{avg:.1f}x', Style.GREEN, bold=True)} faster")


def print_attention_results(results: Dict, device: torch.device):
    """Print attention benchmark results."""
    print_subheader("ATTENTION (per layer forward pass)")

    print(f"  {'Config':<18} {'SDPA':<12} {'Manual':<12} {'Speedup':<10}")
    print(f"  {'-'*52}")

    for name, data in results.items():
        sdpa_ms = data["sdpa"].mean_ms
        manual_ms = data["manual"].mean_ms
        speedup = manual_ms / sdpa_ms
        print(f"  {name:<18} {sdpa_ms:>6.3f} ms    {manual_ms:>6.3f} ms    {format_speedup(speedup)}")

    # Summary
    speedups = [d["manual"].mean_ms / d["sdpa"].mean_ms for d in results.values()]
    avg = np.mean(speedups)
    print()
    print(f"  {colored('→', Style.GREEN)} SDPA avg {colored(f'{avg:.1f}x', Style.GREEN, bold=True)} faster than manual")

    # Check Flash SDP
    if device.type == "cuda":
        from torch.backends.cuda import flash_sdp_enabled
        if flash_sdp_enabled():
            print(f"  {colored('→', Style.GREEN)} Flash Attention: {colored('ENABLED', Style.GREEN, bold=True)} (PyTorch native)")


def print_inference_matrix(results: Dict, device: torch.device):
    """Print inference benchmark matrix."""
    print_subheader("END-TO-END INFERENCE (720p input, batch=1)")

    configs = get_optimization_configs(device)

    # Header
    header = f"  {'Model':<12}"
    for cfg in configs:
        header += f" {cfg.short_name:<14}"
    header += " Best"
    print(header)
    print(f"  {'-'*(14 + 15*len(configs) + 6)}")

    # Results per model
    for model_name, model_results in results.items():
        row = f"  {model_name:<12}"

        best_fps = 0
        best_config = None
        worst_fps = float('inf')

        for cfg in configs:
            if cfg.name in model_results:
                result = model_results[cfg.name]["result"]
                fps = result.fps
                row += f" {fps:>6.1f} img/s  "

                if fps > best_fps:
                    best_fps = fps
                    best_config = cfg
                if fps < worst_fps:
                    worst_fps = fps
            else:
                row += f" {'N/A':<14}"

        # Best indicator
        if best_config:
            row += f" {colored(best_config.short_name, Style.GREEN, bold=True)}"

        print(row)

    # Summary
    print()
    print(f"  {Style.DIM}Legend: GPU=NVJPEG decode, CPU=PIL decode, SDPA=Flash Attention{Style.RESET}")


def print_device_summary(
    device: torch.device,
    preproc_results: Dict,
    attn_results: Dict,
    infer_results: Dict,
):
    """Print summary for a device."""
    print()
    print(colored("─" * 70, Style.CYAN))
    print(colored(f" {device.type.upper()} - OPTIMIZATION SUMMARY", Style.BOLD))
    print(colored("─" * 70, Style.CYAN))

    # Best configuration
    if infer_results:
        print()
        print(f"  {colored('Best configuration per model:', Style.CYAN)}")

        for model_name, model_results in infer_results.items():
            if not model_results:
                continue

            best_name = max(model_results.keys(), key=lambda k: model_results[k]["result"].fps)
            best = model_results[best_name]
            worst_name = min(model_results.keys(), key=lambda k: model_results[k]["result"].fps)
            worst = model_results[worst_name]

            speedup = best["result"].fps / worst["result"].fps if worst["result"].fps > 0 else 1

            print(f"    {model_name:<12} {colored(best['config'].description, Style.GREEN)}")
            print(f"    {'':<12} {best['result'].fps:.1f} img/s ({speedup:.1f}x vs worst)")

    # Recommendations
    print()
    print(f"  {colored('Recommendations:', Style.CYAN)}")

    if device.type == "cuda":
        print(f"    ✓ Use {colored('GPU preprocessing (NVJPEG)', Style.GREEN)} for file inputs")
        print(f"    ✓ {colored('SDPA (Flash Attention)', Style.GREEN)} is enabled by default")
        print(f"    ✓ Pass file paths (not PIL images) to leverage NVJPEG")
    elif device.type == "mps":
        print(f"    ✓ Use {colored('CPU preprocessing', Style.GREEN)} (faster than GPU on MPS)")
        print(f"    ✓ {colored('SDPA', Style.GREEN)} provides moderate speedup")
    else:
        print(f"    ✓ {colored('SDPA', Style.GREEN)} provides speedup over manual attention")
        print(f"    ○ Consider using GPU (CUDA/MPS) for better performance")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DA3 Full Benchmark - Test all optimization combinations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmarks/full_benchmark.py              # Best device only
  python benchmarks/full_benchmark.py -d all       # All devices
  python benchmarks/full_benchmark.py -d cuda      # CUDA only
  python benchmarks/full_benchmark.py --quick      # Quick mode (fewer runs)
  python benchmarks/full_benchmark.py --models da3-small da3-large
        """
    )
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer runs)")
    parser.add_argument("--skip-preprocessing", action="store_true", help="Skip preprocessing benchmark")
    parser.add_argument("--skip-attention", action="store_true", help="Skip attention benchmark")
    parser.add_argument("--skip-inference", action="store_true", help="Skip inference benchmark")
    parser.add_argument("-d", "--device", type=str, default="auto",
                       choices=["auto", "cpu", "mps", "cuda", "all"],
                       help="Device to benchmark (default: auto)")
    parser.add_argument("--models", nargs="+", default=None,
                       help="Models to benchmark (default: all)")
    args = parser.parse_args()

    # Configure runs
    runs_preproc = 3 if args.quick else 5
    runs_attn = 5 if args.quick else 10
    runs_infer = 2 if args.quick else 4

    # Determine models
    if args.models:
        models = args.models
    elif args.quick:
        models = ["da3-small", "da3-large"]
    else:
        models = ["da3-small", "da3-base", "da3-large"]

    # Determine devices
    available_devices = get_available_devices()
    if args.device == "auto":
        devices_to_test = [available_devices[-1]]  # Best available
    elif args.device == "all":
        devices_to_test = available_devices
    else:
        requested = torch.device(args.device)
        if requested in available_devices:
            devices_to_test = [requested]
        else:
            print(f"Error: Device '{args.device}' not available.")
            print(f"Available: {[d.type for d in available_devices]}")
            return

    # Main header
    print()
    print(colored("╔" + "═" * 68 + "╗", Style.CYAN))
    print(colored("║", Style.CYAN) + colored(" DEPTH ANYTHING 3 - FULL BENCHMARK", Style.BOLD).center(77) + colored("║", Style.CYAN))
    print(colored("║", Style.CYAN) + colored(" All Optimization Combinations", Style.DIM).center(77) + colored("║", Style.CYAN))
    print(colored("╚" + "═" * 68 + "╝", Style.CYAN))

    print(f"\n  {Style.DIM}PyTorch{Style.RESET}  : {colored(torch.__version__, Style.CYAN)}")
    print(f"  {Style.DIM}Models{Style.RESET}   : {colored(', '.join(models), Style.CYAN)}")
    print(f"  {Style.DIM}Mode{Style.RESET}     : {colored('Quick' if args.quick else 'Full', Style.CYAN)}")

    print(f"\n  {Style.DIM}Available devices:{Style.RESET}")
    for d in available_devices:
        status = colored("●", Style.GREEN) if d in devices_to_test else colored("○", Style.DIM)
        print(f"    {status} {d.type.upper():<6} {get_device_name(d)}")

    all_results = {}

    # Run benchmarks for each device
    for device in devices_to_test:
        device_name = get_device_name(device)
        all_results[device.type] = {}

        print_header(f"{device.type.upper()} - {device_name}")

        # 1. Preprocessing
        preproc_results = {}
        if not args.skip_preprocessing and device.type != "cpu":
            preproc_results = benchmark_preprocessing_detailed(device, runs=runs_preproc)
            all_results[device.type]["preprocessing"] = preproc_results
            print_preprocessing_results(preproc_results, device)
        elif device.type == "cpu":
            print_subheader("PREPROCESSING")
            print(f"  {Style.DIM}Skipped (CPU only - no GPU comparison){Style.RESET}")

        # 2. Attention
        attn_results = {}
        if not args.skip_attention:
            attn_results = benchmark_attention_detailed(device, runs=runs_attn)
            all_results[device.type]["attention"] = attn_results
            print_attention_results(attn_results, device)

        # 3. Inference Matrix
        infer_results = {}
        if not args.skip_inference:
            infer_results = benchmark_inference_matrix(device, models, runs=runs_infer)
            all_results[device.type]["inference"] = infer_results
            print_inference_matrix(infer_results, device)

        # Device Summary
        print_device_summary(device, preproc_results, attn_results, infer_results)

        cleanup()

    # Cross-device comparison
    if len(devices_to_test) > 1 and not args.skip_inference:
        print_header("CROSS-DEVICE COMPARISON")

        # Find common model
        common_model = models[-1]  # Usually largest tested

        print()
        print(f"  {colored(f'{common_model} (best config per device):', Style.CYAN)}")
        print(f"  {'Device':<10} {'Config':<30} {'Performance':<15}")
        print(f"  {'-'*55}")

        base_fps = None
        for device in devices_to_test:
            if device.type in all_results and "inference" in all_results[device.type]:
                infer = all_results[device.type]["inference"].get(common_model, {})
                if infer:
                    best_name = max(infer.keys(), key=lambda k: infer[k]["result"].fps)
                    best = infer[best_name]
                    fps = best["result"].fps

                    if base_fps is None:
                        base_fps = fps

                    speedup = fps / base_fps if base_fps else 1
                    speedup_str = f"({speedup:.1f}x)" if device != devices_to_test[0] else "(baseline)"

                    print(f"  {device.type.upper():<10} {best['config'].description:<30} {fps:>5.1f} img/s {speedup_str}")

    # Final summary
    print()
    print(colored("═" * 70, Style.CYAN))
    print(colored("║", Style.CYAN) + colored(" BENCHMARK COMPLETE", Style.BOLD).center(77) + colored("║", Style.CYAN))
    print(colored("═" * 70, Style.CYAN))
    print()


if __name__ == "__main__":
    main()
