#!/usr/bin/env python3
"""
Performance benchmarking script for Depth Anything 3.

Compares different optimization configurations to measure performance improvements.
"""

import argparse
import time
import torch
import numpy as np
from pathlib import Path
from PIL import Image

from depth_anything_3.api import DepthAnything3


def create_dummy_images(num_images: int = 5, size: tuple = (504, 280)) -> list:
    """Create dummy images for benchmarking."""
    images = []
    for i in range(num_images):
        img = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
        images.append(Image.fromarray(img))
    return images


def benchmark_config(
    model_name: str,
    enable_compile: bool,
    compile_mode: str,
    device: str, # Added device parameter
    num_images: int,
    num_runs: int = 3
) -> dict:
    """Benchmark a specific configuration."""
    print(f"\n{'='*60}")
    print(f"Configuration:")
    print(f"  - Model: {model_name}")
    print(f"  - Compile enabled: {enable_compile}")
    print(f"  - Compile mode: {compile_mode if enable_compile else 'N/A'}")
    print(f"  - Device: {device}") # Added device to print statement
    print(f"  - Number of images: {num_images}")
    print(f"  - Number of runs: {num_runs}")
    print(f"{'='*60}\n")

    # Initialize model
    print("Initializing model...")
    model = DepthAnything3(
        model_name=model_name,
        enable_compile=enable_compile,
        compile_mode=compile_mode
    )

    print(f"Moving model to {device}...")
    model = model.to(device)

    # Create dummy images
    print(f"Creating {num_images} dummy images...")
    images = create_dummy_images(num_images)

    # Warmup run
    print("\nWarming up (1 run)...")
    _ = model.inference(images, export_dir=None)

    # Benchmark runs
    print(f"\nRunning benchmark ({num_runs} runs)...")
    times = []
    for i in range(num_runs):
        start = time.time()
        prediction = model.inference(images, export_dir=None)
        end = time.time()
        elapsed = end - start
        times.append(elapsed)
        print(f"  Run {i+1}/{num_runs}: {elapsed:.3f}s")

    # Calculate statistics
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)

    results = {
        "model_name": model_name,
        "enable_compile": enable_compile,
        "compile_mode": compile_mode if enable_compile else "N/A",
        "num_images": num_images,
        "device": device,
        "mean_time": mean_time,
        "std_time": std_time,
        "min_time": min_time,
        "max_time": max_time,
        "times": times
    }

    print(f"\nResults:")
    print(f"  Mean: {mean_time:.3f}s ± {std_time:.3f}s")
    print(f"  Min:  {min_time:.3f}s")
    print(f"  Max:  {max_time:.3f}s")
    print(f"  Images/sec: {num_images/mean_time:.2f}")

    return results


def compare_configurations(model_name: str, target_device: str, num_images: int = 5, num_runs: int = 3):
    """Compare different optimization configurations."""

    configurations = [
        {"enable_compile": False, "compile_mode": "default"},
        {"enable_compile": True, "compile_mode": "default"},
        {"enable_compile": True, "compile_mode": "reduce-overhead"},
    ]

    # Only include max-autotune if not MPS (requires Triton)
    if target_device != "mps":
        configurations.append({"enable_compile": True, "compile_mode": "max-autotune"})

    print("\n" + "="*60)
    print("DEPTH ANYTHING 3 - PERFORMANCE BENCHMARK")
    print("="*60)

    all_results = []
    for config in configurations:
        results = benchmark_config(
            model_name=model_name,
            enable_compile=config["enable_compile"],
            compile_mode=config["compile_mode"],
            device=target_device, # Pass the target_device
            num_images=num_images,
            num_runs=num_runs
        )
        all_results.append(results)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY - Performance Comparison")
    print("="*60)
    print(f"\n{'Configuration':<40} {'Mean Time':<15} {'Speedup':<10}")
    print("-" * 65)

    baseline_time = all_results[0]["mean_time"]
    for result in all_results:
        config_name = f"compile={result['enable_compile']}, mode={result['compile_mode']}"
        mean_time = result["mean_time"]
        speedup = baseline_time / mean_time
        print(f"{config_name:<40} {mean_time:.3f}s          {speedup:.2f}x")

    print("\n" + "="*60)
    print(f"Device: {all_results[0]['device']}")
    print(f"Best configuration: {min(all_results, key=lambda x: x['mean_time'])['compile_mode']}")
    print(f"Maximum speedup: {max(baseline_time / r['mean_time'] for r in all_results):.2f}x")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Depth Anything 3 performance with different optimizations"
    )
    parser.add_argument(
        "--model-name",
        default="da3-small",
        help="Model name to benchmark (default: da3-small)"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=5,
        help="Number of images to process (default: 5)"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of benchmark runs (default: 3)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run benchmark on (cpu, cuda, mps)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all optimization configurations (ignores --disable-compile and --compile-mode for orchestration)"
    )
    parser.add_argument(
        "--disable-compile",
        action="store_true",
        help="Disable torch.compile() (enabled by default on CUDA, disabled on MPS/CPU)"
    )
    parser.add_argument(
        "--compile-mode",
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile() mode (default: reduce-overhead)"
    )

    args = parser.parse_args()

    # Determine available devices
    available_devices = ["cpu"]
    if torch.cuda.is_available():
        available_devices.append("cuda")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        available_devices.append("mps")

    if args.compare:
        print("\n" + "="*80)
        print("RUNNING ALL COMPARATIVE BENCHMARKS ACROSS DEVICES AND OPTIMIZATIONS")
        print("="*80)
        all_benchmark_results = []

        for device in available_devices:
            print(f"\n{'#'*70}")
            print(f"BENCHMARKING FOR DEVICE: {device.upper()}")
            print(f"{'#'*70}\n")

            # For each device, run both compiled (optimized) and non-compiled (vanilla) benchmarks
            # We'll run 'reduce-overhead' for compiled as a representative optimized mode
            # max-autotune is skipped for MPS internally in compare_configurations
            results_for_device = compare_configurations(
                model_name=args.model_name,
                target_device=device,
                num_images=args.num_images,
                num_runs=args.num_runs
            )
            all_benchmark_results.extend(results_for_device)

        # Output all results to a temporary file, which the orchestration script will read
        with open(f"/Users/aedelon/.gemini/tmp/da19938fc03f6f915b1d49e2da3d55ebaf0e6c266577cb70344defeca9399abc/benchmark_performance_results.txt", "w") as f:
            for res in all_benchmark_results:
                f.write(str(res) + "\n")

    else:
        # Single benchmark run based on provided arguments
        # If enable_compile is not explicitly set, let DepthAnything3 auto-detect
        enable_compile = not args.disable_compile if args.disable_compile else None

        benchmark_config(
            model_name=args.model_name,
            enable_compile=enable_compile,
            compile_mode=args.compile_mode,
            device=args.device,
            num_images=args.num_images,
            num_runs=args.num_runs
        )
