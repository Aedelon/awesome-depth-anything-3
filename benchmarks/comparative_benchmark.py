#!/usr/bin/env python3
# Copyright (c) Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0
"""
Comparative Benchmark: awesome-depth-anything-3 vs upstream (vanilla)

Compares performance between the optimized fork and the original upstream.

Usage:
    python benchmarks/comparative_benchmark.py --device mps
    python benchmarks/comparative_benchmark.py --device cuda
    python benchmarks/comparative_benchmark.py --device all
    python benchmarks/comparative_benchmark.py --quick
"""

import argparse
import contextlib
import gc
import io
import logging
import os
import shutil
import sys
import time
import warnings

# Suppress ALL logging before any imports
logging.disable(logging.CRITICAL)
os.environ["DA3_LOG_LEVEL"] = "CRITICAL"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import numpy as np
import torch
from PIL import Image

# Suppress all loggers
logging.getLogger("depth_anything_3").disabled = True
logging.getLogger("dinov2").disabled = True
logging.getLogger().setLevel(logging.CRITICAL)


@contextlib.contextmanager
def suppress_output():
    """Context manager to suppress stdout and stderr."""
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        # Also suppress all loggers again
        logging.disable(logging.CRITICAL)
        yield

# ============================================================================
# CONFIGURATION
# ============================================================================

AWESOME_REPO = "/Users/aedelon/Workspace/awesome-depth-anything-3"
UPSTREAM_REPO = "/Users/aedelon/Workspace/depth-anything-3-upstream"
MODEL_NAME = "da3-large"


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


def clear_modules():
    """Clear depth_anything_3 from sys.modules."""
    to_remove = [k for k in sys.modules.keys() if "depth_anything_3" in k]
    for k in to_remove:
        del sys.modules[k]


def suppress_logging():
    """Suppress all logging after module import."""
    logging.disable(logging.CRITICAL)
    try:
        from depth_anything_3.utils.logger import logger
        logger.level = 100
    except:
        pass


def get_available_devices():
    """Get available devices."""
    devices = [torch.device("cpu")]
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    return devices


def get_device_name(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    elif device.type == "mps":
        return "Apple Silicon (MPS)"
    return "CPU"


# ============================================================================
# BENCHMARK: UPSTREAM (VANILLA)
# ============================================================================

def benchmark_upstream(device, pil_images, process_res=504, runs=3):
    """Benchmark upstream/vanilla depth-anything-3."""

    # Setup path
    clear_modules()
    upstream_src = os.path.join(UPSTREAM_REPO, "src")
    if upstream_src in sys.path:
        sys.path.remove(upstream_src)
    sys.path.insert(0, upstream_src)

    with suppress_output():
        from depth_anything_3.api import DepthAnything3
        suppress_logging()

        cleanup()

        # Cold load
        start = time.perf_counter()
        model = DepthAnything3(model_name=MODEL_NAME)
        model = model.to(device)
        model.eval()
        cold_load_time = time.perf_counter() - start

        # Warmup
        for _ in range(2):
            model.inference(pil_images[:1], process_res=process_res)
        sync_device(device)
        cleanup()

        # Benchmark inference
        times = []
        for _ in range(runs):
            cleanup()
            sync_device(device)
            start = time.perf_counter()
            model.inference(pil_images, process_res=process_res)
            sync_device(device)
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times)
        std_time = np.std(times)
        throughput = len(pil_images) / avg_time

        del model
        cleanup()

    # Cleanup path
    sys.path.remove(upstream_src)
    clear_modules()

    return {
        "cold_load": cold_load_time,
        "inference_time": avg_time,
        "inference_std": std_time,
        "throughput": throughput,
    }


# ============================================================================
# BENCHMARK: AWESOME (OPTIMIZED)
# ============================================================================

def benchmark_awesome(device, pil_images, process_res=504, runs=3, use_cache=True):
    """Benchmark awesome (optimized) depth-anything-3."""

    # Setup path
    clear_modules()
    awesome_src = os.path.join(AWESOME_REPO, "src")
    if awesome_src in sys.path:
        sys.path.remove(awesome_src)
    sys.path.insert(0, awesome_src)

    with suppress_output():
        from depth_anything_3.api import DepthAnything3
        from depth_anything_3.cache import get_model_cache
        suppress_logging()

        # Clear cache if testing cold load
        if not use_cache:
            cache = get_model_cache()
            cache.clear()

        cleanup()

        # Cold/warm load
        start = time.perf_counter()
        model = DepthAnything3(model_name=MODEL_NAME, device=device, use_cache=use_cache)
        load_time = time.perf_counter() - start

        # For cache test, do a second load
        cached_load_time = None
        if use_cache:
            del model
            cleanup()
            start = time.perf_counter()
            model = DepthAnything3(model_name=MODEL_NAME, device=device, use_cache=True)
            cached_load_time = time.perf_counter() - start

        # Warmup
        for _ in range(2):
            model.inference(pil_images[:1], process_res=process_res)
        sync_device(device)
        cleanup()

        # Benchmark inference
        times = []
        for _ in range(runs):
            cleanup()
            sync_device(device)
            start = time.perf_counter()
            model.inference(pil_images, process_res=process_res)
            sync_device(device)
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times)
        std_time = np.std(times)
        throughput = len(pil_images) / avg_time

        del model
        cleanup()

    # Cleanup path
    sys.path.remove(awesome_src)
    clear_modules()

    return {
        "cold_load": load_time,
        "cached_load": cached_load_time,
        "inference_time": avg_time,
        "inference_std": std_time,
        "throughput": throughput,
    }


# ============================================================================
# MAIN
# ============================================================================

def run_comparison(device, batch_sizes, process_res=504, runs=3):
    """Run comparison for a specific device."""

    results = {}
    temp_dir = "temp_compare"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Create test images
        max_batch = max(batch_sizes)
        pil_images = []
        for i in range(max_batch):
            img = Image.new("RGB", (1280, 720), color=(100 + i*10, 150, 200))
            pil_images.append(img)

        for batch_size in batch_sizes:
            test_images = pil_images[:batch_size]
            results[batch_size] = {}

            print(f"\n  Batch size: {batch_size}")
            print(f"  {'-'*50}")

            # Upstream
            print(f"  Testing UPSTREAM (vanilla)...", end=" ", flush=True)
            try:
                upstream = benchmark_upstream(device, test_images, process_res, runs)
                results[batch_size]["upstream"] = upstream
                print(f"{upstream['throughput']:.2f} img/s")
            except Exception as e:
                print(f"ERROR: {e}")
                results[batch_size]["upstream"] = None

            # Awesome (no cache - fair comparison)
            print(f"  Testing AWESOME (no cache)...", end=" ", flush=True)
            try:
                awesome_nc = benchmark_awesome(device, test_images, process_res, runs, use_cache=False)
                results[batch_size]["awesome_nocache"] = awesome_nc
                print(f"{awesome_nc['throughput']:.2f} img/s")
            except Exception as e:
                print(f"ERROR: {e}")
                results[batch_size]["awesome_nocache"] = None

            # Awesome (with cache)
            print(f"  Testing AWESOME (cached)...", end=" ", flush=True)
            try:
                awesome_c = benchmark_awesome(device, test_images, process_res, runs, use_cache=True)
                results[batch_size]["awesome_cached"] = awesome_c
                print(f"{awesome_c['throughput']:.2f} img/s")
            except Exception as e:
                print(f"ERROR: {e}")
                results[batch_size]["awesome_cached"] = None

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return results


def print_results_table(results, device):
    """Print formatted results table."""

    print(f"\n{'='*70}")
    print(f" RESULTS: {device.type.upper()}")
    print(f"{'='*70}")

    # Header
    print(f"\n{'Batch':<8} {'Metric':<18} {'Upstream':<12} {'Awesome':<12} {'Speedup':<10}")
    print("-" * 60)

    for batch_size, data in sorted(results.items()):
        upstream = data.get("upstream")
        awesome = data.get("awesome_nocache") or data.get("awesome_cached")

        if not upstream or not awesome:
            continue

        # Inference throughput
        u_thr = upstream["throughput"]
        a_thr = awesome["throughput"]
        speedup = a_thr / u_thr if u_thr > 0 else 0
        print(f"{batch_size:<8} {'Throughput (img/s)':<18} {u_thr:<12.2f} {a_thr:<12.2f} {speedup:<10.2f}x")

        # Inference time
        u_time = upstream["inference_time"] * 1000
        a_time = awesome["inference_time"] * 1000
        speedup = u_time / a_time if a_time > 0 else 0
        print(f"{'':<8} {'Latency (ms)':<18} {u_time:<12.1f} {a_time:<12.1f} {speedup:<10.2f}x")

        # Cold load time
        u_load = upstream["cold_load"]
        a_load = awesome["cold_load"]
        speedup = u_load / a_load if a_load > 0 else 0
        print(f"{'':<8} {'Cold load (s)':<18} {u_load:<12.2f} {a_load:<12.2f} {speedup:<10.2f}x")

        # Cached load (awesome only)
        cached = data.get("awesome_cached")
        if cached and cached.get("cached_load"):
            c_load = cached["cached_load"]
            speedup = u_load / c_load if c_load > 0 else 0
            print(f"{'':<8} {'Cached load (s)':<18} {'-':<12} {c_load:<12.3f} {speedup:<10.1f}x")

        print()


def main():
    parser = argparse.ArgumentParser(description="Comparative Benchmark: Awesome vs Upstream")
    parser.add_argument("--device", "-d", type=str, default="auto",
                       choices=["auto", "cpu", "mps", "cuda", "all"],
                       help="Device to benchmark")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4],
                       help="Batch sizes to test")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per test")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer runs)")
    args = parser.parse_args()

    if args.quick:
        args.batch_sizes = [1, 2]
        args.runs = 2

    # Determine devices
    available = get_available_devices()
    if args.device == "auto":
        devices = [available[-1]]
    elif args.device == "all":
        devices = available
    else:
        requested = torch.device(args.device)
        if requested in available:
            devices = [requested]
        else:
            print(f"Device '{args.device}' not available. Available: {[d.type for d in available]}")
            return

    # Header
    print("\n" + "=" * 70)
    print(" COMPARATIVE BENCHMARK: AWESOME vs UPSTREAM (VANILLA)")
    print("=" * 70)
    print(f" Model: {MODEL_NAME}")
    print(f" PyTorch: {torch.__version__}")
    print(f" Batch sizes: {args.batch_sizes}")
    print(f" Runs per test: {args.runs}")
    print(f" Devices: {[d.type.upper() for d in devices]}")
    for d in available:
        status = "✓" if d in devices else "○"
        print(f"   {status} {d.type.upper()}: {get_device_name(d)}")
    print("=" * 70)

    all_results = {}

    for device in devices:
        print(f"\n{'#'*70}")
        print(f" DEVICE: {device.type.upper()} ({get_device_name(device)})")
        print(f"{'#'*70}")

        results = run_comparison(device, args.batch_sizes, runs=args.runs)
        all_results[device.type] = results
        print_results_table(results, device)

    # Final summary
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)

    for device_type, results in all_results.items():
        print(f"\n {device_type.upper()}:")

        for batch_size, data in sorted(results.items()):
            upstream = data.get("upstream")
            awesome = data.get("awesome_nocache")

            if upstream and awesome:
                speedup = awesome["throughput"] / upstream["throughput"]
                print(f"   Batch {batch_size}: {speedup:.2f}x faster inference")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
