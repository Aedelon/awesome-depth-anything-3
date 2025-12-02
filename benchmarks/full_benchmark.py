#!/usr/bin/env python3
# Copyright (c) 2025 Delanoe Pirard / Aedelon - Apache 2.0
"""
Full Benchmark Suite for Depth Anything 3

Tests:
1. Preprocessing: CPU vs GPU Decode (NVJPEG)
2. Attention: SDPA (Flash) vs Manual
3. End-to-End: All models x All preprocessing methods
4. Adaptive Batching: Auto vs Fixed batch sizes

Usage:
    python benchmarks/full_benchmark.py
    python benchmarks/full_benchmark.py --quick
    python benchmarks/full_benchmark.py --skip-batching
"""

import argparse
import gc
import logging
import os
import shutil
import sys
import time
import warnings

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
# UTILITIES
# ============================================================================

def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


# ============================================================================
# BENCHMARK FUNCTIONS
# ============================================================================

def benchmark_preprocessing(device, runs=5):
    """Benchmark CPU vs GPU preprocessing."""
    from depth_anything_3.utils.io.input_processor import InputProcessor
    from depth_anything_3.utils.io.gpu_input_processor import GPUInputProcessor

    results = {"cpu": {}, "gpu": {}}
    temp_dir = "temp_bench"

    # Setup
    os.makedirs(temp_dir, exist_ok=True)
    cpu_proc = InputProcessor()
    gpu_proc = GPUInputProcessor(device=device) if device.type in ("cuda", "mps") else None

    sizes = [(640, 480), (1920, 1080), (3840, 2160)]

    try:
        for w, h in sizes:
            size_key = f"{w}x{h}"

            # Create test files
            files = []
            for i in range(4):
                img = Image.new("RGB", (w, h), color=(100, 150, 200))
                fpath = f"{temp_dir}/{w}x{h}_{i}.jpg"
                img.save(fpath, quality=95)
                files.append(fpath)

            pil_imgs = [Image.new("RGB", (w, h), color=(100, 150, 200)) for _ in range(4)]

            # CPU benchmark
            cleanup()
            for _ in range(2):  # warmup
                cpu_proc(image=pil_imgs, process_res=504, num_workers=8)

            times = []
            for _ in range(runs):
                start = time.perf_counter()
                cpu_proc(image=pil_imgs, process_res=504, num_workers=8)
                times.append((time.perf_counter() - start) * 1000)
            results["cpu"][size_key] = np.mean(times)

            # GPU benchmark (NVJPEG for CUDA, Kornia for MPS)
            if gpu_proc and gpu_proc.use_gpu:
                cleanup()
                test_input = files if device.type == "cuda" else pil_imgs

                for _ in range(2):  # warmup
                    gpu_proc(image=test_input, process_res=504, num_workers=1)

                times = []
                for _ in range(runs):
                    sync_device(device)
                    start = time.perf_counter()
                    gpu_proc(image=test_input, process_res=504, num_workers=1)
                    sync_device(device)
                    times.append((time.perf_counter() - start) * 1000)
                results["gpu"][size_key] = np.mean(times)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return results


def benchmark_attention(device, runs=10):
    """Benchmark SDPA vs Manual attention."""
    from depth_anything_3.model.dinov2.layers import Attention

    results = {}
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    configs = [
        ("ViT-L 518px", 1024, 16, 529),
        ("ViT-L 1024px", 1024, 16, 1369),
    ]

    for name, dim, heads, seq_len in configs:
        x = torch.randn(1, seq_len, dim, device=device, dtype=dtype)
        results[name] = {}

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

            results[name][backend] = np.mean(times)
            del attn

    return results


def benchmark_inference(device, runs=3, models=None, quick=False):
    """Benchmark end-to-end inference with all optimization combinations."""
    from depth_anything_3.api import DepthAnything3

    if models is None:
        if quick:
            models = ["da3-large"]
        else:
            models = ["da3-small", "da3-base", "da3-large", "da3-giant"]

    # Batch sizes to test
    batch_sizes = [1, 4] if not quick else [1]

    results = {}
    temp_dir = "temp_infer"

    # Create test image files
    os.makedirs(temp_dir, exist_ok=True)
    img_paths = []
    pil_imgs = []
    for i in range(max(batch_sizes)):
        img = Image.new("RGB", (1280, 720), color=(100 + i*10, 150, 200))
        img_path = f"{temp_dir}/test_{i}.jpg"
        img.save(img_path, quality=95)
        img_paths.append(img_path)
        pil_imgs.append(Image.new("RGB", (1280, 720), color=(100 + i*10, 150, 200)))

    # Define optimization profiles based on device
    is_cuda = device.type == "cuda"
    is_mps = device.type == "mps"

    if is_cuda:
        profiles = {
            "max_gpu": {
                "desc": "GPU Decode + SDPA",
                "use_paths": True,  # File paths trigger NVJPEG decode
                "attn_backend": "sdpa",
            },
            "hybrid": {
                "desc": "CPU Preproc + SDPA",
                "use_paths": False,  # PIL triggers CPU preprocessing
                "attn_backend": "sdpa",
            },
            "max_cpu": {
                "desc": "CPU Preproc + Manual",
                "use_paths": False,
                "attn_backend": "manual",
            },
        }
    elif is_mps:
        # MPS: GPU preprocessing is slower, CPU is better
        profiles = {
            "optimal": {
                "desc": "CPU Preproc + SDPA",
                "use_paths": False,
                "attn_backend": "sdpa",
            },
            "manual": {
                "desc": "CPU Preproc + Manual",
                "use_paths": False,
                "attn_backend": "manual",
            },
        }
    else:
        # CPU only
        profiles = {
            "sdpa": {
                "desc": "SDPA",
                "use_paths": False,
                "attn_backend": "sdpa",
            },
            "manual": {
                "desc": "Manual",
                "use_paths": False,
                "attn_backend": "manual",
            },
        }

    try:
        for model_name in models:
            print(f"  {model_name}:", flush=True)
            results[model_name] = {}

            for batch_size in batch_sizes:
                batch_key = f"batch_{batch_size}"
                results[model_name][batch_key] = {}

                for profile_key, profile in profiles.items():
                    cleanup()

                    # Set attention backend via env var
                    os.environ["DA3_ATTENTION_BACKEND"] = profile["attn_backend"]

                    # Load model
                    model = DepthAnything3(
                        model_name=model_name,
                        device=device,
                        use_cache=False  # Force reload for new attention backend
                    )

                    # Prepare input batch
                    if profile["use_paths"]:
                        test_input = img_paths[:batch_size]
                    else:
                        test_input = pil_imgs[:batch_size]

                    # Warmup
                    for _ in range(3):
                        model.inference(test_input, process_res=504)
                    sync_device(device)

                    # Benchmark
                    times = []
                    for _ in range(runs):
                        sync_device(device)
                        start = time.perf_counter()
                        model.inference(test_input, process_res=504)
                        sync_device(device)
                        times.append((time.perf_counter() - start) * 1000)

                    avg_ms = np.mean(times)
                    fps = 1000 / avg_ms * batch_size  # Images per second
                    results[model_name][batch_key][profile_key] = {
                        "ms": avg_ms,
                        "fps": fps,
                        "desc": profile["desc"],
                    }
                    print(f"    B={batch_size} {profile['desc']:<20} {avg_ms:>6.1f} ms  ({fps:>5.1f} img/s)")

                    del model
                    cleanup()

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return results


def benchmark_adaptive_batching(device, model_name="da3-large", num_images=20, process_res=504, runs=2):
    """Benchmark adaptive batching vs fixed batch sizes."""
    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.utils.adaptive_batching import estimate_max_batch_size

    results = {"fixed": {}, "adaptive": {}}
    temp_dir = "temp_batch"

    # Create test images
    os.makedirs(temp_dir, exist_ok=True)
    pil_imgs = []
    for i in range(num_images):
        img = Image.new("RGB", (1280, 720), color=(100 + i % 50, 150, 200))
        pil_imgs.append(img)

    try:
        # Load model once
        cleanup()
        os.environ["DA3_ATTENTION_BACKEND"] = "sdpa"
        model = DepthAnything3(model_name=model_name, device=device, use_cache=False)

        # Get estimated optimal batch size
        optimal_batch = estimate_max_batch_size(model_name, device, process_res)
        print(f"    Estimated optimal batch: {optimal_batch}")

        # Warmup
        model.inference(pil_imgs[:2], process_res=process_res)
        sync_device(device)

        # Test fixed batch sizes
        fixed_sizes = [1, 2, 4]
        if optimal_batch > 4:
            fixed_sizes.append(optimal_batch)

        for batch_size in fixed_sizes:
            cleanup()
            times = []
            oom = False

            for run in range(runs):
                try:
                    sync_device(device)
                    start = time.perf_counter()

                    for i in range(0, num_images, batch_size):
                        end_idx = min(i + batch_size, num_images)
                        model.inference(pil_imgs[i:end_idx], process_res=process_res)

                    sync_device(device)
                    times.append((time.perf_counter() - start) * 1000)

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        oom = True
                        cleanup()
                        break
                    raise

            if oom:
                results["fixed"][f"B={batch_size}"] = {"oom": True}
                print(f"    Fixed B={batch_size}: OOM")
            else:
                avg_ms = np.mean(times)
                fps = num_images / (avg_ms / 1000)
                results["fixed"][f"B={batch_size}"] = {"ms": avg_ms, "fps": fps}
                print(f"    Fixed B={batch_size}: {avg_ms:.0f} ms ({fps:.1f} img/s)")

        # Test adaptive batching
        for utilization in [0.85]:
            cleanup()
            times = []
            batch_sizes_used = []

            for run in range(runs):
                sync_device(device)
                start = time.perf_counter()

                predictions = model.batch_inference(
                    images=pil_imgs,
                    process_res=process_res,
                    batch_size="auto",
                    target_memory_utilization=utilization,
                )

                sync_device(device)
                times.append((time.perf_counter() - start) * 1000)

                if run == 0:
                    for pred in predictions:
                        if hasattr(pred, "depth"):
                            batch_sizes_used.append(len(pred.depth))

            avg_ms = np.mean(times)
            fps = num_images / (avg_ms / 1000)
            results["adaptive"][f"{int(utilization*100)}%"] = {
                "ms": avg_ms,
                "fps": fps,
                "batches": batch_sizes_used,
            }
            batch_str = ",".join(map(str, batch_sizes_used[:3]))
            if len(batch_sizes_used) > 3:
                batch_str += "..."
            print(f"    Adaptive {int(utilization*100)}%: {avg_ms:.0f} ms ({fps:.1f} img/s) [batches: {batch_str}]")

        del model

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return results


# ============================================================================
# MAIN
# ============================================================================

def get_available_devices() -> list[torch.device]:
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
        return "CPU"


def main():
    parser = argparse.ArgumentParser(description="DA3 Full Benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick mode")
    parser.add_argument("--skip-inference", action="store_true", help="Skip inference benchmark")
    parser.add_argument("--skip-batching", action="store_true", help="Skip adaptive batching benchmark")
    parser.add_argument("--batch-images", type=int, default=20, help="Number of images for batching test")
    parser.add_argument("--device", "-d", type=str, default="auto",
                       choices=["auto", "cpu", "mps", "cuda", "all"],
                       help="Device to benchmark (default: auto = best available, all = test all)")
    args = parser.parse_args()

    runs_preprocess = 3 if args.quick else 5
    runs_attention = 5 if args.quick else 10
    runs_inference = 2 if args.quick else 3

    # Determine devices to test
    available_devices = get_available_devices()
    if args.device == "auto":
        devices_to_test = [available_devices[-1]]  # Best available (last in list)
    elif args.device == "all":
        devices_to_test = available_devices
    else:
        requested = torch.device(args.device)
        if requested in available_devices:
            devices_to_test = [requested]
        else:
            print(f"Error: Device '{args.device}' not available.")
            print(f"Available devices: {[d.type for d in available_devices]}")
            return

    # Header
    print("\n" + "=" * 70)
    print(" DEPTH ANYTHING 3 - BENCHMARK")
    print("=" * 70)
    print(f" PyTorch: {torch.__version__}")
    print(f" Devices to test: {[d.type.upper() for d in devices_to_test]}")
    for d in available_devices:
        status = "✓" if d in devices_to_test else "○"
        print(f"   {status} {d.type.upper()}: {get_device_name(d)}")
    print("=" * 70)

    all_results = {}

    # Run benchmarks for each device
    for device in devices_to_test:
        device_name = get_device_name(device)
        all_results[device.type] = {}

        print("\n" + "#" * 70)
        print(f" DEVICE: {device.type.upper()} ({device_name})")
        print("#" * 70)

        # 1. Preprocessing (only for GPU devices)
        if device.type != "cpu":
            print("\n[1] PREPROCESSING (4 images, batch)")
            print("-" * 70)
            preprocess_results = benchmark_preprocessing(device, runs=runs_preprocess)
            all_results[device.type]["preprocessing"] = preprocess_results

            print(f"{'Size':<15} {'CPU':<12} {'GPU':<12} {'Speedup':<12}")
            for size in preprocess_results["cpu"]:
                cpu_ms = preprocess_results["cpu"][size]
                gpu_ms = preprocess_results["gpu"].get(size)
                if gpu_ms:
                    speedup = cpu_ms / gpu_ms
                    print(f"{size:<15} {cpu_ms:>6.1f} ms    {gpu_ms:>6.1f} ms    {speedup:>5.1f}x")
                else:
                    print(f"{size:<15} {cpu_ms:>6.1f} ms    {'N/A':<12}")
        else:
            preprocess_results = {"cpu": {}, "gpu": {}}
            print("\n[1] PREPROCESSING - Skipped (CPU only)")

        # 2. Attention
        print("\n[2] ATTENTION (per layer)")
        print("-" * 70)
        attention_results = benchmark_attention(device, runs=runs_attention)
        all_results[device.type]["attention"] = attention_results

        print(f"{'Config':<20} {'SDPA':<12} {'Manual':<12} {'Speedup':<12}")
        for config, times in attention_results.items():
            sdpa = times["sdpa"]
            manual = times["manual"]
            speedup = manual / sdpa
            print(f"{config:<20} {sdpa:>6.2f} ms    {manual:>6.2f} ms    {speedup:>5.1f}x")

        # 3. End-to-End Inference
        inference_results = {}
        if not args.skip_inference:
            print("\n[3] END-TO-END INFERENCE (1280x720 input)")
            print("-" * 70)
            inference_results = benchmark_inference(device, runs=runs_inference, quick=args.quick)
            all_results[device.type]["inference"] = inference_results

            # Summary table - find best config per model
            print(f"\n  Best configuration per model (batch=1):")
            for model, batches in inference_results.items():
                batch_1 = batches.get("batch_1", {})
                if batch_1:
                    best_key = min(batch_1.keys(), key=lambda k: batch_1[k]["ms"])
                    best = batch_1[best_key]
                    worst_key = max(batch_1.keys(), key=lambda k: batch_1[k]["ms"])
                    worst = batch_1[worst_key]
                    speedup = worst["ms"] / best["ms"] if best["ms"] > 0 else 0
                    print(f"    {model:<12} {best['desc']:<22} {best['fps']:>5.1f} img/s ({speedup:.1f}x vs {worst['desc']})")

        # 4. Adaptive Batching (skip for CPU - too slow)
        batching_results = {}
        if not args.skip_batching and device.type != "cpu":
            num_batch_images = 10 if args.quick else args.batch_images
            runs_batching = 1 if args.quick else 2
            print(f"\n[4] ADAPTIVE BATCHING ({num_batch_images} images, da3-large)")
            print("-" * 70)
            batching_results = benchmark_adaptive_batching(
                device,
                model_name="da3-large",
                num_images=num_batch_images,
                process_res=504,
                runs=runs_batching,
            )
            all_results[device.type]["batching"] = batching_results
        elif device.type == "cpu":
            print("\n[4] ADAPTIVE BATCHING - Skipped (CPU too slow)")

        # Device Summary
        print("\n" + "-" * 70)
        print(f" {device.type.upper()} SUMMARY")
        print("-" * 70)

        # Preprocessing summary
        if preprocess_results["gpu"]:
            speedups = []
            for size in preprocess_results["cpu"]:
                if size in preprocess_results["gpu"]:
                    speedups.append(preprocess_results["cpu"][size] / preprocess_results["gpu"][size])
            if speedups:
                avg_preprocess_speedup = np.mean(speedups)
                if device.type == "cuda":
                    print(f" Preprocessing: GPU Decode (NVJPEG) avg {avg_preprocess_speedup:.1f}x faster")
                else:
                    print(f" Preprocessing: GPU avg {avg_preprocess_speedup:.1f}x vs CPU")

        # Attention summary
        if attention_results:
            speedups = [r["manual"] / r["sdpa"] for r in attention_results.values()]
            avg_attn_speedup = np.mean(speedups)
            print(f" Attention:     SDPA avg {avg_attn_speedup:.1f}x faster than manual")

        # Check Flash SDP
        if device.type == "cuda":
            from torch.backends.cuda import flash_sdp_enabled
            if flash_sdp_enabled():
                print(f"                Flash Attention: ENABLED (PyTorch native)")

        # Inference summary
        if inference_results:
            print(f" Inference:")
            for model, batches in inference_results.items():
                batch_1 = batches.get("batch_1", {})
                if batch_1:
                    best_key = min(batch_1.keys(), key=lambda k: batch_1[k]["ms"])
                    best = batch_1[best_key]
                    print(f"   {model:<12} {best['fps']:>5.1f} img/s")

        # Batching summary
        if batching_results:
            valid_fixed = {k: v for k, v in batching_results["fixed"].items() if not v.get("oom")}
            if valid_fixed:
                best_fixed_key = max(valid_fixed.keys(), key=lambda k: valid_fixed[k]["fps"])
                best_fixed = valid_fixed[best_fixed_key]
                adaptive = batching_results["adaptive"].get("85%", {})
                if adaptive:
                    adaptive_fps = adaptive["fps"]
                    fixed_fps = best_fixed["fps"]
                    if adaptive_fps >= fixed_fps:
                        print(f" Batching:      Adaptive {adaptive_fps:.1f} img/s (optimal)")
                    else:
                        print(f" Batching:      Fixed {best_fixed_key} {fixed_fps:.1f} img/s (better than adaptive)")

        # Cleanup between devices
        cleanup()

    # Cross-device comparison (if multiple devices tested)
    if len(devices_to_test) > 1:
        print("\n" + "=" * 70)
        print(" CROSS-DEVICE COMPARISON")
        print("=" * 70)

        # Compare attention performance
        print("\n Attention (ViT-L 518px, SDPA):")
        for device in devices_to_test:
            if device.type in all_results and "attention" in all_results[device.type]:
                attn = all_results[device.type]["attention"].get("ViT-L 518px", {})
                if "sdpa" in attn:
                    print(f"   {device.type.upper():<8} {attn['sdpa']:>6.2f} ms")

        # Compare inference if available
        if not args.skip_inference:
            print("\n Inference (da3-large, batch=1):")
            for device in devices_to_test:
                if device.type in all_results and "inference" in all_results[device.type]:
                    infer = all_results[device.type]["inference"].get("da3-large", {})
                    batch_1 = infer.get("batch_1", {})
                    if batch_1:
                        best_key = min(batch_1.keys(), key=lambda k: batch_1[k]["ms"])
                        best = batch_1[best_key]
                        print(f"   {device.type.upper():<8} {best['fps']:>5.1f} img/s ({best['desc']})")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
