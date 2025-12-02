# Depth Anything 3 - Benchmark Results

Comprehensive performance benchmarks for Depth Anything 3, testing preprocessing, attention mechanisms, and end-to-end inference across all model sizes.

## Test Environment

| Component | Value |
|-----------|-------|
| GPU | NVIDIA L4 |
| PyTorch | 2.9.0+cu128 |
| Flash Attention | Enabled (PyTorch native SDPA) |

## 1. Preprocessing Performance

Comparison of CPU vs GPU (NVJPEG) image decoding and preprocessing for a batch of 4 images.

| Resolution | CPU | GPU (NVJPEG) | Speedup |
|------------|-----|--------------|---------|
| 640x480 | 25.6 ms | 6.8 ms | **3.8x** |
| 1920x1080 (HD) | 75.8 ms | 17.2 ms | **4.4x** |
| 3840x2160 (4K) | 210.2 ms | 57.4 ms | **3.7x** |

**Key Finding:** GPU decode via NVJPEG is **3.9x faster on average**, with the best speedup at HD resolution (4.4x).

## 2. Attention Performance

Per-layer attention benchmark comparing SDPA (Scaled Dot-Product Attention with Flash Attention) vs Manual implementation.

| Configuration | SDPA | Manual | Speedup |
|---------------|------|--------|---------|
| ViT-L 518px (529 tokens) | 0.52 ms | 0.62 ms | **1.2x** |
| ViT-L 1024px (1369 tokens) | 0.40 ms | 1.50 ms | **3.7x** |

**Key Finding:** SDPA with Flash Attention is **2.5x faster on average**, with larger speedups at higher resolutions (more tokens).

## 3. End-to-End Inference

Full inference benchmarks with 1280x720 input images, testing all model sizes and optimization profiles.

### Optimization Profiles

| Profile | Description |
|---------|-------------|
| **GPU Decode + SDPA** | NVJPEG decode + Flash Attention (maximum GPU optimization) |
| **CPU Preproc + SDPA** | PIL/OpenCV decode + Flash Attention (hybrid) |
| **CPU Preproc + Manual** | PIL/OpenCV decode + Manual attention (baseline) |

### Batch Size = 1

| Model | GPU Decode + SDPA | CPU Preproc + SDPA | CPU Preproc + Manual |
|-------|-------------------|--------------------|-----------------------|
| da3-small | **54.6 ms (18.3 img/s)** | 65.1 ms (15.4 img/s) | 64.0 ms (15.6 img/s) |
| da3-base | **55.1 ms (18.2 img/s)** | 69.9 ms (14.3 img/s) | 67.8 ms (14.7 img/s) |
| da3-large | **97.0 ms (10.3 img/s)** | 107.0 ms (9.3 img/s) | 116.4 ms (8.6 img/s) |
| da3-giant | **172.0 ms (5.8 img/s)** | 181.3 ms (5.5 img/s) | 220.1 ms (4.5 img/s) |

### Batch Size = 4

| Model | GPU Decode + SDPA | CPU Preproc + SDPA | CPU Preproc + Manual |
|-------|-------------------|--------------------|-----------------------|
| da3-small | **92.3 ms (43.4 img/s)** | 111.2 ms (36.0 img/s) | 129.8 ms (30.8 img/s) |
| da3-base | **124.5 ms (32.1 img/s)** | 146.6 ms (27.3 img/s) | 210.9 ms (19.0 img/s) |
| da3-large | **260.5 ms (15.4 img/s)** | 270.3 ms (14.8 img/s) | 444.6 ms (9.0 img/s) |
| da3-giant | **450.0 ms (8.9 img/s)** | 468.7 ms (8.5 img/s) | 917.2 ms (4.4 img/s) |

## Summary

### Best Configuration per Model

| Model | Best Config | Throughput | Speedup vs Baseline |
|-------|-------------|------------|---------------------|
| da3-small | GPU Decode + SDPA | 18.3 img/s | 1.2x |
| da3-base | GPU Decode + SDPA | 18.2 img/s | 1.3x |
| da3-large | GPU Decode + SDPA | 10.3 img/s | 1.2x |
| da3-giant | GPU Decode + SDPA | 5.8 img/s | 1.3x |

### Optimization Impact

| Optimization | Average Speedup |
|--------------|-----------------|
| GPU Decode (NVJPEG) | **3.9x** faster preprocessing |
| SDPA (Flash Attention) | **2.5x** faster attention |
| Combined (batch=4) | **1.7-2.0x** faster end-to-end |

### Recommendations

1. **For CUDA GPUs:** Use file paths as input (triggers NVJPEG decode) with default SDPA attention
2. **For MPS (Apple Silicon):** CPU preprocessing is faster; GPU reserved for model inference
3. **For maximum throughput:** Use batch size 4+ with GPU decode

## Running the Benchmark

```bash
# Quick benchmark (da3-large only, batch=1)
uv run python benchmarks/full_benchmark.py --quick

# Full benchmark (all models, batch 1 & 4)
uv run python benchmarks/full_benchmark.py

# Skip inference (preprocessing + attention only)
uv run python benchmarks/full_benchmark.py --skip-inference
```
