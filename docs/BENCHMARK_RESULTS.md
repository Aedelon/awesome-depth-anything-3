# Depth Anything 3 - Benchmark Results

Comprehensive performance benchmarks for Depth Anything 3, testing preprocessing, attention mechanisms, end-to-end inference, and adaptive batching across all devices.

## Test Environments

### NVIDIA L4 (CUDA)

| Component | Value |
|-----------|-------|
| GPU | NVIDIA L4 |
| PyTorch | 2.9.0+cu128 |
| Flash Attention | Enabled (PyTorch native SDPA) |

### Apple Silicon (MPS)

| Component | Value |
|-----------|-------|
| Device | Apple Silicon (MPS) |
| PyTorch | 2.9.0 |
| SDPA | Enabled |

---

## Quick Comparison

| Device | da3-large Throughput | vs CPU |
|--------|---------------------|--------|
| **NVIDIA L4 (CUDA)** | **10.3 img/s** | **40x** |
| **Apple Silicon (MPS)** | **3.7 img/s** | **14.5x** |
| CPU | 0.3 img/s | baseline |

---

## 1. CUDA Results (NVIDIA L4)

### Preprocessing Performance

| Resolution | CPU | GPU (NVJPEG) | Speedup |
|------------|-----|--------------|---------|
| 640x480 | 25.6 ms | 6.8 ms | **3.8x** |
| 1920x1080 (HD) | 75.8 ms | 17.2 ms | **4.4x** |
| 3840x2160 (4K) | 210.2 ms | 57.4 ms | **3.7x** |

**Key Finding:** GPU decode via NVJPEG is **3.9x faster on average**.

### Attention Performance

| Configuration | SDPA | Manual | Speedup |
|---------------|------|--------|---------|
| ViT-L 518px (529 tokens) | 0.52 ms | 0.62 ms | **1.2x** |
| ViT-L 1024px (1369 tokens) | 0.40 ms | 1.50 ms | **3.7x** |

**Key Finding:** Flash Attention is **2.5x faster on average**, with larger speedups at higher resolutions.

### End-to-End Inference (Batch=1)

| Model | GPU Decode + SDPA | CPU Preproc + SDPA | CPU Preproc + Manual |
|-------|-------------------|--------------------|-----------------------|
| da3-small | **54.6 ms (18.3 img/s)** | 65.1 ms (15.4 img/s) | 64.0 ms (15.6 img/s) |
| da3-base | **55.1 ms (18.2 img/s)** | 69.9 ms (14.3 img/s) | 67.8 ms (14.7 img/s) |
| da3-large | **97.0 ms (10.3 img/s)** | 107.0 ms (9.3 img/s) | 116.4 ms (8.6 img/s) |
| da3-giant | **172.0 ms (5.8 img/s)** | 181.3 ms (5.5 img/s) | 220.1 ms (4.5 img/s) |

### End-to-End Inference (Batch=4)

| Model | GPU Decode + SDPA | CPU Preproc + SDPA | CPU Preproc + Manual |
|-------|-------------------|--------------------|-----------------------|
| da3-small | **92.3 ms (43.4 img/s)** | 111.2 ms (36.0 img/s) | 129.8 ms (30.8 img/s) |
| da3-base | **124.5 ms (32.1 img/s)** | 146.6 ms (27.3 img/s) | 210.9 ms (19.0 img/s) |
| da3-large | **260.5 ms (15.4 img/s)** | 270.3 ms (14.8 img/s) | 444.6 ms (9.0 img/s) |
| da3-giant | **450.0 ms (8.9 img/s)** | 468.7 ms (8.5 img/s) | 917.2 ms (4.4 img/s) |

---

## 2. Apple Silicon Results (MPS)

### Attention Performance

| Configuration | SDPA | Manual | Speedup |
|---------------|------|--------|---------|
| ViT-L 518px | 2.40 ms | 2.10 ms | 0.9x |
| ViT-L 1024px | 9.17 ms | 5.94 ms | 0.6x |

> **Note**: On MPS, Manual attention can be faster than SDPA for larger sequences. This is a known limitation of PyTorch MPS SDPA.

### End-to-End Inference (Batch=1)

| Model | CPU Preproc + SDPA | CPU Preproc + Manual | Best |
|-------|--------------------|-----------------------|------|
| da3-large | **271 ms (3.7 img/s)** | 289 ms (3.5 img/s) | SDPA |

### Adaptive Batching (10 images, da3-large)

| Strategy | Total Time | Throughput | Batch Sizes |
|----------|-----------|------------|-------------|
| Fixed B=1 | 2868 ms | 3.5 img/s | [1,1,1,...] |
| **Fixed B=2** | **2708 ms** | **3.7 img/s** | [2,2,2,...] |
| Fixed B=4 | 2996 ms | 3.3 img/s | [4,4,2] |
| Adaptive 85% | 3245 ms | 3.1 img/s | [4,4,2] |

> **Note**: Fixed batch size B=2 is optimal on this device. Adaptive batching has overhead when memory is abundant.

---

## 3. CPU Results

### Attention Performance

| Configuration | SDPA | Manual | Speedup |
|---------------|------|--------|---------|
| ViT-L 518px | 3.75 ms | 4.96 ms | **1.3x** |
| ViT-L 1024px | 11.73 ms | 16.85 ms | **1.4x** |

### End-to-End Inference (Batch=1)

| Model | SDPA | Manual | Best |
|-------|------|--------|------|
| da3-large | **3927 ms (0.3 img/s)** | 10372 ms (0.1 img/s) | SDPA (2.6x) |

---

## 4. Cross-Device Comparison

### Attention (ViT-L 518px, SDPA)

| Device | Latency | vs CPU |
|--------|---------|--------|
| CPU | 3.75 ms | 1.0x |
| MPS | 2.40 ms | **1.6x** |
| CUDA | 0.52 ms | **7.2x** |

### Inference (da3-large, batch=1)

| Device | Throughput | vs CPU |
|--------|------------|--------|
| CPU | 0.3 img/s | 1.0x |
| MPS | 3.7 img/s | **14.5x** |
| CUDA | 10.3 img/s | **40x** |

---

## 5. Recommendations

### For CUDA GPUs

1. **Use file paths as input** - Triggers NVJPEG decode (3.9x faster preprocessing)
2. **Use SDPA attention** - Flash Attention provides 2.5x speedup
3. **Batch size 4-8** - Optimal for throughput on most GPUs
4. **GPU Decode + SDPA profile** - Best overall configuration

### For Apple Silicon (MPS)

1. **Use MPS backend** - 14.5x faster than CPU
2. **Use SDPA attention** - Slightly faster end-to-end despite per-layer overhead
3. **Batch size 2** - Optimal for memory/throughput balance
4. **CPU preprocessing** - GPU preprocessing not beneficial (unified memory)

### For CPU

1. **Use SDPA attention** - 2.6x faster than manual
2. **Batch size 1** - Memory constraints
3. **Consider MPS or CUDA** - CPU is 14-40x slower

---

## 6. Adaptive Batching

The adaptive batching system automatically selects optimal batch sizes based on available GPU memory.

### When to Use

- **Large image sets** (100+ images)
- **Memory-constrained GPUs**
- **Variable workloads**

### When Fixed Batch is Better

- **Memory-abundant systems** (overhead of estimation)
- **Small image sets** (<20 images)
- **Known optimal batch size**

### API Usage

```python
from depth_anything_3.api import DepthAnything3

model = DepthAnything3(model_name="da3-large")

# Adaptive batching (recommended for large sets)
results = model.batch_inference(
    images=image_paths,
    batch_size="auto",
    target_memory_utilization=0.85,
)

# Fixed batch size (when you know optimal)
results = model.batch_inference(
    images=image_paths,
    batch_size=4,
)

# Get recommended batch size
optimal = model.get_optimal_batch_size(process_res=518)
```

---

## Running the Benchmark

```bash
# Quick benchmark on best available device
python benchmarks/full_benchmark.py --quick

# Full benchmark on all devices (CPU, MPS, CUDA)
python benchmarks/full_benchmark.py --device all

# Specific device
python benchmarks/full_benchmark.py --device mps
python benchmarks/full_benchmark.py --device cuda
python benchmarks/full_benchmark.py --device cpu

# Skip slow tests
python benchmarks/full_benchmark.py --skip-inference --skip-batching

# Custom batch image count
python benchmarks/full_benchmark.py --batch-images 50
```

---

## Changelog

- **2025-12-02**: Added multi-device support (CPU, MPS, CUDA), adaptive batching benchmark
- **2025-12-01**: Initial benchmark with Flash Attention comparison
