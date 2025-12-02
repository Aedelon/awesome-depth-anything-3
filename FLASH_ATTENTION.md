# Flash Attention Integration

Quick guide to Flash Attention support in Depth Anything 3.

## The Short Story

- **NVIDIA GPU (CUDA)** : Install `flash-attn` for 2-3x faster attention layers → ~15-25% overall speedup
- **Apple Silicon (MPS)** : Not supported, PyTorch SDPA is already optimized
- **CPU** : Not supported, not critical anyway

## Installation (CUDA only)

```bash
# Linux / NVIDIA GPU
pip install flash-attn --no-build-isolation
```

**Requirements:**
- CUDA Compute Capability ≥ 7.5 (Tesla V100+, RTX 20/30/40 series, A100, etc.)
- CUDA toolkit
- PyTorch ≥ 2.2
- ninja (usually auto-installed)

**Note:** Does NOT work on macOS or Windows (use Windows CUDA subsystem if you have GPU)

## Verify Installation

```python
from depth_anything_3.model.dinov2.layers import FLASH_ATTN_AVAILABLE, get_attention_backend

print(f"Flash Attention available: {FLASH_ATTN_AVAILABLE}")
print(f"Selected backend: {get_attention_backend()}")
```

Output on NVIDIA GPU with flash-attn:
```
Flash Attention available: True
Selected backend: flash_attn
```

Output on other devices or without flash-attn:
```
Flash Attention available: False
Selected backend: sdpa
```

## How It Works

The code **automatically selects the best backend**:

1. **flash_attn** (if installed + NVIDIA GPU) → Fastest ⚡
2. **sdpa** (PyTorch 2.0+ default) → Fast, built-in ✓
3. **manual** (fallback) → Slow, for debugging

No code changes needed, it just works transparently.

## Override Backend (Advanced)

Force a specific backend with environment variable:

```bash
# Use Flash Attention explicitly
export DA3_ATTENTION_BACKEND=flash_attn
python inference.py

# Force PyTorch SDPA (debug)
export DA3_ATTENTION_BACKEND=sdpa
python inference.py

# Force manual attention (very slow, debugging only)
export DA3_ATTENTION_BACKEND=manual
python inference.py
```

## Benchmark Results

Run the simple benchmark to see differences on your hardware:

```bash
python benchmarks/attention_benchmark_simple.py
```

Example output on NVIDIA A100:
```
📊 512px image (529 tokens)
   flash_attn:   12.34 ms
   sdpa:         28.56 ms
   flash_attn: 2.32x faster than SDPA ✨
```

## Architecture Notes

### Flash Attention Specifics

- **Format**: Expects (Batch, SeqLen, Heads, HeadDim) internally
- **Dtype**: Converts float32 → bfloat16 for computation (more efficient), converts back
- **Attention Mask**: Limited support
  - ✅ No mask (standard self-attention) → Uses Flash Attention
  - ❌ Custom masks → Falls back to SDPA
  - Our RoPE (2D rotary embeddings) → Applied before, fully compatible

### Supported Configurations

| Model | Backbone | Seq Len | Attention |
|-------|----------|---------|-----------|
| DA3-Small | ViT-S | 196 | 12 layers |
| DA3-Base | ViT-B | 256 | 12 layers |
| DA3-Large | ViT-L | 1024 | 24 layers ⭐ |
| DA3-Giant | ViT-G | 1369 | 40 layers ⭐⭐ |

Larger models benefit most from Flash Attention (more attention layers).

## Performance Impact

### On Attention Layers
- **Flash Attention**: 2-3x speedup
- Why: IO-aware algorithm, reduced memory bandwidth

### On Full Inference
- Attention = ~15-20% of total inference time
- Overall speedup = **0.15-0.20 × 2.5 = 15-25%**

Example (DA3-Large on 1024px image):
```
Without flash-attn: 2.5 seconds
With flash-attn:    2.1 seconds
Overall speedup:    ~16%
```

## Troubleshooting

### Q: Flash Attention not detected despite installation?

```python
import flash_attn
print(flash_attn.__version__)
```

If import fails, reinstall:
```bash
pip uninstall flash-attn
pip install flash-attn --no-build-isolation
```

### Q: Getting CUDA errors?

Check CUDA Compute Capability:
```python
import torch
print(torch.cuda.get_device_capability())  # Should be >= (7, 5)
```

If lower, Flash Attention won't work (use SDPA instead).

### Q: On Windows?

Flash Attention requires compilation. Use:
- WSL2 (Windows Subsystem for Linux) with CUDA, OR
- GPU-enabled Colab/cloud notebooks

### Q: On macOS?

Not supported. PyTorch SDPA with Metal optimizations is already good enough.

## Implementation Details

See: `src/depth_anything_3/model/dinov2/layers/attention.py`

Key components:
- `FLASH_ATTN_AVAILABLE`: Detects if flash-attn is installed
- `get_attention_backend()`: Auto-selects best backend
- `Attention._flash_attention()`: Flash Attention implementation
- `Attention._sdpa_attention()`: PyTorch SDPA fallback
- `Attention._manual_attention()`: Classic attention (debug)

All backed by RoPE 2D (rotary position embeddings) support.

## References

- [Flash Attention paper](https://arxiv.org/abs/2205.14135)
- [Flash Attention repo](https://github.com/dao-ailab/flash-attention)
- [PyTorch SDPA docs](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)