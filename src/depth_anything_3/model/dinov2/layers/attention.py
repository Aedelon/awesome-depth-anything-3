# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Flash Attention integration by Delanoe Pirard / Aedelon - Apache 2.0

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os

import torch
import torch.nn.functional as F
from torch import Tensor, nn

logger = logging.getLogger("dinov2")

# Flash Attention availability detection
FLASH_ATTN_AVAILABLE = False
FLASH_ATTN_VERSION = None

try:
    from flash_attn import flash_attn_func
    from flash_attn import __version__ as flash_attn_version

    FLASH_ATTN_AVAILABLE = True
    FLASH_ATTN_VERSION = flash_attn_version
    logger.info(f"Flash Attention v{flash_attn_version} available")
except ImportError:
    logger.debug("flash-attn not installed, using PyTorch SDPA backend")


def get_attention_backend() -> str:
    """
    Determine the best attention backend for current hardware.

    Returns:
        str: 'flash_attn' if flash-attn is available and on CUDA,
             'sdpa' for PyTorch scaled_dot_product_attention,
             'manual' for fallback attention.
    """
    # Check environment override
    env_backend = os.environ.get("DA3_ATTENTION_BACKEND", "").lower()
    if env_backend in ("flash_attn", "sdpa", "manual"):
        return env_backend

    # Auto-detect best backend
    if FLASH_ATTN_AVAILABLE and torch.cuda.is_available():
        return "flash_attn"
    return "sdpa"


class Attention(nn.Module):
    """
    Multi-head attention with Flash Attention support.

    Supports three backends:
    - flash_attn: Flash Attention v2/v3 (fastest on CUDA, requires flash-attn package)
    - sdpa: PyTorch scaled_dot_product_attention (default, may use Flash internally)
    - manual: Classic attention implementation (slowest, for debugging)

    Backend selection:
    - Auto-detected based on hardware and package availability
    - Override via DA3_ATTENTION_BACKEND env var or attn_backend parameter
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # legacy param, kept for compatibility
        attn_backend: str | None = None,  # 'flash_attn', 'sdpa', 'manual', or None for auto
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        # Determine attention backend
        if attn_backend is not None:
            self.attn_backend = attn_backend
        elif not fused_attn:
            self.attn_backend = "manual"
        else:
            self.attn_backend = get_attention_backend()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_drop_p = attn_drop
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

        logger.debug(f"Attention initialized with backend: {self.attn_backend}")

    def _flash_attention(
        self, q: Tensor, k: Tensor, v: Tensor, attn_mask: Tensor | None
    ) -> Tensor:
        """
        Flash Attention v2/v3 forward pass.

        Note: flash_attn_func expects (B, N, H, D) format, not (B, H, N, D).
        Attention mask support is limited in Flash Attention.
        """
        # flash_attn expects (B, N, H, D), we have (B, H, N, D)
        q = q.transpose(1, 2)  # (B, N, H, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Flash Attention requires contiguous tensors in fp16/bf16
        if q.dtype == torch.float32:
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)
            cast_back = True
        else:
            cast_back = False

        dropout_p = self.attn_drop_p if self.training else 0.0

        # Flash Attention v2 does not support arbitrary attention masks
        # It supports causal masking via causal=True flag
        # For non-causal with custom mask, fall back to SDPA
        if attn_mask is not None:
            logger.debug("Flash Attention: custom mask not supported, falling back to SDPA")
            return self._sdpa_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask
            )

        out = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=False)

        if cast_back:
            out = out.to(torch.float32)

        # Back to (B, H, N, D) then (B, N, C)
        return out.transpose(1, 2)

    def _sdpa_attention(
        self, q: Tensor, k: Tensor, v: Tensor, attn_mask: Tensor | None
    ) -> Tensor:
        """PyTorch scaled_dot_product_attention (may use Flash internally on CUDA)."""
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            attn_mask=(
                attn_mask[:, None].repeat(1, self.num_heads, 1, 1)
                if attn_mask is not None
                else None
            ),
        )

    def _manual_attention(
        self, q: Tensor, k: Tensor, v: Tensor, attn_mask: Tensor | None
    ) -> Tensor:
        """Classic attention implementation for debugging."""
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if attn_mask is not None:
            attn = attn + attn_mask[:, None].repeat(1, self.num_heads, 1, 1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        return attn @ v

    def forward(self, x: Tensor, pos=None, attn_mask=None) -> Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q), self.k_norm(k)

        # Apply RoPE if available (before attention)
        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        # Select attention backend
        if self.attn_backend == "flash_attn" and FLASH_ATTN_AVAILABLE:
            x = self._flash_attention(q, k, v, attn_mask)
        elif self.attn_backend == "sdpa" or (
            self.attn_backend == "flash_attn" and not FLASH_ATTN_AVAILABLE
        ):
            x = self._sdpa_attention(q, k, v, attn_mask)
        else:
            x = self._manual_attention(q, k, v, attn_mask)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
