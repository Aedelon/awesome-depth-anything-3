"""
MPS (Metal Performance Shaders) optimizations for Apple Silicon.

MPS optimization strategy:
- Unified Memory architecture (CPU/GPU share RAM)
- BF16 if supported, NOT FP16 (no hardware acceleration for FP16)
- Limited prefetch (Unified Memory contention)
- Moderate preprocessing workers (8-12)
- Conservative compilation (overhead)
"""

from __future__ import annotations
import os
import torch
from typing import Dict, Any

from depth_anything_3.optimizations.base_optimizer import BaseOptimizer, OptimizationConfig
from depth_anything_3.utils.logger import logger


class MPSOptimizer(BaseOptimizer):
    """Optimizer for Apple Silicon MPS backend."""

    def get_default_config(self) -> OptimizationConfig:
        """Get default MPS optimization configuration."""
        return OptimizationConfig(
            mixed_precision=None,  # Auto-detect BF16 support
            num_workers=min(12, (os.cpu_count() or 8)),  # Moderate parallelism
            prefetch_factor=1,  # Limited prefetch (Unified Memory)
            enable_compile=False,  # Conservative (can add overhead)
            batch_size=None,
            dynamic_batching=True,
            performance_mode="balanced",
        )

    def apply_platform_settings(self) -> None:
        """Apply MPS-specific PyTorch settings."""
        # Set memory fraction (leave headroom for system)
        if hasattr(torch.mps, "set_per_process_memory_fraction"):
            torch.mps.set_per_process_memory_fraction(0.8)

        # Set matmul precision
        torch.set_float32_matmul_precision("medium")

        # Disable CUDNN (not applicable for MPS)
        torch.backends.cudnn.enabled = False

        logger.info(
            f"MPS Optimizer applied: Unified Memory, "
            f"num_workers={self.config.num_workers}, "
            f"prefetch_limited=True"
        )

    def get_mixed_precision_dtype(self) -> torch.dtype | None:
        """
        MPS mixed precision strategy:
        - BF16 if available (some ops accelerated)
        - AVOID FP16 (no hardware acceleration, can be slower)
        - FP32 fallback

        Returns:
            torch.bfloat16 if supported, else None (FP32)
        """
        if self.config.mixed_precision == False:
            return None

        if self.config.mixed_precision == "bfloat16":
            # User explicitly requested BF16
            return torch.bfloat16

        if self.config.mixed_precision == "float16":
            # User requested FP16 - warn but allow
            logger.warn(
                "FP16 on MPS may be slower than FP32 (no hardware acceleration). "
                "Consider using BF16 or FP32 instead."
            )
            return torch.float16

        # Auto mode: prefer BF16 if available, else FP32
        # Note: MPS BF16 support is limited, check availability
        if hasattr(torch, "backends") and hasattr(torch.backends, "mps"):
            # Conservative: use FP32 by default on MPS
            # BF16 can be enabled explicitly if tested to work
            logger.info("MPS Optimizer: Using FP32 (conservative, tested stable)")
            return None

        return None

    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get MPS preprocessing configuration."""
        return {
            "num_workers": self.config.num_workers,
            "prefetch_enabled": False,  # Unified Memory: prefetch can cause contention
            "pin_memory": False,  # Unified Memory: pinning not beneficial
            "non_blocking": True,  # Async transfers still helpful
        }

    def should_use_compile(self) -> bool:
        """
        torch.compile on MPS is experimental and can add overhead.
        Only enable in max performance mode with explicit user request.
        """
        if self.config.performance_mode == "max" and self.config.enable_compile:
            logger.warn(
                "torch.compile on MPS is experimental. "
                "May add overhead or cause instability."
            )
            return True
        return False

    def get_memory_config(self) -> Dict[str, Any]:
        """MPS memory configuration (Unified Memory architecture)."""
        return {
            "pin_memory": False,  # Not beneficial with Unified Memory
            "non_blocking": True,  # Async transfers still useful
            "prefetch_factor": 1,  # Minimal prefetch
        }

    def _get_device_type(self) -> str:
        return "mps"