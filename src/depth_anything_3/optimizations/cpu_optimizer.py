"""
CPU-specific performance optimizations.

CPU optimization strategy:
- No mixed precision (FP16 is slower on CPU)
- Threading optimization (OMP, MKL)
- MKLDNN/oneDNN backend
- Reduced preprocessing workers (avoid overhead)
- No compilation for small batches (overhead)
"""

from __future__ import annotations
import os
import torch
from typing import Dict, Any

from depth_anything_3.optimizations.base_optimizer import BaseOptimizer, OptimizationConfig
from depth_anything_3.utils.logger import logger


# Global flag to track if CPU threading has been configured
# torch.set_num_interop_threads() can only be called once
_CPU_THREADING_CONFIGURED = False


class CPUOptimizer(BaseOptimizer):
    """Optimizer for CPU inference."""

    def get_default_config(self) -> OptimizationConfig:
        """Get default CPU optimization configuration."""
        num_cores = os.cpu_count() or 4
        return OptimizationConfig(
            mixed_precision=False,  # CRITICAL: FP16 is SLOWER on CPU
            num_workers=max(1, num_cores - 2),  # Leave 2 cores for system
            prefetch_factor=1,  # Minimal prefetch (low overhead)
            enable_compile=False,  # Usually overhead for small batches
            batch_size=None,
            dynamic_batching=True,
            performance_mode="balanced",
        )

    def apply_platform_settings(self) -> None:
        """Apply CPU-specific PyTorch settings."""
        global _CPU_THREADING_CONFIGURED

        num_cores = os.cpu_count() or 4

        # Threading optimization (can only be set once)
        if not _CPU_THREADING_CONFIGURED:
            torch.set_num_threads(num_cores)
            torch.set_num_interop_threads(min(4, num_cores))
            _CPU_THREADING_CONFIGURED = True
            logger.info(f"CPU threading configured: {num_cores} threads, {min(4, num_cores)} interop threads")
        else:
            logger.debug(f"CPU threading already configured, skipping")

        # Enable MKLDNN/oneDNN for CPU inference
        if hasattr(torch.backends, "mkldnn"):
            torch.backends.mkldnn.enabled = True
            logger.debug(f"CPU Optimizer: MKLDNN enabled")

        # Disable CUDNN (not applicable for CPU)
        torch.backends.cudnn.enabled = False

        logger.info(
            f"CPU Optimizer applied: {num_cores} threads, "
            f"num_workers={self.config.num_workers}"
        )

    def get_mixed_precision_dtype(self) -> torch.dtype | None:
        """
        CPU does not benefit from FP16 mixed precision.

        Returns:
            None to use FP32 (default precision)
        """
        # CRITICAL: Always return None for CPU
        # FP16 on CPU is emulated and SLOWER than FP32
        return None

    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get CPU preprocessing configuration."""
        return {
            "num_workers": self.config.num_workers,
            "prefetch_enabled": False,  # Minimal prefetch overhead
            "pin_memory": False,  # Not applicable for CPU
            "non_blocking": False,
        }

    def should_use_compile(self) -> bool:
        """
        torch.compile often adds overhead on CPU for small batches.
        Only enable for very large batch sizes in max performance mode.
        """
        if self.config.performance_mode == "max" and self.config.batch_size and self.config.batch_size >= 16:
            return self.config.enable_compile
        return False

    def get_memory_config(self) -> Dict[str, Any]:
        """CPU memory configuration."""
        return {
            "pin_memory": False,
            "non_blocking": False,
            "prefetch_factor": 1,
        }

    def _get_device_type(self) -> str:
        return "cpu"