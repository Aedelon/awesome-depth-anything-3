"""
CUDA-specific performance optimizations for NVIDIA GPUs.

CUDA optimization strategy (most aggressive):
- cuDNN benchmark + autotuning
- TF32 tensor cores
- Mixed precision (BF16 if supported, else FP16)
- Aggressive prefetch with pin_memory
- torch.compile for large batches
- CUDA-specific memory optimizations
"""

from __future__ import annotations
import os
import torch
from typing import Dict, Any

from depth_anything_3.optimizations.base_optimizer import BaseOptimizer, OptimizationConfig
from depth_anything_3.utils.logger import logger


class CUDAOptimizer(BaseOptimizer):
    """Optimizer for NVIDIA CUDA GPUs."""

    def get_default_config(self) -> OptimizationConfig:
        """Get default CUDA optimization configuration."""
        return OptimizationConfig(
            mixed_precision=None,  # Auto-detect BF16 support
            num_workers=min(12, (os.cpu_count() or 8)),
            prefetch_factor=2,  # Aggressive prefetch
            enable_compile=False,  # User can enable for max performance
            compile_mode="reduce-overhead",
            batch_size=None,
            dynamic_batching=True,
            performance_mode="balanced",
        )

    def apply_platform_settings(self) -> None:
        """Apply CUDA-specific PyTorch settings (MAXIMUM PERFORMANCE)."""
        # ============================================
        # CRITICAL: cuDNN Benchmark & Autotuning
        # ============================================
        # This is the BIGGEST win - enables autotuned kernels
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True

        # ============================================
        # TF32 Tensor Cores (Ampere+)
        # ============================================
        # Enables TF32 for matmul (5-10% speedup on Ampere/Ada)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        # ============================================
        # Additional CUDA optimizations
        # ============================================
        # Disable deterministic algorithms for speed
        # torch.use_deterministic_algorithms(False)  # If reproducibility not needed

        # Enable CUDA graphs if in max performance mode
        # (requires static graph - advanced feature)

        logger.info(
            "CUDA Optimizer applied: "
            f"cudnn.benchmark=True, TF32=True, "
            f"num_workers={self.config.num_workers}, "
            f"prefetch_factor={self.config.prefetch_factor}"
        )

    def get_mixed_precision_dtype(self) -> torch.dtype | None:
        """
        CUDA mixed precision strategy:
        - BF16 if available (Ampere+ GPUs, more stable)
        - FP16 fallback (all modern GPUs, slightly faster but less stable)
        - FP32 if disabled

        Returns:
            torch.bfloat16 (preferred) or torch.float16
        """
        if self.config.mixed_precision == False:
            return None

        if self.config.mixed_precision == "bfloat16":
            return torch.bfloat16

        if self.config.mixed_precision == "float16":
            return torch.float16

        # Auto mode: prefer BF16 if available
        if torch.cuda.is_bf16_supported():
            logger.info("CUDA Optimizer: Using BF16 (Ampere+ GPU detected)")
            return torch.bfloat16
        else:
            logger.info("CUDA Optimizer: Using FP16 (Pre-Ampere GPU)")
            return torch.float16

    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get CUDA preprocessing configuration (AGGRESSIVE)."""
        return {
            "num_workers": self.config.num_workers,
            "prefetch_enabled": True,  # Critical for hiding data loading latency
            "pin_memory": True,  # Enables async H2D transfers
            "non_blocking": True,  # Async CUDA operations
        }

    def should_use_compile(self) -> bool:
        """
        torch.compile is NOT compatible with Depth Anything 3 model.

        Error: triton.compiler.errors.CompilationError
        The model architecture is too complex for Triton compilation:
        - Nested models with dynamic shapes
        - Complex geometric operations (pose estimation, multi-view)
        - Deep expression trees that Triton cannot optimize

        This is a model-specific limitation, not a CUDA/Triton bug.
        torch.compile works fine on simpler models (ResNet, VGG, etc.)
        """
        if self.config.enable_compile:
            logger.warn(
                "torch.compile is NOT compatible with Depth Anything 3 model. "
                "Triton compilation fails due to model complexity. Disabled automatically."
            )
        return False  # Always disabled for this model

    def get_memory_config(self) -> Dict[str, Any]:
        """CUDA memory optimization configuration."""
        return {
            "pin_memory": True,  # Critical for async transfers
            "non_blocking": True,  # Async CUDA operations
            "prefetch_factor": self.config.prefetch_factor,
            "persistent_workers": True,  # Reuse workers
        }

    def get_compile_config(self) -> Dict[str, str]:
        """Get torch.compile configuration."""
        return {
            "mode": self.config.compile_mode,
            "fullgraph": False,  # Allow graph breaks for flexibility
            "dynamic": True,  # Handle variable shapes
        }

    def _get_device_type(self) -> str:
        return "cuda"

    def get_additional_info(self) -> Dict[str, Any]:
        """Get additional CUDA device information."""
        if not torch.cuda.is_available():
            return {}

        return {
            "device_name": torch.cuda.get_device_name(0),
            "compute_capability": torch.cuda.get_device_capability(0),
            "bf16_supported": torch.cuda.is_bf16_supported(),
            "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        }