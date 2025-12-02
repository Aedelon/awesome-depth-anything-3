"""
Base optimizer interface for device-specific performance optimizations.

This module provides the abstract base class for implementing device-specific
optimizations (CPU, MPS, CUDA). Each device has different performance characteristics
and requires tailored optimization strategies.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict
from dataclasses import dataclass
import torch


@dataclass
class OptimizationConfig:
    """Configuration for performance optimizations."""

    # Mixed precision settings
    mixed_precision: bool | str | None = None  # False, "float16", "bfloat16", or None (auto)

    # Preprocessing settings
    num_workers: int = 8
    prefetch_factor: int = 2

    # Compilation settings
    enable_compile: bool = False
    compile_mode: str = "reduce-overhead"  # "default", "reduce-overhead", "max-autotune"

    # Batch processing
    batch_size: int | None = None
    dynamic_batching: bool = True

    # Performance mode
    performance_mode: str = "balanced"  # "minimal", "balanced", "max"

    # Device-specific overrides
    device_specific: Dict[str, Any] | None = None


class BaseOptimizer(ABC):
    """
    Abstract base class for device-specific optimizers.

    Each optimizer is responsible for:
    - Configuring platform-specific settings (cudnn, mkldnn, etc.)
    - Setting optimal mixed precision strategy
    - Configuring preprocessing workers
    - Managing memory optimizations
    - Applying compilation strategies
    """

    def __init__(self, config: OptimizationConfig | None = None):
        """
        Initialize optimizer with configuration.

        Args:
            config: Optimization configuration. If None, uses device-specific defaults.
        """
        self.config = config or self.get_default_config()
        self._applied = False

    @abstractmethod
    def get_default_config(self) -> OptimizationConfig:
        """Get default optimization configuration for this device."""
        pass

    @abstractmethod
    def apply_platform_settings(self) -> None:
        """Apply platform-specific PyTorch settings (cudnn, mkldnn, etc.)."""
        pass

    @abstractmethod
    def get_mixed_precision_dtype(self) -> torch.dtype | None:
        """
        Get the optimal mixed precision dtype for this device.

        Returns:
            torch.dtype for autocast, or None to disable mixed precision.
        """
        pass

    @abstractmethod
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """
        Get optimal preprocessing configuration.

        Returns:
            Dictionary with keys: num_workers, prefetch_enabled, etc.
        """
        pass

    @abstractmethod
    def should_use_compile(self) -> bool:
        """Determine if torch.compile should be used for this device."""
        pass

    @abstractmethod
    def get_memory_config(self) -> Dict[str, Any]:
        """
        Get memory optimization configuration.

        Returns:
            Dictionary with memory-related settings (pin_memory, non_blocking, etc.)
        """
        pass

    def apply(self) -> None:
        """Apply all optimizations for this device."""
        if self._applied:
            return

        self.apply_platform_settings()
        self._applied = True

    def get_inference_context(self):
        """
        Get context manager for optimized inference.

        Returns:
            Context manager (e.g., autocast) or None
        """
        dtype = self.get_mixed_precision_dtype()
        if dtype is None:
            return torch.inference_mode()

        device_type = self._get_device_type()
        return torch.amp.autocast(device_type=device_type, dtype=dtype)

    @abstractmethod
    def _get_device_type(self) -> str:
        """Get device type string ('cpu', 'cuda', 'mps')."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"
