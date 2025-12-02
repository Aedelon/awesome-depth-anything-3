"""
Device-specific performance optimizations for Depth Anything 3.

This module provides automatic optimization based on the target device (CPU, MPS, CUDA).
Each device has different performance characteristics and requires tailored strategies.

Usage:
    optimizer = get_optimizer(device="cuda", performance_mode="max")
    optimizer.apply()

    # In inference
    with optimizer.get_inference_context():
        output = model(input)
"""

from __future__ import annotations
import torch
from typing import Literal

from depth_anything_3.optimizations.base_optimizer import BaseOptimizer, OptimizationConfig
from depth_anything_3.optimizations.cpu_optimizer import CPUOptimizer
from depth_anything_3.optimizations.mps_optimizer import MPSOptimizer
from depth_anything_3.optimizations.cuda_optimizer import CUDAOptimizer


__all__ = [
    "get_optimizer",
    "BaseOptimizer",
    "OptimizationConfig",
    "CPUOptimizer",
    "MPSOptimizer",
    "CUDAOptimizer",
]


def get_optimizer(
    device: str | torch.device,
    config: OptimizationConfig | None = None,
    performance_mode: Literal["minimal", "balanced", "max"] = "balanced",
) -> BaseOptimizer:
    """
    Factory function to get the appropriate optimizer for the target device.

    Args:
        device: Target device ('cpu', 'cuda', 'mps', or torch.device)
        config: Optional optimization configuration. If None, uses device defaults.
        performance_mode: Performance/compatibility tradeoff:
            - "minimal": Conservative, maximum compatibility
            - "balanced": Good performance with stability (default)
            - "max": Maximum performance, may reduce compatibility

    Returns:
        Device-specific optimizer instance

    Examples:
        >>> # Auto-detect device and use balanced mode
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> optimizer = get_optimizer(device)
        >>> optimizer.apply()

        >>> # Maximum performance on CUDA with custom config
        >>> config = OptimizationConfig(enable_compile=True, mixed_precision="bfloat16")
        >>> optimizer = get_optimizer("cuda", config=config, performance_mode="max")
        >>> optimizer.apply()
    """
    # Normalize device to string
    if isinstance(device, torch.device):
        device_str = device.type
    else:
        device_str = str(device).lower()

    # Update performance mode if config is None
    if config is None:
        config = OptimizationConfig(performance_mode=performance_mode)
    else:
        config.performance_mode = performance_mode

    # Select appropriate optimizer
    if device_str == "cuda":
        return CUDAOptimizer(config)
    elif device_str == "mps":
        return MPSOptimizer(config)
    else:  # cpu or unknown
        return CPUOptimizer(config)


def get_default_device() -> torch.device:
    """
    Get the default device with priority: CUDA > MPS > CPU

    Returns:
        torch.device: The best available device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")