# Copyright (c) Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0
"""
Adaptive Batching Module for Depth Anything 3.

This module provides intelligent batch size selection based on available GPU memory,
model parameters, and input resolution. It maximizes throughput by dynamically
adjusting batch sizes to utilize as much GPU memory as safely possible.

Key features:
- Memory profiling for accurate estimation
- Model-specific memory coefficients
- Resolution-aware scaling
- Safety margins to prevent OOM
- Support for CUDA and MPS devices
"""
from __future__ import annotations

import gc
import math
from dataclasses import dataclass, field
from typing import Callable, Iterator, Sequence, TypeVar

import torch

from depth_anything_3.utils.logger import logger

T = TypeVar("T")


# =============================================================================
# Model Memory Profiles
# =============================================================================

@dataclass
class ModelMemoryProfile:
    """Memory profile for a specific model variant.

    Attributes:
        base_memory_mb: Fixed memory overhead (model weights, buffers)
        per_image_mb_at_504: Memory per image at 504px resolution
        activation_scale: Scaling factor for activations (quadratic with resolution)
        safety_margin: Safety margin to prevent OOM (0.0 to 1.0)
    """
    base_memory_mb: float
    per_image_mb_at_504: float
    activation_scale: float = 1.0
    safety_margin: float = 0.15  # 15% safety margin by default


# Empirically measured memory profiles for each model variant
# Values calibrated on various GPU configurations
MODEL_MEMORY_PROFILES: dict[str, ModelMemoryProfile] = {
    # Small models (ViT-S backbone)
    "da3-small": ModelMemoryProfile(
        base_memory_mb=350,
        per_image_mb_at_504=180,
        activation_scale=0.8,
    ),
    # Base models (ViT-B backbone)
    "da3-base": ModelMemoryProfile(
        base_memory_mb=800,
        per_image_mb_at_504=350,
        activation_scale=1.0,
    ),
    # Large models (ViT-L backbone)
    "da3-large": ModelMemoryProfile(
        base_memory_mb=1600,
        per_image_mb_at_504=600,
        activation_scale=1.2,
    ),
    "da3metric-large": ModelMemoryProfile(
        base_memory_mb=1700,
        per_image_mb_at_504=650,
        activation_scale=1.2,
    ),
    "da3mono-large": ModelMemoryProfile(
        base_memory_mb=1700,
        per_image_mb_at_504=650,
        activation_scale=1.2,
    ),
    # Giant models (ViT-G backbone)
    "da3-giant": ModelMemoryProfile(
        base_memory_mb=4500,
        per_image_mb_at_504=1200,
        activation_scale=1.5,
    ),
    "da3nested-giant-large": ModelMemoryProfile(
        base_memory_mb=6000,
        per_image_mb_at_504=1500,
        activation_scale=1.8,
    ),
}


# =============================================================================
# Memory Utilities
# =============================================================================

def get_available_memory_mb(device: torch.device) -> float:
    """Get available GPU memory in MB.

    Args:
        device: Target device

    Returns:
        Available memory in MB, or float('inf') for CPU
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        total = torch.cuda.get_device_properties(device).total_memory
        reserved = torch.cuda.memory_reserved(device)
        return (total - reserved) / (1024 * 1024)

    elif device.type == "mps":
        # MPS doesn't expose free memory directly
        # Use system memory as a rough proxy (conservative estimate)
        try:
            allocated = torch.mps.current_allocated_memory()
            # Assume 8GB available for MPS (conservative for most Apple Silicon)
            # This can be overridden via environment variable
            import os
            max_mps_memory_gb = float(os.environ.get("DA3_MPS_MAX_MEMORY_GB", "8"))
            max_mps_memory_mb = max_mps_memory_gb * 1024
            return max(0, max_mps_memory_mb - (allocated / (1024 * 1024)))
        except Exception:
            return 6000  # Conservative fallback: 6GB

    else:
        return float("inf")  # CPU has no practical limit for batching


def get_total_memory_mb(device: torch.device) -> float:
    """Get total GPU memory in MB.

    Args:
        device: Target device

    Returns:
        Total memory in MB
    """
    if device.type == "cuda":
        return torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)

    elif device.type == "mps":
        import os
        return float(os.environ.get("DA3_MPS_MAX_MEMORY_GB", "8")) * 1024

    else:
        return float("inf")


# =============================================================================
# Adaptive Batch Size Calculator
# =============================================================================

@dataclass
class AdaptiveBatchConfig:
    """Configuration for adaptive batching.

    Attributes:
        min_batch_size: Minimum batch size (default: 1)
        max_batch_size: Maximum batch size cap (default: 64)
        target_memory_utilization: Target GPU memory usage (0.0 to 1.0)
        enable_profiling: Enable runtime memory profiling for calibration
        profile_warmup_batches: Number of warmup batches before profiling
    """
    min_batch_size: int = 1
    max_batch_size: int = 64
    target_memory_utilization: float = 0.85  # Use 85% of available memory
    enable_profiling: bool = True
    profile_warmup_batches: int = 2


class AdaptiveBatchSizeCalculator:
    """
    Calculates optimal batch sizes based on available GPU memory.

    This class provides intelligent batch size selection that:
    1. Estimates memory requirements based on model and resolution
    2. Measures actual memory usage during runtime (optional profiling)
    3. Adjusts batch sizes dynamically to maximize throughput

    Example:
        >>> from depth_anything_3.utils.adaptive_batching import AdaptiveBatchSizeCalculator
        >>> calc = AdaptiveBatchSizeCalculator(
        ...     model_name="da3-large",
        ...     device=torch.device("cuda"),
        ... )
        >>> # Get optimal batch size for 100 images at 518px
        >>> batch_size = calc.compute_optimal_batch_size(
        ...     num_images=100,
        ...     process_res=518,
        ... )
        >>> print(f"Optimal batch size: {batch_size}")
    """

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        config: AdaptiveBatchConfig | None = None,
    ):
        """Initialize the adaptive batch size calculator.

        Args:
            model_name: Name of the DA3 model variant
            device: Target device for inference
            config: Optional configuration overrides
        """
        self.model_name = model_name
        self.device = device
        self.config = config or AdaptiveBatchConfig()

        # Get memory profile for this model
        self.profile = MODEL_MEMORY_PROFILES.get(
            model_name,
            # Fallback to large model profile for unknown models
            MODEL_MEMORY_PROFILES["da3-large"]
        )

        # Runtime calibration data
        self._measured_per_image_mb: float | None = None
        self._profiling_complete: bool = False
        self._batch_count: int = 0

    def compute_optimal_batch_size(
        self,
        num_images: int,
        process_res: int = 504,
        reserved_memory_mb: float = 0,
    ) -> int:
        """Compute optimal batch size for given workload.

        Args:
            num_images: Total number of images to process
            process_res: Processing resolution
            reserved_memory_mb: Additional memory to reserve (e.g., for other operations)

        Returns:
            Optimal batch size
        """
        # Get available memory
        available_mb = get_available_memory_mb(self.device)

        if available_mb == float("inf"):
            # CPU: return reasonable batch size based on image count
            return min(num_images, self.config.max_batch_size)

        # Apply target utilization and reserve
        usable_mb = (available_mb * self.config.target_memory_utilization) - reserved_memory_mb

        # Subtract base model memory
        usable_mb -= self.profile.base_memory_mb

        if usable_mb <= 0:
            logger.warn(
                f"Insufficient memory for model. "
                f"Available: {available_mb:.0f} MB, "
                f"Model base: {self.profile.base_memory_mb:.0f} MB"
            )
            return self.config.min_batch_size

        # Calculate per-image memory requirement
        per_image_mb = self._estimate_per_image_memory(process_res)

        # Apply safety margin
        per_image_mb *= (1 + self.profile.safety_margin)

        # Calculate optimal batch size
        optimal_batch = int(usable_mb / per_image_mb)

        # Clamp to configured bounds
        optimal_batch = max(self.config.min_batch_size, optimal_batch)
        optimal_batch = min(self.config.max_batch_size, optimal_batch)
        optimal_batch = min(num_images, optimal_batch)

        logger.debug(
            f"Adaptive batch: {optimal_batch} "
            f"(available: {available_mb:.0f} MB, "
            f"per_image: {per_image_mb:.0f} MB @ {process_res}px)"
        )

        return optimal_batch

    def _estimate_per_image_memory(self, process_res: int) -> float:
        """Estimate memory per image at given resolution.

        Memory scales approximately quadratically with resolution.

        Args:
            process_res: Processing resolution

        Returns:
            Estimated memory per image in MB
        """
        # Use measured value if available from profiling
        if self._measured_per_image_mb is not None and self._profiling_complete:
            base_per_image = self._measured_per_image_mb
        else:
            base_per_image = self.profile.per_image_mb_at_504

        # Scale quadratically with resolution
        resolution_scale = (process_res / 504) ** 2

        # Apply model-specific activation scale
        return base_per_image * resolution_scale * self.profile.activation_scale

    def update_from_profiling(self, batch_size: int, memory_used_mb: float, process_res: int) -> None:
        """Update memory estimates from actual profiling data.

        Called after inference to calibrate memory estimates.

        Args:
            batch_size: Batch size used
            memory_used_mb: Actual memory consumed
            process_res: Resolution used
        """
        if not self.config.enable_profiling:
            return

        self._batch_count += 1

        if self._batch_count <= self.config.profile_warmup_batches:
            # Skip warmup batches (memory not stable)
            return

        # Calculate per-image memory at reference resolution (504)
        resolution_scale = (process_res / 504) ** 2
        memory_per_image = (memory_used_mb - self.profile.base_memory_mb) / batch_size
        memory_at_504 = memory_per_image / resolution_scale / self.profile.activation_scale

        # Exponential moving average for stability
        alpha = 0.3
        if self._measured_per_image_mb is None:
            self._measured_per_image_mb = memory_at_504
        else:
            self._measured_per_image_mb = (
                alpha * memory_at_504 +
                (1 - alpha) * self._measured_per_image_mb
            )

        self._profiling_complete = True

        logger.debug(
            f"Profiling update: measured {memory_at_504:.0f} MB/img @ 504px "
            f"(running avg: {self._measured_per_image_mb:.0f} MB)"
        )

    def get_memory_estimate(self, batch_size: int, process_res: int) -> float:
        """Get estimated total memory for a batch.

        Args:
            batch_size: Batch size
            process_res: Processing resolution

        Returns:
            Estimated memory in MB
        """
        per_image = self._estimate_per_image_memory(process_res)
        return self.profile.base_memory_mb + (batch_size * per_image)


# =============================================================================
# Batch Iterator
# =============================================================================

@dataclass
class BatchInfo:
    """Information about a batch for processing.

    Attributes:
        batch_idx: Index of this batch (0-indexed)
        start_idx: Start index in original sequence
        end_idx: End index in original sequence (exclusive)
        items: Items in this batch
        batch_size: Size of this batch
        is_last: Whether this is the last batch
    """
    batch_idx: int
    start_idx: int
    end_idx: int
    items: list
    batch_size: int = field(init=False)
    is_last: bool = False

    def __post_init__(self):
        self.batch_size = len(self.items)


def adaptive_batch_iterator(
    items: Sequence[T],
    calculator: AdaptiveBatchSizeCalculator,
    process_res: int = 504,
    reserved_memory_mb: float = 0,
) -> Iterator[BatchInfo]:
    """
    Iterate over items with adaptive batch sizes.

    This iterator dynamically adjusts batch sizes based on available memory,
    potentially increasing throughput compared to fixed batch sizes.

    Args:
        items: Sequence of items to batch
        calculator: Adaptive batch size calculator
        process_res: Processing resolution
        reserved_memory_mb: Additional memory to reserve

    Yields:
        BatchInfo objects containing batch data and metadata

    Example:
        >>> calc = AdaptiveBatchSizeCalculator("da3-large", device)
        >>> for batch_info in adaptive_batch_iterator(images, calc, process_res=518):
        ...     result = model.inference(batch_info.items, process_res=518)
        ...     # Process result...
    """
    total = len(items)
    idx = 0
    batch_idx = 0

    while idx < total:
        remaining = total - idx

        # Compute optimal batch size for remaining items
        batch_size = calculator.compute_optimal_batch_size(
            num_images=remaining,
            process_res=process_res,
            reserved_memory_mb=reserved_memory_mb,
        )

        end_idx = min(idx + batch_size, total)
        batch_items = list(items[idx:end_idx])

        yield BatchInfo(
            batch_idx=batch_idx,
            start_idx=idx,
            end_idx=end_idx,
            items=batch_items,
            is_last=(end_idx >= total),
        )

        idx = end_idx
        batch_idx += 1


# =============================================================================
# High-Level API
# =============================================================================

def process_with_adaptive_batching(
    items: Sequence[T],
    process_fn: Callable[[list[T]], list],
    model_name: str,
    device: torch.device,
    process_res: int = 504,
    config: AdaptiveBatchConfig | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list:
    """
    Process items with adaptive batching for optimal GPU utilization.

    This function handles the complete workflow of:
    1. Computing optimal batch sizes
    2. Processing batches
    3. Collecting and returning results
    4. Memory cleanup between batches

    Args:
        items: Sequence of items to process
        process_fn: Function to process a batch of items
        model_name: Name of the DA3 model
        device: Target device
        process_res: Processing resolution
        config: Optional batching configuration
        progress_callback: Optional callback(processed, total) for progress updates

    Returns:
        List of all results concatenated

    Example:
        >>> def inference_fn(batch):
        ...     return model.inference(batch, process_res=518)
        >>>
        >>> results = process_with_adaptive_batching(
        ...     items=image_paths,
        ...     process_fn=inference_fn,
        ...     model_name="da3-large",
        ...     device=torch.device("cuda"),
        ...     process_res=518,
        ... )
    """
    calculator = AdaptiveBatchSizeCalculator(
        model_name=model_name,
        device=device,
        config=config,
    )

    all_results = []
    total = len(items)

    for batch_info in adaptive_batch_iterator(items, calculator, process_res):
        # Process batch
        results = process_fn(batch_info.items)
        all_results.extend(results if isinstance(results, list) else [results])

        # Progress callback
        if progress_callback:
            progress_callback(batch_info.end_idx, total)

        # Memory cleanup between batches (except last)
        if not batch_info.is_last:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()

        # Optional: profile memory usage for calibration
        if calculator.config.enable_profiling and device.type == "cuda":
            memory_used = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            calculator.update_from_profiling(
                batch_size=batch_info.batch_size,
                memory_used_mb=memory_used,
                process_res=process_res,
            )
            torch.cuda.reset_peak_memory_stats(device)

    return all_results


# =============================================================================
# Utility Functions
# =============================================================================

def estimate_max_batch_size(
    model_name: str,
    device: torch.device,
    process_res: int = 504,
    target_utilization: float = 0.85,
) -> int:
    """
    Estimate maximum batch size for a given model and resolution.

    Quick utility function for one-off batch size estimation.

    Args:
        model_name: Name of the DA3 model
        device: Target device
        process_res: Processing resolution
        target_utilization: Target memory utilization (0.0 to 1.0)

    Returns:
        Estimated maximum batch size

    Example:
        >>> max_batch = estimate_max_batch_size("da3-large", torch.device("cuda"), 518)
        >>> print(f"Max batch size at 518px: {max_batch}")
    """
    config = AdaptiveBatchConfig(target_memory_utilization=target_utilization)
    calculator = AdaptiveBatchSizeCalculator(model_name, device, config)

    # Return estimate for a large number of images
    return calculator.compute_optimal_batch_size(num_images=1000, process_res=process_res)


def log_batch_plan(
    num_images: int,
    model_name: str,
    device: torch.device,
    process_res: int = 504,
) -> None:
    """
    Log the planned batching strategy for a workload.

    Useful for debugging and understanding how images will be batched.

    Args:
        num_images: Number of images to process
        model_name: Name of the DA3 model
        device: Target device
        process_res: Processing resolution
    """
    calculator = AdaptiveBatchSizeCalculator(model_name, device)

    total_memory = get_total_memory_mb(device)
    available_memory = get_available_memory_mb(device)
    batch_size = calculator.compute_optimal_batch_size(num_images, process_res)
    num_batches = math.ceil(num_images / batch_size)
    memory_per_batch = calculator.get_memory_estimate(batch_size, process_res)

    logger.info(
        f"Batch Plan for {model_name}:\n"
        f"  Images: {num_images} @ {process_res}px\n"
        f"  Device: {device} ({total_memory:.0f} MB total, {available_memory:.0f} MB available)\n"
        f"  Batch Size: {batch_size}\n"
        f"  Num Batches: {num_batches}\n"
        f"  Est. Memory/Batch: {memory_per_batch:.0f} MB"
    )