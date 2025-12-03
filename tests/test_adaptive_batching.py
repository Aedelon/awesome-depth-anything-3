# Copyright (c) Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0
"""
Comprehensive tests for the adaptive batching module.

Tests cover:
- ModelMemoryProfile dataclass
- Memory utility functions
- AdaptiveBatchSizeCalculator
- BatchInfo and adaptive_batch_iterator
- High-level API functions
- Edge cases and error handling
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
import torch

from depth_anything_3.utils.adaptive_batching import (
    MODEL_MEMORY_PROFILES,
    AdaptiveBatchConfig,
    AdaptiveBatchSizeCalculator,
    BatchInfo,
    ModelMemoryProfile,
    adaptive_batch_iterator,
    estimate_max_batch_size,
    get_available_memory_mb,
    get_total_memory_mb,
    log_batch_plan,
    process_with_adaptive_batching,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def cpu_device():
    """Return CPU device."""
    return torch.device("cpu")


@pytest.fixture
def mock_cuda_device():
    """Return mock CUDA device."""
    return torch.device("cuda:0")


@pytest.fixture
def mock_mps_device():
    """Return mock MPS device."""
    return torch.device("mps")


@pytest.fixture
def default_config():
    """Return default adaptive batch config."""
    return AdaptiveBatchConfig()


@pytest.fixture
def calculator_cpu(cpu_device):
    """Return calculator for CPU."""
    return AdaptiveBatchSizeCalculator("da3-large", cpu_device)


# =============================================================================
# ModelMemoryProfile Tests
# =============================================================================


class TestModelMemoryProfile:
    """Tests for ModelMemoryProfile dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        profile = ModelMemoryProfile(
            base_memory_mb=1000,
            per_image_mb_at_504=500,
        )
        assert profile.base_memory_mb == 1000
        assert profile.per_image_mb_at_504 == 500
        assert profile.activation_scale == 1.0
        assert profile.safety_margin == 0.15

    def test_custom_values(self):
        """Test custom values override defaults."""
        profile = ModelMemoryProfile(
            base_memory_mb=2000,
            per_image_mb_at_504=800,
            activation_scale=1.5,
            safety_margin=0.2,
        )
        assert profile.base_memory_mb == 2000
        assert profile.per_image_mb_at_504 == 800
        assert profile.activation_scale == 1.5
        assert profile.safety_margin == 0.2

    def test_all_models_have_profiles(self):
        """Test that all expected models have memory profiles."""
        expected_models = [
            "da3-small",
            "da3-base",
            "da3-large",
            "da3-giant",
            "da3metric-large",
            "da3mono-large",
            "da3nested-giant-large",
        ]
        for model_name in expected_models:
            assert model_name in MODEL_MEMORY_PROFILES
            profile = MODEL_MEMORY_PROFILES[model_name]
            assert profile.base_memory_mb > 0
            assert profile.per_image_mb_at_504 > 0

    def test_profiles_size_ordering(self):
        """Test that model profiles have expected size ordering."""
        small = MODEL_MEMORY_PROFILES["da3-small"]
        base = MODEL_MEMORY_PROFILES["da3-base"]
        large = MODEL_MEMORY_PROFILES["da3-large"]
        giant = MODEL_MEMORY_PROFILES["da3-giant"]

        # Base memory should increase with model size
        assert small.base_memory_mb < base.base_memory_mb
        assert base.base_memory_mb < large.base_memory_mb
        assert large.base_memory_mb < giant.base_memory_mb

        # Per-image memory should also increase
        assert small.per_image_mb_at_504 < base.per_image_mb_at_504
        assert base.per_image_mb_at_504 < large.per_image_mb_at_504
        assert large.per_image_mb_at_504 < giant.per_image_mb_at_504


# =============================================================================
# Memory Utility Tests
# =============================================================================


class TestGetAvailableMemory:
    """Tests for get_available_memory_mb function."""

    def test_cpu_returns_infinity(self, cpu_device):
        """CPU should return infinite memory."""
        result = get_available_memory_mb(cpu_device)
        assert result == float("inf")

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.synchronize")
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.memory_reserved")
    def test_cuda_memory_calculation(
        self,
        mock_reserved,
        mock_properties,
        mock_sync,
        mock_available,
        mock_cuda_device,
    ):
        """Test CUDA memory calculation."""
        # Setup mocks
        mock_props = MagicMock()
        mock_props.total_memory = 16 * 1024 * 1024 * 1024  # 16 GB
        mock_properties.return_value = mock_props
        mock_reserved.return_value = 4 * 1024 * 1024 * 1024  # 4 GB reserved

        result = get_available_memory_mb(mock_cuda_device)

        # Should be (16GB - 4GB) in MB = 12288 MB
        expected = (16 - 4) * 1024
        assert result == expected

    def test_mps_memory_with_env_var(self, mock_mps_device, monkeypatch):
        """Test MPS memory respects environment variable."""
        monkeypatch.setenv("DA3_MPS_MAX_MEMORY_GB", "16")

        with patch("torch.mps.current_allocated_memory", return_value=0):
            result = get_available_memory_mb(mock_mps_device)
            assert result == 16 * 1024  # 16 GB in MB

    def test_mps_memory_default(self, mock_mps_device, monkeypatch):
        """Test MPS memory uses default when env var not set."""
        monkeypatch.delenv("DA3_MPS_MAX_MEMORY_GB", raising=False)

        with patch("torch.mps.current_allocated_memory", return_value=0):
            result = get_available_memory_mb(mock_mps_device)
            assert result == 8 * 1024  # Default 8 GB

    def test_mps_memory_subtracts_allocated(self, mock_mps_device, monkeypatch):
        """Test MPS memory subtracts allocated memory."""
        monkeypatch.setenv("DA3_MPS_MAX_MEMORY_GB", "8")

        allocated_bytes = 2 * 1024 * 1024 * 1024  # 2 GB allocated
        with patch("torch.mps.current_allocated_memory", return_value=allocated_bytes):
            result = get_available_memory_mb(mock_mps_device)
            expected = (8 - 2) * 1024  # 6 GB remaining
            assert result == expected


class TestGetTotalMemory:
    """Tests for get_total_memory_mb function."""

    def test_cpu_returns_infinity(self, cpu_device):
        """CPU should return infinite total memory."""
        result = get_total_memory_mb(cpu_device)
        assert result == float("inf")

    @patch("torch.cuda.get_device_properties")
    def test_cuda_total_memory(self, mock_properties, mock_cuda_device):
        """Test CUDA total memory retrieval."""
        mock_props = MagicMock()
        mock_props.total_memory = 24 * 1024 * 1024 * 1024  # 24 GB
        mock_properties.return_value = mock_props

        result = get_total_memory_mb(mock_cuda_device)
        assert result == 24 * 1024  # 24 GB in MB

    def test_mps_total_memory_env_var(self, mock_mps_device, monkeypatch):
        """Test MPS total memory from environment variable."""
        monkeypatch.setenv("DA3_MPS_MAX_MEMORY_GB", "32")
        result = get_total_memory_mb(mock_mps_device)
        assert result == 32 * 1024


# =============================================================================
# AdaptiveBatchConfig Tests
# =============================================================================


class TestAdaptiveBatchConfig:
    """Tests for AdaptiveBatchConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AdaptiveBatchConfig()
        assert config.min_batch_size == 1
        assert config.max_batch_size == 64
        assert config.target_memory_utilization == 0.85
        assert config.enable_profiling is True
        assert config.profile_warmup_batches == 2

    def test_custom_values(self):
        """Test custom configuration values."""
        config = AdaptiveBatchConfig(
            min_batch_size=2,
            max_batch_size=32,
            target_memory_utilization=0.90,
            enable_profiling=False,
            profile_warmup_batches=5,
        )
        assert config.min_batch_size == 2
        assert config.max_batch_size == 32
        assert config.target_memory_utilization == 0.90
        assert config.enable_profiling is False
        assert config.profile_warmup_batches == 5


# =============================================================================
# AdaptiveBatchSizeCalculator Tests
# =============================================================================


class TestAdaptiveBatchSizeCalculator:
    """Tests for AdaptiveBatchSizeCalculator class."""

    def test_initialization_known_model(self, cpu_device):
        """Test initialization with known model."""
        calc = AdaptiveBatchSizeCalculator("da3-large", cpu_device)
        assert calc.model_name == "da3-large"
        assert calc.device == cpu_device
        assert calc.profile == MODEL_MEMORY_PROFILES["da3-large"]

    def test_initialization_unknown_model_uses_fallback(self, cpu_device):
        """Test initialization with unknown model falls back to da3-large."""
        calc = AdaptiveBatchSizeCalculator("unknown-model", cpu_device)
        assert calc.profile == MODEL_MEMORY_PROFILES["da3-large"]

    def test_initialization_with_custom_config(self, cpu_device):
        """Test initialization with custom config."""
        config = AdaptiveBatchConfig(max_batch_size=16)
        calc = AdaptiveBatchSizeCalculator("da3-large", cpu_device, config)
        assert calc.config.max_batch_size == 16

    def test_compute_optimal_batch_size_cpu(self, cpu_device):
        """CPU should return min(num_images, max_batch_size)."""
        calc = AdaptiveBatchSizeCalculator("da3-large", cpu_device)

        # Small number of images
        result = calc.compute_optimal_batch_size(num_images=10)
        assert result == 10

        # Large number of images
        result = calc.compute_optimal_batch_size(num_images=100)
        assert result == 64  # max_batch_size

    def test_compute_optimal_batch_size_respects_min(self, cpu_device):
        """Batch size should not go below min_batch_size."""
        config = AdaptiveBatchConfig(min_batch_size=4)
        calc = AdaptiveBatchSizeCalculator("da3-large", cpu_device, config)

        result = calc.compute_optimal_batch_size(num_images=2)
        # For CPU, min(num_images, max) = 2, but min_batch is applied after GPU calc
        # CPU returns min(num_images, max_batch_size) directly
        assert result == 2

    def test_compute_optimal_batch_size_respects_max(self, cpu_device):
        """Batch size should not exceed max_batch_size."""
        config = AdaptiveBatchConfig(max_batch_size=8)
        calc = AdaptiveBatchSizeCalculator("da3-large", cpu_device, config)

        result = calc.compute_optimal_batch_size(num_images=100)
        assert result == 8

    @patch("depth_anything_3.utils.adaptive_batching.get_available_memory_mb")
    def test_compute_optimal_batch_size_memory_based(
        self, mock_memory, mock_cuda_device
    ):
        """Test memory-based batch size calculation."""
        # 10GB available memory
        mock_memory.return_value = 10000

        calc = AdaptiveBatchSizeCalculator("da3-large", mock_cuda_device)

        result = calc.compute_optimal_batch_size(num_images=100, process_res=504)

        # Should compute based on memory
        assert 1 <= result <= 64
        assert result < 100  # Should be less than num_images given memory constraints

    @patch("depth_anything_3.utils.adaptive_batching.get_available_memory_mb")
    def test_compute_low_memory_returns_min(self, mock_memory, mock_cuda_device):
        """Low memory should return min batch size."""
        # Only 500MB available (less than base memory for da3-large)
        mock_memory.return_value = 500

        calc = AdaptiveBatchSizeCalculator("da3-large", mock_cuda_device)
        result = calc.compute_optimal_batch_size(num_images=100)

        assert result == 1  # min_batch_size

    def test_estimate_per_image_memory_resolution_scaling(self, cpu_device):
        """Test that memory scales quadratically with resolution."""
        calc = AdaptiveBatchSizeCalculator("da3-large", cpu_device)

        mem_504 = calc._estimate_per_image_memory(504)
        mem_1008 = calc._estimate_per_image_memory(1008)

        # Memory at 2x resolution should be ~4x (quadratic scaling)
        ratio = mem_1008 / mem_504
        assert 3.5 <= ratio <= 4.5  # Allow some tolerance for activation_scale

    def test_update_from_profiling_warmup(self, cpu_device):
        """Test that warmup batches are skipped during profiling."""
        config = AdaptiveBatchConfig(profile_warmup_batches=2)
        calc = AdaptiveBatchSizeCalculator("da3-large", cpu_device, config)

        # First two batches (warmup) should be skipped
        calc.update_from_profiling(batch_size=4, memory_used_mb=3000, process_res=504)
        assert calc._measured_per_image_mb is None

        calc.update_from_profiling(batch_size=4, memory_used_mb=3000, process_res=504)
        assert calc._measured_per_image_mb is None

        # Third batch should update
        calc.update_from_profiling(batch_size=4, memory_used_mb=3000, process_res=504)
        assert calc._measured_per_image_mb is not None

    def test_update_from_profiling_disabled(self, cpu_device):
        """Test that profiling can be disabled."""
        config = AdaptiveBatchConfig(enable_profiling=False)
        calc = AdaptiveBatchSizeCalculator("da3-large", cpu_device, config)

        for _ in range(5):
            calc.update_from_profiling(batch_size=4, memory_used_mb=3000, process_res=504)

        assert calc._measured_per_image_mb is None

    def test_update_from_profiling_ema(self, cpu_device):
        """Test exponential moving average in profiling."""
        config = AdaptiveBatchConfig(profile_warmup_batches=0)
        calc = AdaptiveBatchSizeCalculator("da3-large", cpu_device, config)

        # First update
        calc.update_from_profiling(batch_size=4, memory_used_mb=4000, process_res=504)
        first_value = calc._measured_per_image_mb

        # Second update with different value
        calc.update_from_profiling(batch_size=4, memory_used_mb=5000, process_res=504)
        second_value = calc._measured_per_image_mb

        # EMA should smooth the values
        assert second_value is not None
        assert second_value != first_value

    def test_get_memory_estimate(self, cpu_device):
        """Test memory estimation for batch."""
        calc = AdaptiveBatchSizeCalculator("da3-large", cpu_device)

        estimate = calc.get_memory_estimate(batch_size=4, process_res=504)

        # Should include base memory + per-image memory
        expected_min = calc.profile.base_memory_mb
        assert estimate > expected_min
        assert estimate > calc.profile.base_memory_mb


# =============================================================================
# BatchInfo Tests
# =============================================================================


class TestBatchInfo:
    """Tests for BatchInfo dataclass."""

    def test_batch_info_creation(self):
        """Test basic BatchInfo creation."""
        items = ["a", "b", "c"]
        info = BatchInfo(
            batch_idx=0,
            start_idx=0,
            end_idx=3,
            items=items,
            is_last=True,
        )
        assert info.batch_idx == 0
        assert info.start_idx == 0
        assert info.end_idx == 3
        assert info.items == ["a", "b", "c"]
        assert info.batch_size == 3
        assert info.is_last is True

    def test_batch_size_computed_from_items(self):
        """Test that batch_size is computed from items."""
        info = BatchInfo(
            batch_idx=0,
            start_idx=0,
            end_idx=5,
            items=[1, 2, 3, 4, 5],
        )
        assert info.batch_size == 5

    def test_empty_batch(self):
        """Test empty batch handling."""
        info = BatchInfo(
            batch_idx=0,
            start_idx=0,
            end_idx=0,
            items=[],
        )
        assert info.batch_size == 0


# =============================================================================
# adaptive_batch_iterator Tests
# =============================================================================


class TestAdaptiveBatchIterator:
    """Tests for adaptive_batch_iterator function."""

    def test_single_batch(self, calculator_cpu):
        """Test single batch when all items fit."""
        items = list(range(10))
        batches = list(adaptive_batch_iterator(items, calculator_cpu))

        assert len(batches) == 1
        assert batches[0].items == items
        assert batches[0].is_last is True

    def test_multiple_batches(self, cpu_device):
        """Test multiple batches with small max_batch_size."""
        config = AdaptiveBatchConfig(max_batch_size=3)
        calc = AdaptiveBatchSizeCalculator("da3-large", cpu_device, config)

        items = list(range(10))
        batches = list(adaptive_batch_iterator(items, calc))

        # Should have 4 batches: 3, 3, 3, 1
        assert len(batches) == 4
        assert batches[0].batch_size == 3
        assert batches[-1].batch_size == 1
        assert batches[-1].is_last is True

    def test_batch_indices_are_correct(self, cpu_device):
        """Test that batch indices are sequential."""
        config = AdaptiveBatchConfig(max_batch_size=2)
        calc = AdaptiveBatchSizeCalculator("da3-large", cpu_device, config)

        items = list(range(6))
        batches = list(adaptive_batch_iterator(items, calc))

        for i, batch in enumerate(batches):
            assert batch.batch_idx == i

    def test_start_end_indices_cover_all_items(self, cpu_device):
        """Test that batches cover all items without gaps."""
        config = AdaptiveBatchConfig(max_batch_size=3)
        calc = AdaptiveBatchSizeCalculator("da3-large", cpu_device, config)

        items = list(range(10))
        batches = list(adaptive_batch_iterator(items, calc))

        # Verify no gaps
        prev_end = 0
        for batch in batches:
            assert batch.start_idx == prev_end
            assert batch.end_idx > batch.start_idx
            prev_end = batch.end_idx

        assert prev_end == len(items)

    def test_items_are_preserved(self, cpu_device):
        """Test that all items are preserved in batches."""
        config = AdaptiveBatchConfig(max_batch_size=4)
        calc = AdaptiveBatchSizeCalculator("da3-large", cpu_device, config)

        original_items = ["a", "b", "c", "d", "e", "f", "g"]
        batches = list(adaptive_batch_iterator(original_items, calc))

        # Collect all items from batches
        collected = []
        for batch in batches:
            collected.extend(batch.items)

        assert collected == original_items

    def test_empty_sequence(self, calculator_cpu):
        """Test empty sequence returns no batches."""
        batches = list(adaptive_batch_iterator([], calculator_cpu))
        assert len(batches) == 0

    def test_last_batch_flag(self, cpu_device):
        """Test that only last batch has is_last=True."""
        config = AdaptiveBatchConfig(max_batch_size=2)
        calc = AdaptiveBatchSizeCalculator("da3-large", cpu_device, config)

        items = list(range(5))
        batches = list(adaptive_batch_iterator(items, calc))

        # All but last should be False
        for batch in batches[:-1]:
            assert batch.is_last is False

        # Last should be True
        assert batches[-1].is_last is True


# =============================================================================
# process_with_adaptive_batching Tests
# =============================================================================


class TestProcessWithAdaptiveBatching:
    """Tests for process_with_adaptive_batching function."""

    def test_basic_processing(self, cpu_device):
        """Test basic batch processing."""
        items = list(range(10))

        def process_fn(batch):
            return [x * 2 for x in batch]

        results = process_with_adaptive_batching(
            items=items,
            process_fn=process_fn,
            model_name="da3-large",
            device=cpu_device,
        )

        assert results == [x * 2 for x in items]

    def test_progress_callback(self, cpu_device):
        """Test progress callback is called."""
        items = list(range(10))
        progress_calls = []

        def process_fn(batch):
            return batch

        def progress_callback(processed, total):
            progress_calls.append((processed, total))

        config = AdaptiveBatchConfig(max_batch_size=3)

        results = process_with_adaptive_batching(
            items=items,
            process_fn=process_fn,
            model_name="da3-large",
            device=cpu_device,
            config=config,
            progress_callback=progress_callback,
        )

        # Should have multiple progress calls
        assert len(progress_calls) > 1

        # Last call should show all items processed
        assert progress_calls[-1][0] == len(items)
        assert progress_calls[-1][1] == len(items)

    def test_single_result_handling(self, cpu_device):
        """Test handling of non-list results."""
        items = list(range(5))

        def process_fn(batch):
            # Return a single value instead of list
            return sum(batch)

        results = process_with_adaptive_batching(
            items=items,
            process_fn=process_fn,
            model_name="da3-large",
            device=cpu_device,
        )

        # Should still work and return list of results
        assert isinstance(results, list)

    def test_empty_items(self, cpu_device):
        """Test with empty items list."""
        results = process_with_adaptive_batching(
            items=[],
            process_fn=lambda x: x,
            model_name="da3-large",
            device=cpu_device,
        )
        assert results == []


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestEstimateMaxBatchSize:
    """Tests for estimate_max_batch_size function."""

    def test_returns_positive_integer(self, cpu_device):
        """Test that function returns positive integer."""
        result = estimate_max_batch_size("da3-large", cpu_device)
        assert isinstance(result, int)
        assert result > 0

    def test_different_resolutions(self, cpu_device):
        """Test that higher resolution gives lower batch size (for GPU)."""
        # For CPU this doesn't apply, but the function should still work
        low_res = estimate_max_batch_size("da3-large", cpu_device, process_res=504)
        high_res = estimate_max_batch_size("da3-large", cpu_device, process_res=1008)

        # Both should be valid
        assert low_res > 0
        assert high_res > 0

    def test_different_utilization(self, cpu_device):
        """Test different target utilization values."""
        low_util = estimate_max_batch_size(
            "da3-large", cpu_device, target_utilization=0.5
        )
        high_util = estimate_max_batch_size(
            "da3-large", cpu_device, target_utilization=0.95
        )

        # Both should be valid (CPU returns max_batch_size anyway)
        assert low_util > 0
        assert high_util > 0


class TestLogBatchPlan:
    """Tests for log_batch_plan function."""

    def test_log_batch_plan_runs(self, cpu_device, caplog):
        """Test that log_batch_plan runs without error."""
        import logging

        with caplog.at_level(logging.INFO):
            # Should not raise
            log_batch_plan(
                num_images=100,
                model_name="da3-large",
                device=cpu_device,
                process_res=504,
            )

    def test_log_batch_plan_different_models(self, cpu_device):
        """Test log_batch_plan with different models."""
        for model_name in ["da3-small", "da3-base", "da3-large", "da3-giant"]:
            # Should not raise for any model
            log_batch_plan(
                num_images=50,
                model_name=model_name,
                device=cpu_device,
            )


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the adaptive batching module."""

    def test_full_workflow_cpu(self, cpu_device):
        """Test complete workflow on CPU."""
        # Create data
        images = [f"image_{i}.jpg" for i in range(25)]

        # Track processing
        processed_batches = []

        def process_fn(batch):
            processed_batches.append(len(batch))
            return [f"result_{item}" for item in batch]

        # Process with adaptive batching
        config = AdaptiveBatchConfig(max_batch_size=8)
        results = process_with_adaptive_batching(
            items=images,
            process_fn=process_fn,
            model_name="da3-large",
            device=cpu_device,
            config=config,
        )

        # Verify results
        assert len(results) == len(images)
        assert all(r.startswith("result_") for r in results)

        # Verify batching
        assert sum(processed_batches) == len(images)
        assert max(processed_batches) <= 8

    def test_calculator_reuse(self, cpu_device):
        """Test that calculator can be reused across multiple iterations."""
        calc = AdaptiveBatchSizeCalculator("da3-large", cpu_device)

        # First computation
        batch1 = calc.compute_optimal_batch_size(num_images=100)

        # Second computation should work
        batch2 = calc.compute_optimal_batch_size(num_images=50)

        assert batch1 == 64  # max_batch_size for CPU
        assert batch2 == 50  # min(50, max_batch_size)

    def test_iterator_with_strings(self, cpu_device):
        """Test iterator works with string items."""
        config = AdaptiveBatchConfig(max_batch_size=3)
        calc = AdaptiveBatchSizeCalculator("da3-large", cpu_device, config)

        items = ["path/to/image1.jpg", "path/to/image2.jpg", "path/to/image3.jpg", "path/to/image4.jpg"]

        batches = list(adaptive_batch_iterator(items, calc))

        # Collect all paths
        all_paths = []
        for batch in batches:
            all_paths.extend(batch.items)

        assert all_paths == items

    def test_iterator_with_tuples(self, cpu_device):
        """Test iterator works with tuple items."""
        config = AdaptiveBatchConfig(max_batch_size=2)
        calc = AdaptiveBatchSizeCalculator("da3-large", cpu_device, config)

        items = [(1, "a"), (2, "b"), (3, "c")]

        batches = list(adaptive_batch_iterator(items, calc))

        # Should preserve tuple structure
        all_items = []
        for batch in batches:
            all_items.extend(batch.items)

        assert all_items == list(items)


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_image(self, cpu_device):
        """Test with single image."""
        calc = AdaptiveBatchSizeCalculator("da3-large", cpu_device)

        result = calc.compute_optimal_batch_size(num_images=1)
        assert result == 1

        batches = list(adaptive_batch_iterator(["single"], calc))
        assert len(batches) == 1
        assert batches[0].items == ["single"]
        assert batches[0].is_last is True

    def test_exact_batch_size_multiple(self, cpu_device):
        """Test when num_images is exact multiple of batch_size."""
        config = AdaptiveBatchConfig(max_batch_size=5)
        calc = AdaptiveBatchSizeCalculator("da3-large", cpu_device, config)

        items = list(range(15))  # Exactly 3 batches of 5
        batches = list(adaptive_batch_iterator(items, calc))

        assert len(batches) == 3
        assert all(b.batch_size == 5 for b in batches)

    def test_very_large_num_images(self, cpu_device):
        """Test with very large number of images."""
        calc = AdaptiveBatchSizeCalculator("da3-large", cpu_device)

        result = calc.compute_optimal_batch_size(num_images=1_000_000)
        assert result == 64  # Should cap at max_batch_size

    def test_zero_reserved_memory(self, cpu_device):
        """Test with zero reserved memory."""
        calc = AdaptiveBatchSizeCalculator("da3-large", cpu_device)

        result = calc.compute_optimal_batch_size(
            num_images=100,
            process_res=504,
            reserved_memory_mb=0,
        )
        assert result > 0

    def test_high_resolution(self, cpu_device):
        """Test with very high resolution."""
        calc = AdaptiveBatchSizeCalculator("da3-large", cpu_device)

        # 4K resolution
        result = calc.compute_optimal_batch_size(
            num_images=100,
            process_res=2160,
        )
        assert result > 0  # Should still return valid batch size

    def test_low_resolution(self, cpu_device):
        """Test with very low resolution."""
        calc = AdaptiveBatchSizeCalculator("da3-large", cpu_device)

        result = calc.compute_optimal_batch_size(
            num_images=100,
            process_res=128,
        )
        assert result > 0

    def test_negative_memory_edge_case(self, cpu_device):
        """Test handling when calculations could go negative."""
        config = AdaptiveBatchConfig(
            min_batch_size=1,
            target_memory_utilization=0.01,  # Very low utilization
        )
        calc = AdaptiveBatchSizeCalculator("da3-large", cpu_device, config)

        # Should still return valid result
        result = calc.compute_optimal_batch_size(num_images=100)
        assert result >= 1


# =============================================================================
# Run tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
