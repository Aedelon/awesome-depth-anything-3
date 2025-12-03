# Copyright (c) Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0
"""
Tests for batch_inference and get_optimal_batch_size methods in DepthAnything3 API.

These tests mock the actual model inference to focus on testing the batching logic,
without needing to load heavy model weights.
"""
from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


# =============================================================================
# Mock Prediction Class
# =============================================================================


@dataclass
class MockPrediction:
    """Mock Prediction object for testing."""

    depth: np.ndarray
    processed_images: np.ndarray
    num_images: int

    @classmethod
    def create(cls, num_images: int) -> "MockPrediction":
        """Create a mock prediction for n images."""
        return cls(
            depth=np.zeros((num_images, 256, 256), dtype=np.float32),
            processed_images=np.zeros((num_images, 256, 256, 3), dtype=np.uint8),
            num_images=num_images,
        )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def cpu_device():
    """Return CPU device."""
    return torch.device("cpu")


@pytest.fixture
def mock_model(cpu_device):
    """Create a mock DepthAnything3 model."""
    from depth_anything_3.api import DepthAnything3

    # Create a minimal mock
    model = MagicMock(spec=DepthAnything3)
    model.device = cpu_device
    model.model_name = "da3-large"

    # Setup inference to return mock predictions
    def mock_inference(image, process_res=504, **kwargs):
        num_images = len(image) if isinstance(image, list) else 1
        return MockPrediction.create(num_images)

    model.inference = MagicMock(side_effect=mock_inference)

    return model


@pytest.fixture
def sample_images():
    """Create sample image paths for testing."""
    return [f"image_{i}.jpg" for i in range(10)]


@pytest.fixture
def large_sample_images():
    """Create larger sample of image paths."""
    return [f"image_{i}.jpg" for i in range(100)]


# =============================================================================
# batch_inference Tests
# =============================================================================


class TestBatchInference:
    """Tests for the batch_inference method."""

    def test_batch_inference_empty_list(self, mock_model):
        """Test batch_inference with empty image list."""
        from depth_anything_3.api import DepthAnything3

        # Call the actual method implementation with mocked model
        with patch.object(DepthAnything3, "__init__", lambda x, **k: None):
            api = DepthAnything3()
            api.device = mock_model.device
            api.model_name = mock_model.model_name
            api.inference = mock_model.inference

            results = api.batch_inference([])

            assert results == []
            mock_model.inference.assert_not_called()

    def test_batch_inference_fixed_batch_size(self, mock_model, sample_images):
        """Test batch_inference with fixed batch size."""
        from depth_anything_3.api import DepthAnything3

        with patch.object(DepthAnything3, "__init__", lambda x, **k: None):
            api = DepthAnything3()
            api.device = mock_model.device
            api.model_name = mock_model.model_name
            api.inference = mock_model.inference

            results = api.batch_inference(sample_images, batch_size=3)

            # 10 images with batch size 3 = 4 batches (3, 3, 3, 1)
            assert len(results) == 4
            assert mock_model.inference.call_count == 4

    def test_batch_inference_auto_batch_size(self, mock_model, sample_images):
        """Test batch_inference with auto batch size."""
        from depth_anything_3.api import DepthAnything3

        with patch.object(DepthAnything3, "__init__", lambda x, **k: None):
            api = DepthAnything3()
            api.device = mock_model.device
            api.model_name = mock_model.model_name
            api.inference = mock_model.inference

            results = api.batch_inference(sample_images, batch_size="auto")

            # Should have at least 1 result
            assert len(results) >= 1
            # Should have called inference at least once
            assert mock_model.inference.call_count >= 1

    def test_batch_inference_progress_callback(self, mock_model, sample_images):
        """Test that progress callback is called."""
        from depth_anything_3.api import DepthAnything3

        progress_calls = []

        def progress_callback(processed, total):
            progress_calls.append((processed, total))

        with patch.object(DepthAnything3, "__init__", lambda x, **k: None):
            api = DepthAnything3()
            api.device = mock_model.device
            api.model_name = mock_model.model_name
            api.inference = mock_model.inference

            api.batch_inference(
                sample_images, batch_size=3, progress_callback=progress_callback
            )

            # Should have progress calls
            assert len(progress_calls) == 4  # 4 batches

            # Last call should have all images processed
            assert progress_calls[-1][0] == len(sample_images)
            assert progress_calls[-1][1] == len(sample_images)

    def test_batch_inference_single_image(self, mock_model):
        """Test batch_inference with single image."""
        from depth_anything_3.api import DepthAnything3

        with patch.object(DepthAnything3, "__init__", lambda x, **k: None):
            api = DepthAnything3()
            api.device = mock_model.device
            api.model_name = mock_model.model_name
            api.inference = mock_model.inference

            results = api.batch_inference(["single.jpg"])

            assert len(results) == 1
            mock_model.inference.assert_called_once()

    def test_batch_inference_batch_larger_than_images(self, mock_model):
        """Test when batch size is larger than number of images."""
        from depth_anything_3.api import DepthAnything3

        images = ["img1.jpg", "img2.jpg"]

        with patch.object(DepthAnything3, "__init__", lambda x, **k: None):
            api = DepthAnything3()
            api.device = mock_model.device
            api.model_name = mock_model.model_name
            api.inference = mock_model.inference

            results = api.batch_inference(images, batch_size=10)

            # Should only make one call with all images
            assert len(results) == 1
            mock_model.inference.assert_called_once()

    def test_batch_inference_exact_batch_multiple(self, mock_model):
        """Test when image count is exact multiple of batch size."""
        from depth_anything_3.api import DepthAnything3

        images = [f"img{i}.jpg" for i in range(12)]  # Exactly 4 batches of 3

        with patch.object(DepthAnything3, "__init__", lambda x, **k: None):
            api = DepthAnything3()
            api.device = mock_model.device
            api.model_name = mock_model.model_name
            api.inference = mock_model.inference

            results = api.batch_inference(images, batch_size=3)

            assert len(results) == 4
            assert mock_model.inference.call_count == 4

    def test_batch_inference_respects_process_res(self, mock_model, sample_images):
        """Test that process_res is passed to inference."""
        from depth_anything_3.api import DepthAnything3

        with patch.object(DepthAnything3, "__init__", lambda x, **k: None):
            api = DepthAnything3()
            api.device = mock_model.device
            api.model_name = mock_model.model_name
            api.inference = mock_model.inference

            api.batch_inference(sample_images, batch_size=10, process_res=1024)

            # Check that inference was called with correct process_res
            call_args = mock_model.inference.call_args
            assert call_args.kwargs.get("process_res") == 1024

    def test_batch_inference_max_batch_size_auto(self, mock_model, sample_images):
        """Test max_batch_size parameter with auto batching."""
        from depth_anything_3.api import DepthAnything3

        with patch.object(DepthAnything3, "__init__", lambda x, **k: None):
            api = DepthAnything3()
            api.device = mock_model.device
            api.model_name = mock_model.model_name
            api.inference = mock_model.inference

            # With max_batch_size=2, should split 10 images into more batches
            results = api.batch_inference(
                sample_images, batch_size="auto", max_batch_size=2
            )

            # Should have at least 5 batches (10 images / 2 max)
            assert len(results) >= 5


# =============================================================================
# get_optimal_batch_size Tests
# =============================================================================


class TestGetOptimalBatchSize:
    """Tests for the get_optimal_batch_size method."""

    def test_get_optimal_batch_size_returns_int(self, cpu_device):
        """Test that get_optimal_batch_size returns an integer."""
        from depth_anything_3.api import DepthAnything3

        with patch.object(DepthAnything3, "__init__", lambda x, **k: None):
            api = DepthAnything3()
            api.device = cpu_device
            api.model_name = "da3-large"

            result = api.get_optimal_batch_size()

            assert isinstance(result, int)
            assert result > 0

    def test_get_optimal_batch_size_respects_resolution(self, cpu_device):
        """Test that different resolutions affect the result."""
        from depth_anything_3.api import DepthAnything3

        with patch.object(DepthAnything3, "__init__", lambda x, **k: None):
            api = DepthAnything3()
            api.device = cpu_device
            api.model_name = "da3-large"

            low_res = api.get_optimal_batch_size(process_res=256)
            high_res = api.get_optimal_batch_size(process_res=1024)

            # Both should be valid
            assert low_res > 0
            assert high_res > 0

    def test_get_optimal_batch_size_respects_utilization(self, cpu_device):
        """Test that target_utilization parameter is used."""
        from depth_anything_3.api import DepthAnything3

        with patch.object(DepthAnything3, "__init__", lambda x, **k: None):
            api = DepthAnything3()
            api.device = cpu_device
            api.model_name = "da3-large"

            low_util = api.get_optimal_batch_size(target_utilization=0.5)
            high_util = api.get_optimal_batch_size(target_utilization=0.95)

            # Both should return valid results
            assert low_util > 0
            assert high_util > 0

    def test_get_optimal_batch_size_different_models(self, cpu_device):
        """Test with different model names."""
        from depth_anything_3.api import DepthAnything3

        models = ["da3-small", "da3-base", "da3-large", "da3-giant"]

        for model_name in models:
            with patch.object(DepthAnything3, "__init__", lambda x, **k: None):
                api = DepthAnything3()
                api.device = cpu_device
                api.model_name = model_name

                result = api.get_optimal_batch_size()
                assert result > 0, f"Failed for model {model_name}"


# =============================================================================
# Integration Tests
# =============================================================================


class TestBatchingIntegration:
    """Integration tests for batching functionality."""

    def test_auto_vs_fixed_batching_coverage(self, mock_model, sample_images):
        """Test that both auto and fixed batching process all images."""
        from depth_anything_3.api import DepthAnything3

        with patch.object(DepthAnything3, "__init__", lambda x, **k: None):
            api = DepthAnything3()
            api.device = mock_model.device
            api.model_name = mock_model.model_name
            api.inference = mock_model.inference

            # Track images processed
            auto_images_processed = []
            fixed_images_processed = []

            def track_auto(image, **kwargs):
                batch = image if isinstance(image, list) else [image]
                auto_images_processed.extend(batch)
                return MockPrediction.create(len(batch))

            def track_fixed(image, **kwargs):
                batch = image if isinstance(image, list) else [image]
                fixed_images_processed.extend(batch)
                return MockPrediction.create(len(batch))

            # Test auto batching
            mock_model.inference.side_effect = track_auto
            api.inference = mock_model.inference
            api.batch_inference(sample_images.copy(), batch_size="auto")

            # Test fixed batching
            mock_model.inference.side_effect = track_fixed
            api.inference = mock_model.inference
            api.batch_inference(sample_images.copy(), batch_size=3)

            # Both should process all images
            assert len(auto_images_processed) == len(sample_images)
            assert len(fixed_images_processed) == len(sample_images)

    def test_batch_inference_preserves_order(self, mock_model):
        """Test that batch_inference preserves image order in processing."""
        from depth_anything_3.api import DepthAnything3

        images = ["first.jpg", "second.jpg", "third.jpg", "fourth.jpg", "fifth.jpg"]
        processed_order = []

        def track_order(image, **kwargs):
            batch = image if isinstance(image, list) else [image]
            processed_order.extend(batch)
            return MockPrediction.create(len(batch))

        with patch.object(DepthAnything3, "__init__", lambda x, **k: None):
            api = DepthAnything3()
            api.device = mock_model.device
            api.model_name = mock_model.model_name
            mock_model.inference.side_effect = track_order
            api.inference = mock_model.inference

            api.batch_inference(images, batch_size=2)

            assert processed_order == images

    def test_progress_increases_monotonically(self, mock_model, sample_images):
        """Test that progress always increases."""
        from depth_anything_3.api import DepthAnything3

        progress_values = []

        def progress_callback(processed, total):
            progress_values.append(processed)

        with patch.object(DepthAnything3, "__init__", lambda x, **k: None):
            api = DepthAnything3()
            api.device = mock_model.device
            api.model_name = mock_model.model_name
            api.inference = mock_model.inference

            api.batch_inference(
                sample_images, batch_size=3, progress_callback=progress_callback
            )

            # Progress should always increase
            for i in range(1, len(progress_values)):
                assert progress_values[i] > progress_values[i - 1]


# =============================================================================
# Edge Cases
# =============================================================================


class TestBatchingEdgeCases:
    """Tests for edge cases in batching."""

    def test_batch_size_one(self, mock_model, sample_images):
        """Test with batch size of 1."""
        from depth_anything_3.api import DepthAnything3

        with patch.object(DepthAnything3, "__init__", lambda x, **k: None):
            api = DepthAnything3()
            api.device = mock_model.device
            api.model_name = mock_model.model_name
            api.inference = mock_model.inference

            results = api.batch_inference(sample_images, batch_size=1)

            # Should have one result per image
            assert len(results) == len(sample_images)
            assert mock_model.inference.call_count == len(sample_images)

    def test_very_large_batch_size(self, mock_model, sample_images):
        """Test with very large batch size."""
        from depth_anything_3.api import DepthAnything3

        with patch.object(DepthAnything3, "__init__", lambda x, **k: None):
            api = DepthAnything3()
            api.device = mock_model.device
            api.model_name = mock_model.model_name
            api.inference = mock_model.inference

            results = api.batch_inference(sample_images, batch_size=1000)

            # Should process all in one batch
            assert len(results) == 1

    def test_auto_with_very_low_memory_utilization(self, mock_model, sample_images):
        """Test auto batching with very low memory utilization target."""
        from depth_anything_3.api import DepthAnything3

        with patch.object(DepthAnything3, "__init__", lambda x, **k: None):
            api = DepthAnything3()
            api.device = mock_model.device
            api.model_name = mock_model.model_name
            api.inference = mock_model.inference

            results = api.batch_inference(
                sample_images, batch_size="auto", target_memory_utilization=0.1
            )

            # Should still process all images
            total_processed = sum(r.num_images for r in results)
            assert total_processed == len(sample_images)

    def test_numpy_array_inputs(self, mock_model):
        """Test with numpy array inputs instead of paths."""
        from depth_anything_3.api import DepthAnything3

        # Create dummy numpy arrays
        images = [np.zeros((256, 256, 3), dtype=np.uint8) for _ in range(5)]

        with patch.object(DepthAnything3, "__init__", lambda x, **k: None):
            api = DepthAnything3()
            api.device = mock_model.device
            api.model_name = mock_model.model_name
            api.inference = mock_model.inference

            results = api.batch_inference(images, batch_size=2)

            assert len(results) == 3  # 5 images in batches of 2: 2, 2, 1


# =============================================================================
# Memory Cleanup Tests
# =============================================================================


class TestMemoryCleanup:
    """Tests for memory cleanup during batching."""

    def test_gc_collect_called_between_batches(self, mock_model, sample_images):
        """Test that garbage collection is called between batches."""
        from depth_anything_3.api import DepthAnything3

        with patch.object(DepthAnything3, "__init__", lambda x, **k: None):
            api = DepthAnything3()
            api.device = mock_model.device
            api.model_name = mock_model.model_name
            api.inference = mock_model.inference

            with patch("gc.collect") as mock_gc:
                api.batch_inference(sample_images, batch_size=3)

                # Should call gc.collect between batches (not after last)
                # 4 batches means 3 gc.collect calls
                assert mock_gc.call_count == 3

    def test_cuda_empty_cache_called(self, sample_images):
        """Test that cuda empty_cache is called on CUDA device."""
        from depth_anything_3.api import DepthAnything3

        def mock_inference(image, **kwargs):
            num = len(image) if isinstance(image, list) else 1
            return MockPrediction.create(num)

        mock_model = MagicMock(spec=DepthAnything3)
        mock_model.device = torch.device("cuda:0")
        mock_model.model_name = "da3-large"
        mock_model.inference = MagicMock(side_effect=mock_inference)

        with patch.object(DepthAnything3, "__init__", lambda x, **k: None):
            api = DepthAnything3()
            api.device = mock_model.device
            api.model_name = mock_model.model_name
            api.inference = mock_model.inference

            with patch("torch.cuda.empty_cache") as mock_empty:
                api.batch_inference(sample_images, batch_size=3)

                # Should call empty_cache between batches
                assert mock_empty.call_count == 3

    def test_mps_empty_cache_called(self, sample_images):
        """Test that mps empty_cache is called on MPS device."""
        from depth_anything_3.api import DepthAnything3

        def mock_inference(image, **kwargs):
            num = len(image) if isinstance(image, list) else 1
            return MockPrediction.create(num)

        mock_model = MagicMock(spec=DepthAnything3)
        mock_model.device = torch.device("mps")
        mock_model.model_name = "da3-large"
        mock_model.inference = MagicMock(side_effect=mock_inference)

        with patch.object(DepthAnything3, "__init__", lambda x, **k: None):
            api = DepthAnything3()
            api.device = mock_model.device
            api.model_name = mock_model.model_name
            api.inference = mock_model.inference

            with patch("torch.mps.empty_cache") as mock_empty:
                api.batch_inference(sample_images, batch_size=3)

                assert mock_empty.call_count == 3


# =============================================================================
# Run tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
