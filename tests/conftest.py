# Copyright (c) Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0
"""
Pytest configuration and shared fixtures for Depth Anything 3 tests.
"""
from __future__ import annotations

import pytest
import torch


@pytest.fixture
def cpu_device():
    """Return CPU device."""
    return torch.device("cpu")


@pytest.fixture
def mock_cuda_device():
    """Return mock CUDA device (doesn't require actual CUDA)."""
    return torch.device("cuda:0")


@pytest.fixture
def mock_mps_device():
    """Return mock MPS device (doesn't require actual MPS)."""
    return torch.device("mps")


@pytest.fixture
def sample_image_paths():
    """Create sample image paths for testing."""
    return [f"image_{i}.jpg" for i in range(10)]


@pytest.fixture
def large_sample_image_paths():
    """Create larger sample of image paths."""
    return [f"image_{i}.jpg" for i in range(100)]
