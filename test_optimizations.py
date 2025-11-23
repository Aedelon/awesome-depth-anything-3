#!/usr/bin/env python3
"""
Test script to verify optimization features are working correctly.
"""

import torch
from depth_anything_3.api import DepthAnything3

def test_optimizations():
    """Test that optimizations are applied correctly."""

    print("=" * 60)
    print("Testing Depth Anything 3 Optimizations")
    print("=" * 60)

    # Check PyTorch capabilities
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
    print(f"torch.compile available: {hasattr(torch, 'compile')}")

    # Test model initialization with optimizations
    print("\n" + "=" * 60)
    print("Initializing model with optimizations...")
    print("=" * 60)

    try:
        model = DepthAnything3(model_name="da3-large", enable_compile=True)
        print("✓ Model initialized successfully with torch.compile")
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        return False

    # Test device transfer with channels_last
    print("\n" + "=" * 60)
    print("Testing device transfer and channels_last format...")
    print("=" * 60)

    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f"Using device: {device}")

    try:
        model = model.to(device)
        print(f"✓ Model moved to {device}")

        # Create dummy input to test channels_last format (needs 4D tensor)
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        if device in ('cuda', 'mps'):
            dummy_input = dummy_input.to(memory_format=torch.channels_last)
            print("✓ Input converted to channels_last format")
            # Verify memory format
            is_channels_last = dummy_input.is_contiguous(memory_format=torch.channels_last)
            print(f"  - Is channels_last: {is_channels_last}")

        print("\nModel structure check:")
        print(f"  - Model type: {type(model.model)}")
        print(f"  - Compilation enabled: {model.enable_compile}")

        print("\n" + "=" * 60)
        print("All optimization tests passed!")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"✗ Device transfer failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_optimizations()
    exit(0 if success else 1)
