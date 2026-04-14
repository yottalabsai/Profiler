#!/usr/bin/env python
"""
test_workload_optimized.py — Verification tests for the optimized workload.

Run this to verify:
  1. The workload module can be imported
  2. The backend is registered
  3. The model and input can be created
  4. A forward pass completes (uncompiled)

Usage:
  python test_workload_optimized.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add repo root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def test_import():
    """Test that the workload can be imported."""
    print("[TEST] Importing workload_optimized...")
    try:
        import scripts.workload_optimized as wopt
        print("✓ Workload imported successfully")
        return wopt
    except ImportError as e:
        print(f"✗ Failed to import workload: {e}")
        sys.exit(1)


def test_get_model_and_input(wopt):
    """Test that model and input can be created."""
    print("\n[TEST] Creating model and input...")
    try:
        model, x = wopt.get_model_and_input()
        print(f"✓ Model type: {type(model).__name__}")
        print(f"✓ Model dtype: {next(model.parameters()).dtype}")
        print(f"✓ Input shape: {x.shape}, dtype: {x.dtype}")

        # Verify shapes and dtypes
        assert x.shape == (64, 512), f"Expected input shape [64, 512], got {x.shape}"
        assert x.dtype == torch.bfloat16, f"Expected BF16, got {x.dtype}"
        assert next(model.parameters()).dtype == torch.bfloat16, "Model not in BF16"

        return model, x
    except Exception as e:
        print(f"✗ Failed to create model/input: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def test_forward_pass(model, x):
    """Test an uncompiled forward pass."""
    print("\n[TEST] Running uncompiled forward pass...")
    try:
        import torch
        with torch.no_grad():
            y = model(x)

        print(f"✓ Output shape: {y.shape}")
        print(f"✓ Output dtype: {y.dtype}")

        assert y.shape == (64, 512), f"Expected output shape [64, 512], got {y.shape}"
        assert y.dtype == torch.bfloat16, f"Expected BF16 output, got {y.dtype}"

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def test_backend_registration(wopt):
    """Test that the custom backend is registered."""
    print("\n[TEST] Checking backend registration...")
    try:
        import torch._dynamo
        backends = torch._dynamo.list_backends()

        if "transformer_opt" in backends:
            print("✓ 'transformer_opt' backend is registered")
        else:
            print(f"✗ 'transformer_opt' not in backends: {backends}")
            sys.exit(1)

    except Exception as e:
        print(f"✗ Failed to check backend registration: {e}")
        sys.exit(1)


def main():
    """Run all tests."""
    print("=" * 70)
    print("Optimized TransformerBlock Workload — Verification Tests")
    print("=" * 70)

    # Import torch first to make it available to all tests
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA not available — tests will run on CPU (slower)")
    except ImportError:
        print("✗ PyTorch not installed. Run: pip install torch (with CUDA variant)")
        sys.exit(1)

    # Run tests
    wopt = test_import()
    test_backend_registration(wopt)
    model, x = test_get_model_and_input(wopt)
    test_forward_pass(model, x)

    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Profile with optimizations:")
    print("     python scripts/run_workload.py \\")
    print("         --workload scripts/workload_optimized.py \\")
    print("         --compile-backend transformer_opt \\")
    print("         --warmup-iters 3 --measure-iters 1")
    print("\n  2. Compare against baseline (see WORKLOAD_OPTIMIZED.md)")


if __name__ == "__main__":
    main()
