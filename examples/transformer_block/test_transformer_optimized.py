#!/usr/bin/env python
"""
test_workload_optimized.py — Verification tests for the optimized workload.

Run with pytest:
  cd /home/ubuntu/Profiler
  python -m pytest examples/transformer_block/test_transformer_optimized.py -v

Or directly:
  python examples/transformer_block/test_transformer_optimized.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add repo root and file directory to sys.path so both the nvidia package and
# the workload module are importable regardless of where the test is invoked from.
ROOT = Path(__file__).resolve().parents[2]      # repo root (Profiler/)
FILE_DIR = Path(__file__).resolve().parent       # examples/transformer_block/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(FILE_DIR))

import torch


# ---------------------------------------------------------------------------
# pytest-compatible test functions (each is fully self-contained)
# ---------------------------------------------------------------------------

def test_import():
    """Module imports without error."""
    import transformer_optimized  # noqa: F401


def test_backend_registration():
    """Backend registered with torch._dynamo."""
    import transformer_optimized  # noqa: F401
    backends = str(torch._dynamo.list_backends())
    assert "transformer_opt" in backends, f"Backend not found in: {backends}"


def test_get_model_and_input():
    """Model and input have expected shapes and dtypes."""
    from transformer_optimized import get_model_and_input
    model, x = get_model_and_input()
    assert x.device.type == "cuda", "Input must be on CUDA"
    assert x.shape == (64, 512), f"Expected input shape [64, 512], got {x.shape}"
    assert x.dtype == torch.bfloat16, f"Expected BF16, got {x.dtype}"
    assert next(model.parameters()).dtype == torch.bfloat16, "Model not in BF16"


def test_forward_pass():
    """Uncompiled forward pass completes without NaN/Inf."""
    from transformer_optimized import get_model_and_input
    model, x = get_model_and_input()
    with torch.no_grad():
        out = model(x)
    assert out is not None
    assert out.shape == (64, 512), f"Expected output shape [64, 512], got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN"
    assert not torch.isinf(out).any(), "Output contains Inf"


# ---------------------------------------------------------------------------
# Manual runner (kept for `python test_transformer_optimized.py` usage)
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Optimized TransformerBlock Workload — Verification Tests")
    print("=" * 70)

    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not available — some tests will fail")

    results = []
    for fn in [test_import, test_backend_registration,
               test_get_model_and_input, test_forward_pass]:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            results.append(True)
        except Exception as e:
            print(f"FAIL  {fn.__name__}: {e}")
            results.append(False)

    print("\n" + "=" * 70)
    passed = sum(results)
    print(f"{passed}/{len(results)} tests passed")
    if passed == len(results):
        print("All tests passed — ready to profile with /capture")
    else:
        print("Fix failing tests before profiling")
    print("=" * 70)
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
