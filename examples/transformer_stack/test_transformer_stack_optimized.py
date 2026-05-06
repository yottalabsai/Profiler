"""
test_transformer_stack_optimized.py — Validation tests for the optimized TransformerStack.

Validates:
  1. Module imports without error
  2. Backend is registered with torch._dynamo
  3. get_model_and_input() produces correct shapes and dtypes
  4. Uncompiled forward pass completes without NaN/Inf
"""
from __future__ import annotations

import sys
import pathlib

# Ensure examples/transformer_stack/ is on the path so the module is importable
_EXAMPLE_DIR = str(pathlib.Path(__file__).parent)
if _EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, _EXAMPLE_DIR)


def test_import():
    """Module imports without error."""
    import transformer_stack_optimized  # noqa: F401


def test_backend_registration():
    """Backend is registered with torch._dynamo."""
    import torch
    import transformer_stack_optimized  # noqa: F401 — registers backend as side-effect

    backends = str(torch._dynamo.list_backends())
    assert "transformer_stack_opt" in backends, (
        f"Backend 'transformer_stack_opt' not found in: {backends}"
    )


def test_get_model_and_input():
    """get_model_and_input() returns model and input with expected shapes and dtypes."""
    import torch
    from transformer_stack_optimized import get_model_and_input
    from transformer_stack import BATCH, SEQ_LEN, HIDDEN

    model, x = get_model_and_input()

    # Input must be on CUDA
    assert x.device.type == "cuda", f"Input device must be 'cuda', got: {x.device}"

    # Shape must match BATCH=4, SEQ_LEN=128, HIDDEN=512 (from workload constants)
    assert x.shape == (BATCH, SEQ_LEN, HIDDEN), (
        f"Expected input shape {(BATCH, SEQ_LEN, HIDDEN)}, got {tuple(x.shape)}"
    )

    # OPT-1: dtype must be bfloat16
    assert x.dtype == torch.bfloat16, (
        f"Expected input dtype bfloat16 (OPT-1), got: {x.dtype}"
    )
    param_dtype = next(model.parameters()).dtype
    assert param_dtype == torch.bfloat16, (
        f"Expected model parameters dtype bfloat16 (OPT-1), got: {param_dtype}"
    )

    # Model must be in eval mode
    assert not model.training, "Model must be in eval mode"


def test_forward_pass():
    """Uncompiled forward pass completes without error, NaN, or Inf."""
    import torch
    from transformer_stack_optimized import get_model_and_input
    from transformer_stack import BATCH, SEQ_LEN, HIDDEN

    model, x = get_model_and_input()

    with torch.no_grad():
        out = model(x)

    assert out is not None, "Output is None"

    expected_shape = (BATCH, SEQ_LEN, HIDDEN)
    assert tuple(out.shape) == expected_shape, (
        f"Expected output shape {expected_shape}, got {tuple(out.shape)}"
    )

    assert not torch.isnan(out).any(), (
        "Output contains NaN values — check BF16 cast or SDPA numerics"
    )
    assert not torch.isinf(out).any(), (
        "Output contains Inf values — check BF16 cast or SDPA numerics"
    )


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    tests = [
        test_import,
        test_backend_registration,
        test_get_model_and_input,
        test_forward_pass,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1

    if failed:
        sys.exit(1)
    print(f"\nAll {len(tests)} tests passed.")
