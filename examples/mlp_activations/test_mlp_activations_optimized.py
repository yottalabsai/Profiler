"""
test_mlp_activations_optimized.py — Verification tests for the optimized workload.

Validates:
  1. Module imports successfully
  2. transformer_opt backend is registered with torch._dynamo
  3. get_model_and_input() returns correct shapes and dtypes (BF16 after OPT-001)
  4. Uncompiled forward pass completes without error
  5. torch.compile() with transformer_opt backend runs a forward pass

Usage:
    python test_mlp_activations_optimized.py          # run all tests
    pytest test_mlp_activations_optimized.py -v       # via pytest
"""
from __future__ import annotations

import sys
import traceback
import torch


PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
_results: list[tuple[str, bool, str]] = []


def _run(name: str, fn) -> bool:
    try:
        fn()
        _results.append((name, True, ""))
        print(f"  {PASS}  {name}")
        return True
    except Exception as exc:
        tb = traceback.format_exc()
        _results.append((name, False, tb))
        print(f"  {FAIL}  {name}")
        print(f"       {exc}")
        return False


# ---------------------------------------------------------------------------
# Test 1 — Module import
# ---------------------------------------------------------------------------

def test_import():
    """Module imports without error."""
    import mlp_activations_optimized  # noqa: F401


# ---------------------------------------------------------------------------
# Test 2 — Backend registration
# ---------------------------------------------------------------------------

def test_backend_registration():
    """transformer_opt is registered in torch._dynamo."""
    import torch._dynamo as dynamo
    backends = dynamo.list_backends()
    assert "transformer_opt" in backends, (
        f"transformer_opt not found in registered backends: {backends}"
    )


# ---------------------------------------------------------------------------
# Test 3 — Model and input shapes / dtypes
# ---------------------------------------------------------------------------

def test_get_model_and_input():
    """get_model_and_input() returns correct shapes and BF16 dtype (OPT-001)."""
    from mlp_activations_optimized import get_model_and_input
    from mlp_activations import BATCH_SIZE, DIM_IN, DIM_OUT

    model, x = get_model_and_input()

    # dtype must be BF16 after OPT-001
    assert next(model.parameters()).dtype == torch.bfloat16, (
        f"Model weights expected bfloat16, got {next(model.parameters()).dtype}"
    )
    assert x.dtype == torch.bfloat16, (
        f"Input tensor expected bfloat16, got {x.dtype}"
    )

    # input shape
    assert x.shape == (BATCH_SIZE, DIM_IN), (
        f"Input shape expected ({BATCH_SIZE}, {DIM_IN}), got {x.shape}"
    )

    # model is on CUDA
    assert next(model.parameters()).is_cuda, "Model must be on CUDA"
    assert x.is_cuda, "Input tensor must be on CUDA"

    # smoke forward to confirm shape contract
    model.eval()
    with torch.no_grad():
        y = model(x)
    assert y.shape == (BATCH_SIZE, DIM_OUT), (
        f"Output shape expected ({BATCH_SIZE}, {DIM_OUT}), got {y.shape}"
    )


# ---------------------------------------------------------------------------
# Test 4 — Uncompiled forward pass
# ---------------------------------------------------------------------------

def test_forward_pass_uncompiled():
    """Uncompiled forward pass completes without error."""
    from mlp_activations_optimized import get_model_and_input

    model, x = get_model_and_input()
    model.eval()
    with torch.no_grad():
        y = model(x)
    assert y is not None
    assert not torch.isnan(y).any(), "Output contains NaN values"
    assert not torch.isinf(y).any(), "Output contains Inf values"


# ---------------------------------------------------------------------------
# Test 5 — Compiled forward pass with transformer_opt backend
# ---------------------------------------------------------------------------

def test_forward_pass_compiled():
    """torch.compile(backend='transformer_opt') forward pass completes."""
    from mlp_activations_optimized import get_model_and_input

    model, x = get_model_and_input()
    model.eval()

    compiled = torch.compile(model, backend="transformer_opt", fullgraph=True)

    with torch.no_grad():
        y = compiled(x)

    assert y is not None, "Compiled model returned None"
    assert not torch.isnan(y).any(), "Compiled output contains NaN"
    assert not torch.isinf(y).any(), "Compiled output contains Inf"


# ---------------------------------------------------------------------------
# Test 6 — Output shape consistency between compiled and uncompiled
# ---------------------------------------------------------------------------

def test_output_shape_consistency():
    """Compiled and uncompiled outputs have the same shape and dtype."""
    from mlp_activations_optimized import get_model_and_input

    model, x = get_model_and_input()
    model.eval()

    with torch.no_grad():
        y_eager = model(x)

    compiled = torch.compile(model, backend="transformer_opt", fullgraph=True)
    with torch.no_grad():
        y_compiled = compiled(x)

    assert y_eager.shape == y_compiled.shape, (
        f"Shape mismatch: eager {y_eager.shape} vs compiled {y_compiled.shape}"
    )
    assert y_eager.dtype == y_compiled.dtype, (
        f"Dtype mismatch: eager {y_eager.dtype} vs compiled {y_compiled.dtype}"
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    print("\nMLPActivations Optimized — Verification Tests")
    print("=" * 50)

    if not torch.cuda.is_available():
        print("  SKIP  All tests require CUDA — no GPU detected")
        sys.exit(0)

    _run("test_import", test_import)
    _run("test_backend_registration", test_backend_registration)
    _run("test_get_model_and_input", test_get_model_and_input)
    _run("test_forward_pass_uncompiled", test_forward_pass_uncompiled)
    _run("test_forward_pass_compiled", test_forward_pass_compiled)
    _run("test_output_shape_consistency", test_output_shape_consistency)

    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print("=" * 50)
    print(f"  {passed}/{total} tests passed")

    if passed < total:
        print("\nFailure details:")
        for name, ok, tb in _results:
            if not ok:
                print(f"\n--- {name} ---")
                print(tb)
        sys.exit(1)
    else:
        print("  All tests passed ✓")


if __name__ == "__main__":
    main()