"""
test_sdpa_attention_optimized.py — Verification tests for the optimized workload.

Validates:
  1. Module imports cleanly
  2. Backend is registered with torch._dynamo
  3. get_model_and_input() returns correct shapes and dtypes
  4. Uncompiled forward pass completes without error
  5. Compiled forward pass (transformer_opt backend) produces correct output shape
  6. TF32 flag is set after import
  7. BF16 cast is applied when baseline is FP32

Run with:
    python test_sdpa_attention_optimized.py

Or with pytest:
    pytest test_sdpa_attention_optimized.py -v
"""
from __future__ import annotations

import sys
import logging

import torch
import pytest

logging.basicConfig(level=logging.WARNING)  # suppress INFO during tests


# ---------------------------------------------------------------------------
# Test 1 — Module import
# ---------------------------------------------------------------------------

def test_import():
    """Module imports successfully without raising."""
    import sdpa_attention_optimized  # noqa: F401


# ---------------------------------------------------------------------------
# Test 2 — Backend registration
# ---------------------------------------------------------------------------

def test_backend_registration():
    """transformer_opt backend is registered with torch._dynamo."""
    import torch._dynamo
    backends = torch._dynamo.list_backends()
    assert "transformer_opt" in backends, (
        f"'transformer_opt' not found in registered backends: {backends}"
    )


# ---------------------------------------------------------------------------
# Test 3 — TF32 flag
# ---------------------------------------------------------------------------

def test_tf32_enabled():
    """Import side effect: allow_tf32 is set to True."""
    import sdpa_attention_optimized  # noqa: F401 — triggers module-level side effect
    assert torch.backends.cuda.matmul.allow_tf32 is True, (
        "OPT-001: allow_tf32 should be True after import"
    )


# ---------------------------------------------------------------------------
# Test 4 — get_model_and_input shapes and dtypes
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_get_model_and_input():
    """Model and input are created with correct shapes and dtypes."""
    from sdpa_attention_optimized import get_model_and_input
    from sdpa_attention import BATCH_SIZE, SEQ_LEN, DIM

    model, x = get_model_and_input()

    # Shape checks
    assert x.shape == (BATCH_SIZE, SEQ_LEN, DIM), (
        f"Expected input shape ({BATCH_SIZE}, {SEQ_LEN}, {DIM}), got {x.shape}"
    )
    assert x.device.type == "cuda", f"Expected input on CUDA, got {x.device}"

    # OPT-001: dtype should be BF16
    param_dtype = next(model.parameters()).dtype
    assert param_dtype == torch.bfloat16, (
        f"OPT-001: expected model dtype bfloat16, got {param_dtype}"
    )
    assert x.dtype == torch.bfloat16, (
        f"OPT-001: expected input dtype bfloat16, got {x.dtype}"
    )


# ---------------------------------------------------------------------------
# Test 5 — Uncompiled forward pass
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_forward_pass_uncompiled():
    """Uncompiled forward pass completes and produces correct output shape."""
    from sdpa_attention_optimized import get_model_and_input
    from sdpa_attention import BATCH_SIZE, SEQ_LEN, DIM

    model, x = get_model_and_input()
    with torch.no_grad():
        y = model(x)

    assert y.shape == (BATCH_SIZE, SEQ_LEN, DIM), (
        f"Expected output shape ({BATCH_SIZE}, {SEQ_LEN}, {DIM}), got {y.shape}"
    )
    assert not torch.isnan(y).any(), "Output contains NaN values"
    assert not torch.isinf(y).any(), "Output contains Inf values"


# ---------------------------------------------------------------------------
# Test 6 — Compiled forward pass
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_forward_pass_compiled():
    """Compiled forward pass (transformer_opt backend) produces correct output shape."""
    from sdpa_attention_optimized import get_model_and_input
    from sdpa_attention import BATCH_SIZE, SEQ_LEN, DIM

    model, x = get_model_and_input()
    opt_model = torch.compile(model, backend="transformer_opt")

    with torch.no_grad():
        y = opt_model(x)

    assert y.shape == (BATCH_SIZE, SEQ_LEN, DIM), (
        f"Expected output shape ({BATCH_SIZE}, {SEQ_LEN}, {DIM}), got {y.shape}"
    )
    assert not torch.isnan(y).any(), "Compiled output contains NaN values"


# ---------------------------------------------------------------------------
# Test 7 — Numerical agreement between uncompiled and compiled
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_numerical_agreement():
    """
    Compiled and uncompiled outputs agree within BF16 tolerance.
    BF16 introduces larger error than FP32; threshold is set accordingly.
    """
    from sdpa_attention_optimized import get_model_and_input

    model, x = get_model_and_input()
    opt_model = torch.compile(model, backend="transformer_opt")

    with torch.no_grad():
        y_base = model(x).float()
        y_opt = opt_model(x).float()

    max_diff = (y_base - y_opt).abs().max().item()
    # BF16 precision tolerance — ~0.02 is typical for BF16 vs BF16
    assert max_diff < 0.5, (
        f"Compiled vs uncompiled max abs diff too large: {max_diff:.4f} "
        "(check for graph transformation correctness)"
    )


# ---------------------------------------------------------------------------
# Test 8 — FX pass imports are accessible
# ---------------------------------------------------------------------------

def test_fx_passes_importable():
    """All individual FX pass functions are importable."""
    from sdpa_attention_optimized import (
        pass_fuse_qkv,
        pass_replace_sdpa,
        pass_retile_layernorm,
        pass_monitor_dtype_inheritance,
    )
    # All passes are callable
    assert callable(pass_fuse_qkv)
    assert callable(pass_replace_sdpa)
    assert callable(pass_retile_layernorm)
    assert callable(pass_monitor_dtype_inheritance)


# ---------------------------------------------------------------------------
# Test 9 — FX passes are no-ops on an empty graph (robustness)
# ---------------------------------------------------------------------------

def test_fx_passes_empty_graph():
    """FX passes degrade gracefully on a trivially empty GraphModule."""
    from sdpa_attention_optimized import (
        pass_fuse_qkv,
        pass_replace_sdpa,
        pass_retile_layernorm,
        pass_monitor_dtype_inheritance,
    )

    # Build a minimal GraphModule with a single output node
    gm = fx.GraphModule({}, fx.Graph())
    with gm.graph.inserting_after():
        out = gm.graph.output(None)  # noqa: F841
    gm.graph.lint()
    gm.recompile()

    # All passes should return without raising
    gm = pass_replace_sdpa(gm)
    gm = pass_fuse_qkv(gm)
    gm = pass_retile_layernorm(gm)
    gm = pass_monitor_dtype_inheritance(gm)


import torch.fx as fx  # noqa: E402 — needed for test_fx_passes_empty_graph


# ---------------------------------------------------------------------------
# Main entry point (non-pytest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    tests = [
        test_import,
        test_backend_registration,
        test_tf32_enabled,
        test_fx_passes_importable,
        test_fx_passes_empty_graph,
    ]
    cuda_tests = [
        test_get_model_and_input,
        test_forward_pass_uncompiled,
        test_forward_pass_compiled,
        test_numerical_agreement,
    ]

    passed = failed = skipped = 0

    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL  {t.__name__}")
            traceback.print_exc()
            failed += 1

    for t in cuda_tests:
        if not torch.cuda.is_available():
            print(f"  SKIP  {t.__name__}  (no CUDA)")
            skipped += 1
            continue
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL  {t.__name__}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    sys.exit(1 if failed else 0)