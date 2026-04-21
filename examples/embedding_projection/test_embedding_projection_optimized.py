"""
test_embedding_projection_optimized.py — Verification tests for the optimized
EmbeddingProjection workload.

Validates:
  1. Module imports cleanly
  2. 'transformer_opt' backend is registered with torch._dynamo
  3. get_model_and_input() returns correct shapes and dtypes
  4. Uncompiled forward pass completes without error
  5. FX passes are callable and return GraphModule without raising

Run:
    pytest test_embedding_projection_optimized.py -v
or:
    python test_embedding_projection_optimized.py
"""
from __future__ import annotations

import logging
import sys
import types

import pytest
import torch
import torch.fx as fx

# ---------------------------------------------------------------------------
# Silence INFO logs during testing (warnings/errors still visible)
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


# ============================================================================
# 1. Import
# ============================================================================

def test_import():
    """Module imports successfully without raising."""
    import embedding_projection_optimized  # noqa: F401
    assert True, "Import succeeded"


def test_all_passes_importable():
    """All FX pass functions are importable from the module."""
    from embedding_projection_optimized import (
        pass_insert_bf16_casts,
        pass_batch_sequential_mm,
        pass_propagate_bf16_pointwise,
        pass_detect_embedding_quant,
        transformer_opt,
        get_model_and_input,
    )
    for obj in [
        pass_insert_bf16_casts,
        pass_batch_sequential_mm,
        pass_propagate_bf16_pointwise,
        pass_detect_embedding_quant,
        transformer_opt,
        get_model_and_input,
    ]:
        assert callable(obj), f"{obj} is not callable"


# ============================================================================
# 2. Backend registration
# ============================================================================

def test_backend_registration():
    """'transformer_opt' is registered in torch._dynamo backends."""
    import embedding_projection_optimized  # noqa: F401 — triggers @register_backend
    backends = torch._dynamo.list_backends()
    assert "transformer_opt" in backends, (
        f"'transformer_opt' not found in registered backends: {backends}"
    )


# ============================================================================
# 3. Model and input shapes / dtypes
# ============================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_get_model_and_input_shapes():
    """get_model_and_input() returns tensors with expected shapes."""
    from embedding_projection_optimized import get_model_and_input, BATCH_SIZE, SEQ_LEN, VOCAB_SIZE

    model, token_ids = get_model_and_input()

    assert token_ids.shape == (BATCH_SIZE, SEQ_LEN), (
        f"Expected token_ids shape ({BATCH_SIZE}, {SEQ_LEN}), got {token_ids.shape}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_get_model_and_input_dtypes():
    """get_model_and_input() returns BF16 model weights and integer token IDs."""
    from embedding_projection_optimized import get_model_and_input

    model, token_ids = get_model_and_input()

    param_dtype = next(model.parameters()).dtype
    assert param_dtype == torch.bfloat16, (
        f"Expected model dtype bfloat16, got {param_dtype}"
    )
    assert token_ids.dtype in (torch.int32, torch.int64), (
        f"token_ids should be integer dtype, got {token_ids.dtype}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_get_model_and_input_device():
    """Model and inputs are on CUDA."""
    from embedding_projection_optimized import get_model_and_input

    model, token_ids = get_model_and_input()

    assert next(model.parameters()).is_cuda, "Model parameters not on CUDA"
    assert token_ids.is_cuda, "token_ids not on CUDA"


# ============================================================================
# 4. Forward pass (uncompiled)
# ============================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_forward_pass_uncompiled():
    """Uncompiled forward pass produces correct output shape."""
    from embedding_projection_optimized import get_model_and_input, BATCH_SIZE, SEQ_LEN, VOCAB_SIZE

    model, token_ids = get_model_and_input()

    with torch.no_grad():
        out = model(token_ids)

    assert out.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE), (
        f"Expected output shape ({BATCH_SIZE}, {SEQ_LEN}, {VOCAB_SIZE}), got {out.shape}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_forward_pass_no_nan():
    """Uncompiled BF16 forward pass does not produce NaN outputs."""
    from embedding_projection_optimized import get_model_and_input

    model, token_ids = get_model_and_input()

    with torch.no_grad():
        out = model(token_ids)

    assert not torch.isnan(out).any(), "Output contains NaN values"
    assert not torch.isinf(out).any(), "Output contains Inf values"


# ============================================================================
# 5. FX passes: smoke tests on a synthetic graph
# ============================================================================

def _make_mm_graph() -> fx.GraphModule:
    """Build a minimal GraphModule containing two aten::mm nodes."""

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.W = torch.nn.Parameter(torch.randn(512, 32000))

        def forward(self, x0, x1):
            a = torch.mm(x0, self.W)
            b = torch.mm(x1, self.W)
            return a + b

    model = TinyModel().eval()
    x0 = torch.randn(8, 512)
    x1 = torch.randn(8, 512)
    gm = torch.fx.symbolic_trace(model)
    return gm


def test_pass_insert_bf16_casts_smoke():
    """pass_insert_bf16_casts runs without raising on a synthetic graph."""
    from embedding_projection_optimized import pass_insert_bf16_casts

    gm = _make_mm_graph()
    result = pass_insert_bf16_casts(gm)
    assert isinstance(result, fx.GraphModule)


def test_pass_propagate_bf16_pointwise_smoke():
    """pass_propagate_bf16_pointwise runs without raising."""
    from embedding_projection_optimized import pass_propagate_bf16_pointwise

    gm = _make_mm_graph()
    result = pass_propagate_bf16_pointwise(gm)
    assert isinstance(result, fx.GraphModule)


def test_pass_batch_sequential_mm_smoke():
    """pass_batch_sequential_mm runs without raising."""
    from embedding_projection_optimized import pass_batch_sequential_mm

    gm = _make_mm_graph()
    result = pass_batch_sequential_mm(gm)
    assert isinstance(result, fx.GraphModule)


def test_pass_detect_embedding_quant_smoke():
    """pass_detect_embedding_quant (stub) runs without raising."""
    from embedding_projection_optimized import pass_detect_embedding_quant

    gm = _make_mm_graph()
    result = pass_detect_embedding_quant(gm)
    assert isinstance(result, fx.GraphModule)


def test_pass_insert_bf16_casts_idempotent():
    """Running pass_insert_bf16_casts twice does not raise."""
    from embedding_projection_optimized import pass_insert_bf16_casts

    gm = _make_mm_graph()
    gm = pass_insert_bf16_casts(gm)
    gm = pass_insert_bf16_casts(gm)  # second application should not crash
    assert isinstance(gm, fx.GraphModule)


# ============================================================================
# Standalone runner
# ============================================================================

if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        check=False,
    )
    sys.exit(result.returncode)