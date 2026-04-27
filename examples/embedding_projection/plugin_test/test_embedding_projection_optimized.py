"""
test_embedding_projection_optimized.py — Validation tests for the optimized
EmbeddingProjection workload.

Tests:
  1. test_import          — Module imports without error
  2. test_backend_registration — Backend registered with torch._dynamo
  3. test_get_model_and_input  — Model/input have correct shapes and dtypes
  4. test_forward_pass    — Uncompiled forward pass completes; outputs are finite
  5. test_tied_weight_check    — logits.weight and embed.weight are not aliased
  6. test_output_shape    — Output shape matches (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
  7. test_numerical_sanity     — Outputs contain no NaN or Inf
"""
from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# Test 1: Import
# ---------------------------------------------------------------------------

def test_import():
    """Module imports without error."""
    import examples.embedding_projection.plugin_test.embedding_projection_optimized  # noqa: F401


# ---------------------------------------------------------------------------
# Test 2: Backend registration
# ---------------------------------------------------------------------------

def test_backend_registration():
    """Backend 'embedding_projection_opt' is registered with torch._dynamo."""
    import examples.embedding_projection.plugin_test.embedding_projection_optimized  # noqa: F401

    backends = str(torch._dynamo.list_backends())
    assert "embedding_projection_opt" in backends, (
        f"Backend 'embedding_projection_opt' not found in registered backends: {backends}"
    )


# ---------------------------------------------------------------------------
# Test 3: Model and input shapes / dtypes
# ---------------------------------------------------------------------------

def test_get_model_and_input():
    """Model and input have expected shapes and dtypes after optimizations."""
    from examples.embedding_projection.plugin_test.embedding_projection_optimized import (
        get_model_and_input,
        BATCH_SIZE,
        SEQ_LEN,
        VOCAB_SIZE,
    )

    model, token_ids = get_model_and_input()

    # Input must be on CUDA
    assert token_ids.device.type == "cuda", (
        f"token_ids must be on CUDA, got {token_ids.device}"
    )

    # Token IDs must stay int64 — they are integer indices, cannot be BF16
    assert token_ids.dtype == torch.long, (
        f"token_ids must be int64 (torch.long), got {token_ids.dtype}"
    )

    # Input shape must match (BATCH_SIZE, SEQ_LEN)
    assert token_ids.shape == (BATCH_SIZE, SEQ_LEN), (
        f"Expected token_ids shape ({BATCH_SIZE}, {SEQ_LEN}), got {token_ids.shape}"
    )

    # Model parameters must be in BF16 after OPT-1 cast
    # (Access through the underlying module when compiled)
    raw_model = model
    if hasattr(model, "_orig_mod"):
        raw_model = model._orig_mod
    param_dtype = next(raw_model.parameters()).dtype
    assert param_dtype == torch.bfloat16, (
        f"Model parameters must be bfloat16 after OPT-1 cast, got {param_dtype}"
    )


# ---------------------------------------------------------------------------
# Test 4: Uncompiled forward pass
# ---------------------------------------------------------------------------

def test_forward_pass():
    """Uncompiled forward pass completes without error on CUDA."""
    from examples.embedding_projection.plugin_test.embedding_projection_optimized import (
        get_model_and_input,
        BATCH_SIZE,
        SEQ_LEN,
        VOCAB_SIZE,
    )

    model, token_ids = get_model_and_input()

    with torch.no_grad():
        out = model(token_ids)

    assert out is not None, "Forward pass returned None"
    assert out.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE), (
        f"Expected output shape ({BATCH_SIZE}, {SEQ_LEN}, {VOCAB_SIZE}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# Test 5: Tied-weight check — logits.weight and embed.weight must not alias
# ---------------------------------------------------------------------------

def test_tied_weight_check():
    """
    After get_model_and_input(), logits.weight and embed.weight must have
    different data pointers (i.e., they are not tied / aliased).
    """
    from examples.embedding_projection.embedding_projection import EmbeddingProjection
    import torch

    # Instantiate and apply the same tie-breaking logic manually to verify
    # the check path is exercised correctly.
    model = EmbeddingProjection().to("cuda").eval()

    # In this architecture the weights are NOT tied by default, so this
    # assertion should already pass on a fresh model.
    raw_logits_ptr = model.logits.weight.data_ptr()
    raw_embed_ptr = model.embed.weight.data_ptr()

    assert raw_logits_ptr != raw_embed_ptr, (
        "logits.weight and embed.weight share storage — tied-weight untying logic must fire"
    )

    # Now also verify via get_model_and_input that the compiled model's
    # underlying raw module has independent weights.
    from examples.embedding_projection.plugin_test.embedding_projection_optimized import get_model_and_input

    compiled_model, _ = get_model_and_input()
    raw_model = compiled_model
    if hasattr(compiled_model, "_orig_mod"):
        raw_model = compiled_model._orig_mod

    assert raw_model.logits.weight.data_ptr() != raw_model.embed.weight.data_ptr(), (
        "After get_model_and_input(), logits.weight and embed.weight still share storage"
    )


# ---------------------------------------------------------------------------
# Test 6: Output shape matches original workload shape
# ---------------------------------------------------------------------------

def test_output_shape():
    """
    Output shape must be (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE) — identical to the
    original unoptimized workload.
    """
    from examples.embedding_projection.plugin_test.embedding_projection_optimized import (
        get_model_and_input,
        BATCH_SIZE,
        SEQ_LEN,
        VOCAB_SIZE,
    )

    model, token_ids = get_model_and_input()
    with torch.no_grad():
        out = model(token_ids)

    assert out.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE), (
        f"Output shape mismatch: expected ({BATCH_SIZE}, {SEQ_LEN}, {VOCAB_SIZE}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# Test 7: Numerical sanity — no NaN or Inf in output
# ---------------------------------------------------------------------------

def test_numerical_sanity():
    """Outputs must be finite (no NaN, no Inf)."""
    from examples.embedding_projection.plugin_test.embedding_projection_optimized import get_model_and_input

    model, token_ids = get_model_and_input()
    with torch.no_grad():
        out = model(token_ids)

    assert not torch.isnan(out).any().item(), (
        "Output contains NaN values — possible BF16 underflow or numerical instability"
    )
    assert not torch.isinf(out).any().item(), (
        "Output contains Inf values — possible BF16 overflow"
    )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
