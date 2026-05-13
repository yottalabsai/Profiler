"""
test_embedding_projection_optimized.py — Verification tests for optimized workload.

Validates:
  1. Module imports without error
  2. Backend is registered with torch._dynamo
  3. Model and input have expected shapes and dtypes
  4. Uncompiled forward pass completes without NaN/Inf
"""
from __future__ import annotations

import pytest


def test_import():
    """Module imports without error."""
    import examples.embedding_projection.embedding_projection_optimized  # noqa: F401


def test_backend_registration():
    """Backend 'embedding_projection_opt' is registered with torch._dynamo."""
    import torch
    import examples.embedding_projection.embedding_projection_optimized  # noqa: F401

    backends = str(torch._dynamo.list_backends())
    assert "embedding_projection_opt" in backends, (
        f"Backend 'embedding_projection_opt' not found in: {backends}"
    )


def test_get_model_and_input():
    """Model and input have expected shapes and dtypes after OPT-1."""
    import torch
    from examples.embedding_projection.embedding_projection_optimized import (
        get_model_and_input,
        BATCH_SIZE,
        SEQ_LEN,
        VOCAB_SIZE,
    )

    model, token_ids = get_model_and_input()

    # Input must be on CUDA
    assert token_ids.device.type == "cuda", (
        f"token_ids must be on CUDA, got device={token_ids.device}"
    )

    # Input shape: (BATCH_SIZE, SEQ_LEN) = (64, 128)
    assert token_ids.shape == (BATCH_SIZE, SEQ_LEN), (
        f"Expected token_ids shape ({BATCH_SIZE}, {SEQ_LEN}), got {token_ids.shape}"
    )

    # Token IDs must remain integer (OPT-1 must NOT cast them)
    assert token_ids.dtype in (torch.long, torch.int32, torch.int64), (
        f"token_ids must be integer dtype, got {token_ids.dtype}"
    )

    # Model parameters must be BF16 (OPT-1 applied)
    param_dtype = next(model.parameters()).dtype
    assert param_dtype == torch.bfloat16, (
        f"Expected model parameters to be bfloat16 (OPT-1), got {param_dtype}"
    )

    # Model must be on CUDA
    param_device = next(model.parameters()).device
    assert param_device.type == "cuda", (
        f"Model parameters must be on CUDA, got {param_device}"
    )


def test_forward_pass():
    """Uncompiled forward pass completes without error and produces valid output."""
    import torch
    from examples.embedding_projection.embedding_projection_optimized import (
        get_model_and_input,
        BATCH_SIZE,
        SEQ_LEN,
        VOCAB_SIZE,
    )

    model, token_ids = get_model_and_input()

    with torch.no_grad():
        out = model(token_ids)

    # Output must not be None
    assert out is not None, "Forward pass returned None"

    # Output shape: (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE) = (64, 128, 32000)
    assert out.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE), (
        f"Expected output shape ({BATCH_SIZE}, {SEQ_LEN}, {VOCAB_SIZE}), got {out.shape}"
    )

    # Output dtype should be BF16 (model is BF16 after OPT-1)
    assert out.dtype == torch.bfloat16, (
        f"Expected output dtype bfloat16, got {out.dtype}"
    )

    # Output must be numerically valid
    assert not torch.isnan(out).any(), "Output contains NaN"
    assert not torch.isinf(out).any(), "Output contains Inf"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
