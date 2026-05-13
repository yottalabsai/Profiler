"""
test_gpt2_optimized.py — Validation tests for gpt2_optimized.py.

Covers four assertions:
  1. Module imports without error.
  2. 'gpt2_opt' backend is registered with torch._dynamo.
  3. get_model_and_input() returns the expected device, shape, and dtype.
  4. Uncompiled forward pass completes without error, correct shape, no NaN/Inf.

Run with:
    PYTHONPATH=/root/Profiler pytest examples/gpt2/test_gpt2_optimized.py -v
"""
import pytest


def test_import():
    """Module imports without error."""
    import examples.gpt2.gpt2_optimized  # noqa: F401


def test_backend_registration():
    """'gpt2_opt' backend is registered with torch._dynamo."""
    import torch
    import examples.gpt2.gpt2_optimized  # noqa: F401 — registers backend on import

    backends = torch._dynamo.list_backends()
    assert "gpt2_opt" in backends, (
        f"'gpt2_opt' not found in registered backends: {backends}"
    )


def test_get_model_and_input():
    """Model and input have expected shapes and dtypes after OPT-1 promotion."""
    import torch
    from examples.gpt2.gpt2_optimized import get_model_and_input

    model, input_ids = get_model_and_input()

    # Device assertions
    assert input_ids.device.type == "cuda", (
        f"input_ids must be on CUDA, got {input_ids.device}"
    )

    # Shape assertions — GPT-2 small config: batch=4, seq_len=128
    assert input_ids.shape == (4, 128), (
        f"Expected input_ids shape (4, 128), got {input_ids.shape}"
    )

    # Dtype assertions
    # OPT-1: model parameters should be BF16 after promotion
    first_param_dtype = next(model.parameters()).dtype
    assert first_param_dtype == torch.bfloat16, (
        f"Expected model parameters in BF16, got {first_param_dtype}"
    )
    # input_ids must remain int64 for embedding lookup
    assert input_ids.dtype == torch.int64, (
        f"Expected input_ids dtype int64, got {input_ids.dtype}"
    )


def test_forward_pass():
    """Uncompiled forward pass completes, produces correct shape, no NaN/Inf."""
    import torch
    from examples.gpt2.gpt2_optimized import get_model_and_input

    model, input_ids = get_model_and_input()

    with torch.no_grad():
        out = model(input_ids)

    assert out is not None, "Forward pass returned None"

    # GPT-2 small: hidden_dim = 768; output shape = (batch, seq_len, hidden_dim)
    assert out.shape == (4, 128, 768), (
        f"Expected output shape (4, 128, 768), got {out.shape}"
    )

    # Numerical sanity — BF16 may have larger dynamic range issues than FP32,
    # so check explicitly for NaN and Inf.
    assert not torch.isnan(out).any(), "Output contains NaN values"
    assert not torch.isinf(out).any(), "Output contains Inf values"
