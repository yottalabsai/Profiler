"""
test_transformer_stack_optimized.py — Validation suite for the custom backend.
"""
import pytest
import torch


def test_import():
    """Module imports without error."""
    import transformer_stack_optimized  # noqa: F401


def test_backend_registration():
    """Backend registered with torch._dynamo."""
    import torch
    import transformer_stack_optimized  # noqa: F401
    backends = str(torch._dynamo.list_backends())
    assert "transformer_stack_opt" in backends, f"Backend not found in: {backends}"


def test_get_model_and_input():
    """Model and input have expected shapes and dtypes (BF16 from OPT-1)."""
    import torch
    from transformer_stack_optimized import get_model_and_input
    model, x = get_model_and_input()
    assert x.device.type == "cuda", "Input must be on CUDA"
    assert x.shape == (4, 128, 512), f"Expected (4,128,512) got {x.shape}"
    assert x.dtype == torch.bfloat16, f"Expected bfloat16 got {x.dtype}"
    assert next(model.parameters()).dtype == torch.bfloat16, "Model must be BF16 (OPT-1)"


def test_forward_pass():
    """Uncompiled forward pass completes without error."""
    import torch
    from transformer_stack_optimized import get_model_and_input
    model, x = get_model_and_input()
    with torch.no_grad():
        out = model(x)
    assert out is not None
    assert out.shape == (4, 128, 512), f"Unexpected output shape: {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN"
    assert not torch.isinf(out).any(), "Output contains Inf"
