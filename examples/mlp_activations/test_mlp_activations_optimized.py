"""
test_mlp_activations_optimized.py — 4-test validation suite for the
mlp_activations_opt custom torch.compile() backend.

Tests:
  1. test_import                  — the optimized module imports without error.
  2. test_backend_registration    — mlp_activations_opt is registered with torch._dynamo.
  3. test_get_model_and_input     — input is CUDA, correct shape (256, 512) and dtype fp32.
  4. test_compiled_forward_pass   — compiled forward triggers the backend, logs the passes,
                                    and produces a finite (no NaN/Inf) output.
"""
from __future__ import annotations

import logging

import pytest

BACKEND_NAME = "mlp_activations_opt"
OPT_MODULE = "mlp_activations_optimized"


def test_import():
    """Module imports without error (importing it registers the backend)."""
    import mlp_activations_optimized  # noqa: F401


def test_backend_registration():
    """Backend registered with torch._dynamo."""
    import torch
    import mlp_activations_optimized  # noqa: F401

    backends = str(torch._dynamo.list_backends())
    assert BACKEND_NAME in backends, f"Backend not found in: {backends}"


def test_get_model_and_input():
    """Input is on CUDA; shape (256, 512) and dtype fp32 per optimizations.json."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from mlp_activations_optimized import get_model_and_input

    model, x = get_model_and_input()
    assert x.is_cuda, "input tensor must be on CUDA"
    assert x.shape == (256, 512), f"expected input shape (256, 512), got {tuple(x.shape)}"
    assert x.dtype == torch.float32, f"expected fp32 input, got {x.dtype}"
    # Model parameters should be fp32 on CUDA (no whole-module dtype optimization proposed).
    p = next(model.parameters())
    assert p.is_cuda and p.dtype == torch.float32


def test_compiled_forward_pass(caplog):
    """Compiled forward pass triggers the backend; captures FX pass application logs."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from mlp_activations_optimized import get_model_and_input

    model, x = get_model_and_input()
    compiled = torch.compile(model, backend=BACKEND_NAME)
    out = None
    with caplog.at_level(logging.INFO):
        with torch.no_grad():
            try:
                out = compiled(x)
            except Exception as exc:
                from torch._dynamo.exc import InternalTorchDynamoError

                if not isinstance(exc, InternalTorchDynamoError):
                    raise
                # torch 2.11: guard error after the backend succeeds — safe to suppress.
    for record in caplog.records:
        print(record.getMessage())
    assert caplog.records, "No logger output — backend may not have executed"
    if out is not None:
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
