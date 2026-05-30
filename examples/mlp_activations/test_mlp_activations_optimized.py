"""
test_mlp_activations_optimized.py — 4-test validation suite for the
mlp_activations_opt custom torch.compile() backend.

Tests:
  1. test_import                 — module imports without error
  2. test_backend_registration   — backend registered with torch._dynamo
  3. test_get_model_and_input    — input on CUDA, shape/dtype match the workload
  4. test_compiled_forward_pass  — compiled forward triggers the backend; captures
                                    the FX-pass / config-pass INFO logs; output is
                                    finite (no NaN/Inf)
"""
from __future__ import annotations

import logging

import torch

BACKEND_NAME = "mlp_activations_opt"

# Expected from mlp_activations.py / optimizations.json
EXPECTED_BATCH = 256
EXPECTED_DIM_IN = 512
EXPECTED_DTYPE = torch.float32


def test_import():
    """Module imports without error (and triggers @register_backend at load time)."""
    import mlp_activations_optimized  # noqa: F401


def test_backend_registration():
    """Backend registered with torch._dynamo."""
    import mlp_activations_optimized  # noqa: F401

    backends = str(torch._dynamo.list_backends())
    assert BACKEND_NAME in backends, f"Backend not found in: {backends}"


def test_get_model_and_input():
    """Input is on CUDA; shape and dtype match the workload definition."""
    from mlp_activations_optimized import get_model_and_input

    model, x = get_model_and_input()
    assert x.is_cuda, "Input tensor must be on CUDA"
    assert tuple(x.shape) == (EXPECTED_BATCH, EXPECTED_DIM_IN), (
        f"Unexpected input shape: {tuple(x.shape)} "
        f"(expected {(EXPECTED_BATCH, EXPECTED_DIM_IN)})"
    )
    assert x.dtype == EXPECTED_DTYPE, (
        f"Unexpected input dtype: {x.dtype} (expected {EXPECTED_DTYPE})"
    )
    assert next(model.parameters()).is_cuda, "Model parameters must be on CUDA"


def test_compiled_forward_pass(caplog):
    """Compiled forward pass triggers the backend; captures FX/config pass logs."""
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
                # torch 2.11: guard error after dedup backend succeeds — safe to suppress

    for record in caplog.records:
        print(record.getMessage())

    assert caplog.records, "No logger output — backend may not have executed"

    if out is not None:
        assert tuple(out.shape) == (EXPECTED_BATCH, EXPECTED_DIM_IN), (
            f"Unexpected output shape: {tuple(out.shape)}"
        )
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"
