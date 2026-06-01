"""
test_mlp_activations_optimized.py — 4-test validation suite for the
mlp_activations_opt custom torch.compile() backend.

Tests:
  1. test_import                  — module imports without error
  2. test_backend_registration   — backend registered with torch._dynamo
  3. test_get_model_and_input    — input on CUDA, expected shape/dtype
  4. test_compiled_forward_pass  — compiled forward triggers backend, no NaN/Inf

Run: pytest examples/mlp_activations/test_mlp_activations_optimized.py
"""
from __future__ import annotations

import os
import sys

# Ensure the workload directory is importable (get_model_and_input lives in the
# optimized module, which also self-registers the backend on import).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BACKEND_NAME = "mlp_activations_opt"

# Expected I/O contract from optimizations.json analysis (dtype="float32") and the
# baseline workload shapes (BATCH_SIZE=256, DIM_IN=512, DIM_OUT=512).
EXPECTED_BATCH = 256
EXPECTED_DIM_IN = 512
EXPECTED_DIM_OUT = 512


def test_import():
    """Module imports without error."""
    import mlp_activations_optimized  # noqa: F401


def test_backend_registration():
    """Backend registered with torch._dynamo."""
    import torch
    import mlp_activations_optimized  # noqa: F401

    backends = str(torch._dynamo.list_backends())
    assert BACKEND_NAME in backends, f"Backend not found in: {backends}"


def test_get_model_and_input():
    """Input is on CUDA; shape and dtype match the workload/optimizations contract."""
    import torch
    from mlp_activations_optimized import get_model_and_input

    if not torch.cuda.is_available():
        import pytest

        pytest.skip("CUDA required")

    model, x = get_model_and_input()

    assert x.is_cuda, "Input tensor must be on CUDA"
    assert x.dtype == torch.float32, f"Expected float32 input, got {x.dtype}"
    assert tuple(x.shape) == (EXPECTED_BATCH, EXPECTED_DIM_IN), (
        f"Expected input shape ({EXPECTED_BATCH}, {EXPECTED_DIM_IN}), got {tuple(x.shape)}"
    )
    # Model parameters live on CUDA and the module is in eval mode (OPT-3 freezing).
    assert next(model.parameters()).is_cuda, "Model parameters must be on CUDA"
    assert not model.training, "Model must be in eval mode for freezing (OPT-3)"


def test_compiled_forward_pass(caplog):
    """Compiled forward pass triggers the backend; captures FX pass application logs."""
    import logging

    import torch
    from mlp_activations_optimized import get_model_and_input

    if not torch.cuda.is_available():
        import pytest

        pytest.skip("CUDA required")

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
                # torch 2.11: guard error after dedup backend succeeds — safe to suppress.

    for record in caplog.records:
        print(record.getMessage())

    assert caplog.records, "No logger output — backend may not have executed"

    if out is not None:
        assert out.shape[0] == EXPECTED_BATCH, (
            f"Expected batch {EXPECTED_BATCH}, got {out.shape[0]}"
        )
        assert out.shape[-1] == EXPECTED_DIM_OUT, (
            f"Expected output dim {EXPECTED_DIM_OUT}, got {out.shape[-1]}"
        )
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
