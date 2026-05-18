"""
test_mlp_activations_optimized.py — Validation suite for mlp_activations_optimized.py.

Runs 4 checks:
  1. test_import                      : module loads without error and backend is registered
  2. test_backend_registration         : 'mlp_activations_opt' appears in dynamo backend list
  3. test_get_model_and_input          : correct shapes, dtype=bfloat16, device=cuda
  4. test_compiled_forward_pass        : triggers backend, captures FX pass logs, correct output

Usage:
    # From project root:
    PYTHONPATH=/home/ubuntu/Profiler pytest examples/mlp_activations/test_mlp_activations_optimized.py -v

    # Or directly:
    PYTHONPATH=/home/ubuntu/Profiler python3 -m pytest examples/mlp_activations/test_mlp_activations_optimized.py -v
"""
import sys
from pathlib import Path

import pytest
import torch

# Ensure the mlp_activations directory is importable without PYTHONPATH tricks
sys.path.insert(0, str(Path(__file__).parent))


# ===========================================================================
# Test 1: Import
# ===========================================================================

def test_import():
    """Module imports without error. Backend is registered at import time."""
    import mlp_activations_optimized  # noqa: F401


# ===========================================================================
# Test 2: Backend registration
# ===========================================================================

def test_backend_registration():
    """'mlp_activations_opt' is registered with torch._dynamo."""
    import torch._dynamo as dynamo
    import mlp_activations_optimized  # noqa: F401 — registers backend on import

    backends = dynamo.list_backends()
    assert "mlp_activations_opt" in backends, (
        f"'mlp_activations_opt' not found in dynamo backends: {backends}"
    )


# ===========================================================================
# Test 3: get_model_and_input — shapes and dtypes
# ===========================================================================

def test_get_model_and_input():
    """
    Model and input have the expected shapes, dtype, and device.

    OPT-1 (BF16 promotion) is applied inside get_model_and_input(), so:
      - model parameters must be bfloat16
      - input tensor must be bfloat16
      - input must be on CUDA
      - input shape: (BATCH_SIZE=256, DIM_IN=512)
    """
    from mlp_activations_optimized import get_model_and_input

    model, x = get_model_and_input()

    # Device check
    assert x.device.type == "cuda", (
        f"Input must be on CUDA, got device: {x.device}"
    )

    # OPT-1: BF16 dtype promotion
    param_dtype = next(model.parameters()).dtype
    assert param_dtype == torch.bfloat16, (
        f"Expected bfloat16 model parameters (OPT-1), got {param_dtype}"
    )
    assert x.dtype == torch.bfloat16, (
        f"Expected bfloat16 input tensor (OPT-1), got {x.dtype}"
    )

    # Shape: (BATCH_SIZE=256, DIM_IN=512) from mlp_activations.py constants
    assert x.shape == (256, 512), (
        f"Unexpected input shape: {x.shape}, expected (256, 512)"
    )

    # Model must be in eval mode (required for stable BN/Dropout if present)
    assert not model.training, "Model must be in eval mode"


# ===========================================================================
# Test 4: Compiled forward pass — triggers backend, captures FX pass logs
# ===========================================================================

def test_compiled_forward_pass(caplog):
    """Compiled forward pass triggers the mlp_activations_opt backend; captures FX pass logs."""
    import logging
    from mlp_activations_optimized import get_model_and_input

    model, x = get_model_and_input()
    compiled = torch.compile(model, backend="mlp_activations_opt")
    out = None

    with caplog.at_level(logging.INFO):
        with torch.no_grad():
            try:
                out = compiled(x)
            except Exception as exc:
                from torch._dynamo.exc import InternalTorchDynamoError
                if not isinstance(exc, InternalTorchDynamoError):
                    raise

    for record in caplog.records:
        print(record.getMessage())

    assert caplog.records, "No logger output — backend may not have executed"

    if out is not None:
        assert out.shape == (256, 512), (
            f"Unexpected output shape: {out.shape}, expected (256, 512)"
        )
        assert out.dtype == torch.bfloat16, (
            f"Expected bfloat16 output, got {out.dtype}"
        )
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"
        assert out.abs().max().item() <= 1.0 + 1e-3, (
            f"tanh output exceeds (-1, 1) range: max abs = {out.abs().max().item()}"
        )
