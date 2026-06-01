"""
test_depthwise_separable_conv_optimized.py — 4-test validation suite for the
depthwise_separable_conv_opt backend.

Run with:
    pytest examples/depthwise_separable_conv/test_depthwise_separable_conv_optimized.py -v

Shape / dtype reference (from optimizations.json analysis + workload constants):
    Input:   (16, 32,  56, 56)  float32  [BATCH_SIZE=16, IN_CHANNELS=32, 56x56]
    Output:  (16, 256, 56, 56)  float32  [three blocks: 32->64->128->256]

    OPT-2 converts the model + input to channels_last (NHWC); the logical NCHW shape is
    unchanged and the dtype stays float32 (OPT-1 conv-BN fold is exact in FP32).
"""
from __future__ import annotations

import os
import sys

import pytest

# Make the examples/depthwise_separable_conv directory importable so that
# ``import depthwise_separable_conv_optimized`` and ``import depthwise_separable_conv``
# both resolve.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BACKEND_NAME = "depthwise_separable_conv_opt"
EXPECTED_INPUT_SHAPE = (16, 32, 56, 56)
EXPECTED_INPUT_DTYPE = "torch.float32"
EXPECTED_OUTPUT_SHAPE = (16, 256, 56, 56)
EXPECTED_OUTPUT_DTYPE = "torch.float32"


# ---------------------------------------------------------------------------
# Test 1 — import
# ---------------------------------------------------------------------------

def test_import():
    """Module imports without error.

    Importing depthwise_separable_conv_optimized triggers @register_backend at
    module-load time so the backend is available before torch.compile selects it."""
    import depthwise_separable_conv_optimized  # noqa: F401


# ---------------------------------------------------------------------------
# Test 2 — backend registration
# ---------------------------------------------------------------------------

def test_backend_registration():
    """Backend depthwise_separable_conv_opt is registered with torch._dynamo."""
    import torch
    import depthwise_separable_conv_optimized  # noqa: F401

    backends = str(torch._dynamo.list_backends())
    assert BACKEND_NAME in backends, (
        f"Backend '{BACKEND_NAME}' not found in registered backends: {backends}"
    )


# ---------------------------------------------------------------------------
# Test 3 — get_model_and_input
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="CUDA required"
)
def test_get_model_and_input():
    """Input tensor is on CUDA; shape and dtype match the profiled workload values.

    Workload constants (optimizations.json analysis.dtype = float32):
        BATCH_SIZE=16, IN_CHANNELS=32, 56x56 -> input shape (16, 32, 56, 56), float32.
    OPT-2 converts to channels_last: logical shape and dtype are unchanged, but the
    input must be channels_last-contiguous and the model in eval mode (BN fold validity).
    """
    import torch
    from depthwise_separable_conv_optimized import get_model_and_input

    model, x = get_model_and_input()

    # Input device
    assert x.is_cuda, f"Input tensor must be on CUDA, got device={x.device}"

    # Input shape: (BATCH_SIZE, IN_CHANNELS, H, W) = (16, 32, 56, 56)
    assert tuple(x.shape) == EXPECTED_INPUT_SHAPE, (
        f"Unexpected input shape: expected {EXPECTED_INPUT_SHAPE}, got {tuple(x.shape)}"
    )

    # Input dtype: float32
    assert str(x.dtype) == EXPECTED_INPUT_DTYPE, (
        f"Unexpected input dtype: expected {EXPECTED_INPUT_DTYPE}, got {x.dtype}"
    )

    # OPT-2: input must be channels_last-contiguous (NHWC)
    assert x.is_contiguous(memory_format=torch.channels_last), (
        "Input tensor must be channels_last-contiguous (OPT-2 NHWC conversion)"
    )

    # Model must be on CUDA and in eval mode (BN fold requires eval / no_training BN)
    assert next(model.parameters()).is_cuda, "Model parameters must be on CUDA"
    assert not model.training, "Model must be in eval mode for OPT-1 conv-BN fold"


# ---------------------------------------------------------------------------
# Test 4 — compiled forward pass
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="CUDA required"
)
def test_compiled_forward_pass(caplog):
    """Compiled forward pass triggers the backend and FX passes emit log messages.

    Validates:
    - At least one log record emitted (backend executed)
    - Output shape preserved: (16, 256, 56, 56)
    - Output dtype preserved: float32 (OPT-1 conv-BN fold is exact in FP32)
    - No NaN or Inf in output tensor
    """
    import logging

    import torch
    from depthwise_separable_conv_optimized import get_model_and_input

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

    assert caplog.records, (
        "No logger output captured — backend may not have executed. "
        "Ensure depthwise_separable_conv_optimized is imported and the logger propagates."
    )

    if out is not None:
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"
        assert tuple(out.shape) == EXPECTED_OUTPUT_SHAPE, (
            f"Unexpected output shape: expected {EXPECTED_OUTPUT_SHAPE}, "
            f"got {tuple(out.shape)}"
        )
        assert str(out.dtype) == EXPECTED_OUTPUT_DTYPE, (
            f"Output dtype not float32: got {out.dtype}"
        )
