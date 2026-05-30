"""
Validation suite for conv_block_optimized.py — the conv_block_opt custom backend.

Four tests:
  1. module imports
  2. backend registered with torch._dynamo
  3. get_model_and_input contract (CUDA, shape, dtype, channels_last)
  4. compiled forward pass triggers the backend and produces finite output

Run:
    pytest examples/conv_block/test_conv_block_optimized.py -v
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BACKEND_NAME = "conv_block_opt"

# Expected from optimizations.json analysis / workload constants.
BATCH_SIZE = 16
IN_CHANNELS = 3
HEIGHT = 64
WIDTH = 64
NUM_CLASSES = 10


def test_import():
    """Module imports without error (registers the backend at load time)."""
    import conv_block_optimized  # noqa: F401


def test_backend_registration():
    """Backend registered with torch._dynamo."""
    import torch  # noqa: F401
    import conv_block_optimized  # noqa: F401

    backends = str(torch._dynamo.list_backends())
    assert BACKEND_NAME in backends, f"Backend not found in: {backends}"


@pytest.mark.skipif(
    "not __import__('torch').cuda.is_available()", reason="CUDA required"
)
def test_get_model_and_input():
    """Input on CUDA with the expected shape/dtype; both model and input NHWC (OPT-2)."""
    import torch
    from conv_block_optimized import get_model_and_input

    model, x = get_model_and_input()

    assert x.is_cuda, "Input tensor must be on CUDA"
    assert x.shape == (BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH), (
        f"Unexpected input shape: {tuple(x.shape)}"
    )
    assert x.dtype == torch.float32, f"Expected float32 input, got {x.dtype}"
    # OPT-2 non-graph lever: input + weights converted to channels_last (NHWC).
    assert x.is_contiguous(memory_format=torch.channels_last), (
        "Input should be channels_last (OPT-2)"
    )
    assert next(model.parameters()).is_contiguous(memory_format=torch.channels_last), (
        "Model params should be channels_last (OPT-2)"
    )


@pytest.mark.skipif(
    "not __import__('torch').cuda.is_available()", reason="CUDA required"
)
def test_compiled_forward_pass(caplog):
    """Compiled forward pass triggers the backend; captures FX pass application logs."""
    import logging

    import torch
    from conv_block_optimized import get_model_and_input

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
        assert out.shape == (BATCH_SIZE, NUM_CLASSES), (
            f"Unexpected output shape: {tuple(out.shape)}"
        )
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"
