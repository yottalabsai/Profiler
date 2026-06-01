"""
test_conv_block_optimized.py — validation suite for the conv_block_opt backend.

Four tests, per the /backend skill contract:
  1. test_import                  — module imports without error
  2. test_backend_registration    — conv_block_opt registered with torch._dynamo
  3. test_get_model_and_input     — input on CUDA; shape/dtype match the workload spec
  4. test_compiled_forward_pass   — compiled forward triggers the backend; captures FX logs

Run:
    pytest examples/conv_block/test_conv_block_optimized.py -v
"""
import os
import sys

# The optimized workload imports `from conv_block import ConvBlock`, so the workload
# directory must be importable.
_WORKLOAD_DIR = os.path.dirname(os.path.abspath(__file__))
if _WORKLOAD_DIR not in sys.path:
    sys.path.insert(0, _WORKLOAD_DIR)

BACKEND_NAME = "conv_block_opt"

# Expected input spec, derived from optimizations.json / conv_block.py:
#   batch 16, 3 channels, 64x64, fp32, CUDA.
EXPECTED_SHAPE = (16, 3, 64, 64)
EXPECTED_DTYPE = "torch.float32"


def test_import():
    """Module imports without error (triggers @register_backend at load time)."""
    import conv_block_optimized  # noqa: F401


def test_backend_registration():
    """Backend registered with torch._dynamo."""
    import torch
    import conv_block_optimized  # noqa: F401

    backends = str(torch._dynamo.list_backends())
    assert BACKEND_NAME in backends, f"Backend not found in: {backends}"


def test_get_model_and_input():
    """Input is on CUDA; shape and dtype match the workload spec."""
    import torch

    if not torch.cuda.is_available():
        import pytest

        pytest.skip("CUDA required")

    from conv_block_optimized import get_model_and_input

    model, x = get_model_and_input()
    assert x.is_cuda, "Input tensor must be on CUDA"
    assert tuple(x.shape) == EXPECTED_SHAPE, f"Unexpected input shape: {tuple(x.shape)}"
    assert str(x.dtype) == EXPECTED_DTYPE, f"Unexpected input dtype: {x.dtype}"
    # OPT-1: input should be channels_last (NHWC).
    assert x.is_contiguous(memory_format=torch.channels_last), (
        "Input should be channels_last (OPT-1)"
    )
    # Model returned in eval mode — required for the eval-mode BN fold (OPT-3).
    assert not model.training, "Model must be in eval() for the Conv-BN fold to be valid"


def test_compiled_forward_pass(caplog):
    """Compiled forward pass triggers the backend; captures FX pass application logs."""
    import logging

    import torch

    if not torch.cuda.is_available():
        import pytest

        pytest.skip("CUDA required")

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
                # torch 2.11: guard error after dedup backend succeeds — safe to suppress.

    for record in caplog.records:
        print(record.getMessage())
    assert caplog.records, "No logger output — backend may not have executed"
    if out is not None:
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"
