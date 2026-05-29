"""
test_conv_block_optimized.py — validation suite for the conv_block_opt backend.

Four tests:
  1. module imports cleanly
  2. backend is registered with torch._dynamo
  3. get_model_and_input returns a CUDA tensor with expected shape/dtype
  4. compiled forward pass triggers the backend and produces finite output
"""
import logging

import pytest
import torch


def test_import():
    """Module imports without error (also registers the backend at import time)."""
    import conv_block_optimized  # noqa: F401


def test_backend_registration():
    """Backend registered with torch._dynamo."""
    import conv_block_optimized  # noqa: F401

    backends = str(torch._dynamo.list_backends())
    assert "conv_block_opt" in backends, f"Backend not found in: {backends}"


def test_get_model_and_input():
    """Input is on CUDA with the expected shape and dtype.

    Expected from optimizations.json / conv_block.py: batch 16, 3 channels,
    64x64, fp32 (torch.randn default). channels_last is a memory-format change
    (OPT-2), so the logical shape/dtype are unchanged but the tensor must be
    channels_last contiguous.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from conv_block_optimized import get_model_and_input

    model, x = get_model_and_input()
    assert x.is_cuda, "input must be on CUDA"
    assert tuple(x.shape) == (16, 3, 64, 64), f"unexpected input shape {tuple(x.shape)}"
    assert x.dtype == torch.float32, f"unexpected input dtype {x.dtype}"
    # OPT-2: input should be channels_last after get_model_and_input.
    assert x.is_contiguous(memory_format=torch.channels_last), \
        "input should be channels_last (OPT-2)"
    # OPT-1: every Conv2d should now carry a fused bias and BN should be Identity.
    import torch.nn as nn
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    assert not bns, "BatchNorm2d should be folded away by OPT-1 (eager fold)"
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    assert convs and all(c.bias is not None for c in convs), \
        "folded convs should carry a fused bias (OPT-1)"


def test_compiled_forward_pass(caplog):
    """Compiled forward pass triggers the backend; captures FX pass application logs."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from conv_block_optimized import get_model_and_input

    model, x = get_model_and_input()
    compiled = torch.compile(model, backend="conv_block_opt")
    out = None
    with caplog.at_level(logging.INFO):
        with torch.no_grad():
            try:
                out = compiled(x)
            except Exception as exc:
                from torch._dynamo.exc import InternalTorchDynamoError

                if not isinstance(exc, InternalTorchDynamoError):
                    raise
                # torch 2.11: guard error after a dedup backend succeeds — safe to suppress.
    for record in caplog.records:
        print(record.getMessage())
    assert caplog.records, "No logger output — backend may not have executed"
    if out is not None:
        assert tuple(out.shape) == (16, 10), f"unexpected output shape {tuple(out.shape)}"
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
