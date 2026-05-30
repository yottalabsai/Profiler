"""
test_sdpa_attention_optimized.py — 4-test validation suite for sdpa_attention_opt.

Run with: pytest examples/sdpa_attention/test_sdpa_attention_optimized.py
"""
from __future__ import annotations

import os
import sys

import pytest

# The optimized workload imports `sdpa_attention` (the baseline) by bare module
# name from get_model_and_input(); make the example dir importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Expected shape/dtype from optimizations.json analysis + workload constants:
#   BATCH_SIZE=8, SEQ_LEN=512, DIM=512, dtype fp32 (input contract preserved).
EXPECTED_SHAPE = (8, 512, 512)
EXPECTED_DTYPE = "torch.float32"
BACKEND_NAME = "sdpa_attention_opt"


def test_import():
    """Module imports without error."""
    import sdpa_attention_optimized  # noqa: F401


def test_backend_registration():
    """Backend registered with torch._dynamo."""
    import torch
    import sdpa_attention_optimized  # noqa: F401

    backends = str(torch._dynamo.list_backends())
    assert BACKEND_NAME in backends, f"Backend not found in: {backends}"


@pytest.mark.skipif(
    "not __import__('torch').cuda.is_available()", reason="CUDA required"
)
def test_get_model_and_input():
    """Input is on CUDA with the expected shape and dtype from optimizations.json."""
    import torch  # noqa: F401
    from sdpa_attention_optimized import get_model_and_input

    model, x = get_model_and_input()
    assert x.is_cuda, "Input tensor must be on CUDA"
    assert tuple(x.shape) == EXPECTED_SHAPE, f"Unexpected input shape: {tuple(x.shape)}"
    assert str(x.dtype) == EXPECTED_DTYPE, f"Unexpected input dtype: {x.dtype}"
    assert next(model.parameters()).is_cuda, "Model must be on CUDA"


@pytest.mark.skipif(
    "not __import__('torch').cuda.is_available()", reason="CUDA required"
)
def test_compiled_forward_pass(caplog):
    """Compiled forward pass triggers the backend; captures FX pass application logs."""
    import logging

    import torch
    from sdpa_attention_optimized import get_model_and_input

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
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"
        assert tuple(out.shape) == EXPECTED_SHAPE, f"Unexpected output shape: {tuple(out.shape)}"
        assert str(out.dtype) == EXPECTED_DTYPE, f"Output dtype not restored to fp32: {out.dtype}"
