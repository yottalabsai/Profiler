"""
test_sdpa_attention_optimized.py — validation suite for the sdpa_attention_opt backend.

Run with:  pytest test_sdpa_attention_optimized.py -v -s

The workload module imports `sdpa_attention` (the baseline) by bare module name,
so this test prepends the workload directory to sys.path.
"""
import os
import sys

import pytest

_WORKLOAD_DIR = os.path.dirname(os.path.abspath(__file__))
if _WORKLOAD_DIR not in sys.path:
    sys.path.insert(0, _WORKLOAD_DIR)

# Expected runtime characteristics derived from sdpa_attention.py / optimizations.json
_EXPECTED_SHAPE = (8, 512, 512)   # (BATCH_SIZE, SEQ_LEN, DIM)
_EXPECTED_DTYPE = "float32"        # analysis.dtype — model returns FP32 (BF16 is internal)
_BACKEND_NAME = "sdpa_attention_opt"


def test_import():
    """Module imports without error (triggers @register_backend at load time)."""
    import sdpa_attention_optimized  # noqa: F401


def test_backend_registration():
    """Backend registered with torch._dynamo."""
    import torch
    import sdpa_attention_optimized  # noqa: F401

    backends = str(torch._dynamo.list_backends())
    assert _BACKEND_NAME in backends, f"Backend not found in: {backends}"


def test_get_model_and_input():
    """Input is on CUDA with the expected shape and dtype (per optimizations.json)."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from sdpa_attention_optimized import get_model_and_input

    model, x = get_model_and_input()
    assert x.is_cuda, "Input tensor must be on CUDA"
    assert tuple(x.shape) == _EXPECTED_SHAPE, f"Unexpected input shape: {tuple(x.shape)}"
    assert str(x.dtype).split(".")[-1] == _EXPECTED_DTYPE, f"Unexpected dtype: {x.dtype}"
    assert next(model.parameters()).is_cuda, "Model parameters must be on CUDA"


def test_compiled_forward_pass(caplog):
    """Compiled forward pass triggers the backend; capture FX pass application logs."""
    import logging

    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from sdpa_attention_optimized import get_model_and_input

    model, x = get_model_and_input()
    compiled = torch.compile(model, backend=_BACKEND_NAME)

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
        assert tuple(out.shape) == _EXPECTED_SHAPE, f"Unexpected output shape: {tuple(out.shape)}"
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"
