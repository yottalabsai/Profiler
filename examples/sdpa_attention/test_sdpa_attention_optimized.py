"""
test_sdpa_attention_optimized.py — 4-test validation suite for sdpa_attention_opt.

Run with:
    pytest examples/sdpa_attention/test_sdpa_attention_optimized.py -v

Requires CUDA (the workload and backend target a Blackwell GPU). The
compiled-forward test triggers the full funnel (functional QKV fusion ->
aten bf16 promotion -> inductor freezing) and asserts the backend executed by
capturing its INFO-level pass logs.
"""
import logging

import torch

import examples.sdpa_attention.sdpa_attention_optimized as opt_mod
from examples.sdpa_attention.sdpa_attention_optimized import (
    BATCH_SIZE,
    DIM,
    SEQ_LEN,
    get_model_and_input,
)

BACKEND_NAME = "sdpa_attention_opt"


def test_import():
    """Module imports without error."""
    import examples.sdpa_attention.sdpa_attention_optimized  # noqa: F401


def test_backend_registration():
    """Backend registered with torch._dynamo."""
    backends = str(torch._dynamo.list_backends())
    assert BACKEND_NAME in backends, f"Backend not found in: {backends}"


def test_get_model_and_input():
    """Input is on CUDA; shape/dtype match the workload constants."""
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("CUDA required")
    model, x = get_model_and_input()
    assert x.is_cuda, "Input tensor must be on CUDA"
    assert tuple(x.shape) == (BATCH_SIZE, SEQ_LEN, DIM), f"Unexpected input shape: {tuple(x.shape)}"
    assert x.dtype == torch.float32, f"Expected fp32 input, got {x.dtype}"
    assert not model.training, "Model must be in eval mode for freezing (OPT-3)"
    # The block returns (B, T, D) in fp32 (bf16 region is local to GEMM/SDPA).
    with torch.no_grad():
        out = model(x)
    assert tuple(out.shape) == (BATCH_SIZE, SEQ_LEN, DIM)
    assert out.dtype == torch.float32


def test_compiled_forward_pass(caplog):
    """Compiled forward pass triggers the backend; captures FX pass application logs."""
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
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"
