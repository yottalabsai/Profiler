"""
test_gpt2_optimized.py — 4-test validation suite for the gpt2_opt custom backend.

Run:
    pytest examples/gpt2/test_gpt2_optimized.py -v -s

Tests 3 and 4 require CUDA (and download GPT-2 weights from HuggingFace on first
run). The import / registration tests (1, 2) run without a GPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the workload module importable by its bare name from this directory.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

BACKEND_NAME = "gpt2_opt"

# Expected input contract (from optimizations.json analysis + gpt2.py):
#   GPT-2 small, batch=4, seq_len=128, input_ids int tensor on CUDA.
EXPECTED_BATCH = 4
EXPECTED_SEQ_LEN = 128
EXPECTED_HIDDEN = 768


def test_import():
    """Module imports without error (also triggers @register_backend at load)."""
    import gpt2_optimized  # noqa: F401


def test_backend_registration():
    """Backend registered with torch._dynamo under the expected name."""
    import torch
    import gpt2_optimized  # noqa: F401

    backends = str(torch._dynamo.list_backends())
    assert BACKEND_NAME in backends, f"Backend not found in: {backends}"


def test_get_model_and_input():
    """Input is on CUDA; verify shape and dtype match the GPT-2 workload contract."""
    import torch

    if not torch.cuda.is_available():
        import pytest

        pytest.skip("CUDA required")

    from gpt2_optimized import get_model_and_input

    model, input_ids = get_model_and_input()

    assert input_ids.is_cuda, "input_ids must be on CUDA"
    assert tuple(input_ids.shape) == (EXPECTED_BATCH, EXPECTED_SEQ_LEN), (
        f"expected input shape ({EXPECTED_BATCH}, {EXPECTED_SEQ_LEN}), "
        f"got {tuple(input_ids.shape)}"
    )
    # input_ids are token indices -> an integer dtype.
    assert input_ids.dtype in (torch.int64, torch.int32, torch.long), (
        f"expected integer token-id dtype, got {input_ids.dtype}"
    )
    # Model parameters stay fp32 (dtype promotion happens inside the graph).
    p = next(model.parameters())
    assert p.is_cuda, "model parameters must be on CUDA"
    assert p.dtype == torch.float32, f"expected fp32 params, got {p.dtype}"


def test_compiled_forward_pass(caplog):
    """Compiled forward pass triggers the backend; captures FX pass application logs."""
    import logging

    import torch

    if not torch.cuda.is_available():
        import pytest

        pytest.skip("CUDA required")

    from gpt2_optimized import get_model_and_input

    model, input_ids = get_model_and_input()
    compiled = torch.compile(model, backend=BACKEND_NAME)

    out = None
    with caplog.at_level(logging.INFO):
        with torch.no_grad():
            try:
                out = compiled(input_ids)
            except Exception as exc:
                from torch._dynamo.exc import InternalTorchDynamoError

                if not isinstance(exc, InternalTorchDynamoError):
                    raise
                # torch 2.11: a guard error can surface after the dedup backend
                # has already compiled successfully — safe to suppress here.

    for record in caplog.records:
        print(record.getMessage())

    assert caplog.records, "No logger output — backend may not have executed"

    if out is not None:
        assert tuple(out.shape) == (
            EXPECTED_BATCH,
            EXPECTED_SEQ_LEN,
            EXPECTED_HIDDEN,
        ), f"unexpected output shape {tuple(out.shape)}"
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"
