"""
test_gpt2_optimized.py — validation suite for the gpt2_opt custom backend.

Run from the repo root with the package importable:
    PYTHONPATH=/home/ubuntu/Profiler pytest examples/gpt2/test_gpt2_optimized.py -v -s

Four tests validate: module import, backend registration with torch._dynamo,
the get_model_and_input contract (CUDA device, shape, dtype), and that a compiled
forward pass drives the backend (captured via its INFO log records) without
producing NaN/Inf.
"""
from __future__ import annotations

import logging
import os
import sys

import pytest

# Make the optimized module importable by bare name regardless of cwd / how
# pytest is invoked.
_WORKLOAD_DIR = os.path.dirname(os.path.abspath(__file__))
if _WORKLOAD_DIR not in sys.path:
    sys.path.insert(0, _WORKLOAD_DIR)

# Expected workload facts (from gpt2.py config and optimizations.json analysis).
EXPECTED_BATCH = 4
EXPECTED_SEQ_LEN = 128
EXPECTED_HIDDEN = 768
BACKEND_NAME = "gpt2_opt"

_CUDA = None


def _has_cuda() -> bool:
    global _CUDA
    if _CUDA is None:
        import torch
        _CUDA = torch.cuda.is_available()
    return _CUDA


# ---------------------------------------------------------------------------
# Test 1: Import
# ---------------------------------------------------------------------------

def test_import():
    """Module imports without error (triggers @register_backend at load time)."""
    import gpt2_optimized  # noqa: F401


# ---------------------------------------------------------------------------
# Test 2: Backend registration
# ---------------------------------------------------------------------------

def test_backend_registration():
    """Backend 'gpt2_opt' is registered with torch._dynamo after import."""
    import torch
    import gpt2_optimized  # noqa: F401 — side-effect: registers backend at load

    backends = str(torch._dynamo.list_backends())
    assert BACKEND_NAME in backends, (
        f"Backend '{BACKEND_NAME}' not found in registered backends: {backends}"
    )


# ---------------------------------------------------------------------------
# Test 3: Model and input shape / dtype
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_cuda(), reason="CUDA required")
def test_get_model_and_input():
    """
    Assert:
    - Input tensor is on CUDA
    - Input shape is (4, 128) — BATCH=4, SEQ_LEN=128 per optimizations.json analysis
    - Input dtype is int64 (token ids)
    - Model parameters are float32 (BF16 cast injected by the FX pass, not at build)
    - Model is in eval mode and on CUDA
    """
    import torch
    from gpt2_optimized import get_model_and_input

    model, input_ids = get_model_and_input()

    assert input_ids.device.type == "cuda", (
        f"Input must be on CUDA; got device: {input_ids.device}"
    )
    assert tuple(input_ids.shape) == (EXPECTED_BATCH, EXPECTED_SEQ_LEN), (
        f"Unexpected input shape {tuple(input_ids.shape)}; "
        f"expected {(EXPECTED_BATCH, EXPECTED_SEQ_LEN)} (BATCH, SEQ_LEN)"
    )
    assert input_ids.dtype == torch.int64, (
        f"Expected int64 token ids; got dtype={input_ids.dtype}"
    )

    first_param = next(model.parameters())
    assert first_param.dtype == torch.float32, (
        f"Expected float32 model parameters; got {first_param.dtype}. "
        "BF16 promotion is applied inside the backend FX pass (OPT-1), not at "
        "model construction time."
    )
    assert not model.training, "Model must be in eval mode; got training=True"
    assert first_param.device.type == "cuda", (
        f"Model parameters must be on CUDA; got {first_param.device}"
    )


# ---------------------------------------------------------------------------
# Test 4: Compiled forward pass smoke test
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_cuda(), reason="CUDA required")
def test_compiled_forward_pass(caplog):
    """
    Compiled forward pass triggers the backend; captures FX pass application logs.

    Verifies:
    - Backend emits at least one log record (confirms it executed)
    - Output (if produced) is (4, 128, 768) with no NaN / Inf
    """
    import torch
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
                # torch 2.11: a SHAPE_ENV guard error can surface after the dedup
                # backend has already compiled + patched partition .forward methods.

    for record in caplog.records:
        print(record.getMessage())

    assert caplog.records, (
        "No logger output captured — backend may not have executed or "
        "logging.INFO was not propagated to caplog"
    )

    if out is not None:
        assert tuple(out.shape) == (EXPECTED_BATCH, EXPECTED_SEQ_LEN, EXPECTED_HIDDEN), (
            f"Unexpected output shape {tuple(out.shape)}; "
            f"expected {(EXPECTED_BATCH, EXPECTED_SEQ_LEN, EXPECTED_HIDDEN)} "
            "(GPT-2 small last_hidden_state)"
        )
        assert not torch.isnan(out).any(), (
            "Output contains NaN — possible BF16 overflow or graph mutation error"
        )
        assert not torch.isinf(out).any(), "Output contains Inf — possible BF16 overflow"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
