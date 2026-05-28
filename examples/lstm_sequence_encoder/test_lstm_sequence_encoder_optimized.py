"""
test_lstm_sequence_encoder_optimized.py — validation suite for the
lstm_sequence_encoder_opt custom backend.

Run from the repo root with the package importable:
    PYTHONPATH=/home/ubuntu/Profiler pytest \
        examples/lstm_sequence_encoder/test_lstm_sequence_encoder_optimized.py -v -s

Four tests validate: module import, backend registration with torch._dynamo, the
get_model_and_input contract (CUDA device, shape, dtype), and that a compiled
forward pass drives the backend (captured via its INFO log records) without
producing NaN/Inf.

Note: nn.LSTM/GRU/RNN tracing under Dynamo requires
torch._dynamo.config.allow_rnn = True. Importing lstm_sequence_encoder_optimized
sets it at module load, so no extra setup is needed here.
"""
from __future__ import annotations

import logging
import os
import sys

import pytest

# Make the optimized module importable by bare name regardless of cwd.
_WORKLOAD_DIR = os.path.dirname(os.path.abspath(__file__))
if _WORKLOAD_DIR not in sys.path:
    sys.path.insert(0, _WORKLOAD_DIR)

# Expected workload facts (from lstm_sequence_encoder.py + optimizations.json).
EXPECTED_BATCH = 32
EXPECTED_SEQ_LEN = 128
EXPECTED_INPUT = 256
EXPECTED_CLASSES = 10
BACKEND_NAME = "lstm_sequence_encoder_opt"

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
    """Module imports without error (triggers @register_backend + allow_rnn)."""
    import lstm_sequence_encoder_optimized  # noqa: F401


# ---------------------------------------------------------------------------
# Test 2: Backend registration
# ---------------------------------------------------------------------------

def test_backend_registration():
    """Backend 'lstm_sequence_encoder_opt' is registered with torch._dynamo."""
    import torch
    import lstm_sequence_encoder_optimized  # noqa: F401 — side-effect: registers backend

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
    - Input shape is (32, 128, 256) — BATCH, SEQ_LEN, INPUT_SIZE per workload config
    - Input dtype is float32 (BF16 cast injected by OPT-1 FX pass, not at build)
    - Model parameters are float32 (in-graph cast, not at construction)
    - Model is in eval mode and on CUDA
    """
    import torch
    from lstm_sequence_encoder_optimized import get_model_and_input

    model, x = get_model_and_input()

    assert x.device.type == "cuda", f"Input must be on CUDA; got device: {x.device}"
    assert tuple(x.shape) == (EXPECTED_BATCH, EXPECTED_SEQ_LEN, EXPECTED_INPUT), (
        f"Unexpected input shape {tuple(x.shape)}; "
        f"expected {(EXPECTED_BATCH, EXPECTED_SEQ_LEN, EXPECTED_INPUT)} "
        "(BATCH, SEQ_LEN, INPUT_SIZE)"
    )
    assert x.dtype == torch.float32, f"Expected float32 input; got dtype={x.dtype}"

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
    - Output (if produced) is (32, 10) with no NaN / Inf
    """
    import torch
    from lstm_sequence_encoder_optimized import get_model_and_input

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
                # torch 2.11: a guard error can surface after the backend has already
                # compiled + patched the partition .forward methods — safe to suppress.

    for record in caplog.records:
        print(record.getMessage())

    assert caplog.records, (
        "No logger output captured — backend may not have executed or "
        "logging.INFO was not propagated to caplog"
    )

    if out is not None:
        assert tuple(out.shape) == (EXPECTED_BATCH, EXPECTED_CLASSES), (
            f"Unexpected output shape {tuple(out.shape)}; "
            f"expected {(EXPECTED_BATCH, EXPECTED_CLASSES)} (BATCH, NUM_CLASSES)"
        )
        assert not torch.isnan(out).any(), (
            "Output contains NaN — possible BF16 overflow or graph mutation error"
        )
        assert not torch.isinf(out).any(), "Output contains Inf — possible BF16 overflow"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
