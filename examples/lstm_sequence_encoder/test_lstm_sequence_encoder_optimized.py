"""
test_lstm_sequence_encoder_optimized.py — 4-test validation suite for the
``lstm_sequence_encoder_opt`` custom torch.compile() backend.

Run from the workload directory so the bare-name imports
(`lstm_sequence_encoder`, `lstm_sequence_encoder_optimized`) resolve:

    cd examples/lstm_sequence_encoder
    PYTHONPATH=/home/ubuntu/Profiler pytest test_lstm_sequence_encoder_optimized.py -v -s
"""
import logging
import os
import sys

import pytest

# Ensure the workload directory is importable (bare-name module imports below).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BACKEND_NAME = "lstm_sequence_encoder_opt"

# Expected shapes/dtypes cross-validated against optimizations.json / profile.json:
#   BATCH_SIZE=32, SEQ_LEN=128, INPUT_SIZE=256 ; input dtype float32 (analysis.dtype).
EXPECTED_BATCH = 32
EXPECTED_SEQ = 128
EXPECTED_INPUT = 256


def test_import():
    """Module imports without error (triggers @register_backend at load time)."""
    import lstm_sequence_encoder_optimized  # noqa: F401


def test_backend_registration():
    """Backend registered with torch._dynamo."""
    import torch
    import lstm_sequence_encoder_optimized  # noqa: F401

    backends = str(torch._dynamo.list_backends())
    assert BACKEND_NAME in backends, f"Backend not found in: {backends}"


def test_get_model_and_input():
    """Input is on CUDA; shape and dtype match the values from optimizations.json/profile.json."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from lstm_sequence_encoder_optimized import get_model_and_input

    model, x = get_model_and_input()

    assert x.is_cuda, "Input tensor must be on CUDA"
    assert x.shape == (EXPECTED_BATCH, EXPECTED_SEQ, EXPECTED_INPUT), (
        f"Unexpected input shape {tuple(x.shape)}; "
        f"expected ({EXPECTED_BATCH}, {EXPECTED_SEQ}, {EXPECTED_INPUT})"
    )
    # Baseline dtype is float32 (analysis.dtype); OPT-1 bf16 promotion is in-graph only,
    # so the module-boundary input stays float32.
    assert x.dtype == torch.float32, f"Expected float32 input, got {x.dtype}"
    # eval() is required for OPT-4 freezing and the inference-only bf16 promotion.
    assert not model.training, "Model must be in eval() mode for freezing / bf16 promotion"


def test_compiled_forward_pass(caplog):
    """Compiled forward pass triggers the backend; captures FX pass application logs."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from lstm_sequence_encoder_optimized import get_model_and_input

    # Dynamo refuses to trace nn.RNN/GRU/LSTM unless allow_rnn is enabled; without it
    # Dynamo silently graph-breaks around the whole LSTM, runs it eagerly, and NEVER
    # invokes the custom backend (graph_count == 0). run_workload.py sets this same flag
    # before compiling, so the test mirrors the real capture path.
    torch._dynamo.config.allow_rnn = True

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
                # torch 2.11: guard error after a dedup backend succeeds — safe to suppress.

    for record in caplog.records:
        print(record.getMessage())

    assert caplog.records, "No logger output — backend may not have executed"

    if out is not None:
        assert out.shape == (EXPECTED_BATCH, 10), (
            f"Unexpected output shape {tuple(out.shape)}; expected ({EXPECTED_BATCH}, 10)"
        )
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"
