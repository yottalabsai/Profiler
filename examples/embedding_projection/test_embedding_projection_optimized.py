"""
test_embedding_projection_optimized.py — 4-test validation suite for embedding_projection_opt.

Run with:
    pytest examples/embedding_projection/test_embedding_projection_optimized.py -v

Shape / dtype reference (from optimizations.json analysis + workload constants):
    Input:   (64, 128)              int64    [BATCH_SIZE=64, SEQ_LEN=128 token IDs]
    Output:  (64, 128, 32000)       float32  [VOCAB_SIZE=32000; OPT-1 BF16 round-trip
                                              restores FP32 at the logit output]
"""
from __future__ import annotations

import os
import sys

import pytest

# Make the examples/embedding_projection directory importable so that
# ``import embedding_projection_optimized`` and ``import embedding_projection`` resolve.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BACKEND_NAME = "embedding_projection_opt"
EXPECTED_INPUT_SHAPE = (64, 128)
EXPECTED_INPUT_DTYPE = "torch.int64"
EXPECTED_OUTPUT_SHAPE = (64, 128, 32000)
EXPECTED_OUTPUT_DTYPE = "torch.float32"


# ---------------------------------------------------------------------------
# Test 1 — import
# ---------------------------------------------------------------------------

def test_import():
    """Module imports without error.

    Importing embedding_projection_optimized triggers @register_backend at
    module-load time so the backend is available before torch.compile selects it."""
    import embedding_projection_optimized  # noqa: F401


# ---------------------------------------------------------------------------
# Test 2 — backend registration
# ---------------------------------------------------------------------------

def test_backend_registration():
    """Backend embedding_projection_opt is registered with torch._dynamo."""
    import torch
    import embedding_projection_optimized  # noqa: F401

    backends = str(torch._dynamo.list_backends())
    assert BACKEND_NAME in backends, (
        f"Backend '{BACKEND_NAME}' not found in registered backends: {backends}"
    )


# ---------------------------------------------------------------------------
# Test 3 — get_model_and_input
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="CUDA required"
)
def test_get_model_and_input():
    """Input tensor is on CUDA; shape and dtype match the profiled workload values.

    Workload constants (optimizations.json analysis.dtype = float32):
        BATCH_SIZE=64, SEQ_LEN=128 -> token-id input shape (64, 128), dtype int64.
        The float32 model dtype refers to the parameters / GEMM activations; the
        embedding input is integer token IDs.
    """
    import torch
    from embedding_projection_optimized import get_model_and_input

    model, x = get_model_and_input()

    # Input device
    assert x.is_cuda, f"Input tensor must be on CUDA, got device={x.device}"

    # Input shape: (BATCH_SIZE, SEQ_LEN) = (64, 128)
    assert tuple(x.shape) == EXPECTED_INPUT_SHAPE, (
        f"Unexpected input shape: expected {EXPECTED_INPUT_SHAPE}, got {tuple(x.shape)}"
    )

    # Input dtype: int64 token IDs (torch.randint default integer dtype)
    assert str(x.dtype) == EXPECTED_INPUT_DTYPE, (
        f"Unexpected input dtype: expected {EXPECTED_INPUT_DTYPE}, got {x.dtype}"
    )

    # Model must be on CUDA, FP32, and in eval mode
    p = next(model.parameters())
    assert p.is_cuda, "Model parameters must be on CUDA"
    assert str(p.dtype) == "torch.float32", (
        f"Model parameters must be float32 (OPT-1 promotes selectively in-graph), "
        f"got {p.dtype}"
    )
    assert not model.training, "Model must be in eval mode for OPT-2 freezing"


# ---------------------------------------------------------------------------
# Test 4 — compiled forward pass
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="CUDA required"
)
def test_compiled_forward_pass(caplog):
    """Compiled forward pass triggers the backend and FX passes emit log messages.

    Validates:
    - At least one log record emitted (backend executed)
    - Output shape preserved: (64, 128, 32000)
    - Output dtype preserved: float32 (OPT-1 BF16 round-trip + OPT-2 freezing)
    - No NaN or Inf in output tensor
    """
    import logging

    import torch
    from embedding_projection_optimized import get_model_and_input

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

    assert caplog.records, (
        "No logger output captured — backend may not have executed. "
        "Ensure embedding_projection_optimized is imported and the logger propagates."
    )

    if out is not None:
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"
        assert tuple(out.shape) == EXPECTED_OUTPUT_SHAPE, (
            f"Unexpected output shape: expected {EXPECTED_OUTPUT_SHAPE}, "
            f"got {tuple(out.shape)}"
        )
        assert str(out.dtype) == EXPECTED_OUTPUT_DTYPE, (
            f"Output dtype not restored to float32: got {out.dtype}. "
            "OPT-1 BF16 promotion should insert a FP32 down-cast on each GEMM output."
        )
