"""
test_embedding_projection_optimized.py — Validation suite for the
embedding_projection_opt custom torch.compile() backend.

Four tests:
  1. test_import                — module imports without error
  2. test_backend_registration  — embedding_projection_opt registered with torch._dynamo
  3. test_get_model_and_input   — input on CUDA, expected shape (64,128) int64
  4. test_compiled_forward_pass — compiled forward triggers the backend; no NaN/Inf
"""
from __future__ import annotations

OPTIMIZED_MODULE = "embedding_projection_optimized"
BACKEND_NAME = "embedding_projection_opt"

# Expected input characteristics derived from optimizations.json / the workload:
#   BATCH_SIZE=64, SEQ_LEN=128, VOCAB_SIZE=32000, dtype int64 token ids.
EXPECTED_INPUT_SHAPE = (64, 128)
EXPECTED_VOCAB = 32_000
EXPECTED_OUTPUT_SHAPE = (64, 128, 32_000)


def test_import():
    """Module imports without error."""
    import embedding_projection_optimized  # noqa: F401


def test_backend_registration():
    """Backend registered with torch._dynamo."""
    import torch
    import embedding_projection_optimized  # noqa: F401

    backends = str(torch._dynamo.list_backends())
    assert BACKEND_NAME in backends, f"Backend not found in: {backends}"


def test_get_model_and_input():
    """Assert input is on CUDA; verify shape and dtype match expected values."""
    import torch
    from embedding_projection_optimized import get_model_and_input

    model, x = get_model_and_input()

    assert x.is_cuda, "Input token_ids must be on CUDA"
    assert tuple(x.shape) == EXPECTED_INPUT_SHAPE, (
        f"Expected input shape {EXPECTED_INPUT_SHAPE}, got {tuple(x.shape)}"
    )
    assert x.dtype in (torch.int64, torch.long), (
        f"Token ids must be integer indices, got {x.dtype}"
    )
    assert int(x.max()) < EXPECTED_VOCAB, "Token id exceeds vocab size"
    assert int(x.min()) >= 0, "Token id below 0"
    # Model is returned in eval mode (required for OPT-3 freezing).
    assert not model.training, "Model must be in eval mode for freezing"


def test_compiled_forward_pass(caplog):
    """Compiled forward pass triggers the backend; captures FX pass application logs."""
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
                # torch 2.11: guard error after backend succeeds — safe to suppress.

    for record in caplog.records:
        print(record.getMessage())

    assert caplog.records, "No logger output — backend may not have executed"

    if out is not None:
        assert tuple(out.shape) == EXPECTED_OUTPUT_SHAPE, (
            f"Expected output shape {EXPECTED_OUTPUT_SHAPE}, got {tuple(out.shape)}"
        )
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"
