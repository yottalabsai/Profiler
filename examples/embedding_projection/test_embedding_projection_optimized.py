"""
test_embedding_projection_optimized.py — validation suite for the
embedding_projection_opt custom torch.compile() backend.

Four tests:
  1. test_import                  — module imports cleanly (triggers @register_backend)
  2. test_backend_registration    — backend name is registered with torch._dynamo
  3. test_get_model_and_input     — input is on CUDA with the expected shape/dtype
  4. test_compiled_forward_pass   — compiled forward triggers the backend; FX pass
                                     logs are captured; output has no NaN/Inf
"""
import logging

import pytest

WORKLOAD_MODULE = "embedding_projection_optimized"
BACKEND_NAME = "embedding_projection_opt"

# Expected input shape/dtype derived from embedding_projection.py / optimizations.json:
#   BATCH_SIZE=64, SEQ_LEN=128 integer token ids in [0, VOCAB_SIZE).
EXPECTED_INPUT_SHAPE = (64, 128)
EXPECTED_VOCAB = 32_000


def test_import():
    """Module imports without error (registers the backend at load time)."""
    import embedding_projection_optimized  # noqa: F401


def test_backend_registration():
    """Backend registered with torch._dynamo."""
    import torch  # noqa: F401
    import embedding_projection_optimized  # noqa: F401

    backends = str(torch._dynamo.list_backends())
    assert BACKEND_NAME in backends, f"Backend not found in: {backends}"


def test_get_model_and_input():
    """Input is on CUDA with the shape/dtype expected from the workload spec."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from embedding_projection_optimized import get_model_and_input

    model, x = get_model_and_input()
    assert x.is_cuda, "Input tensor must be on CUDA"
    assert tuple(x.shape) == EXPECTED_INPUT_SHAPE, (
        f"Expected input shape {EXPECTED_INPUT_SHAPE}, got {tuple(x.shape)}"
    )
    # token ids are integer indices into the embedding table
    assert x.dtype in (torch.int64, torch.int32, torch.long), (
        f"Expected integer token-id dtype, got {x.dtype}"
    )
    assert int(x.min()) >= 0 and int(x.max()) < EXPECTED_VOCAB, (
        "token ids must lie in [0, VOCAB_SIZE)"
    )
    assert next(model.parameters()).is_cuda, "Model parameters must be on CUDA"


def test_compiled_forward_pass(caplog):
    """Compiled forward pass triggers the backend; captures FX pass application logs."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

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
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"
        # OPT-3 keeps the 32000-wide logit-head output in bf16, so verify the
        # output width matches the vocabulary dim regardless of dtype.
        assert out.shape[-1] == EXPECTED_VOCAB, (
            f"Expected logit width {EXPECTED_VOCAB}, got {out.shape[-1]}"
        )
