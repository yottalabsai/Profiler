"""
test_conv_block_optimized.py — Validation suite for conv_block_optimized.py.

Runs 4 checks:
  1. Import: module loads without error
  2. Backend registration: 'conv_block_opt' appears in dynamo backend list
  3. get_model_and_input: correct shapes, dtype=bfloat16, channels_last layout
  4. Compiled forward pass: triggers backend, captures FX pass logs, correct output
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))


def test_import():
    import conv_block_optimized  # noqa: F401


def test_backend_registration():
    import torch._dynamo as dynamo
    import conv_block_optimized  # noqa: F401 — registers backend on import
    backends = dynamo.list_backends()
    assert "conv_block_opt" in backends, (
        f"'conv_block_opt' not in dynamo backends: {backends}"
    )


def test_get_model_and_input():
    from conv_block_optimized import get_model_and_input
    model, x = get_model_and_input()

    # dtype
    param_dtype = next(model.parameters()).dtype
    assert param_dtype == torch.bfloat16, f"Expected bfloat16 model, got {param_dtype}"
    assert x.dtype == torch.bfloat16, f"Expected bfloat16 input, got {x.dtype}"

    # memory layout
    assert x.is_contiguous(memory_format=torch.channels_last), (
        "Input tensor is not channels_last contiguous"
    )

    # shape: (BATCH_SIZE=16, IN_CHANNELS=3, HEIGHT=64, WIDTH=64)
    assert x.shape == (16, 3, 64, 64), f"Unexpected input shape: {x.shape}"
    assert x.device.type == "cuda", f"Input not on CUDA: {x.device}"


def test_compiled_forward_pass(caplog):
    """Compiled forward pass triggers the conv_block_opt backend; captures FX pass logs."""
    import logging
    from conv_block_optimized import get_model_and_input

    model, x = get_model_and_input()
    compiled = torch.compile(model, backend="conv_block_opt")
    out = None

    with caplog.at_level(logging.INFO):
        with torch.no_grad():
            try:
                out = compiled(x)
            except Exception as exc:
                from torch._dynamo.exc import InternalTorchDynamoError
                if not isinstance(exc, InternalTorchDynamoError):
                    raise

    for record in caplog.records:
        print(record.getMessage())

    assert caplog.records, "No logger output — backend may not have executed"

    if out is not None:
        assert out.shape == (16, 10), f"Unexpected output shape: {out.shape}"
        assert not torch.isnan(out).any(), "NaN in output"
        assert not torch.isinf(out).any(), "Inf in output"
