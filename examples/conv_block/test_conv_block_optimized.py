"""
test_conv_block_optimized.py — Validation suite for conv_block_optimized.py.

Runs 4 checks:
  1. Import: module loads without error
  2. Backend registration: 'conv_block_opt' appears in dynamo backend list
  3. get_model_and_input: correct shapes, dtype=bfloat16, channels_last layout
  4. Forward pass: uncompiled forward produces finite outputs with correct shape
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


def test_forward_pass():
    from conv_block_optimized import get_model_and_input
    model, x = get_model_and_input()

    with torch.no_grad():
        out = model(x)

    # shape: (BATCH_SIZE=16, NUM_CLASSES=10)
    assert out.shape == (16, 10), f"Unexpected output shape: {out.shape}"
    assert not torch.isnan(out).any(), "NaN in output"
    assert not torch.isinf(out).any(), "Inf in output"
