"""
test_depthwise_separable_conv_optimized.py — Validation suite for the A100-optimized backend.

Validates (per /validate spec):
  1. test_import              — module loads without error
  2. test_backend_registration — 'depthwise_sep_conv_opt' registered in Dynamo
  3. test_get_model_and_input  — shapes/dtypes correct; BN folded; channels_last; BF16
  4. test_forward_pass         — uncompiled forward, correct output shape, no NaN/Inf

Run:
    pytest test_depthwise_separable_conv_optimized.py -v
"""
from __future__ import annotations

import logging
import torch
import torch.fx as fx


def test_import():
    """Module loads and exposes required symbols."""
    import depthwise_separable_conv_optimized as opt  # noqa: F401
    assert hasattr(opt, "get_model_and_input"), "Missing get_model_and_input"
    assert hasattr(opt, "depthwise_sep_conv_opt"), "Missing depthwise_sep_conv_opt backend"
    assert hasattr(opt, "pass_fold_bn"), "Missing pass_fold_bn"


def test_backend_registration():
    """'depthwise_sep_conv_opt' is registered in torch._dynamo."""
    import depthwise_separable_conv_optimized  # noqa: F401 — triggers @register_backend
    import torch._dynamo as dynamo
    backends = dynamo.list_backends()
    assert "depthwise_sep_conv_opt" in backends, (
        f"'depthwise_sep_conv_opt' not found in backends: {list(backends)}"
    )


def test_get_model_and_input():
    """Shapes, dtypes, layout, and BN fold are all correct."""
    import torch.nn as nn
    from depthwise_separable_conv import DepthwiseSepConv, BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH
    from depthwise_separable_conv_optimized import get_model_and_input

    model, x = get_model_and_input()

    # Input
    assert x.shape == (BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH), f"shape: {x.shape}"
    assert x.dtype == torch.bfloat16, f"dtype: {x.dtype}"
    assert x.is_cuda

    # Model dtype and layout
    p = next(model.parameters())
    assert p.dtype == torch.bfloat16, f"model dtype: {p.dtype}"
    assert p.is_contiguous(memory_format=torch.channels_last), "model not channels_last"

    # BN fold: all BN layers must be Identity
    assert isinstance(model, DepthwiseSepConv)
    for block in [model.block1, model.block2, model.block3]:
        assert isinstance(block.bn_dw, nn.Identity), f"bn_dw not folded: {type(block.bn_dw)}"
        assert isinstance(block.bn_pw, nn.Identity), f"bn_pw not folded: {type(block.bn_pw)}"


def test_forward_pass():
    """Uncompiled BF16 forward produces correct shape with no NaN/Inf."""
    from depthwise_separable_conv import BATCH_SIZE, HEIGHT, WIDTH
    from depthwise_separable_conv_optimized import get_model_and_input

    model, x = get_model_and_input()
    with torch.no_grad():
        out = model(x)

    assert out.shape == (BATCH_SIZE, 256, HEIGHT, WIDTH), f"output shape: {out.shape}"
    assert out.dtype == torch.bfloat16, f"output dtype: {out.dtype}"
    assert not torch.isnan(out).any(), "NaN in output"
    assert not torch.isinf(out).any(), "Inf in output"


def test_pass_fold_bn_no_crash():
    """pass_fold_bn runs without exception on a trivial GraphModule."""
    from depthwise_separable_conv_optimized import pass_fold_bn

    class ToyMod(torch.nn.Module):
        def forward(self, x):
            return x + 1.0

    gm = fx.symbolic_trace(ToyMod())
    logging.disable(logging.CRITICAL)
    try:
        gm = pass_fold_bn(gm)
    finally:
        logging.disable(logging.NOTSET)
    gm.graph.lint()