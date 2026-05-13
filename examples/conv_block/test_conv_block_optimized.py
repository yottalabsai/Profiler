"""
test_conv_block_optimized.py — Verification tests for conv_block_optimized.py.

Tests:
  1. Module imports without error.
  2. Backend ``conv_block_opt`` is registered with torch._dynamo.
  3. get_model_and_input() returns tensors with expected shapes and dtypes.
  4. Uncompiled forward pass completes without NaN/Inf.
  5. Output shape matches the baseline ConvBlock.
  6. BN fold removes all BatchNorm2d modules from the model.
  7. OPT-1/OPT-2: output is numerically close to baseline (BF16 atol=1e-1).

Run:
    PYTHONPATH=/root/Profiler pytest examples/conv_block/test_conv_block_optimized.py -v
"""
from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# Test 1: import
# ---------------------------------------------------------------------------

def test_import():
    """Module imports without raising any error."""
    import examples.conv_block.conv_block_optimized  # noqa: F401


# ---------------------------------------------------------------------------
# Test 2: backend registration
# ---------------------------------------------------------------------------

def test_backend_registration():
    """Backend 'conv_block_opt' is registered with torch._dynamo."""
    import torch
    import examples.conv_block.conv_block_optimized  # noqa: F401 — triggers @register_backend

    backends = torch._dynamo.list_backends()
    assert "conv_block_opt" in backends, (
        f"Backend 'conv_block_opt' not found in registered backends: {backends}"
    )


# ---------------------------------------------------------------------------
# Test 3: get_model_and_input shapes and dtypes
# ---------------------------------------------------------------------------

def test_get_model_and_input():
    """Model and input have expected shapes, dtypes, and memory formats."""
    from examples.conv_block.conv_block_optimized import get_model_and_input
    from examples.conv_block.conv_block import BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH

    model, x = get_model_and_input()

    # Device
    assert x.device.type == "cuda", f"Input must be on CUDA, got {x.device}"

    # Shape: [B, C, H, W] for a 4D conv input
    assert x.shape == torch.Size([BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH]), (
        f"Expected input shape {[BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH]}, got {x.shape}"
    )

    # OPT-1: channels_last
    assert x.is_contiguous(memory_format=torch.channels_last), (
        "Input tensor must be in channels_last memory format after OPT-1"
    )
    param0 = next(model.parameters())
    assert param0.is_contiguous(memory_format=torch.channels_last), (
        "Model parameters must be in channels_last memory format after OPT-1"
    )

    # OPT-2: bfloat16
    assert x.dtype == torch.bfloat16, (
        f"Input dtype must be bfloat16 after OPT-2, got {x.dtype}"
    )
    assert param0.dtype == torch.bfloat16, (
        f"Model parameter dtype must be bfloat16 after OPT-2, got {param0.dtype}"
    )


# ---------------------------------------------------------------------------
# Test 4: uncompiled forward pass (correctness guard)
# ---------------------------------------------------------------------------

def test_forward_pass():
    """Uncompiled forward pass completes without error, NaN, or Inf."""
    from examples.conv_block.conv_block_optimized import get_model_and_input
    from examples.conv_block.conv_block import NUM_CLASSES, BATCH_SIZE

    model, x = get_model_and_input()
    model.eval()

    with torch.no_grad():
        out = model(x)

    assert out is not None, "Forward pass returned None"
    assert out.shape == torch.Size([BATCH_SIZE, NUM_CLASSES]), (
        f"Expected output shape [{BATCH_SIZE}, {NUM_CLASSES}], got {out.shape}"
    )
    assert not torch.isnan(out).any(), "Output contains NaN values"
    assert not torch.isinf(out).any(), "Output contains Inf values"


# ---------------------------------------------------------------------------
# Test 5: output shape matches baseline
# ---------------------------------------------------------------------------

def test_output_shape_matches_baseline():
    """Optimized model output shape matches the baseline ConvBlock."""
    from examples.conv_block.conv_block import get_model_and_input as baseline_gmi
    from examples.conv_block.conv_block_optimized import get_model_and_input as opt_gmi
    from examples.conv_block.conv_block import NUM_CLASSES, BATCH_SIZE

    base_model, base_x = baseline_gmi()
    opt_model,  opt_x  = opt_gmi()

    base_model.eval()
    opt_model.eval()

    with torch.no_grad():
        base_out = base_model(base_x)
        opt_out  = opt_model(opt_x)

    assert base_out.shape == opt_out.shape, (
        f"Shape mismatch: baseline {base_out.shape} vs. optimized {opt_out.shape}"
    )


# ---------------------------------------------------------------------------
# Test 6: BN fold removes BatchNorm2d from model
# ---------------------------------------------------------------------------

def test_bn_fold_removes_batchnorm():
    """After get_model_and_input(), the model must contain no BatchNorm2d modules."""
    from examples.conv_block.conv_block_optimized import get_model_and_input
    import torch.nn as nn

    model, _ = get_model_and_input()

    bn_modules = [
        name for name, mod in model.named_modules()
        if isinstance(mod, nn.BatchNorm2d)
    ]
    assert len(bn_modules) == 0, (
        f"BN fold (OPT-3) failed: BatchNorm2d modules still present: {bn_modules}"
    )


# ---------------------------------------------------------------------------
# Test 7: numerical closeness with baseline (BF16 atol=1e-1)
# ---------------------------------------------------------------------------

def test_numerical_closeness_with_baseline():
    """
    Optimized output is numerically close to baseline output.

    The same model weights and input are used for both runs: a deepcopy of the
    baseline model has the optimizations applied, and the baseline input is cast
    to BF16/channels_last.  BF16 introduces ~1e-2 relative error per op; 1e-1
    absolute tolerance is generous enough to be robust across A100 variants.
    """
    import copy
    from examples.conv_block.conv_block import get_model_and_input as baseline_gmi
    from examples.conv_block.conv_block_optimized import fold_all_bn

    base_model, base_x = baseline_gmi()
    base_model.eval()

    # Build optimized model from the same weights via deepcopy
    opt_model = copy.deepcopy(base_model)
    opt_model.eval()
    fold_all_bn(opt_model)
    opt_model = opt_model.to(memory_format=torch.channels_last).to(torch.bfloat16)

    # Cast the same input to match
    opt_x = base_x.to(memory_format=torch.channels_last).to(torch.bfloat16)

    with torch.no_grad():
        base_out = base_model(base_x).float()
        opt_out  = opt_model(opt_x).float()

    assert base_out.shape == opt_out.shape, (
        f"Shape mismatch before allclose: {base_out.shape} vs. {opt_out.shape}"
    )

    close = torch.allclose(base_out, opt_out, atol=1e-1, rtol=1e-2)
    if not close:
        max_diff = (base_out - opt_out).abs().max().item()
        mean_diff = (base_out - opt_out).abs().mean().item()
        pytest.fail(
            f"Numerical mismatch between baseline and optimized model: "
            f"max_diff={max_diff:.4f}, mean_diff={mean_diff:.6f} (atol=1e-1)"
        )
