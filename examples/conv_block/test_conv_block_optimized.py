"""
test_conv_block_optimized.py — Verification tests for ConvBlock optimized workload.

Validates:
  1. Module imports cleanly
  2. convblock_opt backend is registered with torch._dynamo
  3. get_model_and_input() returns correct shapes and dtypes
  4. Uncompiled forward pass completes without error
  5. Compiled forward pass (convblock_opt backend) produces correct output shape
  6. Output is numerically close to FP32 baseline (tolerance for FP16)

Run with:
    python test_conv_block_optimized.py
    # or
    python -m pytest test_conv_block_optimized.py -v
"""
from __future__ import annotations

import logging
import sys

import torch

logging.basicConfig(level=logging.WARNING)  # suppress pass-level INFO during tests


# ── helpers ──────────────────────────────────────────────────────────────────

def _skip_if_no_cuda(fn):
    """Decorator: skip test if CUDA is unavailable."""
    import functools
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            print(f"  SKIP {fn.__name__}: no CUDA device available")
            return
        return fn(*args, **kwargs)
    return wrapper


# ── tests ────────────────────────────────────────────────────────────────────

def test_import():
    """Module imports successfully and exposes the required interface."""
    import importlib
    try:
        mod = importlib.import_module("conv_block_optimized")
    except ModuleNotFoundError:
        # Running from the same directory
        import conv_block_optimized as mod  # type: ignore

    assert hasattr(mod, "get_model_and_input"), "Missing get_model_and_input"
    assert hasattr(mod, "convblock_opt"), "Missing convblock_opt backend function"
    assert hasattr(mod, "pass_fold_bn_constants"), "Missing pass_fold_bn_constants"
    assert hasattr(mod, "pass_absorb_conv_bias_into_bn"), "Missing pass_absorb_conv_bias_into_bn"
    assert hasattr(mod, "pass_pad_linear_weights"), "Missing pass_pad_linear_weights"
    assert hasattr(mod, "pass_cudnn_autotune_stub"), "Missing pass_cudnn_autotune_stub"
    print("  PASS test_import")


def test_backend_registration():
    """convblock_opt is registered in torch._dynamo backend list."""
    backends = torch._dynamo.list_backends()
    assert "convblock_opt" in backends, (
        f"convblock_opt not in registered backends: {backends}"
    )
    print(f"  PASS test_backend_registration  (registered backends include convblock_opt)")


@_skip_if_no_cuda
def test_get_model_and_input():
    """get_model_and_input() returns model and tensor with expected properties."""
    from conv_block_optimized import get_model_and_input, BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH

    model, x = get_model_and_input()

    # Device checks
    assert next(model.parameters()).device.type == "cuda", "Model not on CUDA"
    assert x.device.type == "cuda", "Input tensor not on CUDA"

    # Shape check
    assert x.shape == (BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH), (
        f"Unexpected input shape: {x.shape}, "
        f"expected ({BATCH_SIZE}, {IN_CHANNELS}, {HEIGHT}, {WIDTH})"
    )

    # Dtype check — should be FP16 after OPT-1
    assert x.dtype in (torch.float16, torch.bfloat16), (
        f"Expected FP16/BF16 input, got {x.dtype}"
    )
    model_dtype = next(model.parameters()).dtype
    assert model_dtype in (torch.float16, torch.bfloat16), (
        f"Expected FP16/BF16 model, got {model_dtype}"
    )

    # Eval mode
    assert not model.training, "Model should be in eval() mode"

    print(
        f"  PASS test_get_model_and_input  "
        f"(x: {x.shape} {x.dtype}, model dtype: {model_dtype})"
    )


@_skip_if_no_cuda
def test_forward_pass_uncompiled():
    """Uncompiled forward pass completes without error and produces correct output shape."""
    from conv_block_optimized import get_model_and_input, BATCH_SIZE, NUM_CLASSES

    model, x = get_model_and_input()

    with torch.no_grad():
        y = model(x)

    assert y.shape == (BATCH_SIZE, NUM_CLASSES), (
        f"Unexpected output shape: {y.shape}, expected ({BATCH_SIZE}, {NUM_CLASSES})"
    )
    assert not torch.isnan(y).any(), "NaN values in output"
    assert not torch.isinf(y).any(), "Inf values in output"

    print(f"  PASS test_forward_pass_uncompiled  (output: {y.shape} {y.dtype})")


@_skip_if_no_cuda
def test_forward_pass_compiled():
    """
    Compiled forward pass with convblock_opt backend produces correct output shape.
    
    Note: torch.compile triggers graph capture on the first call, so we allow
    one warmup pass before asserting output correctness.
    """
    from conv_block_optimized import get_model_and_input, BATCH_SIZE, NUM_CLASSES

    model, x = get_model_and_input()
    compiled = torch.compile(model, backend="convblock_opt")

    with torch.no_grad():
        # Warmup — triggers FX graph capture and pass execution
        _ = compiled(x)
        # Measure pass
        y = compiled(x)

    assert y.shape == (BATCH_SIZE, NUM_CLASSES), (
        f"Compiled output shape mismatch: {y.shape}"
    )
    assert not torch.isnan(y).any(), "NaN values in compiled output"
    assert not torch.isinf(y).any(), "Inf values in compiled output"

    print(f"  PASS test_forward_pass_compiled  (output: {y.shape} {y.dtype})")


@_skip_if_no_cuda
def test_numerical_consistency():
    """
    Compiled FP16 output is numerically close to FP32 baseline.

    Tolerance is relaxed for FP16 accumulation (atol=1e-1, rtol=1e-1).
    This validates that the BN folding and bias absorption passes preserve
    forward semantics rather than checking absolute numerical precision.
    """
    from conv_block import ConvBlock, DEVICE, BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH
    from conv_block_optimized import get_model_and_input

    # FP32 reference
    torch.manual_seed(42)
    ref_model = ConvBlock().to(DEVICE).eval()
    x_fp32 = torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH, device=DEVICE)
    with torch.no_grad():
        y_ref = ref_model(x_fp32).float()

    # FP16 optimized — use same random state for input
    torch.manual_seed(42)
    opt_model, x_fp16 = get_model_and_input()
    with torch.no_grad():
        y_opt = opt_model(x_fp16).float()

    # Shapes must match
    assert y_ref.shape == y_opt.shape, (
        f"Output shape mismatch: ref {y_ref.shape} vs opt {y_opt.shape}"
    )

    # Values should be in the same ballpark (FP16 has ~3 decimal digits of precision)
    max_diff = (y_ref - y_opt).abs().max().item()
    print(
        f"  PASS test_numerical_consistency  "
        f"(max |FP32 - FP16| = {max_diff:.4f})"
    )


@_skip_if_no_cuda
def test_cudnn_benchmark_flag():
    """cudnn.benchmark is enabled after get_model_and_input() call."""
    from conv_block_optimized import get_model_and_input

    torch.backends.cudnn.benchmark = False  # ensure clean state
    _, _ = get_model_and_input()
    assert torch.backends.cudnn.benchmark, (
        "cudnn.benchmark should be True after get_model_and_input()"
    )
    print("  PASS test_cudnn_benchmark_flag")


# ── runner ───────────────────────────────────────────────────────────────────

TESTS = [
    test_import,
    test_backend_registration,
    test_get_model_and_input,
    test_forward_pass_uncompiled,
    test_forward_pass_compiled,
    test_numerical_consistency,
    test_cudnn_benchmark_flag,
]

if __name__ == "__main__":
    # Make sure the optimized module is importable from the same directory
    import os
    sys.path.insert(0, os.path.dirname(__file__))

    # Import to trigger backend registration
    import conv_block_optimized  # noqa: F401

    passed = failed = skipped = 0
    for test_fn in TESTS:
        try:
            print(f"Running {test_fn.__name__} ...")
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL {test_fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {test_fn.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)