"""
test_conv_block_optimized.py — Validation tests for ConvBlock optimized workload.

Validates:
  1. Module imports without error
  2. conv_block_opt backend is registered with torch._dynamo
  3. get_model_and_input() returns correct shapes, dtypes, and device
  4. Uncompiled forward pass completes without NaN or Inf

Run with:
    cd /home/ubuntu/Profiler/examples/conv_block
    python -m pytest test_conv_block_optimized.py -v
    # or directly:
    python test_conv_block_optimized.py
"""
from __future__ import annotations

import logging
import sys
import os

import torch

# Suppress INFO-level pass logging during tests for cleaner output
logging.basicConfig(level=logging.WARNING)

# Ensure the examples/conv_block directory is on the path so that
# both conv_block and conv_block_optimized are importable by name.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


# ── helpers ───────────────────────────────────────────────────────────────────

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


# ── tests ─────────────────────────────────────────────────────────────────────

def test_import():
    """Module imports without error and exposes the required interface."""
    import conv_block_optimized as mod  # noqa: F401

    assert hasattr(mod, "get_model_and_input"), (
        "conv_block_optimized missing get_model_and_input"
    )
    assert hasattr(mod, "conv_block_opt"), (
        "conv_block_optimized missing conv_block_opt backend function"
    )
    assert hasattr(mod, "pass_fold_bn_into_conv"), (
        "conv_block_optimized missing pass_fold_bn_into_conv"
    )
    print("  PASS test_import")


def test_backend_registration():
    """conv_block_opt is registered with torch._dynamo after module import."""
    # Import the module to trigger @register_backend decoration
    import conv_block_optimized  # noqa: F401

    backends = str(torch._dynamo.list_backends())
    assert "conv_block_opt" in backends, (
        f"Backend 'conv_block_opt' not found in registered backends: {backends}"
    )
    print(f"  PASS test_backend_registration  (conv_block_opt in backends)")


@_skip_if_no_cuda
def test_get_model_and_input():
    """get_model_and_input() returns model and tensor with expected shapes and dtypes."""
    from conv_block_optimized import get_model_and_input

    # These match the baseline constants imported by conv_block_optimized
    EXPECTED_BATCH   = 16
    EXPECTED_CHANNELS = 3
    EXPECTED_HEIGHT  = 64
    EXPECTED_WIDTH   = 64

    model, x = get_model_and_input()

    # Device checks
    assert x.device.type == "cuda", (
        f"Input must be on CUDA, got device: {x.device}"
    )
    assert next(model.parameters()).device.type == "cuda", (
        "Model parameters must be on CUDA"
    )

    # Shape check — must match baseline BATCH_SIZE × IN_CHANNELS × HEIGHT × WIDTH
    assert x.shape == (EXPECTED_BATCH, EXPECTED_CHANNELS, EXPECTED_HEIGHT, EXPECTED_WIDTH), (
        f"Unexpected input shape: {x.shape}, "
        f"expected ({EXPECTED_BATCH}, {EXPECTED_CHANNELS}, {EXPECTED_HEIGHT}, {EXPECTED_WIDTH})"
    )

    # Dtype check — OPT-3 promotes model and input to BF16
    assert x.dtype == torch.bfloat16, (
        f"Expected BF16 input after OPT-3, got {x.dtype}"
    )
    model_dtype = next(model.parameters()).dtype
    assert model_dtype == torch.bfloat16, (
        f"Expected BF16 model parameters after OPT-3, got {model_dtype}"
    )

    # Eval mode
    assert not model.training, "Model must be in eval() mode"

    # channels_last check — OPT-1 converts 4-D conv weights to NHWC
    first_param = next(model.parameters())
    if first_param.dim() == 4:
        assert first_param.is_contiguous(memory_format=torch.channels_last), (
            "Model conv weights should be channels_last after OPT-1"
        )

    print(
        f"  PASS test_get_model_and_input  "
        f"(x: {x.shape} {x.dtype} {'channels_last' if x.is_contiguous(memory_format=torch.channels_last) else 'contiguous'}, "
        f"model dtype: {model_dtype})"
    )


@_skip_if_no_cuda
def test_forward_pass():
    """Uncompiled forward pass completes without error and produces no NaN or Inf."""
    from conv_block_optimized import get_model_and_input

    EXPECTED_BATCH  = 16
    EXPECTED_CLASSES = 10

    model, x = get_model_and_input()

    with torch.no_grad():
        out = model(x)

    assert out is not None, "Forward pass returned None"
    assert out.shape == (EXPECTED_BATCH, EXPECTED_CLASSES), (
        f"Unexpected output shape: {out.shape}, "
        f"expected ({EXPECTED_BATCH}, {EXPECTED_CLASSES})"
    )
    assert not torch.isnan(out).any(), (
        "Output contains NaN values — BN fold or BF16 cast may have broken numerics"
    )
    assert not torch.isinf(out).any(), (
        "Output contains Inf values — BN fold or BF16 cast may have caused overflow"
    )

    print(
        f"  PASS test_forward_pass  "
        f"(output: {out.shape} {out.dtype}, "
        f"max abs: {out.abs().max().item():.4f})"
    )


# ── runner ────────────────────────────────────────────────────────────────────

TESTS = [
    test_import,
    test_backend_registration,
    test_get_model_and_input,
    test_forward_pass,
]

if __name__ == "__main__":
    passed = failed = 0
    for test_fn in TESTS:
        try:
            print(f"Running {test_fn.__name__} ...")
            test_fn()
            passed += 1
        except AssertionError as exc:
            print(f"  FAIL {test_fn.__name__}: {exc}")
            failed += 1
        except Exception as exc:
            print(f"  ERROR {test_fn.__name__}: {type(exc).__name__}: {exc}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
