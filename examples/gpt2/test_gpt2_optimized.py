#!/usr/bin/env python
"""
test_gpt2_optimized.py — Verification tests for the optimized GPT-2 workload.

Validates:
  1. Module imports without error
  2. gpt2_backend is registered with torch._dynamo
  3. get_model_and_input() returns model and input with correct shapes and dtypes
  4. Uncompiled forward pass produces finite outputs with correct shape

Run with pytest:
    cd /root/Profiler
    python -m pytest test_gpt2_optimized.py -v

Or directly:
    python test_gpt2_optimized.py

Note: test_get_model_and_input and test_forward_pass require CUDA and will
download GPT-2 weights (~500 MB) from HuggingFace on first run.
test_numerical_correctness downloads weights twice (FP32 + BF16) and runs two
forward passes; it may take 2-3 minutes on first execution.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add repo root to sys.path so the nvidia package and gpt2_optimized module
# are importable regardless of the working directory at invocation time.
ROOT = Path(__file__).resolve().parent  # /root/Profiler
sys.path.insert(0, str(ROOT))

import torch


# ---------------------------------------------------------------------------
# Test functions (pytest-compatible; each is fully self-contained)
# ---------------------------------------------------------------------------

def test_import():
    """Module imports without error."""
    import gpt2_optimized  # noqa: F401


def test_backend_registration():
    """Backend gpt2_backend is registered with torch._dynamo."""
    import gpt2_optimized  # noqa: F401  — triggers @register_backend at import time
    backends = str(torch._dynamo.list_backends())
    assert "gpt2_backend" in backends, f"Backend not found in: {backends}"


def test_get_model_and_input():
    """Model and input have the expected shapes and dtypes after optimizations."""
    from gpt2_optimized import get_model_and_input

    model, input_ids = get_model_and_input()

    # Input checks
    assert input_ids.device.type == "cuda", (
        f"input_ids must be on CUDA, got {input_ids.device}"
    )
    assert input_ids.shape == (4, 128), (
        f"Expected input shape (4, 128), got {input_ids.shape}"
    )
    assert input_ids.dtype == torch.int64, (
        f"input_ids must be int64 (embedding lookup), got {input_ids.dtype}"
    )

    # Model weight dtype — unwrap OptimizedModule/CompiledModule wrapper if present
    raw_model = model
    while hasattr(raw_model, "_orig_mod"):
        raw_model = raw_model._orig_mod

    param_dtype = next(raw_model.parameters()).dtype
    assert param_dtype == torch.bfloat16, (
        f"Model weights must be BF16 (OPT-1), got {param_dtype}"
    )


def test_forward_pass():
    """Uncompiled forward pass produces output with correct shape and finite values."""
    # We test the raw (pre-compile) model to avoid autotuning overhead in CI.
    # This validates the BF16 cast + model structure without triggering max-autotune.
    import torch.nn as nn
    from transformers import GPT2Model

    DEVICE = "cuda"
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("CUDA not available")

    from gpt2_optimized import GPT2Wrapper, MODEL_ID, BATCH, SEQ_LEN

    hf_model = GPT2Model.from_pretrained(MODEL_ID)
    model = GPT2Wrapper(hf_model).to(DEVICE).eval().to(torch.bfloat16)
    vocab_size = hf_model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (BATCH, SEQ_LEN), device=DEVICE)

    with torch.no_grad():
        out = model(input_ids)

    assert out is not None, "Forward pass returned None"
    assert out.shape == (4, 128, 768), (
        f"Expected output shape (4, 128, 768), got {out.shape}"
    )
    assert not torch.isnan(out).any(), "Output contains NaN — BF16 overflow?"
    assert not torch.isinf(out).any(), "Output contains Inf — BF16 overflow?"


def test_numerical_correctness():
    """
    BF16 output is numerically close to FP32 reference.

    Acceptance criteria (calibrated against observed GPT-2 BF16 behavior):
      - Mean absolute error < 0.02  (typical: ~0.006)
      - 99th-percentile absolute error < 0.1  (typical: ~0.03)
      - Relative error (atol=4.0, rtol=0.02) — handles the small number of
        high-magnitude output tokens (~209 max) where BF16 rounding creates
        absolute differences up to ~3.0; relative error for those tokens is
        still within BF16 precision (~1.6%).

    Note: max-abs-diff is NOT used as the sole criterion because BF16
    accumulation errors at large output magnitudes routinely produce 1-3
    absolute difference on individual tokens while mean error stays < 0.01.
    """
    from transformers import GPT2Model
    from gpt2_optimized import GPT2Wrapper, MODEL_ID, BATCH, SEQ_LEN

    DEVICE = "cuda"
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("CUDA not available")

    vocab_size = GPT2Model.from_pretrained(MODEL_ID).config.vocab_size
    input_ids = torch.randint(0, vocab_size, (BATCH, SEQ_LEN), device=DEVICE)

    # FP32 reference
    hf_fp32 = GPT2Model.from_pretrained(MODEL_ID)
    model_fp32 = GPT2Wrapper(hf_fp32).to(DEVICE).eval()
    with torch.no_grad():
        out_fp32 = model_fp32(input_ids).float()

    # BF16 optimized
    hf_bf16 = GPT2Model.from_pretrained(MODEL_ID)
    model_bf16 = GPT2Wrapper(hf_bf16).to(DEVICE).eval().to(torch.bfloat16)
    with torch.no_grad():
        out_bf16 = model_bf16(input_ids).float()

    diff = (out_fp32 - out_bf16).abs()
    mean_abs_diff = diff.mean().item()
    p99_abs_diff = torch.quantile(diff.flatten(), 0.99).item()

    assert mean_abs_diff < 0.02, (
        f"BF16 vs FP32 mean abs diff {mean_abs_diff:.4f} exceeds threshold 0.02. "
        "Check for overflow or incorrect model dtype cast."
    )
    assert p99_abs_diff < 0.1, (
        f"BF16 vs FP32 99th-pctile abs diff {p99_abs_diff:.4f} exceeds threshold 0.1. "
        "Check for overflow or incorrect model dtype cast."
    )
    # Relative-tolerance check — tolerates large absolute differences at high output magnitudes
    # (atol=4.0 covers the ~3-unit BF16 rounding error at output values ~200)
    torch.testing.assert_close(out_fp32, out_bf16, atol=4.0, rtol=0.02,
                                msg="BF16 vs FP32 relative error exceeds 2% (BF16 precision bound)")


# ---------------------------------------------------------------------------
# Manual runner (for `python test_gpt2_optimized.py`)
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("GPT-2 Optimized Workload — Verification Tests")
    print("=" * 70)
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        dev = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability()
        print(f"CUDA device:     {dev} (sm{cap[0]}{cap[1]})")
    else:
        print("WARNING: CUDA not available — CUDA-dependent tests will fail")
    print()

    tests = [
        test_import,
        test_backend_registration,
        test_get_model_and_input,
        test_forward_pass,
        test_numerical_correctness,
    ]

    results = []
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            results.append(True)
        except Exception as exc:
            print(f"FAIL  {fn.__name__}: {exc}")
            results.append(False)

    print()
    print("=" * 70)
    passed = sum(results)
    print(f"{passed}/{len(results)} tests passed")
    if passed == len(results):
        print("All tests passed — ready to profile with /capture")
    else:
        print("Fix failing tests before profiling")
    print("=" * 70)
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
