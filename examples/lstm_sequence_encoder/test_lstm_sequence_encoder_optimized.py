"""
Validation tests for lstm_sequence_encoder_optimized.py.

Run:
    python -m pytest test_lstm_sequence_encoder_optimized.py -v
    python test_lstm_sequence_encoder_optimized.py       # standalone
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import pytest

# Ensure both the example dir and project root are on the path
_DIR = Path(__file__).parent
_ROOT = _DIR.parent.parent
sys.path.insert(0, str(_DIR))
sys.path.insert(0, str(_ROOT))


def test_import():
    """Module imports without error."""
    import lstm_sequence_encoder_optimized  # noqa: F401


def test_backend_registration():
    """'lstm_sequence_encoder_opt' backend is registered with torch._dynamo."""
    import torch
    import lstm_sequence_encoder_optimized  # noqa: F401
    backends = str(torch._dynamo.list_backends())
    assert "lstm_sequence_encoder_opt" in backends, (
        f"Backend 'lstm_sequence_encoder_opt' not found in: {backends}"
    )


def test_get_model_and_input():
    """Model and input have expected shapes and dtypes after BF16 optimization."""
    from lstm_sequence_encoder import BATCH_SIZE, SEQ_LEN, INPUT_SIZE
    from lstm_sequence_encoder_optimized import get_model_and_input

    model, x = get_model_and_input()
    assert x.device.type == "cuda", f"Input must be on CUDA, got {x.device}"
    assert x.shape == (BATCH_SIZE, SEQ_LEN, INPUT_SIZE), (
        f"Expected input shape ({BATCH_SIZE}, {SEQ_LEN}, {INPUT_SIZE}), got {x.shape}"
    )
    assert x.dtype == torch.bfloat16, (
        f"Expected BF16 input after OPT-1 dtype promotion, got {x.dtype}"
    )
    assert next(model.parameters()).dtype == torch.bfloat16, (
        f"Expected BF16 model parameters after OPT-1, got {next(model.parameters()).dtype}"
    )


def test_forward_pass():
    """Uncompiled forward pass completes without error, no NaN/Inf in output."""
    from lstm_sequence_encoder import BATCH_SIZE, NUM_CLASSES
    from lstm_sequence_encoder_optimized import get_model_and_input

    model, x = get_model_and_input()
    with torch.no_grad():
        out = model(x)

    assert out is not None, "Output is None"
    assert out.shape == (BATCH_SIZE, NUM_CLASSES), (
        f"Expected output shape ({BATCH_SIZE}, {NUM_CLASSES}), got {out.shape}"
    )
    assert not torch.isnan(out).any(), f"Output contains NaN: {out}"
    assert not torch.isinf(out).any(), f"Output contains Inf: {out}"


if __name__ == "__main__":
    tests = [
        test_import,
        test_backend_registration,
        test_get_model_and_input,
        test_forward_pass,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {t.__name__}: {exc}")
            failed += 1
    print(f"\n{passed}/{passed + failed} tests passed")
    sys.exit(0 if failed == 0 else 1)
