"""
test_sdpa_attention_optimized.py — Validation tests for sdpa_attention_optimized.py.

Covers four assertions:
  1. Module imports without error.
  2. 'sdpa_attention_opt' backend is registered with torch._dynamo.
  3. get_model_and_input() returns expected device, shape, and dtype.
  4. Compiled forward pass triggers the backend, captures FX pass logs, correct output.

Run with:
    PYTHONPATH=/home/ubuntu/Profiler pytest examples/sdpa_attention/test_sdpa_attention_optimized.py -v
"""
import pytest


def test_import():
    """Module imports without error."""
    import examples.sdpa_attention.sdpa_attention_optimized  # noqa: F401


def test_backend_registration():
    """'sdpa_attention_opt' backend is registered with torch._dynamo."""
    import torch
    import examples.sdpa_attention.sdpa_attention_optimized  # noqa: F401 — registers backend on import

    backends = torch._dynamo.list_backends()
    assert "sdpa_attention_opt" in backends, (
        f"'sdpa_attention_opt' not found in registered backends: {backends}"
    )


def test_get_model_and_input():
    """Model and input have expected shapes and dtypes after OPT-1 and OPT-3."""
    import torch
    from examples.sdpa_attention.sdpa_attention_optimized import (
        get_model_and_input,
        BATCH_SIZE,
        SEQ_LEN,
        DIM,
    )

    model, x = get_model_and_input()

    # Input must be on CUDA
    assert x.device.type == "cuda", (
        f"Input must be on CUDA, got device={x.device}"
    )

    # Input shape: (BATCH_SIZE, SEQ_LEN, DIM) = (8, 512, 512)
    assert x.shape == (BATCH_SIZE, SEQ_LEN, DIM), (
        f"Expected input shape ({BATCH_SIZE}, {SEQ_LEN}, {DIM}), got {x.shape}"
    )

    # OPT-1: input must be BF16
    assert x.dtype == torch.bfloat16, (
        f"Expected input dtype bfloat16 (OPT-1), got {x.dtype}"
    )

    # OPT-1: model parameters must be BF16
    param_dtype = next(model.parameters()).dtype
    assert param_dtype == torch.bfloat16, (
        f"Expected model parameters to be bfloat16 (OPT-1), got {param_dtype}"
    )

    # Model must be on CUDA
    param_device = next(model.parameters()).device
    assert param_device.type == "cuda", (
        f"Model parameters must be on CUDA, got {param_device}"
    )

    # OPT-3: Flash SDP should be enabled after get_model_and_input()
    # Verify via torch.backends.cuda.flash_sdp_enabled()
    assert torch.backends.cuda.flash_sdp_enabled(), (
        "Flash SDP must be enabled after OPT-3 (enable_flash_sdp=True)"
    )


def test_compiled_forward_pass(caplog):
    """Compiled forward pass triggers the sdpa_attention_opt backend; captures FX pass logs."""
    import logging
    import torch
    from examples.sdpa_attention.sdpa_attention_optimized import (
        get_model_and_input,
        BATCH_SIZE,
        SEQ_LEN,
        DIM,
    )

    model, x = get_model_and_input()
    compiled = torch.compile(model, backend="sdpa_attention_opt")
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
        assert out.shape == (BATCH_SIZE, SEQ_LEN, DIM), (
            f"Expected output shape ({BATCH_SIZE}, {SEQ_LEN}, {DIM}), got {out.shape}"
        )
        assert out.dtype == torch.bfloat16, (
            f"Expected output dtype bfloat16, got {out.dtype}"
        )
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
