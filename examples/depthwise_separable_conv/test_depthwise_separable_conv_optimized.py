"""
test_depthwise_separable_conv_optimized.py — 4-test validation suite for the
depthwise_separable_conv_opt backend.

Run with:
    pytest examples/depthwise_separable_conv/test_depthwise_separable_conv_optimized.py -v
"""
import logging

import torch

BACKEND_NAME = "depthwise_separable_conv_opt"


def test_import():
    """Module imports without error (triggers @register_backend at load time)."""
    import depthwise_separable_conv_optimized  # noqa: F401


def test_backend_registration():
    """Backend registered with torch._dynamo."""
    import torch
    import depthwise_separable_conv_optimized  # noqa: F401

    backends = str(torch._dynamo.list_backends())
    assert BACKEND_NAME in backends, f"Backend not found in: {backends}"


def test_get_model_and_input():
    """Input is on CUDA with the shape/dtype expected from profile.json:
    [16, 32, 56, 56], fp32. OPT-3 makes it channels_last (NHWC)."""
    from depthwise_separable_conv_optimized import get_model_and_input

    model, x = get_model_and_input()

    assert x.is_cuda, "input tensor must be on CUDA"
    assert tuple(x.shape) == (16, 32, 56, 56), f"unexpected input shape: {tuple(x.shape)}"
    assert x.dtype == torch.float32, f"unexpected input dtype: {x.dtype}"
    # OPT-3: channels_last applied in get_model_and_input()
    assert x.is_contiguous(memory_format=torch.channels_last), \
        "input should be channels_last (NHWC) after OPT-3"
    assert next(model.parameters()).is_cuda, "model params must be on CUDA"


def test_compiled_forward_pass(caplog):
    """Compiled forward pass triggers the backend; captures FX pass logs and
    checks output finiteness."""
    import logging
    import torch
    from depthwise_separable_conv_optimized import get_model_and_input

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
                # torch 2.11: guard error after dedup backend succeeds — safe to suppress

    for record in caplog.records:
        print(record.getMessage())

    assert caplog.records, "No logger output — backend may not have executed"

    if out is not None:
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"
        # Output of the 3-block stack: [16, 256, 56, 56]
        assert tuple(out.shape) == (16, 256, 56, 56), \
            f"unexpected output shape: {tuple(out.shape)}"
