"""
lstm_sequence_encoder_optimized.py — LSTMSequenceEncoder with custom torch.compile() backend.

Implements 3 operator-level optimizations derived from ncu profiling:

  1. BF16 dtype promotion (get_model_and_input) — HIGH confidence
       Routes cuDNN LSTM from legacy SIMT 'Kernel2' to WGMMA sm90_xmma_gemm_bf16bf16_*
       path on H100. Expected 2-4x speedup on aten::_cudnn_rnn (98.9% of baseline time).

  2. cudnn.benchmark + allow_tf32 (get_model_and_input) — HIGH confidence
       Enables cuDNN to auto-select fastest BF16 LSTM algorithm for this exact input
       configuration. Sets allow_tf32 for the linear head (aten::addmm).

  3. pass_pretranspose_classifier (FX pass) — MEDIUM confidence
       Pre-stores the transposed Linear classifier weight as a buffer, eliminating the
       per-call aten.t() on the [512, 10] weight. Impact is negligible (<0.1ms) but
       demonstrates FX passes are executing. Degrades gracefully if pattern not found.

Architecture note: nn.LSTM with torch.compile+inductor on PyTorch 2.11/H100 stays as
aten::_cudnn_rnn (cuDNN handles the full LSTM). Inductor does NOT decompose it into
Triton kernels. FX graph passes only see aten::mean and aten::addmm (the linear head).

To validate:
    python -m pytest test_lstm_sequence_encoder_optimized.py -v

To profile the optimized workload:
    operator-profiler profile examples/lstm_sequence_encoder/lstm_sequence_encoder_optimized.py \\
        --model-name LSTMSequenceEncoderOpt --compile-mode lstm_sequence_encoder_opt \\
        --output runs/lstm_optimized
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.fx as fx
from torch._dynamo import register_backend
from torch._inductor.compile_fx import compile_fx  # function, not module

# Allow importing baseline workload from the same directory
sys.path.insert(0, str(Path(__file__).parent))
from lstm_sequence_encoder import (
    LSTMSequenceEncoder,
    DEVICE,
    BATCH_SIZE,
    SEQ_LEN,
    INPUT_SIZE,
    NUM_CLASSES,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FX Graph Pass: Pre-transpose classifier weight (OPT-3)
# ---------------------------------------------------------------------------

def pass_pretranspose_classifier(gm: fx.GraphModule) -> fx.GraphModule:
    """
    Eliminate aten.t() on the Linear classifier weight in the FX graph.

    In Inductor-traced graphs, nn.Linear appears as:
        bias_node = get_attr('classifier.bias')
        weight_t  = aten.t(get_attr('classifier.weight'))   ← on-the-fly transpose
        out       = aten.addmm(bias_node, input, weight_t)

    This pass pre-computes weight.T.contiguous() and registers it as a buffer,
    replacing the per-call aten.t() with a direct get_attr on the pre-transposed copy.

    Confidence: medium — degrades gracefully if the pattern is not found.
    Impact: negligible (<0.1ms, 0.04% of total), included for completeness.
    """
    try:
        matched = False
        for node in list(gm.graph.nodes):
            if node.target != torch.ops.aten.addmm.default:
                continue
            # args: (bias, input, weight_or_t_of_weight)
            if len(node.args) < 3:
                continue
            weight_node = node.args[2]
            # Check for aten.t(get_attr('...')) pattern
            if not (
                weight_node.target == torch.ops.aten.t.default
                and weight_node.args[0].op == "get_attr"
            ):
                continue
            param_name = weight_node.args[0].target
            try:
                weight = gm.get_parameter(param_name)
            except AttributeError:
                continue
            # Only apply to classifier head (small weight [out, in])
            if weight.numel() > 512 * 512:
                continue
            w_T = weight.T.contiguous()
            buf_name = param_name.replace(".", "_") + "_T"
            gm.register_buffer(buf_name, w_T)
            with gm.graph.inserting_before(node):
                new_get = gm.graph.get_attr(buf_name)
                new_get.meta = {}
            # Replace the t(get_attr) with get_attr(buf_name)
            weight_node.replace_all_uses_with(new_get)
            gm.graph.erase_node(weight_node)
            matched = True
            logger.info(
                "[pass_pretranspose_classifier] Applied — pre-transposed '%s' [%s]",
                param_name,
                list(weight.shape),
            )
        if not matched:
            logger.warning(
                "[pass_pretranspose_classifier] Pattern not found — "
                "aten.addmm with transposed get_attr weight not in graph. "
                "cuDNN may have fused the classifier into _cudnn_rnn or Inductor "
                "decomposed it differently. Pass not applied."
            )
            return gm
        gm.graph.lint()
        gm.recompile()
    except Exception as exc:
        logger.warning("[pass_pretranspose_classifier] Failed: %s — returning unchanged graph", exc)
    return gm


# ---------------------------------------------------------------------------
# Backend Registration
# ---------------------------------------------------------------------------

@register_backend
def lstm_sequence_encoder_opt(gm: fx.GraphModule, example_inputs) -> Callable:
    """
    Custom torch.compile() backend for LSTMSequenceEncoder.

    Applies:
      - pass_pretranspose_classifier (OPT-3, medium confidence)

    Non-graph optimizations (OPT-1, OPT-2) are in get_model_and_input().
    cuDNN handles aten::_cudnn_rnn opaquely — no FX passes can reach inside it.
    """
    logger.info("lstm_sequence_encoder_opt backend: starting")
    gm = pass_pretranspose_classifier(gm)
    logger.info("lstm_sequence_encoder_opt backend: delegating to Inductor")
    return compile_fx(gm, example_inputs)


# ---------------------------------------------------------------------------
# Workload Interface
# ---------------------------------------------------------------------------

def get_model_and_input() -> tuple[nn.Module, torch.Tensor]:
    """
    Return (optimized model, input tensor) for profiling.

    Non-graph optimizations applied here:
      OPT-1: BF16 dtype promotion — routes cuDNN LSTM to WGMMA Tensor Core path
      OPT-2: cudnn.benchmark — auto-selects fastest BF16 algorithm for this shape
    """
    assert torch.cuda.is_available(), "CUDA required"

    # OPT-2a: Enable cuDNN benchmark BEFORE model construction so first-call
    # algorithm selection uses the benchmark mode from the start.
    torch.backends.cudnn.benchmark = True
    # OPT-2b: Allow TF32 for aten::addmm (linear classifier head)
    torch.backends.cuda.matmul.allow_tf32 = True

    model = LSTMSequenceEncoder().to(DEVICE).eval()
    x = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE, device=DEVICE)

    # OPT-1: BF16 dtype promotion
    # Routes cuDNN from 'Kernel2' (FP32 SIMT) to sm90_xmma_gemm_bf16bf16_* (WGMMA)
    if next(model.parameters()).dtype != torch.bfloat16:
        model = model.to(torch.bfloat16)
        x = x.to(torch.bfloat16)

    return model, x


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch._dynamo.reset()
    model, x = get_model_and_input()
    compiled = torch.compile(model, backend="lstm_sequence_encoder_opt")
    with torch.no_grad():
        out = compiled(x)
    assert out.shape == (BATCH_SIZE, NUM_CLASSES), f"Unexpected shape: {out.shape}"
    assert not torch.isnan(out).any(), "NaN in output"
    assert not torch.isinf(out).any(), "Inf in output"
    print(f"Smoke test passed — output shape: {out.shape}, dtype: {out.dtype}")
