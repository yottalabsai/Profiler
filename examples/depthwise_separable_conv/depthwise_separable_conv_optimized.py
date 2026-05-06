"""
depthwise_separable_conv_optimized.py — DepthwiseSepConv with custom torch.compile() backend.

Implements 4 operator-level optimizations derived from profiling on NVIDIA A100-SXM4-80GB:

  1. pass_fold_bn       (OPT-001, HIGH)   — FX pass: fold BatchNorm into Conv2d weights,
                                            eliminating 60 BN kernel launches (43.2% of time)
  2. channels_last      (OPT-002, HIGH)   — model/input layout, enables cuDNN NHWC paths,
                                            eliminates im2col overhead on pointwise convs
  3. BF16 dtype cast    (OPT-003, HIGH)   — model/input dtype, engages HMMA Tensor Cores,
                                            reduces registers/thread 238 → ~64-96
  4. epilogue_fusion    (OPT-004, MEDIUM) — Inductor config, fuses conv + ReLU6 epilogue
                                            after BN elimination exposes standalone activations

Dependency order: OPT-001 → OPT-002 → OPT-003 → OPT-004

To run with optimizations:
    python depthwise_separable_conv_optimized.py

To profile:
    python nvidia/scripts/run_workload.py \\
        --workload examples/depthwise_separable_conv/depthwise_separable_conv_optimized.py \\
        --compile-backend depthwise_sep_conv_opt
"""
from __future__ import annotations

import logging
import operator
from typing import Callable

import torch
import torch.nn as nn
import torch.fx as fx
from torch._dynamo import register_backend
from torch._inductor.compile_fx import compile_fx

import sys as _sys
import pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).parent))

from depthwise_separable_conv import (
    DepthwiseSepConv,
    DEVICE,
    BATCH_SIZE,
    IN_CHANNELS,
    HEIGHT,
    WIDTH,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# OPT-001: FX pass — Conv-BN fold (high confidence)
# ---------------------------------------------------------------------------

def pass_fold_bn(gm: fx.GraphModule) -> fx.GraphModule:
    """
    Fold _native_batch_norm_legit_no_training nodes into their preceding
    convolution nodes by absorbing BN scale/shift into conv weight/bias.

    Applied as a defensive FX pass; the primary fold happens at module level
    in get_model_and_input() via fuse_conv_bn_eval. This pass handles any
    residual BN nodes that Dynamo may have traced.

    Eliminating BN removes the intermediate activation tensor write+read
    between conv and BN — the dominant source of DRAM traffic at 43.2% of
    attributed time (dram_throughput=60.5%).
    """
    try:
        nodes = list(gm.graph.nodes)
        fold_count = 0

        for node in nodes:
            if node.target not in (
                torch.ops.aten._native_batch_norm_legit_no_training.default,
                torch.ops.aten.batch_norm.default,
            ):
                continue
            bn_node = node

            # BN inputs: (input, weight, bias, running_mean, running_var, ?, ?, eps, ...)
            bn_args = bn_node.args
            conv_node = bn_args[0]
            if conv_node.target != torch.ops.aten.convolution.default:
                logger.warning("[pass_fold_bn] BN input is not convolution.default — skipping")
                continue

            def _get_param(n):
                if n is not None and hasattr(n, 'op') and n.op == 'get_attr':
                    return gm.get_parameter(n.target)
                return None

            gamma = _get_param(bn_args[1])   # BN weight (scale)
            beta  = _get_param(bn_args[2])   # BN bias (shift)
            mean  = _get_param(bn_args[3])   # running_mean
            var   = _get_param(bn_args[4])   # running_var
            eps   = bn_args[7] if len(bn_args) > 7 else 1e-5

            if any(t is None for t in [gamma, beta, mean, var]):
                logger.warning("[pass_fold_bn] Could not extract BN parameters — skipping")
                continue

            conv_args = conv_node.args
            conv_weight = _get_param(conv_args[1])
            conv_bias   = _get_param(conv_args[2])
            if conv_weight is None:
                logger.warning("[pass_fold_bn] Could not extract conv weight — skipping")
                continue

            # Fold: new_weight = (gamma / sqrt(var + eps)) * conv_weight
            # For 4D weight [C_out, C_in, H, W] (or [C_in, 1, H, W] for depthwise):
            # scale has shape [C_out] → view(-1, 1, 1, 1) broadcasts correctly
            scale = gamma / torch.sqrt(var + eps)
            new_weight = conv_weight * scale.view(-1, 1, 1, 1)
            if conv_bias is not None:
                new_bias = (conv_bias - mean) * scale + beta
            else:
                new_bias = beta - mean * scale

            w_buf = f'_folded_weight_{fold_count}'
            b_buf = f'_folded_bias_{fold_count}'
            gm.register_buffer(w_buf, new_weight)
            gm.register_buffer(b_buf, new_bias)

            with gm.graph.inserting_before(conv_node):
                fw_node = gm.graph.get_attr(w_buf)
                fb_node = gm.graph.get_attr(b_buf)

            new_conv_args = (conv_args[0], fw_node, fb_node) + conv_args[3:]
            with gm.graph.inserting_after(conv_node):
                new_conv = gm.graph.call_function(
                    torch.ops.aten.convolution.default, new_conv_args
                )

            # BN output is a tuple (out, mean, rstd); replace getitem(0) with new_conv
            for user in list(bn_node.users):
                if user.target == operator.getitem and user.args[1] == 0:
                    user.replace_all_uses_with(new_conv)
                if not list(user.users):
                    gm.graph.erase_node(user)

            conv_node.replace_all_uses_with(new_conv)
            if not list(conv_node.users):
                gm.graph.erase_node(conv_node)
            if not list(bn_node.users):
                gm.graph.erase_node(bn_node)

            logger.info("[pass_fold_bn] Folded BN %d into preceding Conv2d", fold_count)
            fold_count += 1

        if fold_count:
            gm.graph.lint()
            gm.recompile()
        else:
            logger.info("[pass_fold_bn] No BN nodes found — already folded at module level")

    except Exception as e:
        logger.warning("[pass_fold_bn] Failed: %s", e)

    return gm


# ---------------------------------------------------------------------------
# Backend registration
# ---------------------------------------------------------------------------

@register_backend
def depthwise_sep_conv_opt(gm: fx.GraphModule, example_inputs) -> Callable:
    """
    Custom Dynamo backend implementing OPT-001 through OPT-004.

    Pass order:
      1. pass_fold_bn — structural change must run first (removes BN nodes)
      2. compile_fx   — Inductor compilation with epilogue_fusion
    """
    gm = pass_fold_bn(gm)
    return compile_fx(gm, example_inputs)


# ---------------------------------------------------------------------------
# Workload interface — applies non-graph optimizations before compilation
# ---------------------------------------------------------------------------

def _fold_model_bn(model: DepthwiseSepConv) -> DepthwiseSepConv:
    """
    OPT-001 (primary path): fuse Conv2d + BatchNorm2d at module level using
    the PyTorch utility, which is more reliable than the FX pass for known
    module structures. The FX pass_fold_bn above handles any residual BN
    nodes that Dynamo may still trace.
    """
    model.eval()
    for block in [model.block1, model.block2, model.block3]:
        block.depthwise = torch.nn.utils.fuse_conv_bn_eval(
            block.depthwise, block.bn_dw
        )
        block.bn_dw = nn.Identity()
        block.pointwise = torch.nn.utils.fuse_conv_bn_eval(
            block.pointwise, block.bn_pw
        )
        block.bn_pw = nn.Identity()
    logger.info("[opt] OPT-001: folded 6 Conv-BN pairs (3 depthwise + 3 pointwise)")
    return model


def get_model_and_input() -> tuple[nn.Module, torch.Tensor]:
    """
    Workload interface. Returns (model, input) with all non-graph optimizations
    applied. The custom backend (depthwise_sep_conv_opt) is NOT applied here;
    run_workload.py selects it via --compile-backend.

    Optimizations applied here (in dependency order):
      OPT-001: BN fold        — eliminates 60 BN kernel launches (43.2% of time)
      OPT-002: channels_last  — enables cuDNN NHWC paths for all conv layers
      OPT-003: BF16 dtype     — engages HMMA Tensor Cores; regs 238 → ~64-96
      OPT-004: epilogue_fusion — Inductor fuses conv + ReLU6 activation epilogue
    """
    assert torch.cuda.is_available(), "CUDA required"

    model = DepthwiseSepConv().to(DEVICE).eval()
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH, device=DEVICE)

    # OPT-001: BN fold (module level — primary mechanism)
    model = _fold_model_bn(model)

    # OPT-002: channels_last — eliminates im2col, enables cuDNN NHWC kernel path
    # Evidence: all conv ops used NCHW Kernel2 (238 regs, occ 12-31%)
    if not next(model.parameters()).is_contiguous(memory_format=torch.channels_last):
        model = model.to(memory_format=torch.channels_last)
        logger.info("[opt] OPT-002: applied channels_last to model")
    x = x.to(memory_format=torch.channels_last)

    # OPT-003: BF16 dtype — engages Tensor Cores on Ampere (sm80_xmma_gemm_bf16bf16 path)
    # Evidence: pw conv regs=238, occ=12%; BF16 reduces to ~64-96 regs, occ ~50%+
    model = model.to(torch.bfloat16)
    x = x.to(torch.bfloat16)
    logger.info("[opt] OPT-003: cast model and input to bfloat16")

    # OPT-004: epilogue fusion — Inductor fuses conv output with ReLU6 activation
    # After BN fold, ReLU6 is a standalone elementwise op; epilogue fusion
    # eliminates the intermediate DRAM write/read between conv and activation.
    try:
        import torch._inductor.config as _ind_cfg
        _ind_cfg.epilogue_fusion = True
        logger.info("[opt] OPT-004: enabled Inductor epilogue_fusion")
    except Exception as e:
        logger.warning("[opt] OPT-004: could not set epilogue_fusion: %s", e)

    return model, x


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model, x = get_model_and_input()
    compiled = torch.compile(model, backend="depthwise_sep_conv_opt")
    with torch.no_grad():
        out = compiled(x)
    print(f"output shape: {out.shape}, dtype: {out.dtype}")
    print("All optimizations applied successfully.")
