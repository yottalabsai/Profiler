"""
conv_block_optimized.py — ConvBlock with custom torch.compile() backend.

Implements 3 operator-level optimizations:
  1. bn_fold        — folds BatchNorm2d weights into Conv2d via fuse_conv_bn_eval(),
                      eliminating 14 kernel launches (67.9 µs). Applied before
                      torch.compile so the FX graph never contains BN nodes.
  2. channels_last  — eliminates ~15 convertTensor_kernel launches per forward pass.
  3. bf16           — routes convolutions to Blackwell WGMMA (1,457 TFLOPS) instead
                      of Ampere sm80_xmma TF32; halves BatchNorm DRAM traffic.

All three are applied in get_model_and_input() before compilation. The custom backend
conv_block_opt wraps compile_fx with UniqueSubgraphRegistry for dedup correctness.

Note on FX-pass BN fold: adding new placeholder or get_attr nodes to the aot_autograd
graph requires _dynamo_source metadata that only Dynamo can set during tracing. Module-
level fold via fuse_conv_bn_eval() is the safe alternative — the modified nn.Module is
traced cleanly with no BN nodes in the resulting FX graph.

Backend name: conv_block_opt

To profile:
    PYTHONPATH=/home/ubuntu/Profiler/nvidia:/home/ubuntu/Profiler \\
    nsys profile --trace=cuda,nvtx \\
        --output=profiler_output/conv_block_opt \\
        --force-overwrite=true \\
        python3 nvidia/scripts/run_workload.py \\
            --workload examples/conv_block/conv_block_optimized.py \\
            --compile-backend conv_block_opt \\
            --warmup-iters 2 --measure-iters 2 \\
            --output-prefix profiler_output/conv_block_opt \\
            --inductor-debug-dir profiler_output/conv_block_opt_inductor_debug
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Callable

import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
from torch._dynamo import register_backend
from torch._inductor.compile_fx import compile_fx
from torch.nn.utils.fusion import fuse_conv_bn_eval

from nvidia.operator_profiler.fx import UniqueSubgraphRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OPT-1: Module-level BatchNorm fold
# ---------------------------------------------------------------------------

def _fold_all_bn(model: nn.Module) -> nn.Module:
    """
    Fold every ConvBnRelu-style submodule's BN into its Conv2d weight/bias.

    Uses torch.nn.utils.fusion.fuse_conv_bn_eval (official PyTorch API) to
    compute folded weight = conv_w * gamma/sqrt(var+eps) and folded bias.
    Replaces the BN with nn.Identity() so the forward path passes through with
    no kernel launch (Inductor traces Identity as an alias, not a computation).

    Safe only at eval() time — running_mean/var must be frozen.
    """
    for module in model.modules():
        conv = getattr(module, "conv", None)
        bn   = getattr(module, "bn",   None)
        if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
            module.conv = fuse_conv_bn_eval(conv, bn)
            module.bn   = nn.Identity()
            logger.info(
                "[fold_all_bn] Folded BN into Conv2d (%s in_ch=%d out_ch=%d)",
                type(module).__name__, conv.in_channels, conv.out_channels,
            )
    return model


# ---------------------------------------------------------------------------
# Partition input capture utility (for UniqueSubgraphRegistry dedup path)
# ---------------------------------------------------------------------------

def _capture_partition_inputs(
    split_gm: fx.GraphModule,
    example_inputs: list,
) -> dict[str, list]:
    """Run split_gm once to capture input tensors for each partition."""
    captured: dict[str, list] = {}
    hooks = []
    for name, submod in split_gm.named_children():
        if isinstance(submod, fx.GraphModule):
            def _hook(mod, args, _name=name):
                captured[_name] = list(args)
            hooks.append(submod.register_forward_pre_hook(_hook))
    with torch.no_grad():
        split_gm(*example_inputs)
    for h in hooks:
        h.remove()
    return captured


# ---------------------------------------------------------------------------
# Backend registration
# ---------------------------------------------------------------------------

@register_backend
def conv_block_opt(gm: fx.GraphModule, example_inputs: list) -> Callable:
    """
    Custom torch.compile() backend for ConvBlock.

    BN fold, channels_last, and BF16 are all applied before this backend is
    invoked (in get_model_and_input()). The backend uses UniqueSubgraphRegistry
    for correctness with the built-in dedup infrastructure and delegates final
    compilation to compile_fx.
    """
    registry = UniqueSubgraphRegistry(gm)
    equiv_map = registry.build_partition_equivalence_map()

    if not equiv_map:
        # No repeated layers — flat compile (expected for ConvBlock)
        logger.info("[conv_block_opt] Flat compile path (no duplicate partitions)")
        return compile_fx(gm, example_inputs)

    # Dedup path for models with repeated blocks
    logger.info(
        "[conv_block_opt] Dedup path: %d unique partition(s)",
        len(list(registry.unique_reps)),
    )
    partition_inputs = _capture_partition_inputs(registry.split, example_inputs)
    for rep_name, rep_mod in registry.unique_reps:
        inputs = partition_inputs.get(rep_name, example_inputs)
        compiled = compile_fx(rep_mod, inputs)
        rep_mod.forward = compiled
        for _, dup_mod in registry.duplicates_of(rep_name):
            dup_mod.forward = compiled

    return lambda *args: registry.split(*args)


# ---------------------------------------------------------------------------
# Workload interface
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent))
from conv_block import ConvBlock, DEVICE, BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH


def get_model_and_input() -> tuple:
    """
    Return (uncompiled ConvBlock, input tensor) with all three optimizations applied.

    OPT-1 (bn_fold): Fold BN into Conv2d via fuse_conv_bn_eval before Dynamo traces
      the model. The resulting FX graph has no aten._native_batch_norm_legit_no_training
      nodes — all 14 BN kernel launches are eliminated.

    OPT-2 (channels_last): Eliminates NCHW<->NHWC convertTensor_kernel round-trips.
      cuDNN receives NHWC tensors directly and selects NHWC-optimized GEMM kernels.

    OPT-3 (bf16): Routes convolutions from Ampere sm80_xmma TF32 kernels to
      Blackwell-native BF16 WGMMA (1,457 TFLOPS peak vs 91 TFLOPS FP32 SIMT).
    """
    assert torch.cuda.is_available(), "CUDA required"
    model = ConvBlock().to(DEVICE).eval()

    # OPT-1: fold BN into Conv2d weights before any dtype/layout conversion
    _fold_all_bn(model)

    # OPT-2: channels_last
    if not next(model.parameters()).is_contiguous(memory_format=torch.channels_last):
        model = model.to(memory_format=torch.channels_last)

    # OPT-3: BF16
    if next(model.parameters()).dtype != torch.bfloat16:
        model = model.to(torch.bfloat16)

    x = torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH, device=DEVICE)
    x = x.to(memory_format=torch.channels_last)
    x = x.to(torch.bfloat16)

    return model, x


if __name__ == "__main__":
    model, x = get_model_and_input()
    compiled = torch.compile(model, backend="conv_block_opt")
    out = compiled(x)
    print(f"output shape: {out.shape}  dtype: {out.dtype}")
