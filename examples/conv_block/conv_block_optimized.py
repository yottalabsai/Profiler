"""
conv_block_optimized.py — VGG-style ConvBlock with four operator-level optimizations.

Implements the following transformations derived from ncu profiling of the
baseline FP32 NCHW ConvBlock:

  1. OPT-3 (BN fold, high confidence) — Folds each BatchNorm2d's running
     statistics and affine parameters into the preceding Conv2d weight/bias.
     Eliminates aten::_native_batch_norm_legit_no_training entirely from the FX
     graph.  Applied to the nn.Module in FP32 before any layout/dtype cast.

  2. OPT-1 (channels_last, high confidence) — Converts model and input tensors
     to channels_last (NHWC) memory format.  Eliminates cuDNN convertTensor_kernel
     launches on every conv forward pass.

  3. OPT-2 (BF16, high confidence) — Casts model parameters and input to
     bfloat16 after channels_last conversion so cuDNN selects
     sm80_xmma_gemm_bf16bf16 (HMMA Tensor Core) instead of the FP32 SIMT path.

  4. OPT-4 (max-autotune + TF32, medium confidence) — Compiles with
     mode='max-autotune' so Inductor benchmarks all available cuDNN/cuBLAS
     algorithms for the specific conv/GEMM shapes.  TF32 flags are set as a
     fallback for any residual FP32 paths.

Application order (from optimizations.json):
  OPT-3 → OPT-1 → OPT-2 → torch.compile(mode='max-autotune')

All four optimizations are applied before torch.compile() is called, so no
custom FX graph pass is needed in the backend.  The backend's role is to
register under the name ``conv_block_opt`` and delegate to Inductor, with the
dedup-aware structure that gracefully handles any models that happen to have
repeated layer structure.

To profile with optimizations:
    python nvidia/scripts/run_workload.py \\
        --workload examples/conv_block/conv_block_optimized.py \\
        --compile-backend conv_block_opt \\
        --output-prefix runs/conv_block_opt \\
        --warmup-iters 3 --measure-iters 10
"""
from __future__ import annotations

import logging
from typing import Callable

import torch
import torch.nn as nn
import torch.fx as fx
from torch._dynamo import register_backend
from torch._inductor.compile_fx import compile_fx

from nvidia.operator_profiler.fx import UniqueSubgraphRegistry, FxPassRunner

# ---------------------------------------------------------------------------
# Re-export baseline model classes so this file is a self-contained workload
# ---------------------------------------------------------------------------
from examples.conv_block.conv_block import (
    ConvBnRelu,
    ConvBlock,
    get_model_and_input as _baseline_get_model_and_input,
    DEVICE,
    BATCH_SIZE,
    IN_CHANNELS,
    HEIGHT,
    WIDTH,
    NUM_CLASSES,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ============================================================================
# BN-fold utility (OPT-3)
# ============================================================================

def fold_bn_into_conv(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """
    Fold BatchNorm2d parameters into the preceding Conv2d, returning the
    modified Conv2d.

    Closed-form formulas (inference mode, training=False):
        scale   = gamma / sqrt(running_var + eps)
        W_new   = W_conv * scale.view(-1, 1, 1, 1)
        b_new   = (b_conv - running_mean) * scale + beta

    After folding the BN module should be removed from the graph so that the
    FX trace never generates aten::_native_batch_norm_legit_no_training.

    Arithmetic is performed in FP32 regardless of the model's current dtype so
    that accumulated BN statistics are not truncated before the fold.
    """
    bn.eval()
    with torch.no_grad():
        gamma = bn.weight.float()        # (C_out,)
        beta  = bn.bias.float()          # (C_out,)
        mean  = bn.running_mean.float()  # (C_out,)
        var   = bn.running_var.float()   # (C_out,)
        eps   = bn.eps

        scale   = gamma / (var + eps).sqrt()          # (C_out,)
        scale_w = scale.view(-1, 1, 1, 1)             # broadcast over kernel dims

        w = conv.weight.float()
        new_weight = w * scale_w

        if conv.bias is not None:
            b = conv.bias.float()
        else:
            b = torch.zeros(conv.out_channels, device=conv.weight.device)

        new_bias = (b - mean) * scale + beta

        conv.weight = nn.Parameter(new_weight.to(conv.weight.dtype))
        conv.bias   = nn.Parameter(new_bias.to(conv.weight.dtype))

    logger.info(
        "fold_bn_into_conv: folded BN(%d) into Conv2d(%d→%d)",
        bn.num_features, conv.in_channels, conv.out_channels,
    )
    return conv


def fold_all_bn(model: nn.Module) -> nn.Module:
    """
    Walk ``model`` and fold every (Conv2d, BatchNorm2d) sequential pair.

    Strategy:
      - For each ``nn.Sequential`` (and the top-level module), scan children in
        order.  When child[i] is Conv2d and child[i+1] is BatchNorm2d, fold and
        replace child[i+1] with nn.Identity.
      - Recurse into all child modules so nested Sequentials are handled.

    Using nn.Identity as a placeholder rather than removing the child by key
    keeps the Sequential indexing stable for any code that addresses children by
    index, and avoids mutating a dict while iterating over it.
    """
    _fold_sequential_pairs(model)
    return model


def _fold_sequential_pairs(module: nn.Module) -> None:
    """Recursively fold Conv+BN pairs inside all nn.Sequential children."""
    for name, child in module.named_children():
        if isinstance(child, nn.Sequential):
            _fold_in_sequential(child)
        # Recurse regardless — nested structures like ConvBnRelu are not Sequential
        # but their children may be Sequential, or they may contain conv+bn as
        # direct attributes (handled by _fold_direct_attrs below).
        _fold_direct_attrs(child)
        _fold_sequential_pairs(child)


def _fold_in_sequential(seq: nn.Sequential) -> None:
    """
    Fold Conv+BN pairs in a flat nn.Sequential.

    Iterates by index; when pair (i, i+1) = (Conv2d, BatchNorm2d), folds the
    BN into the Conv and replaces the BN with nn.Identity.
    """
    children = list(seq.named_children())
    for i in range(len(children) - 1):
        name_i,   mod_i   = children[i]
        name_i1,  mod_i1  = children[i + 1]
        if isinstance(mod_i, nn.Conv2d) and isinstance(mod_i1, nn.BatchNorm2d):
            fold_bn_into_conv(mod_i, mod_i1)
            seq._modules[name_i1] = nn.Identity()
            logger.info("fold_all_bn: replaced %s[%s] with Identity", type(seq).__name__, name_i1)


def _fold_direct_attrs(module: nn.Module) -> None:
    """
    Fold Conv+BN pairs stored as direct child modules of a non-Sequential module.

    ConvBnRelu stores them as self.conv and self.bn.  ``named_children()``
    iterates in registration order (the order ``self.x = ...`` assignments
    appear in ``__init__``), which is the correct causal order for Conv→BN.

    We scan the children in order: when child[i] is Conv2d and child[i+1] is
    BatchNorm2d, fold the BN into the Conv and replace the BN child with
    nn.Identity.  Any intervening non-Conv2d, non-BatchNorm2d child resets the
    conv candidate so we do not mis-pair distant Conv/BN siblings.
    """
    # nn.Module._modules is an OrderedDict; named_children() iterates it in
    # insertion order — same order as __init__ assignments.
    children = list(module.named_children())
    conv_name: str | None = None
    conv_mod:  nn.Conv2d | None = None

    for child_name, child_mod in children:
        if isinstance(child_mod, nn.Conv2d):
            conv_name = child_name
            conv_mod  = child_mod
        elif isinstance(child_mod, nn.BatchNorm2d) and conv_mod is not None:
            fold_bn_into_conv(conv_mod, child_mod)
            module._modules[child_name] = nn.Identity()
            logger.info(
                "fold_all_bn: replaced %s.%s with Identity (paired with .%s)",
                type(module).__name__, child_name, conv_name,
            )
            conv_name = None
            conv_mod  = None
        else:
            # Any other module between Conv and BN breaks the pair assumption
            conv_name = None
            conv_mod  = None


# ============================================================================
# _capture_partition_inputs — utility for dedup compile path
# ============================================================================

def _capture_partition_inputs(
    split_gm: fx.GraphModule,
    example_inputs: list,
) -> dict[str, list]:
    """
    Capture real input tensors for each partition by running one forward pass
    through the split graph with forward-pre hooks attached.

    Returns a dict mapping partition submodule name → list of tensors so that
    each partition can be compiled with its actual input shapes/dtypes rather
    than the top-level example_inputs.
    """
    partition_inputs: dict[str, list] = {}
    hooks = []
    for name, submod in split_gm.named_children():
        if isinstance(submod, fx.GraphModule):
            def _hook(mod, args, _name=name):
                partition_inputs[_name] = list(args)
            hooks.append(submod.register_forward_pre_hook(_hook))
    with torch.no_grad():
        split_gm(*example_inputs)
    for h in hooks:
        h.remove()
    return partition_inputs


# ============================================================================
# Backend Registration
# ============================================================================

@register_backend
def conv_block_opt(gm: fx.GraphModule, example_inputs, **kwargs) -> Callable:
    """
    Custom torch.compile() backend for ConvBlock.

    All four optimizations (BN fold, channels_last, BF16, max-autotune) are
    applied before torch.compile() is invoked, so the FX graph that arrives
    here is already optimized at the nn.Module level.  The backend's job is to:

      1. Detect whether the graph has repeated-layer structure (dedup path) or
         not (flat path).
      2. Flat path (ConvBlock): compile the whole graph with Inductor so that
         cross-layer operator fusion is preserved.
      3. Dedup path: compile each unique representative partition once, share
         the compiled callable with structural duplicates.

    No additional FX passes are applied here because:
      - BN fold is done in get_model_and_input() — BN nodes are absent from the
        FX graph by the time this backend is called.
      - channels_last and BF16 are tensor properties visible to Inductor via
        the input metadata; no graph surgery needed.
      - max-autotune is requested via the torch.compile() call site in
        get_model_and_input().
    """
    logger.info("conv_block_opt backend: starting")

    registry = UniqueSubgraphRegistry(gm)
    equiv_map = registry.build_partition_equivalence_map()

    if not equiv_map:
        # ConvBlock has no repeated layers — flat compile preserves cross-layer
        # Inductor fusion opportunities (e.g. BN-folded conv + ReLU can be
        # fused across stage boundaries in a flat graph).
        logger.info("conv_block_opt: no repeated layers, flat compile path")
        logger.info("conv_block_opt: delegating to Inductor")
        return compile_fx(gm, example_inputs)

    # Dedup path — for future models with repeated structure loaded through this
    # backend.  Apply any replace_pattern-compatible passes via FxPassRunner,
    # then compile unique reps with their real partition inputs.
    logger.info(
        "conv_block_opt: %d duplicate partition(s) detected, dedup path",
        len(equiv_map),
    )
    runner = FxPassRunner(registry)  # no replace_pattern passes for ConvBlock

    partition_inputs = _capture_partition_inputs(registry.split, example_inputs)
    for rep_name, rep_mod in registry.unique_reps:
        compiled = compile_fx(
            rep_mod,
            partition_inputs.get(rep_name, example_inputs),
        )
        rep_mod.forward = compiled
        for _, dup_mod in registry.duplicates_of(rep_name):
            dup_mod.forward = compiled

    logger.info("conv_block_opt: all partitions compiled, returning split graph")
    return lambda *args: registry.split(*args)


# ============================================================================
# Workload Interface
# ============================================================================

def get_model_and_input() -> tuple:
    """
    Workload interface for run_workload.py.

    Applies all four optimizations to the baseline model before returning:

      Step 1 — OPT-3: Fold each BatchNorm2d into its preceding Conv2d so that
        the FX graph never contains aten::_native_batch_norm_legit_no_training.
        Done in FP32 for numerical accuracy.

      Step 2 — OPT-1: Convert model and input to channels_last memory format.
        Must happen before torch.compile() so TorchDynamo traces channels_last
        tensors and cuDNN sees NHWC inputs from the start.

      Step 3 — OPT-2: Cast model parameters and input to bfloat16.
        Applied after channels_last so the layout flag is preserved; cuDNN then
        selects the NHWC + BF16 HMMA kernel path.

      Step 4 — OPT-4: Enable TF32 flags (max-autotune mode is set at the
        torch.compile() call site in the profiling driver).

    Each step checks the current state of the model/input before applying so
    that double-applying is safe if this function is called on an already-
    optimized model.
    """
    assert torch.cuda.is_available(), "CUDA required"

    model, x = _baseline_get_model_and_input()

    # ------------------------------------------------------------------
    # Step 1: BN fold (OPT-3) — FP32, before any layout/dtype cast
    # ------------------------------------------------------------------
    logger.info("get_model_and_input: applying OPT-3 (BN fold)")
    model = fold_all_bn(model)
    logger.info("get_model_and_input: OPT-3 complete")

    # ------------------------------------------------------------------
    # Step 2: channels_last (OPT-1)
    # ------------------------------------------------------------------
    param0 = next(model.parameters())
    if not param0.is_contiguous(memory_format=torch.channels_last):
        logger.info("get_model_and_input: applying OPT-1 (channels_last)")
        model = model.to(memory_format=torch.channels_last)
    if not x.is_contiguous(memory_format=torch.channels_last):
        x = x.to(memory_format=torch.channels_last)

    # ------------------------------------------------------------------
    # Step 3: BF16 cast (OPT-2)
    # ------------------------------------------------------------------
    if next(model.parameters()).dtype != torch.bfloat16:
        logger.info("get_model_and_input: applying OPT-2 (BF16)")
        model = model.to(torch.bfloat16)
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)

    # ------------------------------------------------------------------
    # Step 4: TF32 flags (OPT-4 partial — max-autotune set at compile time)
    # ------------------------------------------------------------------
    logger.info("get_model_and_input: applying OPT-4 (TF32 flags)")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    logger.info(
        "get_model_and_input: ready — dtype=%s, channels_last=%s",
        next(model.parameters()).dtype,
        next(model.parameters()).is_contiguous(memory_format=torch.channels_last),
    )
    return model, x


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    m, x = get_model_and_input()
    compiled = torch.compile(m, backend="conv_block_opt", mode="max-autotune")
    with torch.no_grad():
        y = compiled(x)
    print(f"Output shape : {y.shape}")
    print(f"Output dtype : {y.dtype}")
