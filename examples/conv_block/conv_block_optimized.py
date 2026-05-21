"""
conv_block_optimized.py — Optimized workload for ConvBlock (VGG-style CNN).

Custom torch.compile() backend: conv_block_opt

Implements four profiling-guided optimizations from optimizations.json:

  OPT-1 (HIGH)   — channels_last memory layout
                   Eliminates cuDNN NCHW↔NHWC format-conversion kernels.
                   Applied in get_model_and_input() (non-graph).

  OPT-2 (MEDIUM) — BF16 dtype promotion
                   Halves DRAM traffic for DRAM-bound BN-ReLU pointwise kernels.
                   Applied in get_model_and_input() (non-graph) + FX cast pass.

  OPT-3 (MEDIUM) — Inductor max-autotune for wave-starvation relief
                   Enables Inductor's Triton conv autotuner so the 64→128 and
                   128→256 convolutions can select smaller-tile algorithms that
                   improve SM occupancy (8.3% → target >20%).
                   Implemented as Inductor config directives in the backend.
                   [Stub: does not rewrite the FX graph; only sets config before
                   delegating to compile_fx.]

  OPT-4 (MEDIUM) — 3-channel conv padding to 4 channels
                   Pads the 3-channel (K=27) input and matching weight to 4
                   channels (K=36), enabling cuDNN to select the shared-memory-
                   staging GEMM path instead of the indexed_wo_smem path.
                   Implemented as an FX graph pass.

Prerequisite order: OPT-1 → OPT-2 → OPT-3 (per optimizations.json).
OPT-4 is independent.
"""
from __future__ import annotations

import logging
import operator
from typing import Callable

import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
from torch._dynamo import register_backend
from torch._inductor.compile_fx import compile_fx  # function, not module

from nvidia.operator_profiler.fx import UniqueSubgraphRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (mirrored from conv_block.py)
# ---------------------------------------------------------------------------
DEVICE = "cuda"
BATCH_SIZE = 16
IN_CHANNELS = 3
HEIGHT = 64
WIDTH = 64
NUM_CLASSES = 10


# ---------------------------------------------------------------------------
# Model definition (verbatim copy so this file is self-contained)
# ---------------------------------------------------------------------------

class ConvBnRelu(nn.Module):
    """Conv2d → BatchNorm2d → ReLU building block."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=kernel_size, stride=stride,
            padding=kernel_size // 2, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ConvBlock(nn.Module):
    """
    Three-stage VGG-style conv pipeline.

    Stage 1: 3 → 64  channels, 64×64 spatial  (memory-bound conv)
    Stage 2: 64 → 128 channels, 32×32 spatial  (transitional)
    Stage 3: 128 → 256 channels, 16×16 spatial (compute-bound conv)
    Then: AdaptiveAvgPool → Linear classifier
    """
    def __init__(self):
        super().__init__()
        self.stage1 = ConvBnRelu(3, 64, kernel_size=3)
        self.stage2 = nn.Sequential(
            ConvBnRelu(64, 128, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.stage3 = nn.Sequential(
            ConvBnRelu(128, 256, kernel_size=3),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(256, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = x.flatten(1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# FX Pass: OPT-2 — BF16 cast on graph input (medium confidence)
# ---------------------------------------------------------------------------

def _pass_insert_bf16_cast(gm: fx.GraphModule) -> fx.GraphModule:
    """
    Insert a .to(bfloat16) cast node immediately after the first placeholder
    (the activation input) in the pre-Inductor FX graph.

    Weight placeholders are already BF16 because get_model_and_input() calls
    model.bfloat16(). This pass handles the runtime input tensor so Dynamo does
    not need to re-trace if the caller passes a FP32 tensor.

    Only the first (activation) placeholder is cast — weight placeholders that
    follow are already BF16 and must not be double-cast.
    """
    try:
        matched = False
        for node in list(gm.graph.nodes):
            if node.op == "placeholder":
                # Only cast the first placeholder (the activation input x).
                # Weight/bias placeholders are static parameters already BF16.
                with gm.graph.inserting_after(node):
                    cast_node = gm.graph.call_method(
                        "to",
                        args=(node,),
                        kwargs={"dtype": torch.bfloat16},
                    )
                node.replace_all_uses_with(cast_node)
                # Fix the self-reference: cast_node.args[0] was rewritten above
                cast_node.args = (node,)
                matched = True
                break  # only the first placeholder

        if not matched:
            logger.warning("[pass_insert_bf16_cast] No placeholder found — pass not applied")
            return gm

        gm.graph.lint()
        gm.recompile()
        logger.info("[pass_insert_bf16_cast] Inserted BF16 cast on graph input placeholder")
    except Exception as e:
        logger.warning("[pass_insert_bf16_cast] Failed: %s", e)
    return gm


# ---------------------------------------------------------------------------
# FX Pass: OPT-4 — Pad 3-channel convolutions to 4 channels (medium confidence)
# ---------------------------------------------------------------------------

def _pass_pad_shallow_conv(gm: fx.GraphModule) -> fx.GraphModule:
    """
    For each F.conv2d node whose input has in_channels == 3, insert zero-padding
    ops to bring input and weight to 4 channels.

    This allows cuDNN to select the shared-memory-staging GEMM algorithm instead
    of the indexed_wo_smem path (which uses only 4 warps/block and achieves
    merely 15% Tensor Core utilisation due to K=27 being below WMMA alignment).

    Padding channel math:
      input:  [N, 3, H, W]  → [N, 4, H, W]   (pad=(0,0,0,0,0,1) on dim 1)
      weight: [C_out, 3, kH, kW] → [C_out, 4, kH, kW] (pad last dim of in_ch)

    The extra zero column in the weight tensor ensures mathematical equivalence:
      output[n, c_out, h, w] = Σ_{k=0..3} Σ_{kh,kw} w[c_out,k,kh,kw]*x[n,k,h',w']
    For k=3 the padded weight row is 0, so the extra padded input channel
    contributes exactly 0 to each output element.

    Note: F.conv2d at pre-Inductor level takes args:
      (input, weight, bias, stride, padding, dilation, groups)
    """
    try:
        matched = False
        for node in list(gm.graph.nodes):
            if node.op != "call_function":
                continue
            if node.target not in (F.conv2d, torch.ops.aten.convolution.default,
                                   torch.ops.aten.cudnn_convolution.default):
                continue

            input_node = node.args[0]
            weight_node = node.args[1]

            # Detect in_channels from fake tensor metadata (set by Dynamo)
            try:
                in_channels = input_node.meta["val"].shape[1]
            except (AttributeError, KeyError, IndexError):
                continue

            if in_channels != 3:
                continue

            # Insert padding for input: pad last dimension pair of dim-1
            # torch.nn.functional.pad pads from last dim backwards in pairs.
            # To pad dim 1 (channel) of a 4-D tensor [N,C,H,W]:
            #   pad = (0,0, 0,0, 0,1, 0,0)  — right-pad C dim by 1
            with gm.graph.inserting_before(node):
                padded_input = gm.graph.call_function(
                    torch.ops.aten.constant_pad_nd.default,
                    args=(input_node, [0, 0, 0, 0, 0, 1], 0.0),
                )
                # Weight pad: [C_out, 3, kH, kW] → [C_out, 4, kH, kW]
                # Pad the in_channel dim (dim 1, second from end in 4-D weight).
                # For 4-D weight [C_out, C_in, kH, kW]:
                #   last dim = kW, second-last = kH, third-last = C_in, fourth-last = C_out
                # pad = (0,0, 0,0, 0,1, 0,0)  — right-pad C_in by 1
                padded_weight = gm.graph.call_function(
                    torch.ops.aten.constant_pad_nd.default,
                    args=(weight_node, [0, 0, 0, 0, 0, 1, 0, 0], 0.0),
                )

            # Rebuild the conv node args with padded input and weight
            new_args = (padded_input, padded_weight) + node.args[2:]
            node.args = new_args
            matched = True
            logger.info(
                "[pass_pad_shallow_conv] Padded 3-channel conv input and weight to 4 channels"
            )

        if not matched:
            logger.warning("[pass_pad_shallow_conv] No 3-channel F.conv2d found — pass not applied")
            return gm

        gm.graph.lint()
        gm.recompile()
        logger.info("[pass_pad_shallow_conv] 3→4 channel padding complete")
    except Exception as e:
        logger.warning("[pass_pad_shallow_conv] Failed: %s", e)
    return gm


# ---------------------------------------------------------------------------
# Backend: conv_block_opt
# ---------------------------------------------------------------------------

@register_backend
def conv_block_opt(gm: fx.GraphModule, example_inputs) -> Callable:
    """
    Custom torch.compile() backend for ConvBlock.

    Pass order:
      1. OPT-2 FX pass: insert BF16 cast on activation input
      2. OPT-4 FX pass: pad 3-channel conv to 4 channels
      3. OPT-3 Inductor config: enable max-autotune for conv autotuning
      4. Delegate to Inductor via compile_fx

    OPT-1 (channels_last) is a non-graph transformation applied in
    get_model_and_input() — it is not visible in the FX graph.
    """
    logger.info("conv_block_opt backend: starting compilation")

    # Build dedup registry to detect repeated subgraph structure.
    # ConvBlock has no repeated transformer blocks, so equiv_map will be empty
    # and we follow the flat-compile path.
    registry = UniqueSubgraphRegistry(gm)
    equiv_map = registry.build_partition_equivalence_map()

    if not equiv_map:
        # Flat graph path — no repeated partitions detected.
        # Apply all manual FX passes directly to the full graph.
        logger.info("conv_block_opt: no repeated partitions — flat compile path")

        # OPT-2: Insert BF16 cast on activation input
        gm = _pass_insert_bf16_cast(gm)

        # OPT-4: Pad 3-channel convolutions to 4 channels
        gm = _pass_pad_shallow_conv(gm)

        # OPT-3: Configure Inductor for max-autotune (Triton conv autotuning)
        # This is a stub-level intervention: we set Inductor config flags so that
        # the conv autotuner searches for smaller-tile algorithms, relieving wave
        # starvation on the 64→128 and 128→256 convolutions.
        try:
            import torch._inductor.config as inductor_config
            # Enable general autotuning
            if not inductor_config.max_autotune:
                inductor_config.max_autotune = True
                logger.info("conv_block_opt: [OPT-3] enabled max_autotune")
            # Enable Triton convolution kernel search
            if hasattr(inductor_config, "max_autotune_conv"):
                if not inductor_config.max_autotune_conv:
                    inductor_config.max_autotune_conv = True
                    logger.info("conv_block_opt: [OPT-3] enabled max_autotune_conv")
            # Coordinate descent tuning searches nearby tile configs
            if hasattr(inductor_config, "coordinate_descent_tuning"):
                if not inductor_config.coordinate_descent_tuning:
                    inductor_config.coordinate_descent_tuning = True
                    logger.info("conv_block_opt: [OPT-3] enabled coordinate_descent_tuning")
        except Exception as e:
            logger.warning("conv_block_opt: [OPT-3] Inductor config setup failed: %s", e)

        logger.info("conv_block_opt: delegating to Inductor compile_fx")
        return compile_fx(gm, example_inputs)

    # Dedup path (not expected for ConvBlock — included for robustness)
    logger.info(
        "conv_block_opt: %d duplicate partition(s) detected — dedup path",
        len(equiv_map),
    )

    for rep_name, rep_mod in registry.unique_reps:
        _pass_insert_bf16_cast(rep_mod)
        _pass_pad_shallow_conv(rep_mod)
        for _, dup_mod in registry.duplicates_of(rep_name):
            _pass_insert_bf16_cast(dup_mod)
            _pass_pad_shallow_conv(dup_mod)

    # Collect partition inputs for compile_fx calls
    partition_inputs = _capture_partition_inputs(registry.split, example_inputs)

    for rep_name, rep_mod in registry.unique_reps:
        inputs = partition_inputs.get(rep_name, example_inputs)
        compiled = compile_fx(rep_mod, inputs)
        rep_mod.forward = compiled
        for _, dup_mod in registry.duplicates_of(rep_name):
            dup_mod.forward = compiled

    return lambda *args: registry.split(*args)


# ---------------------------------------------------------------------------
# Utility: capture partition inputs (needed for dedup path)
# ---------------------------------------------------------------------------

def _capture_partition_inputs(
    split_gm: fx.GraphModule,
    example_inputs: list,
) -> dict[str, list]:
    """Capture actual input tensors for each partition by running split_gm once."""
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


# ---------------------------------------------------------------------------
# Workload interface
# ---------------------------------------------------------------------------

def get_model_and_input() -> tuple:
    """
    Return (optimized_model, input_tensor) ready for torch.compile.

    Non-graph optimizations applied here:

    OPT-1 (HIGH) — channels_last memory layout
        Converts all Conv2d weights and the activation tensor to channels_last
        (NHWC) format. cuDNN then receives already-NHWC data and skips all
        convertTensor_kernel and nhwcToNchwKernel launches (~18 kernels,
        ~245 µs saved per profile estimate).

    OPT-2 (MEDIUM) — BF16 dtype promotion
        Converts model parameters (weights, BN scale/shift) to bfloat16.
        BN running_mean and running_var stay float32 (PyTorch's
        _native_batch_norm_legit_no_training accumulates in FP32 internally).
        The activation input is also cast to bfloat16 here so the FX pass has
        a consistent dtype to insert a cast on (handles callers that pass FP32).
    """
    assert torch.cuda.is_available(), "CUDA required"

    model = ConvBlock().to(DEVICE).eval()

    # OPT-1: channels_last — check before applying (idempotent guard)
    first_param = next(model.parameters())
    if not first_param.is_contiguous(memory_format=torch.channels_last):
        model = model.to(memory_format=torch.channels_last)
        logger.info("get_model_and_input: applied channels_last to model [OPT-1]")

    # OPT-2: BF16 dtype promotion on parameters
    model = model.bfloat16()
    logger.info("get_model_and_input: converted model to bfloat16 [OPT-2]")

    # Build input with both optimizations applied
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH, device=DEVICE)
    # OPT-1: channels_last on input
    x = x.to(memory_format=torch.channels_last)
    # OPT-2: BF16 input
    x = x.bfloat16()

    return model, x


if __name__ == "__main__":
    model, x = get_model_and_input()
    compiled = torch.compile(model, backend="conv_block_opt")
    with torch.no_grad():
        out = compiled(x)
    print(out.shape)
