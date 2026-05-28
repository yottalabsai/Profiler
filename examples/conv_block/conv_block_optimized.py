"""
conv_block_optimized.py — VGG-style CNN with a custom torch.compile() backend.

Source workload: conv_block.py (three Conv2d→BatchNorm2d→ReLU stages, MaxPool,
AdaptiveAvgPool, Linear head; eval/inference, batch 16).

Backend name (registered with torch._dynamo via @register_backend):
    conv_block_opt

Optimizations implemented (from optimizations.json):
    OPT-1  Conv-BN fold (high)            — Aten IR pass in _aten_fw_compiler
    OPT-2  channels_last layout (high)    — non-graph (get_model_and_input) + Aten IR verify pass
    OPT-3  ReLU → clamp_min epilogue (med)— Aten IR pass in _aten_fw_compiler
    OPT-4  classifier addmm              — documented no-op (NOT implemented)

All graph passes run at the Aten IR level inside _aten_fw_compiler, which
aot_autograd calls with the fully decomposed graph. nn.Module parameters are
placeholder nodes at this level; their tensors are positionally matched from
fw_example_inputs.

compile_mode is "inductor" (per optimizations.json analysis.compile_mode), so
this writes a full FX-pass backend.
"""
from __future__ import annotations

import logging
import operator
from functools import partial
from typing import Callable

import torch
import torch.fx as fx
import torch.nn as nn
from torch._dynamo import register_backend
from torch._dynamo.backends.common import aot_autograd
from torch._inductor.compile_fx import compile_fx  # function, not module
from torch._subclasses.fake_tensor import unset_fake_temporarily

from nvidia.operator_profiler.fx import UniqueSubgraphRegistry

logger = logging.getLogger(__name__)

DEVICE      = "cuda"
BATCH_SIZE  = 16
IN_CHANNELS = 3
HEIGHT      = 64
WIDTH       = 64
NUM_CLASSES = 10

_BN_TARGET = torch.ops.aten._native_batch_norm_legit_no_training.default
_CONV_TARGETS = frozenset({
    torch.ops.aten.convolution.default,
    torch.ops.aten.cudnn_convolution.default,
})


# ----------------------------------------------------------------------------
# Model definition (identical architecture to conv_block.py)
# ----------------------------------------------------------------------------
class ConvBnRelu(nn.Module):
    """Conv2d → BatchNorm2d → ReLU building block."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=kernel_size, stride=stride,
            padding=kernel_size // 2, bias=False,
        )
        self.bn  = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ConvBlock(nn.Module):
    """Three-stage VGG-style conv pipeline."""
    def __init__(self):
        super().__init__()
        self.stage1 = ConvBnRelu(3,   64,  kernel_size=3)
        self.stage2 = nn.Sequential(
            ConvBnRelu(64,  128, kernel_size=3),
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


# ----------------------------------------------------------------------------
# OPT-1: Conv-BN fold (high confidence, Aten IR pass)
# ----------------------------------------------------------------------------
def _pass_fold_bn(gm: fx.GraphModule, ph_to_tensor: dict) -> fx.GraphModule:
    """Fold each eval-mode BatchNorm2d into the preceding convolution.

    scale       = bn_weight / sqrt(running_var + eps)
    W_folded    = conv_weight * scale.view(-1, 1, 1, 1)
    b_folded    = bn_bias - running_mean * scale   (conv had bias=False)

    aten._native_batch_norm_legit_no_training returns a 3-tuple
    (output, save_mean, save_rstd); the live output is getitem(bn, 0).

    Confidence: high — assume the conv→BN adjacency exists; an exception is a
    real error (warn + return gm unchanged).
    """
    try:
        folded = 0
        for bn_node in list(gm.graph.nodes):
            if not (bn_node.op == "call_function" and bn_node.target is _BN_TARGET):
                continue

            conv_node = bn_node.args[0]
            if not (conv_node.op == "call_function" and conv_node.target in _CONV_TARGETS):
                continue

            # Conv must be the sole consumer feeding this BN (folding identity
            # assumes one downstream BN per conv).
            if len(conv_node.users) != 1:
                logger.warning(
                    "[pass_fold_bn] conv %s has %d users — not folding",
                    conv_node.name, len(conv_node.users),
                )
                continue

            # aten._native_batch_norm_legit_no_training(
            #     input, weight, bias, running_mean, running_var, momentum, eps)
            bn_weight = ph_to_tensor.get(bn_node.args[1])
            bn_bias   = ph_to_tensor.get(bn_node.args[2])
            run_mean  = ph_to_tensor.get(bn_node.args[3])
            run_var   = ph_to_tensor.get(bn_node.args[4])
            eps       = bn_node.args[6] if len(bn_node.args) > 6 else 1e-5
            if any(t is None for t in (bn_weight, bn_bias, run_mean, run_var)):
                logger.warning("[pass_fold_bn] BN params not in fw_example_inputs — skipping")
                continue

            # aten.convolution.default args:
            #   (input, weight, bias, stride, padding, dilation,
            #    transposed, output_padding, groups)
            conv_weight = ph_to_tensor.get(conv_node.args[1])
            conv_bias   = (ph_to_tensor.get(conv_node.args[2])
                           if conv_node.args[2] is not None else None)
            if conv_weight is None:
                logger.warning("[pass_fold_bn] conv weight not in fw_example_inputs — skipping")
                continue

            # CRITICAL (two distinct fake-tensor hazards):
            #
            # (1) conv_weight/bn_* MUST be the REAL parameter tensors, NOT the
            #     FakeTensors aot_autograd passes as fw_example_inputs. ph_to_tensor
            #     is built from the REAL dynamo-level example_inputs (see
            #     _aten_fw_compiler), so these are genuine CUDA tensors.
            #
            # (2) The fold runs while compile_fx's FakeTensorMode is still ACTIVE
            #     on the dispatch stack. fuse_conv_bn_weights internally calls
            #     aten.zeros_like / arithmetic, which the active FakeTensorMode
            #     intercepts and rejects ("convert all Tensors to FakeTensors
            #     first") because the inputs are real. unset_fake_temporarily()
            #     pops fake mode for the duration of the real-tensor math, so the
            #     folded weight/bias are computed eagerly and stay real.
            #
            # Folding under fake mode (either hazard) bakes a FakeTensor constant
            # into the graph; Inductor's runtime then fails at .data_ptr()
            # ("Cannot access data pointer of FakeTensor").
            with unset_fake_temporarily():
                fused_w, fused_b = torch.nn.utils.fuse_conv_bn_weights(
                    conv_weight, conv_bias, run_mean, run_var, eps, bn_weight, bn_bias,
                )
                # Detach from autograd and own the storage before registering as
                # a constant buffer.
                fused_w = fused_w.detach().clone()
                fused_b = fused_b.detach().clone()
                # Keep folded weights in the conv input's memory format so OPT-2
                # (channels_last) does not re-trigger a one-time layout convert.
                if conv_weight.is_contiguous(memory_format=torch.channels_last):
                    fused_w = fused_w.contiguous(memory_format=torch.channels_last)

            w_name = f"_folded_conv_weight_{folded}"
            b_name = f"_folded_conv_bias_{folded}"
            gm.register_buffer(w_name, fused_w)
            gm.register_buffer(b_name, fused_b)

            with gm.graph.inserting_before(conv_node):
                fw = gm.graph.get_attr(w_name)
                fb = gm.graph.get_attr(b_name)

            new_conv_args = (conv_node.args[0], fw, fb) + tuple(conv_node.args[3:])
            with gm.graph.inserting_before(bn_node):
                new_conv = gm.graph.call_function(
                    conv_node.target, new_conv_args, dict(conv_node.kwargs),
                )

            # Re-route getitem(bn, 0) consumers to the folded conv output.
            for user in list(bn_node.users):
                if (user.op == "call_function"
                        and user.target is operator.getitem
                        and user.args[1] == 0):
                    user.replace_all_uses_with(new_conv)
                    gm.graph.erase_node(user)

            if not bn_node.users:
                gm.graph.erase_node(bn_node)
            if not conv_node.users:
                gm.graph.erase_node(conv_node)
            folded += 1

        if folded:
            gm.graph.eliminate_dead_code()
            gm.graph.lint()
            gm.recompile()
            logger.info("[pass_fold_bn] Folded %d BatchNorm into Conv2d [Aten IR]", folded)
        else:
            logger.warning("[pass_fold_bn] No conv→BN pair found — pass not applied")
    except Exception as e:
        logger.warning("[pass_fold_bn] Failed: %s", e)
    return gm


# ----------------------------------------------------------------------------
# OPT-3: ReLU → clamp_min epilogue (medium confidence, Aten IR pass)
# ----------------------------------------------------------------------------
def _pass_relu_to_clamp_min(gm: fx.GraphModule) -> fx.GraphModule:
    """Rewrite aten.relu.default consuming a (folded) conv output to
    aten.clamp_min.default(x, 0), which Inductor schedules into the conv-output
    pointwise group rather than a standalone full-tensor launch.

    Depends on OPT-1: after BN folding the conv→relu adjacency exists.

    Confidence: medium — include the `matched` no-op guard; the win overlaps
    Inductor's default epilogue fusion, so absence of the pattern is benign.
    """
    try:
        matched = False
        for node in list(gm.graph.nodes):
            if not (node.op == "call_function" and node.target is torch.ops.aten.relu.default):
                continue
            producer = node.args[0]
            if not (producer.op == "call_function" and producer.target in _CONV_TARGETS):
                continue
            with gm.graph.inserting_after(node):
                fused = gm.graph.call_function(
                    torch.ops.aten.clamp_min.default, (producer, 0),
                )
            node.replace_all_uses_with(fused)
            gm.graph.erase_node(node)
            matched = True

        if not matched:
            logger.warning(
                "[pass_relu_to_clamp_min] No relu-on-conv pattern found — pass not applied"
            )
            return gm
        gm.graph.lint()
        gm.recompile()
        logger.info("[pass_relu_to_clamp_min] Rewrote relu → clamp_min on conv epilogue [Aten IR]")
    except Exception as e:
        logger.warning("[pass_relu_to_clamp_min] Failed: %s", e)
    return gm


# ----------------------------------------------------------------------------
# OPT-2 (verification half): assert channels_last propagated, no NCHW reinsert
# ----------------------------------------------------------------------------
def _pass_verify_channels_last(gm: fx.GraphModule) -> fx.GraphModule:
    """Non-mutating verification pass for OPT-2.

    The layout switch itself is non-graph (get_model_and_input). This pass only
    confirms no aten.clone / aten.contiguous forcing contiguous (NCHW) format
    was reinserted between conv nodes, which would re-trigger convertTensor
    reformatting. Never transforms the graph.
    """
    try:
        suspect = 0
        for n in gm.graph.nodes:
            if n.op != "call_function":
                continue
            if n.target in (torch.ops.aten.clone.default, torch.ops.aten.contiguous.default):
                mf = n.kwargs.get("memory_format", None)
                if mf in (None, torch.contiguous_format):
                    suspect += 1
        if suspect:
            logger.warning(
                "[pass_verify_channels_last] %d clone/contiguous(NCHW) node(s) present — "
                "channels_last may not be fully propagated", suspect,
            )
        else:
            logger.info("[pass_verify_channels_last] No NCHW re-layout reinserted [Aten IR]")
    except Exception as e:
        logger.warning("[pass_verify_channels_last] Failed: %s", e)
    return gm


# ----------------------------------------------------------------------------
# Aten IR forward compiler — all graph passes run here
# ----------------------------------------------------------------------------
def _aten_fw_compiler(gm: fx.GraphModule, fw_example_inputs, real_inputs=None) -> Callable:
    """Receives the fully decomposed Aten IR graph from aot_autograd.

    Pass order (Rule 6 + optimizations.json dependency DAG):
      OPT-1 (fold BN)  -> OPT-3 (relu→clamp_min) -> OPT-2 verify
    OPT-1 first: node-count-reducing fusion; creates the conv→relu adjacency
    that OPT-3 depends on. OPT-2's layout switch is non-graph; only its verify
    half runs here, last, after the graph shape is final.

    `fw_example_inputs` are FakeTensors (aot_autograd traces under FakeTensorMode).
    Any pass that READS weight VALUES (OPT-1 BN fold) must use `real_inputs` —
    the genuine Parameter/buffer tensors captured at the dynamo backend level —
    so the folded constants it materializes are real tensors. fw_example_inputs
    are still handed to compile_fx unchanged for Inductor's meta tracing.

    real_inputs and fw_example_inputs share the same length and positional order
    (both correspond 1:1 to the aten graph placeholders); real_inputs falls back
    to fw_example_inputs only if the backend did not supply it.
    """
    weight_source = real_inputs if real_inputs is not None else fw_example_inputs
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    ph_to_tensor = {ph: t for ph, t in zip(placeholders, weight_source)}

    gm = _pass_fold_bn(gm, ph_to_tensor)        # OPT-1 (weight-access pass)
    gm = _pass_relu_to_clamp_min(gm)            # OPT-3 (op-target pass)
    gm = _pass_verify_channels_last(gm)         # OPT-2 verification (no-op)

    # compile_fx returns a callable that uses Inductor's BOXED calling
    # convention: it expects a single list/tuple of inputs, e.g. f([a, b, c]).
    # aot_autograd's runtime, however, invokes the fw_compiler result with
    # UNPACKED positional args, f(a, b, c). Without bridging the two, the 18
    # flat args arrive as one positional list and Inductor's
    # copy_misaligned_inputs trips on `Expected tensors only, but got: list`.
    # Re-box the args here so the calling conventions line up.
    compiled = compile_fx(gm, fw_example_inputs)

    def _boxed_adapter(*args):
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            return compiled(*args[0])
        return compiled(*args)

    return _boxed_adapter


def _capture_partition_inputs(split_gm: fx.GraphModule, example_inputs: list) -> dict:
    """Capture actual input tensors for each partition by running split_gm once."""
    partition_inputs: dict = {}
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


# ----------------------------------------------------------------------------
# Registered backend
# ----------------------------------------------------------------------------
@register_backend
def conv_block_opt(gm: fx.GraphModule, example_inputs) -> Callable:
    """Custom torch.compile backend for ConvBlock.

    ConvBlock has three structurally DISTINCT ConvBnRelu stages (3→64, 64→128,
    128→256), so no repeated-layer dedup is expected; the flat compile path is
    taken, preserving cross-layer Inductor fusion. The dedup path (Rule 9) is
    retained for robustness if the registry detects equivalence classes.
    """
    logger.info("conv_block_opt backend: starting")
    registry = UniqueSubgraphRegistry(gm)
    equiv_map = registry.build_partition_equivalence_map()

    if not equiv_map:
        logger.info("conv_block_opt: no repeated layers, flat compile path")
        # example_inputs are REAL Parameter/Tensor objects at the dynamo level;
        # thread them to _aten_fw_compiler so weight-reading passes (OPT-1) fold
        # against real values, not the FakeTensors aot_autograd traces with.
        fw = partial(_aten_fw_compiler, real_inputs=example_inputs)
        return aot_autograd(fw_compiler=fw)(gm, example_inputs)

    logger.info("conv_block_opt: %d duplicate partition(s), dedup path", len(equiv_map))
    partition_inputs = _capture_partition_inputs(registry.split, example_inputs)
    for rep_name, rep_mod in registry.unique_reps:
        inputs = partition_inputs.get(rep_name, example_inputs)
        fw = partial(_aten_fw_compiler, real_inputs=inputs)
        compiled = aot_autograd(fw_compiler=fw)(rep_mod, inputs)
        rep_mod.forward = compiled
        for _, dup_mod in registry.duplicates_of(rep_name):
            dup_mod.forward = compiled

    return lambda *args: registry.split(*args)


# ----------------------------------------------------------------------------
# Workload interface
# ----------------------------------------------------------------------------
def get_model_and_input() -> tuple:
    """Return (raw_model, input_tensor) on CUDA — uncompiled, unwarmed.

    OPT-2 (non-graph half): convert the module and input to channels_last (NHWC)
    so data stays in cuDNN's preferred layout end-to-end, eliminating the
    convertTensor NCHW<->NHWC reformatting kernels around every conv.
    Checked-before-applied per Rule 7 (baseline may already be channels_last).
    """
    assert torch.cuda.is_available(), "CUDA required"
    model = ConvBlock().to(DEVICE).eval()
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH, device=DEVICE)

    # OPT-2: channels_last (NHWC) layout — non-graph optimization.
    if not next(model.parameters()).is_contiguous(memory_format=torch.channels_last):
        model = model.to(memory_format=torch.channels_last)
    if not x.is_contiguous(memory_format=torch.channels_last):
        x = x.to(memory_format=torch.channels_last)

    return model, x


if __name__ == "__main__":
    model, x = get_model_and_input()
    compiled = torch.compile(model, backend="conv_block_opt")
    with torch.no_grad():
        out = compiled(x)
    print(out.shape)
