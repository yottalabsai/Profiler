"""
depthwise_separable_conv_optimized.py — Custom torch.compile() backend for
DepthwiseSepConv (MobileNet-style depthwise-separable conv blocks).

Registered backend: ``depthwise_sep_conv_opt``
(capture stage drives it with ``--compile-backend=depthwise_sep_conv_opt``)

Implements the four optimizations from optimizations.json. Dependency DAG
(from the proposal global_notes): OPT-1 -> {OPT-2, OPT-3, OPT-4}, OPT-2 -> OPT-4.
Apply order: OPT-1, OPT-2, OPT-3, OPT-4.

  OPT-1  dtype_promotion  — Promote the 1x1 POINTWISE convolutions (weight shape
                            [Cout,Cin,1,1], groups==1) to bfloat16 at Aten IR: cast the
                            activation + weight to bf16 before the conv and cast the
                            result back to fp32 after. This routes Inductor's conv-as-mm
                            lowering to a bf16 tensor-core GEMM template autotuned for
                            sm_120 instead of the prebuilt sm_80 cutlass s1688 (TF32)
                            kernel that pins occupancy at ~8%. Paired with
                            ``max_autotune_gemm`` so the GEMM is retuned for Blackwell.
                            Confidence: medium (changes numerics fp32->bf16). MUST run
                            first so OPT-2's folded weight buffers materialize at the
                            bf16 runtime dtype. Depthwise (groups==C) convs stay fp32.
  OPT-2  fusion           — Conv-BatchNorm folding (inference). Matches the
                            ``aten.convolution -> _native_batch_norm_legit_no_training``
                            pair at Aten IR, bakes the BN per-channel affine into the
                            conv weight/bias from the REAL parameter tensors (threaded via
                            ``real_inputs``), registers folded buffers at the conv's
                            runtime dtype (bf16 for pointwise after OPT-1, fp32 for
                            depthwise), rewires a fresh biased conv, and deletes the BN.
                            Eliminates every standalone DRAM-bound BN+ReLU6 Triton kernel.
                            Confidence: high. Numerically exact. Prerequisite for OPT-4.
  OPT-3  memory_layout    — channels_last (NHWC) propagation. Primary lever is the
                            eager-side model + input ``.to(memory_format=channels_last)``
                            in get_model_and_input() (Rule 7): the conv/GEMM kernels then
                            consume native NHWC without per-block permute kernels. The
                            graph pass strips now-redundant ``aten.clone`` /
                            ``aten._to_copy(memory_format=channels_last)`` copies whose
                            source is already NHWC. Confidence: medium. Runs AFTER OPT-1
                            so the folded/bf16 conv nodes are the ones re-evaluated.
  OPT-4  fusion           — Depthwise ReLU6 (hardtanh) epilogue fusion. Primary lever is
                            config-level (max_autotune + max_autotune_conv_backends=
                            'TRITON,ATEN') so the scheduler lowers the depthwise conv to a
                            Triton template and fuses the clamp(0,6) epilogue (and the
                            OPT-2-folded bias) into it, removing one full-tensor DRAM
                            round-trip per depthwise stage. The graph pass DETECTS the
                            (depthwise conv -> hardtanh) pairs and tags the conv node meta
                            so the contract is asserted/logged. Confidence: medium. MUST
                            run AFTER OPT-2 — the ReLU6 can only fuse into the conv once BN
                            has been folded out (otherwise BN sits between conv and
                            hardtanh and blocks epilogue fusion).

IR-level mechanics (torch 2.11, RTX PRO 6000 Blackwell sm_120):
  The graph torch.compile hands a @register_backend function is the *functional* Dynamo
  graph, NOT Aten IR. Aten IR (aten.convolution,
  aten._native_batch_norm_legit_no_training, aten.hardtanh) only appears after AOTAutograd
  decomposition. We install the pass chain via ``compile_fx``'s ``inner_compile`` hook
  (Strategy D, Rule 9): ``compile_fx`` owns AOTAutograd, the decomposition table, the
  boxed calling convention, and the partitioner; we only swap the leaf compiler
  ``compile_fx_inner`` and run the Aten-IR passes just before it. The ``inner_compile``
  ``example_inputs`` may be FakeTensors, so the weight-VALUE-reading OPT-2 fold uses the
  genuine parameter tensors threaded as ``real_inputs``.

compile_mode = "inductor" (from optimizations.json): standard FX pass approach.
"""
from __future__ import annotations

import functools
import logging
import operator
from typing import Callable, Optional

import torch
import torch.fx as fx
from torch._dynamo import register_backend
from torch._inductor.compile_fx import compile_fx, compile_fx_inner  # functions, not module

from nvidia.operator_profiler.fx import UniqueSubgraphRegistry

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Global Inductor / backend config (cheap, process-scoped).
#   OPT-1: retune the bf16 pointwise GEMM template for Blackwell sm_120 instead of
#          reusing the prebuilt sm_80 cutlass s1688 (TF32) kernel.
#   OPT-4: let the scheduler lower the depthwise conv to a Triton template and fuse the
#          ReLU6 (hardtanh) + folded-bias epilogue into it.
# --------------------------------------------------------------------------- #
import torch._inductor.config as inductor_config

inductor_config.max_autotune_gemm = True            # OPT-1: retune bf16 GEMM for sm_120
inductor_config.layout_optimization = True          # OPT-3: NHWC propagation (default on)
try:
    inductor_config.max_autotune = True             # OPT-4: enable template autotune
    # OPT-4: Triton conv template enables hardtanh+bias epilogue fusion; ATEN fallback
    # keeps the faster library kernel when the Triton depthwise conv loses autotune.
    inductor_config.max_autotune_conv_backends = "TRITON,ATEN"
except Exception as _e:  # pragma: no cover - config key drift across torch versions
    logger.warning("[config] max_autotune conv backend keys unavailable: %s", _e)

# TF32 tensor cores for any residual fp32 conv/matmul (depthwise stays fp32).
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --------------------------------------------------------------------------- #
# Aten IR targets.
# --------------------------------------------------------------------------- #
_CONV = torch.ops.aten.convolution.default
_CONV_CUDNN = torch.ops.aten.cudnn_convolution.default
_CONV_TARGETS = frozenset({_CONV, _CONV_CUDNN})

_BN_NO_TRAIN = torch.ops.aten._native_batch_norm_legit_no_training.default
_HARDTANH = torch.ops.aten.hardtanh.default

_CLONE = torch.ops.aten.clone.default
_TO_COPY = torch.ops.aten._to_copy.default

# dtype-cast op form we insert / unwrap.
_TO_DTYPE = torch.ops.aten.to.dtype
_CAST_TARGETS = frozenset({_TO_DTYPE, _TO_COPY})


def _node_shape(node) -> Optional[tuple]:
    """Best-effort static shape from FX meta['val']; None if unavailable."""
    try:
        if not isinstance(node, fx.Node):
            return None
        val = node.meta.get("val", None)
        if val is not None and hasattr(val, "shape"):
            return tuple(val.shape)
    except Exception:
        pass
    return None


def _unwrap_cast(node):
    """Walk back through inserted dtype casts (aten.to.dtype / _to_copy) to the
    underlying producer/placeholder so OPT-2/OPT-4 can read the real node."""
    cur = node
    seen = 0
    while (
        isinstance(cur, fx.Node)
        and cur.op == "call_function"
        and cur.target in _CAST_TARGETS
        and seen < 4
    ):
        cur = cur.args[0]
        seen += 1
    return cur


# =========================================================================== #
# OPT-1 — dtype promotion of the 1x1 pointwise convolutions to bfloat16.
# Confidence: medium. MUST run first (prerequisite for OPT-2/3/4).
#
# aten.convolution(input, weight, bias, stride, padding, dilation,
#                  transposed, output_padding, groups)
#   pointwise  := weight.shape[2:] == (1, 1) and groups == 1
#   depthwise  := groups == C_in (left in fp32 — memory-bound, no GEMM win)
#
# Insert bf16 casts on the conv input + weight, run the conv in bf16, cast the
# output back to fp32 so the surrounding graph (and any unfolded BN) stays fp32.
# =========================================================================== #
def _pass_bf16_pointwise_conv(gm: fx.GraphModule) -> fx.GraphModule:
    """Cast the 1x1 pointwise conv operands to bf16 so Inductor lowers them to an
    autotuned bf16 tensor-core GEMM (sm_120) instead of the sm_80 s1688 TF32 kernel.

    Medium confidence: graceful no-op if no pointwise conv is present. The output is
    cast back to fp32 so OPT-2's BN fold (which may not have folded the trailing BN
    yet) and downstream consumers keep their expected dtype; consecutive bf16 stages
    are still coalesced by Inductor's dtype propagation."""
    try:
        promoted = 0
        for conv in list(gm.graph.nodes):
            if not (conv.op == "call_function" and conv.target in _CONV_TARGETS):
                continue
            groups = conv.args[8] if len(conv.args) > 8 else 1
            weight = conv.args[1]
            wshape = _node_shape(weight)
            # pointwise: 1x1 spatial kernel AND not grouped (groups == 1).
            is_pointwise = (
                groups == 1
                and wshape is not None
                and len(wshape) == 4
                and tuple(wshape[2:]) == (1, 1)
            )
            if not is_pointwise:
                continue

            x_in = conv.args[0]
            with gm.graph.inserting_before(conv):
                x_bf16 = gm.graph.call_function(_TO_DTYPE, (x_in, torch.bfloat16))
                w_bf16 = gm.graph.call_function(_TO_DTYPE, (weight, torch.bfloat16))
            conv.update_arg(0, x_bf16)
            conv.update_arg(1, w_bf16)

            with gm.graph.inserting_after(conv):
                back = gm.graph.call_function(_TO_DTYPE, (conv, torch.float32))
            conv.replace_all_uses_with(back)
            # replace_all_uses_with rewired the new cast's own input too; restore it.
            back.update_arg(0, conv)
            promoted += 1
            logger.info(
                "[OPT-1 bf16_pointwise] Promoted 1x1 pointwise conv to bf16 "
                "(Cout=%s) [Aten IR]",
                wshape[0] if wshape else "?",
            )

        if promoted:
            gm.graph.lint()
            gm.recompile()
            logger.info(
                "[OPT-1 bf16_pointwise] Applied — promoted %d pointwise conv(s) to bf16",
                promoted,
            )
        else:
            logger.warning(
                "[OPT-1 bf16_pointwise] No 1x1 pointwise (groups==1) conv found "
                "— pass not applied"
            )
    except Exception as e:  # never crash the compile
        logger.warning("[OPT-1 bf16_pointwise] Failed: %s", e)
    return gm


# =========================================================================== #
# OPT-2 — Conv-BatchNorm folding (inference). Confidence: high.
# Prerequisite for OPT-4. Runs AFTER OPT-1 so folded buffers inherit the bf16
# runtime dtype of the (already-cast) pointwise convs.
#
# aten._native_batch_norm_legit_no_training(input, weight(gamma), bias(beta),
#                                           running_mean, running_var, momentum, eps)
#   -> returns (output, save_mean, save_rstd); only getitem(node, 0) is live.
# =========================================================================== #
def _pass_fold_conv_bn(gm: fx.GraphModule, fw_example_inputs: list) -> fx.GraphModule:
    """Fold an inference-mode BatchNorm into the preceding convolution.

    Reads REAL parameter tensors (gamma/beta/running_mean/running_var, conv weight/bias)
    from ``fw_example_inputs`` (the threaded ``real_inputs`` under Strategy D, so the
    values are genuine, not FakeTensors). The conv weight arg may be an OPT-1 bf16 cast
    node and the BN input may be the OPT-1 fp32-cast of the conv output; both are
    unwrapped to reach the real placeholder / conv. Computes the numerically-exact
    inference fold:

        scale      = gamma / sqrt(running_var + eps)
        new_weight = conv_weight * scale[:, None, None, None]
        new_bias   = (conv_bias - running_mean) * scale + beta   (conv_bias=0 here)

    Folded buffers are registered at the conv's runtime dtype: bf16 for OPT-1-promoted
    pointwise convs, fp32 for depthwise convs. A fresh biased conv replaces
    ``getitem(bn, 0)``; the BN node is eliminated.

    High confidence: assume the pattern exists; a genuine error logs a warning and
    returns the graph unchanged."""
    try:
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        ph_to_tensor = {ph: t for ph, t in zip(placeholders, fw_example_inputs)}

        folded = 0
        for bn_node in list(gm.graph.nodes):
            if not (bn_node.op == "call_function" and bn_node.target is _BN_NO_TRAIN):
                continue

            # The BN input may be the OPT-1 fp32 re-cast of the conv output; unwrap it.
            conv_node = _unwrap_cast(bn_node.args[0])
            if not (
                isinstance(conv_node, fx.Node)
                and conv_node.op == "call_function"
                and conv_node.target in _CONV_TARGETS
            ):
                continue
            # Conv output must feed only this BN (possibly via the single fp32 cast).
            consumer = bn_node.args[0]
            cast_in_between = (
                isinstance(consumer, fx.Node)
                and consumer.op == "call_function"
                and consumer.target in _CAST_TARGETS
            )
            chain_tail = consumer if cast_in_between else conv_node
            if len(conv_node.users) != 1:
                continue
            if cast_in_between and len(consumer.users) != 1:
                continue

            gamma = ph_to_tensor.get(bn_node.args[1])
            beta = ph_to_tensor.get(bn_node.args[2])
            run_mean = ph_to_tensor.get(bn_node.args[3])
            run_var = ph_to_tensor.get(bn_node.args[4])
            eps = bn_node.args[6] if len(bn_node.args) > 6 else 1e-5
            if any(t is None for t in (gamma, beta, run_mean, run_var)):
                logger.warning(
                    "[OPT-2 fold_conv_bn] BN params not resolvable from real_inputs "
                    "— skipping this BN"
                )
                continue

            # Conv weight may be an OPT-1 bf16 cast node; recover the placeholder value.
            weight_arg = conv_node.args[1]
            weight_ph = _unwrap_cast(weight_arg)
            conv_weight = ph_to_tensor.get(weight_ph)
            conv_bias_node = conv_node.args[2] if len(conv_node.args) > 2 else None
            conv_bias = (
                ph_to_tensor.get(_unwrap_cast(conv_bias_node))
                if isinstance(conv_bias_node, fx.Node)
                else None
            )
            if conv_weight is None:
                logger.warning(
                    "[OPT-2 fold_conv_bn] Conv weight not resolvable — skipping"
                )
                continue

            # Compute the fold in fp32 for accuracy, then match the conv runtime dtype.
            cw32 = conv_weight.float()
            scale = gamma.float() / torch.sqrt(run_var.float() + eps)
            new_weight = cw32 * scale.reshape(-1, 1, 1, 1)
            if conv_bias is not None:
                new_bias = (conv_bias.float() - run_mean.float()) * scale + beta.float()
            else:
                new_bias = beta.float() - run_mean.float() * scale

            # Runtime dtype: bf16 if the conv operands were cast by OPT-1, else fp32.
            ran_bf16 = (
                isinstance(weight_arg, fx.Node)
                and weight_arg.op == "call_function"
                and weight_arg.target in _CAST_TARGETS
            )
            runtime_dtype = torch.bfloat16 if ran_bf16 else torch.float32
            new_weight = new_weight.to(runtime_dtype)
            new_bias = new_bias.to(runtime_dtype)

            c_out = int(new_weight.shape[0])
            wname = f"_folded_conv_weight_{folded}"
            bname = f"_folded_conv_bias_{folded}"
            gm.register_buffer(wname, new_weight)
            gm.register_buffer(bname, new_bias)

            # Fresh biased conv reusing the original conv input + spatial args. When the
            # conv ran bf16 (OPT-1) its input is already the bf16-cast activation node, so
            # the folded bf16 weight/bias match.
            new_conv_args = list(conv_node.args)
            with gm.graph.inserting_before(bn_node):
                fw = gm.graph.get_attr(wname)
                fb = gm.graph.get_attr(bname)
                new_conv_args[1] = fw
                new_conv_args[2] = fb
                new_conv = gm.graph.call_function(_CONV, tuple(new_conv_args))
                # Cast the folded output back to fp32 if the conv ran bf16, so the BN
                # consumers' dtype contract is preserved.
                conv_out = new_conv
                if ran_bf16:
                    conv_out = gm.graph.call_function(
                        _TO_DTYPE, (new_conv, torch.float32)
                    )

            # BN returns a tuple; redirect the live getitem(bn, 0) consumers.
            for user in list(bn_node.users):
                if (
                    user.op == "call_function"
                    and user.target is operator.getitem
                    and user.args[1] == 0
                ):
                    user.replace_all_uses_with(conv_out)
                    gm.graph.erase_node(user)
            if not bn_node.users:
                gm.graph.erase_node(bn_node)
            # Drop the now-dead intermediate fp32 cast (and old conv) if unused.
            if cast_in_between and not chain_tail.users:
                gm.graph.erase_node(chain_tail)
            if not conv_node.users:
                gm.graph.erase_node(conv_node)

            folded += 1
            logger.info(
                "[OPT-2 fold_conv_bn] Folded BatchNorm into conv (C_out=%d, dtype=%s) "
                "[Aten IR]",
                c_out,
                runtime_dtype,
            )

        if folded:
            gm.graph.eliminate_dead_code()
            gm.graph.lint()
            gm.recompile()
            logger.info("[OPT-2 fold_conv_bn] Applied — folded %d conv+BN pair(s)", folded)
        else:
            logger.warning(
                "[OPT-2 fold_conv_bn] No conv->_native_batch_norm_legit_no_training "
                "pair found — pass not applied"
            )
    except Exception as e:  # never crash the compile
        logger.warning("[OPT-2 fold_conv_bn] Failed: %s", e)
    return gm


# =========================================================================== #
# OPT-3 — channels_last propagation: strip redundant layout-copy nodes.
# Confidence: medium. Primary (non-graph) lever is the eager-side conversion in
# get_model_and_input(); this pass removes graph-level NCHW<->NHWC copies that are
# already no-ops once producer/consumer agree on layout. Runs AFTER OPT-1/OPT-2 so the
# folded/bf16 conv nodes are the ones whose layout is re-evaluated (DAG order).
# =========================================================================== #
def _pass_strip_layout_copies(gm: fx.GraphModule) -> fx.GraphModule:
    """Erase ``aten.clone`` / ``aten._to_copy(memory_format=channels_last)`` whose input
    is already channels_last contiguous. Medium confidence: graceful no-op if no such
    redundant copy is present (the eager-side conversion already aligned layouts)."""
    try:
        stripped = 0
        for node in list(gm.graph.nodes):
            if not (node.op == "call_function" and node.target in (_CLONE, _TO_COPY)):
                continue
            if node.kwargs.get("memory_format", None) is not torch.channels_last:
                continue
            src = node.args[0]
            try:
                meta = src.meta["val"]
                if meta.is_contiguous(memory_format=torch.channels_last):
                    node.replace_all_uses_with(src)
                    stripped += 1
            except Exception:
                continue

        if stripped:
            gm.graph.eliminate_dead_code()
            gm.graph.lint()
            gm.recompile()
            logger.info(
                "[OPT-3 strip_layout_copies] Removed %d redundant channels_last copy "
                "node(s) [Aten IR]",
                stripped,
            )
        else:
            logger.info(
                "[OPT-3 strip_layout_copies] No redundant channels_last copy nodes — "
                "layout handled eager-side in get_model_and_input()"
            )
    except Exception as e:
        logger.warning("[OPT-3 strip_layout_copies] Failed: %s", e)
    return gm


# =========================================================================== #
# OPT-4 — Depthwise ReLU6 (hardtanh) epilogue fusion. Confidence: medium.
# MUST run AFTER OPT-2 (the clamp can only fuse into the conv once BN is folded out).
#
# The PRIMARY lever is config-level (max_autotune + max_autotune_conv_backends=
# 'TRITON,ATEN', set at module load) so the scheduler lowers the depthwise conv to a
# Triton template and fuses the clamp(0,6) epilogue. This graph pass DETECTS the
# (depthwise conv -> hardtanh) pairs and tags the conv node meta so the contract is
# asserted/logged for the Inductor debug dir (no node surgery needed for fusion).
# =========================================================================== #
def _pass_mark_depthwise_relu6_fusion(gm: fx.GraphModule) -> fx.GraphModule:
    """Detect (depthwise aten.convolution -> aten.hardtanh) pairs and tag the conv for
    Triton-template lowering so the ReLU6 (and OPT-2-folded bias) epilogue fuses into the
    conv kernel. After OPT-2 the hardtanh is the conv's direct consumer; before OPT-2 a
    BN node sits between them and fusion is blocked (hence the OPT-2 -> OPT-4 ordering).

    Detection only — never mutates op semantics; medium confidence, graceful no-op."""
    try:
        marked = 0
        for ht in list(gm.graph.nodes):
            if not (ht.op == "call_function" and ht.target is _HARDTANH):
                continue
            # The hardtanh input may be a dtype cast (OPT-1 bf16 round-trip).
            conv = _unwrap_cast(ht.args[0])
            if not (
                isinstance(conv, fx.Node)
                and conv.op == "call_function"
                and conv.target in _CONV_TARGETS
            ):
                continue
            groups = conv.args[8] if len(conv.args) > 8 else 1
            wshape = _node_shape(conv.args[1])
            # depthwise: groups > 1 and weight is [C,1,k,k] (out channels == groups).
            is_depthwise = (
                isinstance(groups, int)
                and groups > 1
                and wshape is not None
                and len(wshape) == 4
                and int(wshape[0]) == groups
            )
            if not is_depthwise:
                continue
            conv.meta["fuse_relu6_epilogue"] = True
            marked += 1
            logger.info(
                "[OPT-4 depthwise_relu6_fuse] Tagged depthwise conv (C=%s) -> ReLU6 for "
                "Triton epilogue fusion [Aten IR]",
                groups,
            )

        if marked:
            logger.info(
                "[OPT-4 depthwise_relu6_fuse] Applied — %d depthwise->ReLU6 pair(s) "
                "tagged; Triton conv epilogue fusion via max_autotune_conv_backends",
                marked,
            )
        else:
            logger.warning(
                "[OPT-4 depthwise_relu6_fuse] No depthwise conv -> hardtanh pair found "
                "(BN may not be folded, or layout differs) — pass not applied"
            )
    except Exception as e:
        logger.warning("[OPT-4 depthwise_relu6_fuse] Failed: %s", e)
    return gm


# =========================================================================== #
# Strategy D — install the Aten-IR pass chain via compile_fx's inner_compile hook.
# Order respects the DAG: OPT-1, OPT-2, OPT-3, OPT-4.
# =========================================================================== #
def _aten_inner_compile(
    gm: fx.GraphModule, example_inputs, *, real_inputs=None, **kwargs
) -> Callable:
    """Inductor ``inner_compile`` hook. ``compile_fx`` calls this with the fully
    decomposed Aten IR graph (after AOTAutograd). Run the Aten-IR passes here, then
    delegate to the real ``compile_fx_inner`` (Aten -> Triton).

    ``example_inputs`` may be FakeTensors; the weight-VALUE-reading OPT-2 fold uses the
    genuine ``real_inputs`` threaded from the backend for the ph_to_tensor lookup. All
    ``**kwargs`` are forwarded verbatim to stay forward-compatible.
    """
    weight_source = real_inputs if real_inputs is not None else example_inputs
    gm = _pass_bf16_pointwise_conv(gm)              # OPT-1 (medium) — must run first
    gm = _pass_fold_conv_bn(gm, weight_source)      # OPT-2 (high)   — prereq for OPT-4
    gm = _pass_strip_layout_copies(gm)              # OPT-3 (medium) — re-eval layout
    gm = _pass_mark_depthwise_relu6_fusion(gm)      # OPT-4 (medium) — after OPT-2
    return compile_fx_inner(gm, example_inputs, **kwargs)


def _compile_with_aten_passes(gm: fx.GraphModule, example_inputs) -> Callable:
    """Compile a (sub)graph through Inductor with the Aten-IR passes installed via
    ``inner_compile``. ``compile_fx`` owns AOTAutograd, the decomp table, the boxed
    calling convention, and the partitioner — we only swap the leaf compiler. Scoped
    per call (no process-global state)."""
    inner = functools.partial(_aten_inner_compile, real_inputs=list(example_inputs))
    return compile_fx(gm, example_inputs, inner_compile=inner)


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


@register_backend
def depthwise_sep_conv_opt(gm: fx.GraphModule, example_inputs) -> Callable:
    """Custom torch.compile backend for DepthwiseSepConv.

    Installs the Aten-IR pass chain (OPT-1 bf16_pointwise, OPT-2 fold_conv_bn,
    OPT-3 strip_layout_copies, OPT-4 mark_depthwise_relu6_fusion) via ``compile_fx``'s
    ``inner_compile`` hook, then delegates AOTAutograd + lowering to ``compile_fx``.
    OPT-3's primary (non-graph) lever is the channels_last conversion in
    get_model_and_input() (Rule 7).

    Dedup-aware per Rule 9: the three DWSepBlocks have DISTINCT channel counts
    (32->64, 64->128, 128->256), so there are no structurally repeated layers and the
    flat compile path is taken (preserving cross-block Inductor fusion). The dedup
    branch is retained for structural reuse if the model grows to repeated blocks.
    """
    logger.info("depthwise_sep_conv_opt backend: starting")
    registry = UniqueSubgraphRegistry(gm)
    equiv_map = registry.build_partition_equivalence_map()

    if not equiv_map:
        logger.info("depthwise_sep_conv_opt: no repeated layers, flat compile path")
        return _compile_with_aten_passes(gm, example_inputs)

    logger.info(
        "depthwise_sep_conv_opt: %d duplicate partition(s), dedup path", len(equiv_map)
    )
    partition_inputs = _capture_partition_inputs(registry.split, example_inputs)
    for rep_name, rep_mod in registry.unique_reps:
        inputs = partition_inputs.get(rep_name, example_inputs)
        compiled = _compile_with_aten_passes(rep_mod, inputs)
        rep_mod.forward = compiled
        for _, dup_mod in registry.duplicates_of(rep_name):
            dup_mod.forward = compiled

    return lambda *args: registry.split(*args)


# =========================================================================== #
# Workload interface — non-graph optimizations live here (Rule 7).
#   OPT-3 (channels_last): model + input cast to torch.channels_last so the conv/GEMM
#          kernels run native NHWC and Inductor drops the per-block permute kernels.
#          This is the PRIMARY lever for OPT-3; the graph pass only cleans up residual
#          copies. Idempotent — checked before converting (Rule 7).
# OPT-1 (bf16 pointwise), OPT-2 (conv-BN fold), OPT-4 (depthwise ReLU6 epilogue) are
# realized as Aten-IR graph passes at compile time, NOT here; leaving the model in fp32
# with the BN intact lets the passes demonstrably transform the graph.
# =========================================================================== #
DEVICE = "cuda"
BATCH_SIZE = 16
IN_CHANNELS = 32
HEIGHT = 56
WIDTH = 56


def get_model_and_input() -> tuple:
    """Return (model, input_tensor) on CUDA — uncompiled, unwarmed.

    Applies OPT-3's non-graph half: channels_last for both the model weights and the
    input (idempotent — checked before converting per Rule 7). OPT-1 (bf16 pointwise),
    OPT-2 (conv-BN fold) and OPT-4 (depthwise ReLU6 epilogue) run inside the
    depthwise_sep_conv_opt backend at compile time.
    """
    assert torch.cuda.is_available(), "CUDA required"
    from depthwise_separable_conv import DepthwiseSepConv

    model = DepthwiseSepConv().to(DEVICE).eval()
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH, device=DEVICE)

    # OPT-3 (channels_last propagation) — eager-side, only if not already NHWC.
    if not next(model.parameters()).is_contiguous(memory_format=torch.channels_last):
        model = model.to(memory_format=torch.channels_last)
        logger.info("[OPT-3 channels_last] Model cast to channels_last (NHWC)")
    if not x.is_contiguous(memory_format=torch.channels_last):
        x = x.to(memory_format=torch.channels_last)
        logger.info("[OPT-3 channels_last] Input cast to channels_last (NHWC)")

    return model, x


if __name__ == "__main__":
    model, x = get_model_and_input()
    compiled = torch.compile(model, backend="depthwise_sep_conv_opt")
    with torch.no_grad():
        out = compiled(x)
    print("output shape:", out.shape, "dtype:", out.dtype)
