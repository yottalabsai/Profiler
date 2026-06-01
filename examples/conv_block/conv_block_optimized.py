"""
conv_block_optimized.py — Custom torch.compile() backend for ConvBlock
(VGG-style Conv2d -> BatchNorm2d -> ReLU pipeline + Linear head).

Registered backend: ``conv_block_opt``
(capture stage drives it with ``--compile-backend=conv_block_opt``)

Implements the three optimizations from optimizations.json. All three are
``ir_level: aten`` (or boundary/non-graph). Application order within the aten
level is OPT-2 (dtype) -> OPT-3 (conv-BN fold), per the proposal's
register_buffer-after-dtype-promotion rule (OPT-3 depends on OPT-2). OPT-1 is a
layout-only transform with no ordering constraint and is realized at the boundary.

  OPT-1  memory_layout   — channels_last (NHWC). The cuDNN tensor-op conv kernels
                           this workload selects are already native NHWC
                           (_nhwc_align4 / _nhwckrsc_nhwc), but the framework
                           tensors are NCHW-contiguous, so cuDNN wraps every conv
                           in two convertTensor NCHW<->NHWC transpose kernels (12
                           launches, ~49,600 ns, zero useful math). PRIMARY lever
                           is the eager-side model + input
                           ``.to(memory_format=torch.channels_last)`` in
                           get_model_and_input() (Rule 7): the framework tensors
                           become NHWC-contiguous and cuDNN consumes them directly,
                           eliminating the transposes. The aten-level FX pass is a
                           DETECT/cleanup fallback (Rule 5 canonical clone form):
                           it inserts ``aten.clone(memory_format=channels_last)`` on
                           conv activation inputs only when the boundary layout was
                           NOT set; when the eager-side conversion already aligned
                           layouts it logs a graceful no-op. Confidence: high.
                           Numerically exact, layout only.
  OPT-2  dtype_promotion — Cast the conv activation+weight operands to bfloat16 so
                           cuDNN dispatches the BF16 tensor-op path (~2x TF32
                           tensor-core rate on Blackwell + lower register pressure,
                           recovering the ~8.3% occupancy). Also halves the DRAM
                           bytes of the memory-bound BN+ReLU+pool epilogue group.
                           PRIMARY lever is the eager-side ``model.bfloat16()`` +
                           ``input.bfloat16()`` combined with OPT-1 in one boundary
                           cast. The aten-level FX pass is the FALLBACK: it wraps
                           each ``aten.cudnn_convolution.default`` /
                           ``aten.convolution.default`` operand in bf16 casts and
                           restores fp32 on the output; a graceful no-op when the
                           operands are already bf16 (boundary cast applied).
                           Confidence: medium (changes numerics). Prerequisite for
                           OPT-3 — folded buffers must materialize at the runtime
                           dtype.
  OPT-3  fusion          — Eval-mode conv-BatchNorm fold. Matches each
                           ``aten._native_batch_norm_legit_no_training`` whose input
                           is a single-consumer convolution and bakes the BN
                           per-channel affine into the conv weight/bias from the REAL
                           parameter tensors (threaded via ``real_inputs``). The
                           model's convs have bias=False, so the fold INTRODUCES a
                           bias' term; a fresh biased conv replaces the BN's
                           ``getitem(bn, 0)`` consumer and the BN node is erased.
                           Runs AFTER OPT-2 so the folded weight/bias buffers are
                           registered at the conv's runtime dtype. Confidence: low —
                           Inductor may already bake the eval-mode affine into the
                           elementwise epilogue, in which case the explicit fold is a
                           near no-op (graceful).

IR-level mechanics (torch 2.11, RTX PRO 6000 Blackwell sm_120):
  The funnel is ``_run_functional_passes(gm) -> compile_fx(inner_compile=
  _aten_inner_compile, config_patches=...)`` (Rule 9). ``compile_fx`` owns
  AOTAutograd, the decomposition table, the boxed calling convention and the
  partitioner; we only run functional passes ahead of it (none here — all opts are
  aten/boundary), swap the leaf compiler at its ``inner_compile`` seam to run the
  aten passes on the fully decomposed Aten IR graph, and delegate to
  ``compile_fx_inner`` (Aten -> Triton). ``inner_compile``'s ``example_inputs`` may
  be FakeTensors, so the weight-VALUE-reading OPT-3 fold uses the genuine parameter
  tensors threaded as ``real_inputs``. ``_repropagate_meta`` repopulates
  ``meta['val']`` on inserted nodes after each structural rewrite.

compile_mode = "dedup-inductor" (from optimizations.json analysis.compile_mode):
  standard FX pass approach; dedup-aware backend per Rule 9.
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

# TF32 tensor cores for any residual fp32 conv/matmul (the 3->64 stage-1 conv stays
# fp32-ish under sm80_xmma; BF16 helps it less, see OPT-2 notes).
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --------------------------------------------------------------------------- #
# Aten IR targets.
# --------------------------------------------------------------------------- #
_CONV = torch.ops.aten.convolution.default
_CONV_CUDNN = torch.ops.aten.cudnn_convolution.default
_CONV_TARGETS = frozenset({_CONV, _CONV_CUDNN})

_BN_NO_TRAIN = torch.ops.aten._native_batch_norm_legit_no_training.default

_CLONE = torch.ops.aten.clone.default
_TO_COPY = torch.ops.aten._to_copy.default
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


def _node_meta_tensor(node):
    """Return the FakeTensor in node.meta['val'] (for dtype/layout queries), or None."""
    try:
        if isinstance(node, fx.Node):
            val = node.meta.get("val", None)
            if val is not None and hasattr(val, "dtype"):
                return val
    except Exception:
        pass
    return None


def _unwrap_cast(node):
    """Walk back through inserted dtype casts (aten.to.dtype / _to_copy) to the
    underlying producer/placeholder so OPT-3 can read the real node."""
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
# OPT-1 — channels_last (NHWC). aten-level FX fallback/cleanup pass.
# Confidence: high. No ordering constraint (layout-only, independent of OPT-2/3).
#
# PRIMARY lever is the eager-side conversion in get_model_and_input() (Rule 7).
# This pass is the documented aten-level fallback (optimizations.json OPT-1
# fx_steps): when the conv activation is NOT already channels_last-contiguous it
# inserts aten.clone(memory_format=channels_last) on the conv input so cuDNN
# consumes NHWC directly and the convertTensor transposes vanish. When the
# boundary conversion already aligned layouts (the normal path) every conv input
# is already NHWC and the pass is a graceful no-op.
# =========================================================================== #
def _pass_channels_last_conv_inputs(gm: fx.GraphModule) -> fx.GraphModule:
    """Insert aten.clone(memory_format=channels_last) on conv activation inputs that
    are not already NHWC-contiguous. Graceful no-op when the eager-side conversion
    (get_model_and_input) already set the layout — the canonical and preferred path."""
    try:
        cloned = 0
        skipped_already_nhwc = 0
        for node in list(gm.graph.nodes):
            if not (node.op == "call_function" and node.target in _CONV_TARGETS):
                continue
            act = node.args[0]
            act_meta = _node_meta_tensor(_unwrap_cast(act))
            # If the activation is already channels_last-contiguous (boundary cast
            # applied), do nothing — inserting a clone would be redundant churn.
            if act_meta is not None and act_meta.dim() == 4:
                try:
                    if act_meta.is_contiguous(memory_format=torch.channels_last):
                        skipped_already_nhwc += 1
                        continue
                except Exception:
                    pass
            else:
                # Non-4D or unknown layout — not a spatial conv input we should clone.
                continue
            with gm.graph.inserting_before(node):
                act_cl = gm.graph.call_function(
                    _CLONE, (act,), {"memory_format": torch.channels_last}
                )
            node.update_arg(0, act_cl)
            cloned += 1
            logger.info(
                "[OPT-1 channels_last] Inserted channels_last clone on conv input "
                "(shape=%s) [Aten IR fallback]",
                _node_shape(node),
            )

        if cloned:
            gm.graph.lint()
            gm.recompile()
            logger.info(
                "[OPT-1 channels_last] Applied — inserted %d NHWC clone(s) "
                "(boundary layout was not set)",
                cloned,
            )
        else:
            logger.info(
                "[OPT-1 channels_last] No NCHW conv inputs found "
                "(%d already NHWC) — layout handled eager-side in "
                "get_model_and_input(); pass is a no-op",
                skipped_already_nhwc,
            )
    except Exception as e:
        logger.warning("[OPT-1 channels_last] Failed: %s", e)
    return gm


# =========================================================================== #
# OPT-2 — dtype promotion to bfloat16 around the convolutions. Confidence: medium.
# Prerequisite for OPT-3. PRIMARY lever is the eager-side model.bfloat16() +
# input.bfloat16() boundary cast (combined with OPT-1). This aten-level FX pass is
# the documented FALLBACK (optimizations.json OPT-2 fx_steps): cast each conv
# operand to bf16 and restore fp32 on the output. Graceful no-op when operands are
# already bf16 (the boundary cast was applied).
# =========================================================================== #
def _pass_bf16_conv_operands(gm: fx.GraphModule) -> fx.GraphModule:
    """Cast conv activation+weight to bf16 so cuDNN selects the BF16 tensor-op path,
    restoring fp32 on the conv output. Graceful no-op when the operands are already
    bf16 (eager-side model.bfloat16() applied)."""
    try:
        promoted = 0
        already_bf16 = 0
        for conv in list(gm.graph.nodes):
            if not (conv.op == "call_function" and conv.target in _CONV_TARGETS):
                continue
            w = conv.args[1]
            w_meta = _node_meta_tensor(_unwrap_cast(w))
            # If the weight is already bf16 (boundary cast), the conv already runs the
            # BF16 path — nothing to do.
            if w_meta is not None and w_meta.dtype == torch.bfloat16:
                already_bf16 += 1
                continue

            x_in = conv.args[0]
            with gm.graph.inserting_before(conv):
                x_bf16 = gm.graph.call_function(_TO_DTYPE, (x_in, torch.bfloat16))
                w_bf16 = gm.graph.call_function(_TO_DTYPE, (w, torch.bfloat16))
            conv.update_arg(0, x_bf16)
            conv.update_arg(1, w_bf16)
            # Bias (if present) must match the conv compute dtype.
            bias = conv.args[2] if len(conv.args) > 2 else None
            if isinstance(bias, fx.Node):
                with gm.graph.inserting_before(conv):
                    b_bf16 = gm.graph.call_function(_TO_DTYPE, (bias, torch.bfloat16))
                conv.update_arg(2, b_bf16)

            with gm.graph.inserting_after(conv):
                back = gm.graph.call_function(_TO_DTYPE, (conv, torch.float32))
            conv.replace_all_uses_with(back)
            back.update_arg(0, conv)  # replace_all_uses_with rewired back's own input
            promoted += 1
            logger.info(
                "[OPT-2 bf16_conv] Promoted conv operands to bf16 (Cout=%s) [Aten IR]",
                (_node_shape(conv.args[1]) or ("?",))[0],
            )

        if promoted:
            gm.graph.lint()
            gm.recompile()
            logger.info(
                "[OPT-2 bf16_conv] Applied — promoted %d conv(s) to bf16", promoted
            )
        else:
            logger.info(
                "[OPT-2 bf16_conv] %d conv(s) already bf16 — dtype handled eager-side "
                "in get_model_and_input(); pass is a no-op",
                already_bf16,
            )
    except Exception as e:
        logger.warning("[OPT-2 bf16_conv] Failed: %s", e)
    return gm


# =========================================================================== #
# OPT-3 — eval-mode conv-BatchNorm fold. Confidence: low.
# Runs AFTER OPT-2 (register_buffer-after-dtype-promotion rule): folded weight/bias
# buffers are registered at the conv's runtime dtype.
#
# aten._native_batch_norm_legit_no_training(input, weight(gamma), bias(beta),
#                                           running_mean, running_var, momentum, eps)
#   -> returns (output, save_mean, save_rstd); only getitem(node, 0) is live.
#
# The model convs have bias=False, so the fold MUST add a bias':
#   scale = gamma / sqrt(running_var + eps)
#   W'    = W * scale.reshape(out_ch, 1, 1, 1)
#   bias' = beta - gamma * running_mean / sqrt(running_var + eps)
# =========================================================================== #
def _pass_fold_conv_bn(gm: fx.GraphModule, fw_example_inputs: list) -> fx.GraphModule:
    """Fold an inference-mode BatchNorm into its preceding convolution, reading the
    REAL parameter tensors threaded via real_inputs (inner_compile's example_inputs
    may be FakeTensors). The conv has no bias, so the fold introduces bias'.

    Low confidence: Inductor may already bake the eval-mode affine into the
    elementwise epilogue, so if no foldable conv->BN pair is found the pass logs a
    graceful no-op."""
    try:
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        ph_to_tensor = {ph: t for ph, t in zip(placeholders, fw_example_inputs)}

        folded = 0
        for bn_node in list(gm.graph.nodes):
            if not (bn_node.op == "call_function" and bn_node.target is _BN_NO_TRAIN):
                continue

            # BN input may be the OPT-2 fp32 re-cast of the conv output; unwrap it.
            consumer = bn_node.args[0]
            conv_node = _unwrap_cast(consumer)
            if not (
                isinstance(conv_node, fx.Node)
                and conv_node.op == "call_function"
                and conv_node.target in _CONV_TARGETS
            ):
                continue

            cast_in_between = (
                isinstance(consumer, fx.Node)
                and consumer.op == "call_function"
                and consumer.target in _CAST_TARGETS
                and consumer is not conv_node
            )
            chain_tail = consumer if cast_in_between else conv_node
            # Conv output must feed only the BN (possibly via the single fp32 cast).
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
                    "[OPT-3 fold_conv_bn] BN params not resolvable from real_inputs "
                    "— skipping this BN"
                )
                continue

            # Conv weight may be an OPT-2 bf16 cast node; recover the placeholder value.
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
                    "[OPT-3 fold_conv_bn] Conv weight not resolvable — skipping"
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

            # Runtime dtype: bf16 if the conv operands were cast (OPT-2 or boundary).
            w_meta = _node_meta_tensor(weight_ph)
            ran_bf16 = (
                isinstance(weight_arg, fx.Node)
                and weight_arg.op == "call_function"
                and weight_arg.target in _CAST_TARGETS
            ) or (w_meta is not None and w_meta.dtype == torch.bfloat16)
            runtime_dtype = torch.bfloat16 if ran_bf16 else torch.float32
            new_weight = new_weight.to(runtime_dtype)
            new_bias = new_bias.to(runtime_dtype)

            c_out = int(new_weight.shape[0])
            wname = f"_folded_conv_weight_{folded}"
            bname = f"_folded_conv_bias_{folded}"
            gm.register_buffer(wname, new_weight)
            gm.register_buffer(bname, new_bias)

            # Fresh biased conv reusing the original conv input + spatial args.
            new_conv_args = list(conv_node.args)
            with gm.graph.inserting_before(bn_node):
                fw = gm.graph.get_attr(wname)
                fb = gm.graph.get_attr(bname)
                new_conv_args[1] = fw
                if len(new_conv_args) > 2:
                    new_conv_args[2] = fb
                else:
                    new_conv_args.append(fb)
                new_conv = gm.graph.call_function(_CONV, tuple(new_conv_args))
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
            if cast_in_between and not chain_tail.users:
                gm.graph.erase_node(chain_tail)
            if not conv_node.users:
                gm.graph.erase_node(conv_node)

            folded += 1
            logger.info(
                "[OPT-3 fold_conv_bn] Folded BatchNorm into conv (C_out=%d, dtype=%s) "
                "[Aten IR]",
                c_out,
                runtime_dtype,
            )

        if folded:
            gm.graph.eliminate_dead_code()
            gm.graph.lint()
            gm.recompile()
            logger.info(
                "[OPT-3 fold_conv_bn] Applied — folded %d conv+BN pair(s)", folded
            )
        else:
            logger.warning(
                "[OPT-3 fold_conv_bn] No foldable conv->_native_batch_norm_legit_"
                "no_training pair found (Inductor may already bake the affine) "
                "— pass not applied"
            )
    except Exception as e:
        logger.warning("[OPT-3 fold_conv_bn] Failed: %s", e)
    return gm


# =========================================================================== #
# Pass registry — all three optimizations are aten-level (or boundary/non-graph).
# Within the aten level the order is OPT-1 (layout, independent) -> OPT-2 (dtype,
# prereq for OPT-3) -> OPT-3 (conv-BN fold, reads runtime dtype). OPT-3 reads real
# weight values; OPT-1/OPT-2 are op-target passes.
# =========================================================================== #
PASS_REGISTRY = [
    {"id": "OPT-1", "level": "aten", "fn": _pass_channels_last_conv_inputs, "reads_weights": False},
    {"id": "OPT-2", "level": "aten", "fn": _pass_bf16_conv_operands, "reads_weights": False},
    {"id": "OPT-3", "level": "aten", "fn": _pass_fold_conv_bn, "reads_weights": True},
]


def _passes(level):
    return [p for p in PASS_REGISTRY if p["level"] == level]


def _run_functional_passes(gm: fx.GraphModule) -> fx.GraphModule:
    """LEVEL 1 — Dynamo (functional) graph, before compile_fx. No functional passes
    for this workload (all opts are aten/boundary), so this is a structural pass-through
    that keeps the funnel uniform per Rule 9."""
    for p in _passes("functional"):
        try:
            gm = p["fn"](gm)
        except Exception as e:
            logger.warning("[%s] functional pass no-op: %s", p["id"], e)
    return gm


def _repropagate_meta(gm: fx.GraphModule, example_inputs) -> None:
    """Re-run FakeTensorProp after a structural graph rewrite so inserted nodes
    (clones, dtype casts, folded convs) get meta['val'] before compile_fx_inner."""
    try:
        from torch.fx.passes.fake_tensor_prop import FakeTensorProp

        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        fake_inputs = []
        fake_mode = None
        for ph, ex in zip(placeholders, example_inputs):
            val = ph.meta.get("val", ex)
            fake_inputs.append(val)
            fm = getattr(val, "fake_mode", None)
            if fm is not None:
                fake_mode = fm
        if fake_mode is not None:
            with fake_mode:
                FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(
                    *fake_inputs
                )
        else:
            FakeTensorProp(gm).propagate_dont_convert_inputs(*fake_inputs)
    except Exception as e:
        logger.warning("[conv_block_opt] meta re-propagation skipped: %s", e)


def _aten_inner_compile(
    gm: fx.GraphModule, example_inputs, *, real_inputs=None, **kwargs
) -> Callable:
    """LEVEL 2 — Inductor inner_compile hook. compile_fx calls this with the fully
    decomposed Aten IR graph (post-AOTAutograd). Run aten-level passes in DAG order,
    re-propagating meta after each structural rewrite, then delegate to
    compile_fx_inner (Aten -> Triton).

    example_inputs may be FakeTensors; the weight-VALUE-reading OPT-3 fold uses the
    genuine real_inputs threaded from the backend for the ph_to_tensor lookup.
    **kwargs is forwarded verbatim to compile_fx_inner for forward-compatibility."""
    weight_source = real_inputs if real_inputs is not None else example_inputs
    for p in _passes("aten"):
        try:
            if p["reads_weights"]:
                gm = p["fn"](gm, weight_source)
            else:
                gm = p["fn"](gm)
            _repropagate_meta(gm, example_inputs)
        except Exception as e:
            logger.warning("[%s] aten pass no-op: %s", p["id"], e)
    return compile_fx_inner(gm, example_inputs, **kwargs)


def _config_patches() -> dict:
    """LEVEL 3 — scoped Inductor config_patches merged into THIS compile_fx call only.
    No inductor_config-level optimizations for this workload."""
    patches: dict = {}
    for p in _passes("inductor_config"):
        try:
            patches.update(p["fn"]() or {})
        except Exception as e:
            logger.warning("[%s] config pass skipped: %s", p["id"], e)
    return patches


def _compile_unit(gm: fx.GraphModule, example_inputs) -> Callable:
    """The fixed three-stage funnel (Rule 9): functional passes -> compile_fx with the
    aten passes installed at inner_compile and scoped config_patches. compile_fx owns
    AOTAutograd, the decomp table, the boxed calling convention and the partitioner —
    we only run functional passes ahead of it and swap the leaf compiler."""
    gm = _run_functional_passes(gm)
    inner = functools.partial(_aten_inner_compile, real_inputs=list(example_inputs))
    return compile_fx(
        gm, example_inputs, inner_compile=inner, config_patches=_config_patches()
    )


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
def conv_block_opt(gm: fx.GraphModule, example_inputs) -> Callable:
    """Custom torch.compile backend for ConvBlock.

    Installs the aten-IR pass chain (OPT-1 channels_last fallback, OPT-2 bf16 conv
    operands, OPT-3 conv-BN fold) via compile_fx's inner_compile hook, then delegates
    AOTAutograd + lowering to compile_fx. OPT-1 and OPT-2's PRIMARY levers are the
    channels_last + bfloat16 boundary casts in get_model_and_input() (Rule 7); the
    graph passes are the documented aten-level fallbacks and degrade to no-ops once the
    boundary cast is applied.

    Dedup-aware per Rule 9: ConvBlock's three ConvBnRelu stages have DISTINCT channel
    counts (3->64, 64->128, 128->256), so there are no structurally repeated layers and
    the flat compile path is taken (preserving cross-stage Inductor fusion). The dedup
    branch is retained for structural reuse should the model grow repeated blocks.
    """
    logger.info(
        "conv_block_opt backend: starting (functional -> aten -> inductor_config)"
    )
    registry = UniqueSubgraphRegistry(gm)
    equiv_map = registry.build_partition_equivalence_map()

    if not equiv_map:
        logger.info("conv_block_opt: no repeated layers, flat compile path")
        return _compile_unit(gm, example_inputs)

    logger.info(
        "conv_block_opt: %d duplicate partition(s), dedup path", len(equiv_map)
    )
    partition_inputs = _capture_partition_inputs(registry.split, example_inputs)
    for rep_name, rep_mod in registry.unique_reps:
        inputs = partition_inputs.get(rep_name, example_inputs)
        compiled = _compile_unit(rep_mod, inputs)
        rep_mod.forward = compiled
        for _, dup_mod in registry.duplicates_of(rep_name):
            dup_mod.forward = compiled

    return lambda *args: registry.split(*args)


# =========================================================================== #
# Workload interface — non-graph (boundary) optimizations live here (Rule 7).
#   OPT-1 (channels_last): model + input cast to torch.channels_last so the cuDNN
#          tensor-op conv kernels run native NHWC and the convertTensor NCHW<->NHWC
#          transposes (~49,600 ns, 12 launches) are eliminated. PRIMARY lever.
#   OPT-2 (bfloat16): model + input cast to bfloat16 so cuDNN dispatches the BF16
#          tensor-op conv path (~2x TF32 rate + occupancy recovery) and the
#          DRAM-bound BN+ReLU+pool epilogue halves its bytes. PRIMARY lever, combined
#          with OPT-1 in one boundary cast per the proposal application order.
# Both are checked-before-applied (idempotent, Rule 7). OPT-3 (conv-BN fold) runs as
# an aten-IR graph pass inside the backend at compile time.
# =========================================================================== #
DEVICE = "cuda"
BATCH_SIZE = 16
IN_CHANNELS = 3
HEIGHT = 64
WIDTH = 64


def get_model_and_input() -> tuple:
    """Return (model, input_tensor) on CUDA — uncompiled, unwarmed.

    Applies OPT-1 (channels_last) and OPT-2 (bfloat16) as the preferred boundary casts
    (idempotent — checked before converting per Rule 7). OPT-3 (conv-BN fold) runs
    inside the conv_block_opt backend at compile time.
    """
    assert torch.cuda.is_available(), "CUDA required"
    from conv_block import ConvBlock

    model = ConvBlock().to(DEVICE).eval()
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH, device=DEVICE)

    # OPT-1 (channels_last / NHWC) — eager-side, only if not already NHWC.
    if not next(model.parameters()).is_contiguous(memory_format=torch.channels_last):
        model = model.to(memory_format=torch.channels_last)
        logger.info("[OPT-1 channels_last] Model cast to channels_last (NHWC)")
    if not x.is_contiguous(memory_format=torch.channels_last):
        x = x.to(memory_format=torch.channels_last)
        logger.info("[OPT-1 channels_last] Input cast to channels_last (NHWC)")

    # OPT-2 (bfloat16) — eager-side, combined with OPT-1 per the proposal order.
    if next(model.parameters()).dtype != torch.bfloat16:
        model = model.to(torch.bfloat16)
        logger.info("[OPT-2 bfloat16] Model cast to bfloat16")
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
        logger.info("[OPT-2 bfloat16] Input cast to bfloat16")

    return model, x


if __name__ == "__main__":
    model, x = get_model_and_input()
    compiled = torch.compile(model, backend="conv_block_opt")
    with torch.no_grad():
        out = compiled(x)
    print("output shape:", out.shape, "dtype:", out.dtype)
