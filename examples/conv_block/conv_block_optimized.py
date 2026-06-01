"""
conv_block_optimized.py — Custom torch.compile() backend for ConvBlock.

Registered backend: ``conv_block_opt``

Implements the three optimizations from optimizations.json, routed to the correct IR
level via the three-stage funnel (functional -> aten -> inductor_config). Each pass runs
at the level where its pattern is unambiguous and the rewrite is sound.

Backend name: conv_block_opt  (model "conv_block" -> snake-case + _opt)

Pass summary (execution order: non-graph (module) -> functional -> aten -> inductor_config):

  OPT-1  non-graph / high    — channels_last (NHWC) memory format
      cuDNN's TF32 / bf16 tensor-op fprop path is native NHWC. In default contiguous NCHW
      cuDNN inserts a convertTensor (NCHW<->NHWC) permute before AND after every conv
      (12 zero-math repack kernels = ~10.5% of attributed time) plus a 4-kernel triton
      input-prep repack. Per the funnel rules, whole-module memory_format is a NON-GRAPH
      optimization applied in get_model_and_input() (model + input both converted) so the
      NHWC format propagates through the conv stack without inserting per-op transpose
      kernels. Applying it in-graph (aten.contiguous per conv) risks stray repack kernels
      that reintroduce the very DRAM round-trip we are removing.

  OPT-2  aten / medium       — bf16 dtype promotion (prereq for OPT-3)
      The four large cutlass_tensorop_s1688fprop_optimized_tf32 conv GEMMs are genuinely
      compute/tensor-core bound (~74% of attributed time, tensor-core active 66-73%, DRAM
      6-28%). TF32 keeps the FP32 byte width but uses the half-rate tensor op. Casting conv
      and the linear-head (addmm) operands to bf16 routes cuDNN to the half-rate (s16816)
      tensor-op path — roughly halving tensor-op cycles AND halving the bytes the
      DRAM-bound BN/ReLU triton kernels stage. Implemented as an aten op-target pass: in
      front of every aten.convolution.default / aten.addmm.default cast activation+weight to
      bf16 with aten._to_copy, then cast the result back to fp32. Inductor folds adjacent
      redundant casts and fuses the back-cast into the downstream epilogue; accumulation
      stays fp32.

  OPT-3  functional / high   — Conv-BN fold (inference) [routed functional, not aten]
      Fold every eval-mode BatchNorm into the weight/bias of its immediately-preceding
      convolution: w' = w * gamma / sqrt(var+eps); b' = beta - gamma*mean/sqrt(var+eps).
      Exact and lossless in eval mode.

      CRITICAL ROUTING NOTE: optimizations.json marks OPT-3 ir_level="aten", but at the
      aten seam the fold is a SILENT NO-OP. compile_fx runs AOTAutograd BEFORE the
      inner_compile seam, and AOTAutograd decomposes eval-mode BatchNorm into primitive ops
      — so by the time _aten_inner_compile sees the graph there are ZERO
      aten._native_batch_norm_legit_no_training nodes left to match. The fold must run on
      the Dynamo FUNCTIONAL graph the backend receives BEFORE handing off to compile_fx,
      where eval-mode BN is still a single torch.nn.functional.batch_norm node (training=
      False) fed directly by a conv2d node. A defensive aten-level fallback
      ("OPT-3-fallback") remains registered and gracefully no-ops on torch 2.11.

      PREREQUISITE OPT-2 -> OPT-3 (register_buffer-after-dtype rule): the folded weight/bias
      buffers must be created at the bf16 RUNTIME dtype so the bf16 conv (OPT-2) consumes
      them without an extra cast. Although OPT-3 runs at the functional level (before the
      aten bf16 pass), the cross-level prerequisite is satisfied by computing the fold in
      fp32 for numerical accuracy and then registering the folded buffers as bf16 — matching
      the dtype OPT-2's _to_copy would have produced. The OPT-2 cast in front of the conv
      then sees an already-bf16 weight and folds to a no-op.

      DTYPE RECONCILIATION (input cast): because the folded buffers are bf16 but the conv
      INPUT (graph input / prior fp32 activation) is fp32, the fold ALSO wraps the rewritten
      conv with an input cast (fp32 -> bf16) before the conv and an output cast (bf16 -> fp32)
      after it. Without this, a bf16-weight conv fed an fp32 input raises
      "Input type (float) and bias type (BFloat16) should be the same" at AOTAutograd trace
      time. The wrap keeps the surrounding fp32 functional graph dtype-consistent; the casts
      OPT-2 adds at the aten level become redundant adjacent casts that Inductor folds away.

Prerequisite / ordering rationale:
  - OPT-1 (channels_last) is independent (non-graph, module level).
  - OPT-2 (bf16, aten) is a prerequisite for OPT-3 (BN fold). Because OPT-3 is forced to the
    functional level (BN is decomposed before the aten seam), the prerequisite is honored by
    allocating the folded buffers at bf16 inside the functional fold (matching OPT-2's
    runtime dtype) rather than relying on within-level sequencing.

IR-level mechanics (torch 2.11):
  compile_fx owns AOTAutograd, the decomp table, the boxed calling convention and the
  partitioner. We do NOT use aot_autograd(fw_compiler=compile_fx) — on torch 2.11 that
  raises AssertionError inside copy_misaligned_inputs. The funnel runs functional-level
  passes BEFORE compile_fx, aten-level passes through its inner_compile seam, and
  inductor_config passes as scoped config_patches.

compile_mode = "inductor" (from optimizations.json analysis.compile_mode).
dtype = bf16 conv/linear compute (OPT-2), fp32 elsewhere; channels_last layout (OPT-1).
"""
from __future__ import annotations

import functools
import logging
import operator
from typing import Callable

import torch
import torch.fx as fx
from torch._dynamo import register_backend
from torch._inductor.compile_fx import compile_fx, compile_fx_inner

from nvidia.operator_profiler.fx import UniqueSubgraphRegistry

logger = logging.getLogger(__name__)

_BF16 = torch.bfloat16
_FP32 = torch.float32

# ---------------------------------------------------------------------------
# Op targets
# ---------------------------------------------------------------------------
_CONV_TARGETS = frozenset(
    {
        torch.ops.aten.convolution.default,
        torch.ops.aten.cudnn_convolution.default,
    }
)
_ADDMM_TARGET = torch.ops.aten.addmm.default
_MM_TARGET = torch.ops.aten.mm.default
_BN_TARGET = torch.ops.aten._native_batch_norm_legit_no_training.default
# Cast op for backend-inserted nodes. NOTE: a literal aten._to_copy.default inserted into
# the graph trips Inductor's "both a fallback and a decomp for same op: aten._to_copy.default"
# assertion during lowering (it has BOTH a registered decomp and a registered fallback, so
# a hand-inserted node is ambiguous). prims.convert_element_type.default is the dtype-cast
# primitive Inductor lowers cleanly, has no such collision, and adjacent redundant casts are
# folded by Inductor's CSE/peephole. Used by BOTH the functional fold (OPT-3) and the
# aten-level bf16 promotion (OPT-2).
_CONVERT = torch.ops.prims.convert_element_type.default


def _is_functional_conv(node: fx.Node) -> bool:
    """True if `node` is a functional-level convolution call.

    At the Dynamo graph level a Conv2d traces to a single high-level node whose target is
    the builtin ``torch.conv2d``. We also accept F.conv2d and the aten conv overloads to
    stay robust across Dynamo lowering variants. Matched by identity first, then by name."""
    if node.op != "call_function":
        return False
    t = node.target
    if t in (torch.conv2d, torch.nn.functional.conv2d):
        return True
    if t in _CONV_TARGETS:
        return True
    return getattr(t, "__name__", "") in ("conv2d", "convolution")


def _is_functional_batch_norm(node: fx.Node) -> bool:
    """True if `node` is a functional-level batch_norm call.

    At the Dynamo graph level an eval-mode BatchNorm2d traces to a single
    torch.nn.functional.batch_norm node (training=False). Identity first, then by name."""
    if node.op != "call_function":
        return False
    t = node.target
    if t is torch.nn.functional.batch_norm:
        return True
    return getattr(t, "__name__", "") == "batch_norm"


# ---------------------------------------------------------------------------
# OPT-3 — Conv-BN fold (inference). ir_level=functional. Confidence: high.
#
# Functional batch_norm args:
#   (input, running_mean, running_var, weight, bias, training, momentum, eps)
# Functional conv2d args:
#   (input, weight, bias, stride, padding, dilation, groups)
# ---------------------------------------------------------------------------

def _register_synthetic_source(gm: fx.GraphModule, attr_name: str) -> None:
    """Give a backend-introduced functional-level get_attr a source AOTAutograd accepts.

    Dynamo stores ``_param_name_to_source`` on the GraphModule it hands the backend;
    aot_export later asserts every lifted get_attr target is present there with a unique,
    non-None source. New buffers added by a functional pass are absent, so we register a
    unique LocalSource keyed on the buffer name. Best-effort no-op if the API is absent."""
    try:
        src_map = getattr(gm, "_param_name_to_source", None)
        if src_map is None:
            return
        from torch._dynamo.source import LocalSource

        if attr_name not in src_map:
            src_map[attr_name] = LocalSource(attr_name)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(
            "[OPT-3 fold_conv_bn] could not register source for %s: %s", attr_name, e
        )


def _fpass_fold_conv_bn(gm: fx.GraphModule, ph_to_tensor: dict) -> fx.GraphModule:
    """OPT-3: Fold eval-mode BatchNorm into the preceding convolution. Functional IR level.

    ph_to_tensor maps placeholder nodes -> real parameter tensors (Dynamo hands the backend
    real tensors here). The fold is exact in eval mode. Folded buffers are registered at the
    bf16 runtime dtype to honor the OPT-2 -> OPT-3 prerequisite (register-buffer-after-dtype):
    the bf16 conv (OPT-2) then consumes an already-bf16 weight with no extra cast."""
    try:
        g = gm.graph
        folded = 0
        buf_idx = 0

        for bn_node in list(g.nodes):
            if not _is_functional_batch_norm(bn_node):
                continue

            args = bn_node.args
            kwargs = bn_node.kwargs

            def _arg(pos, name, default=None):
                if len(args) > pos:
                    return args[pos]
                return kwargs.get(name, default)

            conv_node = _arg(0, "input")
            run_mean_n = _arg(1, "running_mean")
            run_var_n = _arg(2, "running_var")
            bn_weight_n = _arg(3, "weight")
            bn_bias_n = _arg(4, "bias")
            training = _arg(5, "training", False)
            eps = _arg(7, "eps", 1e-5)

            if training:
                continue
            if not (isinstance(conv_node, fx.Node) and _is_functional_conv(conv_node)):
                continue
            # BN must be the conv's sole consumer to fold safely.
            if len(conv_node.users) != 1:
                continue

            conv_weight_n = conv_node.args[1] if len(conv_node.args) > 1 else None
            conv_bias_n = conv_node.args[2] if len(conv_node.args) > 2 else None

            def _resolve(n):
                if n is None:
                    return None
                if isinstance(n, fx.Node):
                    return ph_to_tensor.get(n)
                return n

            conv_weight = _resolve(conv_weight_n)
            conv_bias = _resolve(conv_bias_n)
            run_mean = _resolve(run_mean_n)
            run_var = _resolve(run_var_n)
            bn_weight = _resolve(bn_weight_n)
            bn_bias = _resolve(bn_bias_n)

            if conv_weight is None:
                logger.warning(
                    "[OPT-3 fold_conv_bn] Conv weight not resolvable from real inputs "
                    "— skipping this site"
                )
                continue
            if any(t is None for t in (run_mean, run_var, bn_weight, bn_bias)):
                logger.warning(
                    "[OPT-3 fold_conv_bn] BN constants not resolvable from real inputs "
                    "— skipping this site"
                )
                continue

            # Exact, lossless eval-mode fold computed in FP32 for accuracy.
            try:
                from torch.nn.utils import fuse_conv_bn_weights

                fused_w, fused_b = fuse_conv_bn_weights(
                    conv_weight.float(),
                    conv_bias.float() if conv_bias is not None else None,
                    run_mean.float(),
                    run_var.float(),
                    eps,
                    bn_weight.float(),
                    bn_bias.float(),
                )
            except Exception:
                scale = bn_weight.float() / torch.sqrt(run_var.float() + eps)
                fused_w = conv_weight.float() * scale.view(-1, 1, 1, 1)
                if conv_bias is not None:
                    fused_b = (conv_bias.float() - run_mean.float()) * scale + bn_bias.float()
                else:
                    fused_b = bn_bias.float() - run_mean.float() * scale

            # OPT-2 prerequisite: allocate folded buffers at the bf16 runtime dtype so the
            # bf16 conv (OPT-2) consumes them directly. Preserve channels_last (OPT-1).
            fused_w = fused_w.to(_BF16)
            fused_b = fused_b.to(_BF16)
            if fused_w.dim() == 4 and conv_weight.is_contiguous(
                memory_format=torch.channels_last
            ):
                fused_w = fused_w.contiguous(memory_format=torch.channels_last)

            w_name = f"_folded_conv_weight_{buf_idx}"
            b_name = f"_folded_conv_bias_{buf_idx}"
            buf_idx += 1
            gm.register_buffer(w_name, fused_w)
            gm.register_buffer(b_name, fused_b)
            _register_synthetic_source(gm, w_name)
            _register_synthetic_source(gm, b_name)

            with g.inserting_before(conv_node):
                fw = g.get_attr(w_name)
                fb = g.get_attr(b_name)

            # DTYPE RECONCILIATION (fixes float-input / bf16-weight mismatch):
            # The folded weight/bias buffers are bf16 (OPT-2 prerequisite), but the conv
            # input (e.g. the graph input l_x_, or a previous fp32 activation) is fp32. A
            # conv with fp32 input and bf16 weight raises at runtime/trace time. Cast the
            # conv input to bf16 BEFORE the conv, and cast the conv output back to fp32 AFTER,
            # so the surrounding fp32 functional graph is dtype-consistent. This is exactly
            # the wrap OPT-2 would have inserted at the aten level; doing it here keeps the
            # functional graph self-consistent before AOTAutograd traces it. Inductor folds
            # the redundant adjacent casts that OPT-2 adds later.
            conv_in = conv_node.args[0]
            tail = tuple(conv_node.args[3:])
            with g.inserting_before(bn_node):
                in_cast = g.call_function(_CONVERT, (conv_in, _BF16))
                new_conv = g.call_function(
                    conv_node.target, (in_cast, fw, fb) + tail, dict(conv_node.kwargs)
                )
                out_cast = g.call_function(_CONVERT, (new_conv, _FP32))

            # F.batch_norm returns a single tensor at the functional level (no getitem).
            bn_node.replace_all_uses_with(out_cast)
            g.erase_node(bn_node)
            if not conv_node.users:
                g.erase_node(conv_node)

            folded += 1

        if folded == 0:
            logger.warning(
                "[OPT-3 fold_conv_bn] No foldable conv -> batch_norm pair at functional "
                "level — pass not applied"
            )
            return gm

        g.eliminate_dead_code()
        g.lint()
        gm.recompile()
        logger.info(
            "[OPT-3 fold_conv_bn] APPLIED — folded %d eval-mode batch_norm node(s) into "
            "the preceding convolution [functional IR; bf16 folded buffers honor OPT-2 "
            "prerequisite]",
            folded,
        )
    except Exception as e:
        logger.warning("[OPT-3 fold_conv_bn] Failed: %s", e)
    return gm


# ---------------------------------------------------------------------------
# OPT-2 — bf16 dtype promotion. ir_level=aten. Confidence: medium.
#
# Runs inside _aten_inner_compile on the fully decomposed Aten graph. In front of every
# aten.convolution.default / aten.addmm.default cast the activation + weight operands to
# bf16, run the op in bf16, and cast the result back to fp32. Inductor folds the redundant
# adjacent casts (e.g. the bf16 conv weight produced by OPT-3 -> its _to_copy is a no-op)
# and fuses the back-cast into the downstream BN/ReLU triton epilogue. This is an op-TARGET
# pass (no weight VALUES read), so it does not need the ph_to_tensor lookup.
# ---------------------------------------------------------------------------

def _maybe_cast(g: fx.Graph, before: fx.Node, src: fx.Node, dtype) -> fx.Node:
    """Insert a prims.convert_element_type(src, dtype) before `before`. Returns the cast node.
    Uses the convert_element_type primitive (not aten._to_copy) to avoid Inductor's
    decomp/fallback collision on hand-inserted _to_copy nodes."""
    with g.inserting_before(before):
        return g.call_function(_CONVERT, (src, dtype))


def _apass_bf16_promotion(gm: fx.GraphModule) -> fx.GraphModule:
    """OPT-2: Promote conv and linear-head compute to bf16 (fp32 in/out). Aten IR level.

    Op-target pass: matches aten.convolution.default and aten.addmm.default / aten.mm.default.
    Casts the activation (arg0) and weight (arg1 for conv) to bf16; for addmm casts all three
    matrix operands; then casts the op result back to fp32 so the surrounding fp32 graph is
    numerically unchanged except for the bf16 tensor-op precision."""
    try:
        promoted = 0
        for node in list(gm.graph.nodes):
            if node.op != "call_function":
                continue

            if node.target in _CONV_TARGETS:
                act, weight = node.args[0], node.args[1]
                bias = node.args[2] if len(node.args) > 2 else None
                a_cast = _maybe_cast(gm.graph, node, act, _BF16)
                w_cast = _maybe_cast(gm.graph, node, weight, _BF16)
                node.update_arg(0, a_cast)
                node.update_arg(1, w_cast)
                if isinstance(bias, fx.Node):
                    b_cast = _maybe_cast(gm.graph, node, bias, _BF16)
                    node.update_arg(2, b_cast)
            elif node.target is _ADDMM_TARGET:
                # addmm(bias, mat1, mat2): cast all three matrix operands to bf16.
                bias, mat1, mat2 = node.args[0], node.args[1], node.args[2]
                b_cast = _maybe_cast(gm.graph, node, bias, _BF16)
                m1_cast = _maybe_cast(gm.graph, node, mat1, _BF16)
                m2_cast = _maybe_cast(gm.graph, node, mat2, _BF16)
                node.update_arg(0, b_cast)
                node.update_arg(1, m1_cast)
                node.update_arg(2, m2_cast)
            elif node.target is _MM_TARGET:
                mat1, mat2 = node.args[0], node.args[1]
                m1_cast = _maybe_cast(gm.graph, node, mat1, _BF16)
                m2_cast = _maybe_cast(gm.graph, node, mat2, _BF16)
                node.update_arg(0, m1_cast)
                node.update_arg(1, m2_cast)
            else:
                continue

            # Cast the op result back to fp32 for the downstream fp32 graph.
            with gm.graph.inserting_after(node):
                back = gm.graph.call_function(_CONVERT, (node, _FP32))
            node.replace_all_uses_with(back)
            back.update_arg(0, node)
            promoted += 1

        if promoted == 0:
            logger.warning(
                "[OPT-2 bf16_promotion] No conv/addmm/mm nodes found — pass not applied"
            )
            return gm

        gm.graph.lint()
        gm.recompile()
        logger.info(
            "[OPT-2 bf16_promotion] Promoted %d conv/linear op(s) to bf16 tensor-op path "
            "[aten IR; fp32 accumulate, fp32 in/out]",
            promoted,
        )
    except Exception as e:
        logger.warning("[OPT-2 bf16_promotion] Failed: %s", e)
    return gm


# ---------------------------------------------------------------------------
# OPT-3 fallback — Conv-BN fold at the aten level (defensive; no-op on torch 2.11).
# Reads weight VALUES, so it uses the ph_to_tensor lookup from threaded real_inputs.
# ---------------------------------------------------------------------------

def _apass_fold_conv_bn(gm: fx.GraphModule, ph_to_tensor: dict) -> fx.GraphModule:
    """OPT-3 fallback: Fold eval-mode BatchNorm into the preceding conv. Aten IR level.

    Only fires if an aten._native_batch_norm_legit_no_training node survives to the
    decomposed graph (it does not on torch 2.11 — eval BN is decomposed before this seam,
    which is why OPT-3 folds at the functional level instead). Graceful no-op otherwise."""
    try:
        g = gm.graph
        folded = 0
        buf_idx = 0

        for bn_node in list(g.nodes):
            if not (bn_node.op == "call_function" and bn_node.target is _BN_TARGET):
                continue
            conv_node = bn_node.args[0]
            if not (
                isinstance(conv_node, fx.Node)
                and conv_node.op == "call_function"
                and conv_node.target in _CONV_TARGETS
            ):
                continue
            if len(conv_node.users) != 1:
                continue

            bn_weight = ph_to_tensor.get(bn_node.args[1])
            bn_bias = ph_to_tensor.get(bn_node.args[2])
            run_mean = ph_to_tensor.get(bn_node.args[3])
            run_var = ph_to_tensor.get(bn_node.args[4])
            eps = bn_node.args[6] if len(bn_node.args) > 6 else 1e-5

            if any(t is None for t in (bn_weight, bn_bias, run_mean, run_var)):
                logger.warning(
                    "[OPT-3-fallback fold_conv_bn] BN constants not resolvable — skipping"
                )
                continue

            conv_weight = ph_to_tensor.get(conv_node.args[1])
            conv_bias = (
                ph_to_tensor.get(conv_node.args[2])
                if len(conv_node.args) > 2 and conv_node.args[2] is not None
                else None
            )
            if conv_weight is None:
                logger.warning(
                    "[OPT-3-fallback fold_conv_bn] Conv weight not resolvable — skipping"
                )
                continue

            try:
                from torch.nn.utils import fuse_conv_bn_weights

                fused_w, fused_b = fuse_conv_bn_weights(
                    conv_weight.float(),
                    conv_bias.float() if conv_bias is not None else None,
                    run_mean.float(),
                    run_var.float(),
                    eps,
                    bn_weight.float(),
                    bn_bias.float(),
                )
            except Exception:
                scale = bn_weight.float() / torch.sqrt(run_var.float() + eps)
                fused_w = conv_weight.float() * scale.view(-1, 1, 1, 1)
                if conv_bias is not None:
                    fused_b = (conv_bias.float() - run_mean.float()) * scale + bn_bias.float()
                else:
                    fused_b = bn_bias.float() - run_mean.float() * scale

            # Match the conv weight's dtype at the aten seam (already bf16 if OPT-2 ran).
            target_dtype = conv_weight.dtype
            fused_w = fused_w.to(target_dtype)
            fused_b = fused_b.to(target_dtype)

            w_name = f"_folded_conv_weight_aten_{buf_idx}"
            b_name = f"_folded_conv_bias_aten_{buf_idx}"
            buf_idx += 1
            gm.register_buffer(w_name, fused_w)
            gm.register_buffer(b_name, fused_b)

            with g.inserting_before(conv_node):
                fw = g.get_attr(w_name)
                fb = g.get_attr(b_name)

            new_conv_args = (conv_node.args[0], fw, fb) + tuple(conv_node.args[3:])
            with g.inserting_before(bn_node):
                new_conv = g.call_function(
                    torch.ops.aten.convolution.default, new_conv_args
                )

            for user in list(bn_node.users):
                if (
                    user.op == "call_function"
                    and user.target is operator.getitem
                    and user.args[1] == 0
                ):
                    user.replace_all_uses_with(new_conv)
                    g.erase_node(user)
            if not bn_node.users:
                g.erase_node(bn_node)
            if not conv_node.users:
                g.erase_node(conv_node)

            folded += 1

        if folded == 0:
            logger.warning(
                "[OPT-3-fallback fold_conv_bn] No aten BatchNorm node (expected — eval BN "
                "is decomposed before the aten seam on torch 2.11); folded at functional "
                "level instead"
            )
            return gm

        g.eliminate_dead_code()
        g.lint()
        gm.recompile()
        logger.info(
            "[OPT-3-fallback fold_conv_bn] Folded %d aten BatchNorm node(s) into the "
            "preceding convolution(s)",
            folded,
        )
    except Exception as e:
        logger.warning("[OPT-3-fallback fold_conv_bn] Failed: %s", e)
    return gm


# ---------------------------------------------------------------------------
# Pass registry — routed by ir_level
# ---------------------------------------------------------------------------
_WEIGHT_VALUE_PASSES = {"OPT-3", "OPT-3-fallback"}

PASS_REGISTRY = [
    # Functional level (before compile_fx, on the Dynamo graph).
    # OPT-3 conv-BN fold MUST run here — AOTAutograd decomposes eval BN before the aten seam.
    {"id": "OPT-3", "level": "functional", "fn": _fpass_fold_conv_bn},
    # Aten level (inside compile_fx inner_compile, post-decomposition).
    # OPT-2 bf16 promotion runs on the decomposed conv/addmm; OPT-3-fallback is defensive.
    {"id": "OPT-2", "level": "aten", "fn": _apass_bf16_promotion},
    {"id": "OPT-3-fallback", "level": "aten", "fn": _apass_fold_conv_bn},
    # OPT-1 (channels_last) is non-graph — applied in get_model_and_input().
]

_FUNCTIONAL_PASSES = [p for p in PASS_REGISTRY if p["level"] == "functional"]
_ATEN_PASSES = [p for p in PASS_REGISTRY if p["level"] == "aten"]
_CONFIG_PASSES = [p for p in PASS_REGISTRY if p["level"] == "inductor_config"]


def _reads_weight_values(p: dict) -> bool:
    return p["id"] in _WEIGHT_VALUE_PASSES


# ---------------------------------------------------------------------------
# LEVEL 1 — Functional passes (Dynamo graph, pre-AOTAutograd)
# ---------------------------------------------------------------------------

def _run_functional_passes(gm: fx.GraphModule, example_inputs) -> fx.GraphModule:
    """Run all functional-level passes on the Dynamo graph before compile_fx.

    At this level convolutions and BatchNorm (F.batch_norm) are single high-level nodes, and
    weight/BN parameters are placeholder nodes whose REAL tensors are positionally matched to
    ``example_inputs`` (Dynamo hands the backend real tensors here). Weight-VALUE-reading
    passes (OPT-3 conv-BN fold) use that placeholder->tensor lookup. AOTAutograd recomputes
    meta when it traces the rewritten graph, so no FakeTensorProp is needed here."""
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    ph_to_tensor = {ph: t for ph, t in zip(placeholders, example_inputs)}
    for p in _FUNCTIONAL_PASSES:
        try:
            if _reads_weight_values(p):
                gm = p["fn"](gm, ph_to_tensor)
            else:
                gm = p["fn"](gm)
        except Exception as e:
            logger.warning("[%s] functional pass error: %s", p["id"], e)
    return gm


# ---------------------------------------------------------------------------
# LEVEL 3 — Inductor config patches (none for this workload)
# ---------------------------------------------------------------------------

def _build_config_patches() -> dict:
    """Collect/merge inductor_config-level patches, scoped to this compile_fx call only.
    This workload proposes no inductor_config pass, so the dict is empty."""
    patches: dict = {}
    for p in _CONFIG_PASSES:
        try:
            result = p["fn"]()
            if result:
                patches.update(result)
        except Exception as e:
            logger.warning("[%s] config pass error: %s", p["id"], e)
    return patches


# ---------------------------------------------------------------------------
# LEVEL 2 — Aten-level passes (inside compile_fx inner_compile hook)
# ---------------------------------------------------------------------------

def _repropagate_meta(gm: fx.GraphModule, example_inputs) -> None:
    """Re-run FakeTensorProp after a structural graph rewrite so inserted nodes (the bf16
    _to_copy casts, the folded conv + its get_attr buffers) get meta['val'] before
    compile_fx_inner runs."""
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
    """LEVEL 2 — Inductor inner_compile hook.

    compile_fx calls this with the fully decomposed Aten IR graph (post-AOTAutograd). Run
    aten-level passes (OPT-2 bf16 promotion, then OPT-3 fallback fold), re-propagating meta
    after each structural rewrite, then delegate to compile_fx_inner (Aten -> Triton).

    ``example_inputs`` may be FakeTensors under FakeTensorMode. Weight-VALUE-reading passes
    (OPT-3 fallback) use the threaded ``real_inputs`` for the placeholder->tensor lookup.
    ``**kwargs`` is forwarded verbatim to compile_fx_inner for forward-compatibility."""
    weight_source = real_inputs if real_inputs is not None else example_inputs
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    ph_to_tensor = {ph: t for ph, t in zip(placeholders, weight_source)}

    for p in _ATEN_PASSES:
        try:
            if _reads_weight_values(p):
                gm = p["fn"](gm, ph_to_tensor)
            else:
                gm = p["fn"](gm)
            _repropagate_meta(gm, example_inputs)
        except Exception as e:
            logger.warning("[%s] aten pass error: %s", p["id"], e)

    return compile_fx_inner(gm, example_inputs, **kwargs)


# ---------------------------------------------------------------------------
# Three-stage funnel: functional -> (AOTAutograd decomposition) -> aten -> config
# ---------------------------------------------------------------------------

def _compile_unit(gm: fx.GraphModule, example_inputs) -> Callable:
    """Fixed three-stage funnel for one (sub)graph.

    Stage 1 (functional): OPT-3 conv-BN fold on the Dynamo graph (folds eval-mode
             F.batch_norm into the preceding conv, reading real weight values; folded buffers
             are allocated bf16 to honor OPT-2's runtime dtype).
    Stage 2 (aten): compile_fx owns AOTAutograd + decomp; _aten_inner_compile runs OPT-2 bf16
             promotion on the decomposed conv/addmm, then the defensive OPT-3 aten fallback.
    Stage 3 (config): no inductor_config patches for this workload.
    OPT-1 (channels_last) is non-graph, applied in get_model_and_input()."""
    gm = _run_functional_passes(gm, list(example_inputs))
    inner = functools.partial(_aten_inner_compile, real_inputs=list(example_inputs))
    config_patches = _build_config_patches()
    return compile_fx(
        gm, example_inputs, inner_compile=inner, config_patches=config_patches
    )


# ---------------------------------------------------------------------------
# Partition input capture (dedup path)
# ---------------------------------------------------------------------------

def _capture_partition_inputs(
    split_gm: fx.GraphModule, example_inputs: list
) -> dict[str, list]:
    """Run split_gm once under no_grad to capture per-partition input tensors so each unique
    representative is compiled with correct (real-value) example inputs."""
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
# Backend: conv_block_opt
# ---------------------------------------------------------------------------

@register_backend
def conv_block_opt(gm: fx.GraphModule, example_inputs) -> Callable:
    """Custom torch.compile() backend for ConvBlock.

    Implements three optimizations from optimizations.json via the three-stage funnel:

      OPT-1 (non-graph):  channels_last (NHWC) memory format — applied in get_model_and_input()
      OPT-2 (aten):       bf16 dtype promotion of conv + linear-head compute (prereq for OPT-3)
      OPT-3 (functional): Conv-BN fold (inference) — folds eval-mode F.batch_norm into the
                          conv weight/bias on the Dynamo graph (BEFORE AOTAutograd decomposes
                          BN away; the aten placement marked in the JSON is a silent no-op).
                          Folded buffers are allocated bf16 to honor the OPT-2 prerequisite.

    Dedup-aware: ConvBlock stacks three ConvBnRelu blocks with different channel counts
    (3->64, 64->128, 128->256), so the blocks are NOT structurally identical and
    UniqueSubgraphRegistry returns an empty equivalence map -> flat compile path. The dedup
    branch is preserved for models with repeated identical blocks."""
    logger.info(
        "conv_block_opt backend: starting "
        "(functional[OPT-3 conv-BN fold] -> aten[OPT-2 bf16 -> OPT-3 fallback no-op]); "
        "OPT-1 channels_last applied non-graph"
    )

    registry = UniqueSubgraphRegistry(gm)
    equiv_map = registry.build_partition_equivalence_map()

    if not equiv_map:
        logger.info("conv_block_opt: no repeated layers, flat compile path")
        return _compile_unit(gm, example_inputs)

    logger.info(
        "conv_block_opt: %d duplicate partition(s), dedup compile path", len(equiv_map)
    )

    partition_inputs = _capture_partition_inputs(registry.split, example_inputs)
    for rep_name, rep_mod in registry.unique_reps:
        inputs = partition_inputs.get(rep_name, example_inputs)
        compiled = _compile_unit(rep_mod, inputs)
        rep_mod.forward = compiled
        for _, dup_mod in registry.duplicates_of(rep_name):
            dup_mod.forward = compiled

    return lambda *args: registry.split(*args)


# ---------------------------------------------------------------------------
# Workload interface
# ---------------------------------------------------------------------------
DEVICE = "cuda"
BATCH_SIZE = 16
IN_CHANNELS = 3
HEIGHT = 64
WIDTH = 64


def get_model_and_input() -> tuple:
    """Return (raw_model, input_tensor) on CUDA — uncompiled, unwarmed.

    OPT-1 (channels_last / NHWC) is a NON-GRAPH whole-module optimization applied here: the
    model and input are converted to channels_last so cuDNN consumes activations directly on
    its native NHWC tensor-op path and drops all 12 convertTensor permute kernels (plus the
    triton input-prep repack). The format propagates through the conv -> (folded BN) -> ReLU
    -> conv chain without Inductor inserting per-op transpose kernels.

    Model is returned in eval() — required for the eval-mode
    (_native_batch_norm_legit_no_training) BN fold (OPT-3) to be valid. The model is returned
    in FP32; OPT-2 (bf16) is applied in-graph by the backend (conv/linear compute only), so
    the public interface stays FP32 and numerically comparable to the baseline."""
    assert torch.cuda.is_available(), "CUDA required"
    from conv_block import ConvBlock

    model = ConvBlock().to(DEVICE).eval()
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH, device=DEVICE)

    # OPT-1 — channels_last (NHWC). Check current state first before converting.
    has_conv = any(isinstance(m, torch.nn.Conv2d) for m in model.modules())
    if has_conv:
        first_param = next(model.parameters(), None)
        already_cl = first_param is not None and first_param.is_contiguous(
            memory_format=torch.channels_last
        )
        if not already_cl:
            model = model.to(memory_format=torch.channels_last)
        if not x.is_contiguous(memory_format=torch.channels_last):
            x = x.to(memory_format=torch.channels_last)

    return model, x


if __name__ == "__main__":
    model, x = get_model_and_input()
    compiled = torch.compile(model, backend="conv_block_opt")
    with torch.no_grad():
        out = compiled(x)
    print("output shape:", out.shape, "dtype:", out.dtype)
