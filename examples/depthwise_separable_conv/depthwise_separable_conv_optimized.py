"""
depthwise_separable_conv_optimized.py — Custom torch.compile() backend for DepthwiseSepConv.

Registered backend: ``depthwise_separable_conv_opt``

Implements three optimizations from optimizations.json, routed to the correct IR level
via the three-stage funnel (functional -> aten -> inductor_config). Each pass executes at
the level where its pattern is unambiguous and the rewrite is sound.

Backend name: depthwise_separable_conv_opt  (model "depthwise_separable_conv" -> snake-case + _opt)

Pass summary (execution order: functional then aten then inductor_config):

  OPT-3  functional / medium  — Conv -> ReLU6 (hardtanh) epilogue fusion enablement
      At the Dynamo functional graph level, confirm that each convolution feeding a
      hardtanh (ReLU6) clamp is the clamp's sole producer/consumer so that, once OPT-1
      folds BN away at the aten level, Inductor's pointwise epilogue scheduler fuses the
      clamp directly onto the conv kernel rather than emitting a standalone elementwise
      kernel. This pass is structurally non-destructive (it does not rewrite nodes — the
      conv -> hardtanh chain is already a clean single-producer/single-consumer pair in
      this workload); it tags the conv producer with epilogue metadata and verifies the
      precondition Inductor needs. Depends on OPT-1: only after the BN affine kernel is
      removed does conv -> hardtanh become directly adjacent. Cross-level ordering
      (functional precondition check -> aten BN fold) is satisfied by the funnel: the
      functional pass only annotates, and the actual fusion is realized by Inductor after
      the aten-level BN fold has run.

  OPT-1  functional / high  — Conv-BN fold (inference)   [moved from aten -> functional]
      Fold every eval-mode BatchNorm into the weight and bias of its immediately-preceding
      convolution. In eval mode the BN is a pure affine with constant per-channel
      scale/bias, so scale = gamma / sqrt(var + eps), W' = W * scale[:,None,None,None],
      b' = (b - mean) * scale + beta is exact and lossless.

      CRITICAL CORRECTION: this fold was originally placed at the aten level inside
      _aten_inner_compile, but it was a SILENT NO-OP there. compile_fx runs AOTAutograd
      BEFORE the inner_compile seam, and AOTAutograd decomposes the eval-mode BatchNorm
      into primitive ops — so at the aten seam there are ZERO
      aten._native_batch_norm_legit_no_training nodes to match. The fold must therefore
      run on the Dynamo FUNCTIONAL graph the backend receives BEFORE handing off to
      compile_fx, where the eval-mode BN is still a single high-level
      torch.nn.functional.batch_norm node (training=False) fed directly by a conv2d node.

      At the functional level the conv weight / BN params are PLACEHOLDER nodes whose real
      tensors are positionally matched to example_inputs (Dynamo hands the backend REAL
      tensors here, not FakeTensors), so we read real values directly. The folded
      weight/bias are registered as buffers, wired in via get_attr nodes, and a synthetic
      LocalSource is registered in gm._param_name_to_source for each (AOTAutograd asserts
      every backend-introduced get_attr has a unique non-None source). The single
      F.batch_norm node is then replaced by the new conv (no getitem at this level) and
      erased. AOTAutograd decomposes the now-BN-free graph normally; the standalone BN
      affine DRAM round-trip (~30% of attributed time) is gone, and Inductor epilogue-fuses
      the residual hardtanh (OPT-3). A defensive aten-level fallback ("OPT-1-fallback")
      remains registered and gracefully no-ops on torch 2.11.

  OPT-2  non-graph / medium  — channels_last (NHWC) memory format
      A 1x1 pointwise conv is a per-pixel matmul over channels; in NCHW the contraction
      stride is H*W, forcing the strided '_tn' cutlass GEMM path with zero L1 reuse. In
      channels_last the channel axis is innermost (stride 1), giving coalesced loads and
      better tensor-core feeding. The depthwise 3x3 convs and the (folded) BN/ReLU6
      epilogues also prefer NHWC. Per the funnel rules, whole-module memory_format is a
      NON-GRAPH optimization applied in get_model_and_input() (model + input both
      converted), letting the format propagate through the conv stack without inserting
      per-op transpose kernels. Applying it in-graph would risk a stray aten._to_copy
      transpose that reintroduces a DRAM round-trip and negates the gain.

Prerequisite / ordering rationale:
  - OPT-1 and OPT-3 are BOTH functional-level passes and are sequenced within the level:
    OPT-1 (BN fold) runs FIRST, then OPT-3. The conv -> hardtanh epilogue precondition
    only exists once BN is folded away (after the fold, hardtanh's producer is the new
    conv directly). The registry order [OPT-1, OPT-3] enforces this prerequisite.
  - OPT-3 only annotates / verifies the conv -> hardtanh precondition; Inductor's scheduler
    performs the actual epilogue fusion after the (now BN-free) conv is lowered.
  - OPT-1 reads weight values from the REAL example_inputs (Dynamo passes real tensors to
    the backend at the functional level) and must run on the Dynamo graph where the BN is
    still a single F.batch_norm node — NOT at the aten seam, where AOTAutograd has already
    decomposed it (the prior aten placement found zero BN nodes and silently no-op'd).
  - OPT-2 (non-graph) is independent and applied at the module level before tracing.

IR-level mechanics (torch 2.11):
  compile_fx owns AOTAutograd, the decomp table, the boxed calling convention and the
  partitioner. We do NOT use aot_autograd(fw_compiler=compile_fx) — on torch 2.11 that
  raises AssertionError inside copy_misaligned_inputs. The funnel runs functional-level
  passes BEFORE compile_fx, aten-level passes through its inner_compile seam, and
  inductor_config passes as scoped config_patches.

compile_mode = "inductor" (from optimizations.json analysis.compile_mode).
dtype = float32 (no dtype-promotion pass proposed; BN fold is exact in FP32).
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

# ---------------------------------------------------------------------------
# Op targets
# ---------------------------------------------------------------------------
# Aten-level (post-decomposition) targets — used by the defensive aten fallback only.
_BN_TARGET = torch.ops.aten._native_batch_norm_legit_no_training.default
_CONV_TARGETS = frozenset(
    {
        torch.ops.aten.convolution.default,
        torch.ops.aten.cudnn_convolution.default,
    }
)
_HARDTANH_TARGETS = frozenset(
    {
        torch.ops.aten.hardtanh.default,
        torch.ops.aten.hardtanh_.default,
    }
)


def _is_functional_conv(node: fx.Node) -> bool:
    """True if `node` is a functional-level convolution call.

    At the Dynamo graph level a Conv2d traces to a single high-level node whose target
    is the builtin ``torch.conv2d`` (a builtin method object). We also accept
    ``F.conv2d`` / ``torch.nn.functional.conv2d`` and the aten conv overloads to stay
    robust across Dynamo lowering variants. Matched by identity first, then by name."""
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
    ``torch.nn.functional.batch_norm`` node (the BN module's eval path calls
    F.batch_norm with training=False). Matched by identity first, then by name."""
    if node.op != "call_function":
        return False
    t = node.target
    if t is torch.nn.functional.batch_norm:
        return True
    return getattr(t, "__name__", "") == "batch_norm"


# ---------------------------------------------------------------------------
# OPT-1 — Conv-BN fold (inference). ir_level=functional. Confidence: high.
#
# RATIONALE FOR FUNCTIONAL LEVEL (corrected from the original aten placement):
# AOTAutograd decomposes the eval-mode BatchNorm into primitive ops BEFORE the
# inner_compile (aten) seam runs, so by the time _aten_inner_compile sees the graph
# there are ZERO aten._native_batch_norm_legit_no_training nodes left to fold — the
# aten pass was a silent no-op. The robust fold point is the Dynamo functional graph
# the backend receives BEFORE compile_fx/AOTAutograd, where the eval-mode BN is still a
# single high-level torch.nn.functional.batch_norm node fed directly by a conv2d node.
#
# Functional batch_norm args:
#   (input, running_mean, running_var, weight, bias, training, momentum, eps)
# Functional conv2d args:
#   (input, weight, bias, stride, padding, dilation, groups)
#
# At this level the conv weight / BN params are PLACEHOLDER nodes whose real tensors are
# positionally matched to example_inputs (which are REAL Parameters here, not FakeTensors
# — Dynamo passes real inputs to the backend). We read those real values, fold with
# torch.nn.utils.fuse_conv_bn_weights, register the fused weight/bias as buffers, wire
# them in via get_attr nodes, and rewire the BN's consumer to the new conv. AOTAutograd
# then decomposes the (now BN-free) conv normally and the standalone BN affine kernel is
# gone. Inductor epilogue-fuses the residual hardtanh (OPT-3).
# ---------------------------------------------------------------------------

def _register_synthetic_source(gm: fx.GraphModule, attr_name: str) -> None:
    """Give a backend-introduced functional-level get_attr a source AOTAutograd accepts.

    Dynamo stores `_param_name_to_source` (name -> torch._guards.Source) on the
    GraphModule it hands the backend; aot_export later asserts every lifted get_attr
    target is present there with a unique, non-None source. New buffers added by a
    functional pass are absent, so we register a unique LocalSource keyed on the buffer
    name. No-op (best effort) if the map/source API is unavailable on this torch build."""
    try:
        src_map = getattr(gm, "_param_name_to_source", None)
        if src_map is None:
            return
        from torch._dynamo.source import LocalSource

        if attr_name not in src_map:
            src_map[attr_name] = LocalSource(attr_name)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(
            "[OPT-1 fold_conv_bn] could not register source for %s: %s", attr_name, e
        )


def _fpass_fold_conv_bn(
    gm: fx.GraphModule, ph_to_tensor: dict
) -> fx.GraphModule:
    """OPT-1: Fold eval-mode BatchNorm into the preceding convolution. Functional IR level.

    ph_to_tensor maps placeholder nodes -> real parameter tensors (from example_inputs).
    Folding is exact and lossless in eval mode (training=False uses fixed running stats)."""
    try:
        g = gm.graph
        folded = 0
        buf_idx = 0

        for bn_node in list(g.nodes):
            if not _is_functional_batch_norm(bn_node):
                continue

            # F.batch_norm(input, running_mean, running_var, weight, bias,
            #              training, momentum, eps)
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

            # Only fold the eval-mode (training=False) BN.
            if training:
                continue

            if not (isinstance(conv_node, fx.Node) and _is_functional_conv(conv_node)):
                continue
            # BN must be the conv's sole consumer to fold safely.
            if len(conv_node.users) != 1:
                continue

            # conv2d(input, weight, bias, stride, padding, dilation, groups)
            conv_weight_n = conv_node.args[1] if len(conv_node.args) > 1 else None
            conv_bias_n = conv_node.args[2] if len(conv_node.args) > 2 else None

            def _resolve(n):
                if n is None:
                    return None
                if isinstance(n, fx.Node):
                    return ph_to_tensor.get(n)
                return n  # already a constant

            conv_weight = _resolve(conv_weight_n)
            conv_bias = _resolve(conv_bias_n)
            run_mean = _resolve(run_mean_n)
            run_var = _resolve(run_var_n)
            bn_weight = _resolve(bn_weight_n)
            bn_bias = _resolve(bn_bias_n)

            if conv_weight is None:
                logger.warning(
                    "[OPT-1 fold_conv_bn] Conv weight not resolvable from real inputs "
                    "— skipping this site"
                )
                continue
            if any(t is None for t in (run_mean, run_var, bn_weight, bn_bias)):
                logger.warning(
                    "[OPT-1 fold_conv_bn] BN constants not resolvable from real inputs "
                    "— skipping this site"
                )
                continue

            # Exact, lossless eval-mode fold. Prefer the canonical helper.
            try:
                from torch.nn.utils import fuse_conv_bn_weights

                fused_w, fused_b = fuse_conv_bn_weights(
                    conv_weight,
                    conv_bias,
                    run_mean,
                    run_var,
                    eps,
                    bn_weight,
                    bn_bias,
                )
            except Exception:
                scale = bn_weight / torch.sqrt(run_var + eps)
                fused_w = conv_weight * scale.view(-1, 1, 1, 1)
                if conv_bias is not None:
                    fused_b = (conv_bias - run_mean) * scale + bn_bias
                else:
                    fused_b = bn_bias - run_mean * scale

            w_name = f"_folded_conv_weight_{buf_idx}"
            b_name = f"_folded_conv_bias_{buf_idx}"
            buf_idx += 1
            # Keep the conv weight's memory format so OPT-2 (channels_last) is preserved.
            gm.register_buffer(w_name, fused_w.contiguous(
                memory_format=torch.channels_last
            ) if fused_w.dim() == 4 and conv_weight.is_contiguous(
                memory_format=torch.channels_last
            ) else fused_w)
            gm.register_buffer(b_name, fused_b)

            # Dynamo hands AOTAutograd a `_param_name_to_source` map; every get_attr the
            # backend introduces at the functional level must have a UNIQUE non-None source
            # there or aot_export raises "<name> not found in param_name_to_source". Register
            # a synthetic LocalSource for each new folded buffer.
            _register_synthetic_source(gm, w_name)
            _register_synthetic_source(gm, b_name)

            with g.inserting_before(conv_node):
                fw = g.get_attr(w_name)
                fb = g.get_attr(b_name)

            # Rebuild the conv call with folded weight + new bias, preserving the
            # remaining args (stride, padding, dilation, groups).
            tail = tuple(conv_node.args[3:])
            new_conv_args = (conv_node.args[0], fw, fb) + tail
            with g.inserting_before(bn_node):
                new_conv = g.call_function(
                    conv_node.target, new_conv_args, dict(conv_node.kwargs)
                )

            # F.batch_norm returns a single tensor at the functional level (no getitem).
            bn_node.replace_all_uses_with(new_conv)
            g.erase_node(bn_node)
            if not conv_node.users:
                g.erase_node(conv_node)

            folded += 1

        if folded == 0:
            logger.warning(
                "[OPT-1 fold_conv_bn] No foldable conv -> batch_norm pair at functional "
                "level — pass not applied"
            )
            return gm

        g.eliminate_dead_code()
        g.lint()
        gm.recompile()
        logger.info(
            "[OPT-1 fold_conv_bn] APPLIED — folded %d eval-mode batch_norm node(s) into "
            "the preceding convolution [functional IR; matched "
            "torch.nn.functional.batch_norm <- conv2d]",
            folded,
        )
    except Exception as e:
        logger.warning("[OPT-1 fold_conv_bn] Failed: %s", e)
    return gm


# ---------------------------------------------------------------------------
# OPT-3 — Conv -> ReLU6 (hardtanh) epilogue fusion enablement. ir_level=functional.
# Confidence: medium.
#
# At the functional level the activation is a single hardtanh (ReLU6) node fed by a
# single convolution node. This pass verifies the conv -> hardtanh chain is a clean
# single-producer / single-consumer pair (the precondition Inductor's pointwise
# epilogue scheduler needs to fuse the clamp onto the conv kernel) and tags the conv
# producer with epilogue metadata. It does NOT rewrite nodes — forcing a manual fusion
# here would fight Inductor's scheduler; instead we ensure the conv output is not
# materialized by an intervening multi-user view/clone, which (after OPT-1 removes the
# BN affine kernel) lets the default scheduler epilogue-fuse the clamp.
# ---------------------------------------------------------------------------

def _fpass_mark_conv_relu6_epilogue(gm: fx.GraphModule) -> fx.GraphModule:
    """OPT-3: Verify/annotate conv -> hardtanh single-consumer chains for epilogue fusion.

    Functional IR level. Non-destructive: tags conv producers and confirms the
    precondition so Inductor can fuse the ReLU6 clamp into the conv epilogue once
    OPT-1 has folded BN away. Graceful no-op if the pattern is absent."""
    try:
        g = gm.graph
        marked = 0
        for ht in list(g.nodes):
            is_hardtanh = ht.op == "call_function" and (
                ht.target is torch.nn.functional.hardtanh
                or ht.target in _HARDTANH_TARGETS
                or getattr(ht.target, "__name__", "") in ("hardtanh", "hardtanh_")
            )
            if not is_hardtanh:
                continue
            if not ht.args:
                continue
            producer = ht.args[0]
            if not isinstance(producer, fx.Node):
                continue
            # After OPT-1 has folded BN away, the hardtanh's producer is the (new) conv
            # node directly. Accept only a direct conv producer; if a batch_norm node is
            # still present here, OPT-1 did not fold this site, so the epilogue fusion
            # precondition is not yet met and we leave it alone.
            if not _is_functional_conv(producer):
                continue
            # Require hardtanh to be the conv's sole consumer (no forced materialization).
            if len(producer.users) != 1:
                continue
            min_val = ht.args[1] if len(ht.args) > 1 else 0.0
            max_val = ht.args[2] if len(ht.args) > 2 else 6.0
            producer.meta["epilogue_activation"] = ("hardtanh", min_val, max_val)
            marked += 1

        if marked == 0:
            logger.warning(
                "[OPT-3 conv_relu6_epilogue] No direct conv -> hardtanh pair at functional "
                "level (BN likely still present) — Inductor handles epilogue fusion after "
                "OPT-1 BN fold; pass is a no-op annotation here"
            )
            return gm

        logger.info(
            "[OPT-3 conv_relu6_epilogue] Annotated %d conv -> ReLU6 epilogue site(s) "
            "[functional IR]; Inductor fuses the clamp after OPT-1 BN fold",
            marked,
        )
    except Exception as e:
        logger.warning("[OPT-3 conv_relu6_epilogue] Failed: %s", e)
    return gm


# ---------------------------------------------------------------------------
# OPT-1 — Conv-BN fold (inference). ir_level=aten. Confidence: high.
#
# Runs inside _aten_inner_compile after AOTAutograd has fully decomposed the graph.
# Every aten._native_batch_norm_legit_no_training whose sole producer is a convolution
# is folded into that convolution's weight/bias. Reads actual weight VALUES, so it uses
# the placeholder->tensor lookup built from the threaded real_inputs.
# ---------------------------------------------------------------------------

def _apass_fold_conv_bn(gm: fx.GraphModule, ph_to_tensor: dict) -> fx.GraphModule:
    """OPT-1: Fold eval-mode BatchNorm into the preceding convolution. Aten IR level.

    ph_to_tensor maps placeholder nodes -> real parameter tensors (from real_inputs).
    Folding is exact and lossless in eval mode (no_training BN uses fixed running stats)."""
    try:
        g = gm.graph
        folded = 0
        buf_idx = 0

        for bn_node in list(g.nodes):
            if not (bn_node.op == "call_function" and bn_node.target is _BN_TARGET):
                continue

            # aten._native_batch_norm_legit_no_training(
            #     input, weight, bias, running_mean, running_var, momentum, eps)
            conv_node = bn_node.args[0]
            if not (
                isinstance(conv_node, fx.Node)
                and conv_node.op == "call_function"
                and conv_node.target in _CONV_TARGETS
            ):
                continue
            # BN must be the conv's sole consumer to fold safely.
            if len(conv_node.users) != 1:
                continue

            bn_weight = ph_to_tensor.get(bn_node.args[1])
            bn_bias = ph_to_tensor.get(bn_node.args[2])
            run_mean = ph_to_tensor.get(bn_node.args[3])
            run_var = ph_to_tensor.get(bn_node.args[4])
            eps = bn_node.args[6] if len(bn_node.args) > 6 else 1e-5

            if any(t is None for t in (bn_weight, bn_bias, run_mean, run_var)):
                logger.warning(
                    "[OPT-1 fold_conv_bn] BN constants not resolvable from real_inputs "
                    "— skipping this site"
                )
                continue

            # aten.convolution.default args:
            #   (input, weight, bias, stride, padding, dilation,
            #    transposed, output_padding, groups)
            conv_weight = ph_to_tensor.get(conv_node.args[1])
            conv_bias = (
                ph_to_tensor.get(conv_node.args[2])
                if len(conv_node.args) > 2 and conv_node.args[2] is not None
                else None
            )
            if conv_weight is None:
                logger.warning(
                    "[OPT-1 fold_conv_bn] Conv weight not resolvable from real_inputs "
                    "— skipping this site"
                )
                continue

            # Exact fold. Prefer torch.nn.utils.fuse_conv_bn_weights for the canonical
            # (fused_w, fused_b); fall back to the explicit formula if unavailable.
            try:
                from torch.nn.utils import fuse_conv_bn_weights

                fused_w, fused_b = fuse_conv_bn_weights(
                    conv_weight,
                    conv_bias,
                    run_mean,
                    run_var,
                    eps,
                    bn_weight,
                    bn_bias,
                )
            except Exception:
                scale = bn_weight / torch.sqrt(run_var + eps)
                fused_w = conv_weight * scale.view(-1, 1, 1, 1)
                if conv_bias is not None:
                    fused_b = (conv_bias - run_mean) * scale + bn_bias
                else:
                    fused_b = bn_bias - run_mean * scale

            w_name = f"_folded_conv_weight_{buf_idx}"
            b_name = f"_folded_conv_bias_{buf_idx}"
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

            # bn_node returns a 3-tuple; redirect the getitem(bn_node, 0) consumer.
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
                "[OPT-1 fold_conv_bn] No foldable conv -> BatchNorm pair found — "
                "pass not applied"
            )
            return gm

        g.eliminate_dead_code()
        g.lint()
        gm.recompile()
        logger.info(
            "[OPT-1 fold_conv_bn] Folded %d BatchNorm node(s) into preceding "
            "convolution(s) [aten IR]",
            folded,
        )
    except Exception as e:
        logger.warning("[OPT-1 fold_conv_bn] Failed: %s", e)
    return gm


# ---------------------------------------------------------------------------
# Pass registry — routed by ir_level
# ---------------------------------------------------------------------------

# Passes that read actual weight VALUES need the ph_to_tensor lookup (real inputs).
# Applies at both the functional level (Dynamo example_inputs are real tensors) and the
# defensive aten fallback (threaded real_inputs).
_WEIGHT_VALUE_PASSES = {"OPT-1", "OPT-1-fallback"}

PASS_REGISTRY = [
    # Functional-level passes (run before compile_fx, on the Dynamo graph).
    # OPT-1 MUST precede OPT-3: the conv -> hardtanh epilogue precondition only exists
    # once BN is folded away (OPT-1 -> OPT-3 prerequisite, satisfied by registry order).
    {"id": "OPT-1", "level": "functional", "fn": _fpass_fold_conv_bn},
    {"id": "OPT-3", "level": "functional", "fn": _fpass_mark_conv_relu6_epilogue},
    # Defensive aten-level fallback for OPT-1: only fires if BN somehow survives to the
    # decomposed graph (it does not on torch 2.11 — eval BN is decomposed pre-inner_compile).
    {"id": "OPT-1-fallback", "level": "aten", "fn": _apass_fold_conv_bn},
    # OPT-2 (channels_last) is non-graph — applied in get_model_and_input().
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

    At this level convolutions, BatchNorm (F.batch_norm) and ReLU6 (hardtanh) are single
    high-level nodes, and weight/BN parameters are placeholder nodes whose REAL tensors are
    positionally matched to ``example_inputs`` (Dynamo hands the backend real tensors here,
    not FakeTensors). Weight-VALUE-reading passes (OPT-1 conv-BN fold) use that
    placeholder->tensor lookup. AOTAutograd recomputes meta when it traces the rewritten
    graph, so no FakeTensorProp is needed at this level."""
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
    """Collect and merge all inductor_config-level patches. Scoped to this compile_fx
    call only — no global Inductor config mutation. This workload proposes no
    inductor_config pass, so the dict is empty."""
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
    """Re-run FakeTensorProp after a structural graph rewrite so inserted nodes
    (the new aten.convolution and its get_attr buffers) get meta['val'] before
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
        logger.warning(
            "[depthwise_separable_conv_opt] meta re-propagation skipped: %s", e
        )


def _aten_inner_compile(
    gm: fx.GraphModule, example_inputs, *, real_inputs=None, **kwargs
) -> Callable:
    """LEVEL 2 — Inductor inner_compile hook.

    compile_fx calls this with the fully decomposed Aten IR graph (post-AOTAutograd).
    Run aten-level passes (OPT-1 conv-BN fold), re-propagating meta after each
    structural rewrite, then delegate to compile_fx_inner (Aten -> Triton).

    ``example_inputs`` may be FakeTensors under FakeTensorMode. Weight-VALUE-reading
    passes (OPT-1) use the threaded ``real_inputs`` for the placeholder->tensor lookup.
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

    Stage 1: run functional passes on the Dynamo graph — OPT-1 conv-BN fold (folds the
             eval-mode F.batch_norm into the preceding conv, reading real weight values),
             then OPT-3 epilogue annotation.
    Stage 2: compile_fx owns AOTAutograd + decomp; our _aten_inner_compile hook runs the
             defensive OPT-1 aten fallback (a no-op on torch 2.11 — eval BN is already
             decomposed before this seam, which is why OPT-1 now folds at the functional
             level instead).
    Stage 3: OPT-2 is non-graph; no inductor_config patches for this workload."""
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
    """Run split_gm once under no_grad to capture per-partition input tensors so each
    unique representative is compiled with correct (and real-value) example inputs."""
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
# Backend: depthwise_separable_conv_opt
# ---------------------------------------------------------------------------

@register_backend
def depthwise_separable_conv_opt(gm: fx.GraphModule, example_inputs) -> Callable:
    """Custom torch.compile() backend for DepthwiseSepConv.

    Implements three optimizations from optimizations.json via the three-stage funnel
    (functional -> aten -> inductor_config):

      OPT-1 (functional): Conv-BN fold (inference) — folds eval-mode F.batch_norm into
                          conv weight/bias on the Dynamo graph (BEFORE AOTAutograd
                          decomposes BN away; the prior aten placement was a no-op)
      OPT-3 (functional): Conv -> ReLU6 epilogue fusion enablement (annotation), after fold
      OPT-2 (non-graph):  channels_last memory format — applied in get_model_and_input()

    Dedup-aware: DepthwiseSepConv stacks three DWSepBlocks with different channel counts
    (32->64, 64->128, 128->256), so the blocks are NOT structurally identical and
    UniqueSubgraphRegistry returns an empty equivalence map -> flat compile path. The
    dedup branch is preserved for models with repeated identical blocks.
    """
    logger.info(
        "depthwise_separable_conv_opt backend: starting "
        "(functional[OPT-1 conv-BN fold -> OPT-3 epilogue] -> aten[OPT-1 fallback no-op]); "
        "OPT-2 channels_last applied non-graph"
    )

    registry = UniqueSubgraphRegistry(gm)
    equiv_map = registry.build_partition_equivalence_map()

    if not equiv_map:
        # No repeated layers — flat compile preserves cross-layer Inductor fusion.
        logger.info(
            "depthwise_separable_conv_opt: no repeated layers, flat compile path"
        )
        return _compile_unit(gm, example_inputs)

    logger.info(
        "depthwise_separable_conv_opt: %d duplicate partition(s), dedup compile path",
        len(equiv_map),
    )

    # Compile each unique representative through the same funnel; share the compiled
    # callable with all structural duplicates. Functional passes run per-rep (inside
    # _compile_unit), never on the pre-split graph.
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
IN_CHANNELS = 32
HEIGHT = 56
WIDTH = 56


def get_model_and_input() -> tuple:
    """Return (raw_model, input_tensor) on CUDA — uncompiled, unwarmed.

    OPT-2 (channels_last / NHWC) is a NON-GRAPH whole-module optimization applied here:
    the model and input are converted to channels_last so the format propagates through
    the conv -> (folded BN) -> ReLU6 -> conv chain without Inductor inserting per-op
    transpose kernels. A 1x1 pointwise conv is a per-pixel channel matmul; NHWC makes the
    channel (contraction) axis stride-1, giving coalesced loads for the cutlass GEMM path.

    Model dtype: FP32 (matches optimizations.json analysis.dtype = "float32"). OPT-1
    (conv-BN fold) is exact and lossless in FP32 and is applied in-graph; no dtype
    promotion is proposed for this workload. The model is returned with .eval() set —
    required for the eval-mode (_native_batch_norm_legit_no_training) BN fold to be valid.
    """
    assert torch.cuda.is_available(), "CUDA required"
    from depthwise_separable_conv import DepthwiseSepConv

    model = DepthwiseSepConv().to(DEVICE).eval()
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH, device=DEVICE)

    # OPT-2 — channels_last (NHWC). Check current state first (the baseline may already
    # be channels_last) before converting model and input.
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
    compiled = torch.compile(model, backend="depthwise_separable_conv_opt")
    with torch.no_grad():
        out = compiled(x)
    print("output shape:", out.shape, "dtype:", out.dtype)
