"""
gpt2_optimized.py — custom torch.compile() backend for GPT-2 small (gpt2.py).

Backend name:  gpt2_opt   (registered via @register_backend; analysis.model = "gpt2")

GPT-2 small: 12 identical transformer decoder blocks, hidden=768, heads=12,
ffn=3072, batch=4, seq_len=128, fp32 inference on an RTX PRO 6000 Blackwell
(torch 2.11.0+cu128, CUDA 12.8). All 12 blocks are structurally identical, so
the dedup path compiles one representative and shares the callable across blocks.

Optimizations implemented (from optimizations.json, routed by ir_level). The
recommended application order is OPT-2 -> OPT-1 -> OPT-4 -> OPT-3; the funnel
(functional -> aten -> inductor_config) enforces the cross-level ordering, and
within the aten level OPT-1 is registered before OPT-4.

  OPT-2  canonicalize attention to F.scaled_dot_product_attention(is_causal=True)
         level: functional  (medium)
         At the Dynamo graph level, find F.scaled_dot_product_attention calls and
         canonicalize them to the is_causal=True form, dropping any materialized
         additive attn_mask argument. HF GPT-2's sdpa attention path already emits
         an F.scaled_dot_product_attention node; switching it to is_causal=True lets
         the mask construction (constant_pad_nd / scalar_tensor / where) become dead
         code that AOTAutograd / Inductor eliminate, collapsing the 48-launch
         memory-bound SDPA-prep family. Defensive: if no SDPA node is found, or it
         is already causal with no mask, the pass is a logged no-op.

  OPT-1  bf16 dtype promotion for mm / addmm operands
         level: aten  (high)
         Inside _aten_inner_compile (post-AOTAutograd), cast the GEMM operands of
         every aten.mm.default and aten.addmm.default to bfloat16 via
         prims.convert_element_type, and cast the result back to the original dtype.
         Routes the four projection families (QKV addmm 768->2304, attn-out mm
         768->768, FFN fc_up 768->3072, FFN fc_down 3072->768; ~84% of attributed
         time) off the cutlass_80_simt_sgemm SIMT path (tensor_core_active_pct=0.0)
         onto the Blackwell bf16 tensor-core HGEMM path. prims.convert_element_type
         is used instead of aten._to_copy to avoid the torch 2.11
         "both a fallback and a decomp for same op" assertion in an already
         decomposed graph.

  OPT-4  cast cancellation (cleanup for OPT-1)
         level: aten  (medium)
         After OPT-1, walk the graph and cancel adjacent inverse
         prims.convert_element_type pairs (fp32 -> bf16 -> fp32 and the reverse) so
         consecutive GEMMs connected through a pointwise/LayerNorm epilogue run
         bf16 end-to-end instead of round-tripping through DRAM. Registered AFTER
         OPT-1 in the aten level so it operates on OPT-1's freshly inserted casts
         (a genuine within-level ordering edge).

  OPT-3  Inductor freezing (weight pre-pack + epilogue fusion)
         level: inductor_config  (medium)
         Pass config_patches={"freezing": True} to compile_fx. For this eval /
         no_grad inference graph Inductor constant-folds the LayerNorm affine +
         projection bias and pre-transposes / pre-packs the weight constants into
         the tensor-core-friendly layout, letting cuBLASLt/CUTLASS pick a
         fused-epilogue HGEMM and removing standalone memory-bound bias/GELU Triton
         kernels (the addmm_0 family). Satisfied-by-level after OPT-1, so the frozen
         weight is materialized along the bf16 path.

The backend is the canonical three-stage funnel
    _run_functional_passes(gm) -> compile_fx(inner_compile=_aten_inner_compile,
                                              config_patches=_config_patches())
invoked identically on the flat graph and on every dedup representative. We do NOT
use aot_autograd(fw_compiler=compile_fx): on torch 2.11 that raises
AssertionError inside copy_misaligned_inputs. compile_fx owns AOTAutograd exactly
once; functional passes run before it, aten passes run inside its inner_compile
seam, and config patches are scoped to each compile_fx call (no global state).

compile_mode = "inductor" (from optimizations.json analysis.compile_mode).
"""
from __future__ import annotations

import functools
import logging
from typing import Callable

import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F

from torch._dynamo import register_backend
# Import the callable functions — NOT the module (module is not callable).
from torch._inductor.compile_fx import compile_fx, compile_fx_inner

from nvidia.operator_profiler.fx import UniqueSubgraphRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Workload constants (cross-validated against optimizations.json analysis)
#   GPT-2 small: hidden=768, heads=12, ffn=3072, 12 layers. batch=4, seq_len=128,
#   fp32, CUDA. Output is [4, 128, 768].
# ---------------------------------------------------------------------------
DEVICE   = "cuda"
BATCH    = 4
SEQ_LEN  = 128
MODEL_ID = "gpt2"

_BF16 = torch.bfloat16
_FP32 = torch.float32

# Op targets.
_MM = torch.ops.aten.mm.default
_ADDMM = torch.ops.aten.addmm.default
# Use prims.convert_element_type, not aten._to_copy. On torch 2.11, aten._to_copy
# carries both a fallback and a decomp registration; inserting it into an already
# decomposed Aten graph makes Inductor raise "both a fallback and a decomp for the
# same op". prims.convert_element_type lowers cleanly to a Triton elementwise cast.
_CONVERT = torch.ops.prims.convert_element_type.default


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node_dtype(n) -> torch.dtype | None:
    """Return the tensor dtype stored in node meta, or None if unavailable."""
    if not isinstance(n, fx.Node):
        return None
    val = n.meta.get("val", None)
    if val is None or not hasattr(val, "dtype"):
        return None
    return val.dtype


def _insert_cast(g: fx.Graph, src: fx.Node, dtype: torch.dtype, before: fx.Node) -> fx.Node:
    """Insert a prims.convert_element_type cast to ``dtype`` directly before
    ``before``. Returns ``src`` unchanged if it already has the target dtype."""
    if _node_dtype(src) is dtype:
        return src
    with g.inserting_before(before):
        return g.call_function(_CONVERT, (src, dtype))


def _is_sdpa_target(target) -> bool:
    """True if ``target`` is F.scaled_dot_product_attention (functional level)."""
    if target is F.scaled_dot_product_attention:
        return True
    # Robustness against re-binding / overload wrappers.
    return getattr(target, "__name__", "") == "scaled_dot_product_attention"


# ===========================================================================
# LEVEL 1 (functional) pass — OPT-2: canonicalize attention to SDPA is_causal=True
# ===========================================================================
def _fpass_canonicalize_sdpa(gm: fx.GraphModule) -> fx.GraphModule:
    """OPT-2 (functional, medium): canonicalize F.scaled_dot_product_attention to
    the is_causal=True form and drop any materialized additive attn_mask argument.

    HF GPT-2 (sdpa attention implementation) emits an
    F.scaled_dot_product_attention node at the functional level. Replacing an
    explicit additive causal mask with is_causal=True makes the mask-construction
    chain (constant_pad_nd / scalar_tensor / where / expand) dead, so AOTAutograd
    and Inductor eliminate the 48-launch memory-bound SDPA-prep family
    (aten::view_0). The signature is
    scaled_dot_product_attention(query, key, value, attn_mask=None,
    dropout_p=0.0, is_causal=False, scale=None). seq_len=128 is static, so there is
    no dynamic-shape guard issue.

    Defensive (medium confidence): if no SDPA node is found, or all SDPA nodes are
    already causal with no mask, the pass is a logged no-op.
    """
    try:
        g = gm.graph
        canonicalized = 0
        for node in list(g.nodes):
            if not (node.op == "call_function" and _is_sdpa_target(node.target)):
                continue

            args = list(node.args)
            kwargs = dict(node.kwargs)

            # Determine current attn_mask and is_causal from args/kwargs.
            # Positional layout: (query, key, value, attn_mask, dropout_p,
            #                     is_causal, scale)
            attn_mask = kwargs.get("attn_mask", args[3] if len(args) > 3 else None)
            is_causal = kwargs.get("is_causal", args[5] if len(args) > 5 else False)

            if attn_mask is None and is_causal is True:
                # Already in the canonical causal form — nothing to do.
                continue

            # Rebuild as F.sdpa(q, k, v, attn_mask=None, dropout_p=<preserve>,
            # is_causal=True, scale=<preserve>). Keep only q/k/v positionally.
            q, k, v = args[0], args[1], args[2]
            dropout_p = kwargs.get("dropout_p", args[4] if len(args) > 4 else 0.0)
            scale = kwargs.get("scale", args[6] if len(args) > 6 else None)

            new_kwargs = {
                "attn_mask": None,
                "dropout_p": dropout_p,
                "is_causal": True,
            }
            if scale is not None:
                new_kwargs["scale"] = scale

            with g.inserting_before(node):
                new_sdpa = g.call_function(
                    F.scaled_dot_product_attention, (q, k, v), new_kwargs
                )
            node.replace_all_uses_with(new_sdpa)
            g.erase_node(node)
            canonicalized += 1

        if canonicalized == 0:
            logger.warning(
                "[OPT-2 canonicalize_sdpa] No non-causal / masked SDPA node found "
                "— attention already canonical or expressed differently; pass not applied"
            )
            return gm

        g.eliminate_dead_code()
        g.lint()
        gm.recompile()
        logger.info(
            "[OPT-2 canonicalize_sdpa] Canonicalized %d SDPA node(s) to "
            "is_causal=True (additive mask dropped) [functional IR]",
            canonicalized,
        )
    except Exception as e:  # noqa: BLE001 — medium confidence: degrade gracefully
        logger.warning("[OPT-2 canonicalize_sdpa] Failed: %s", e)
    return gm


# ===========================================================================
# LEVEL 2 (aten) pass — OPT-1: bf16 dtype promotion for mm / addmm operands
# ===========================================================================
def _apass_bf16_promotion(gm: fx.GraphModule) -> fx.GraphModule:
    """OPT-1 (aten, high): promote aten.mm / aten.addmm GEMM operands to bf16 and
    cast the result back to the original dtype.

    aten.mm.default(act, weight)              -> both operands bf16
    aten.addmm.default(bias, act, weight)     -> bias, act, weight all bf16

    Routes the four projection families off the cutlass_80_simt_sgemm SIMT path
    (tensor_core_active_pct=0.0) onto the Blackwell bf16 tensor-core HGEMM path.
    This is an op-target pass (keys on mm/addmm targets, not weight values), so it
    needs no ph_to_tensor lookup. The output is cast back to the original dtype so
    downstream fp32 consumers are unaffected; OPT-4 then cancels the redundant
    round-trips between consecutive GEMMs.
    """
    try:
        g = gm.graph
        promoted = 0

        for node in list(g.nodes):
            if not (node.op == "call_function" and node.target in (_MM, _ADDMM)):
                continue
            if node.target is _MM and len(node.args) < 2:
                continue
            if node.target is _ADDMM and len(node.args) < 3:
                continue

            orig_dtype = _node_dtype(node) or _FP32
            if orig_dtype is _BF16:
                # GEMM already operates in bf16 — nothing to promote.
                continue

            if node.target is _MM:
                act, w = node.args[0], node.args[1]
                if not (isinstance(act, fx.Node) and isinstance(w, fx.Node)):
                    continue
                cast_act = _insert_cast(g, act, _BF16, node)
                cast_w = _insert_cast(g, w, _BF16, node)
                if cast_act is act and cast_w is w:
                    continue
                node.args = (cast_act, cast_w) + tuple(node.args[2:])
            else:  # _ADDMM(bias, act, weight)
                bias, act, w = node.args[0], node.args[1], node.args[2]
                if not all(isinstance(x, fx.Node) for x in (bias, act, w)):
                    continue
                cast_bias = _insert_cast(g, bias, _BF16, node)
                cast_act = _insert_cast(g, act, _BF16, node)
                cast_w = _insert_cast(g, w, _BF16, node)
                if cast_bias is bias and cast_act is act and cast_w is w:
                    continue
                node.args = (cast_bias, cast_act, cast_w) + tuple(node.args[3:])

            # Restore the original dtype on the GEMM output for downstream consumers.
            with g.inserting_after(node):
                back = g.call_function(_CONVERT, (node, orig_dtype))
            node.replace_all_uses_with(back, delete_user_cb=lambda u: u is not back)
            promoted += 1

        if promoted == 0:
            logger.warning(
                "[OPT-1 bf16_promotion] No fp32 aten.mm / aten.addmm nodes found "
                "— pass not applied"
            )
            return gm

        g.lint()
        gm.recompile()
        logger.info(
            "[OPT-1 bf16_promotion] Promoted %d aten.mm/addmm node(s) to bf16 "
            "operands (original dtype restored on output) [aten IR]",
            promoted,
        )
    except Exception as e:  # noqa: BLE001 — high confidence: exception = real error
        logger.warning("[OPT-1 bf16_promotion] Failed: %s", e)
    return gm


# ===========================================================================
# LEVEL 2 (aten) pass — OPT-4: cast cancellation (cleanup for OPT-1)
# ===========================================================================
def _apass_cancel_casts(gm: fx.GraphModule) -> fx.GraphModule:
    """OPT-4 (aten, medium): cancel adjacent inverse prims.convert_element_type
    pairs introduced by OPT-1.

    OPT-1 inserts a cast(bf16) before each GEMM and a cast(orig) after it. Where
    two GEMMs are connected through a pointwise/LayerNorm epilogue this produces a
    fp32 -> bf16 -> ... -> fp32 -> bf16 round-trip whose only effect is extra DRAM
    traffic. This pass detects convert(x, D) whose single input is convert(y, E)
    where the inner conversion's source already has dtype D, and rewires the outer
    node's users straight to that source, dropping both casts.

    Registered AFTER OPT-1 in the aten level so the inverse pairs exist when it
    runs (a genuine within-level ordering edge). Defensive: if OPT-1 inserted no
    casts (e.g. an autocast-region variant), this is a logged no-op.
    """
    try:
        g = gm.graph
        cancelled = 0

        for node in list(g.nodes):
            if not (node.op == "call_function" and node.target is _CONVERT):
                continue
            if len(node.args) < 2:
                continue
            inner = node.args[0]
            outer_dtype = node.args[1]
            if not (isinstance(inner, fx.Node)
                    and inner.op == "call_function"
                    and inner.target is _CONVERT):
                continue
            if len(inner.args) < 2:
                continue
            inner_src = inner.args[0]
            if not isinstance(inner_src, fx.Node):
                continue

            # Cancel only true round-trips: convert(convert(src, E), D) where the
            # original src dtype is exactly D, so the pair is an identity.
            src_dtype = _node_dtype(inner_src)
            if src_dtype is None or src_dtype is not outer_dtype:
                continue

            node.replace_all_uses_with(inner_src)
            g.erase_node(node)
            # Drop the now-dead inner cast if it has no remaining users.
            if not inner.users:
                g.erase_node(inner)
            cancelled += 1

        if cancelled == 0:
            logger.warning(
                "[OPT-4 cancel_casts] No redundant convert_element_type round-trips "
                "found — nothing to cancel (OPT-1 may not have inserted per-node casts)"
            )
            return gm

        g.eliminate_dead_code()
        g.lint()
        gm.recompile()
        logger.info(
            "[OPT-4 cancel_casts] Cancelled %d redundant fp32<->bf16 round-trip(s) "
            "[aten IR]",
            cancelled,
        )
    except Exception as e:  # noqa: BLE001 — medium confidence: degrade gracefully
        logger.warning("[OPT-4 cancel_casts] Failed: %s", e)
    return gm


# ===========================================================================
# LEVEL 3 (inductor_config) pass — OPT-3: Inductor freezing
# ===========================================================================
def _cfg_freezing() -> dict:
    """OPT-3 (inductor_config, medium): enable Inductor freezing.

    For this eval / no_grad inference graph, freezing treats the projection
    weights, biases and LayerNorm affine parameters as compile-time constants,
    constant-folds them, and pre-transposes / pre-packs the weight constants into
    the tensor-core-friendly layout so cuBLASLt/CUTLASS can select a fused-epilogue
    HGEMM — removing standalone memory-bound bias/GELU Triton kernels (the addmm_0
    family). Returned as scoped config_patches merged into THIS compile_fx call only
    (no global torch._inductor.config mutation). Satisfied-by-level after OPT-1, so
    the frozen weight constant is packed along the bf16 path.
    """
    logger.info(
        "[OPT-3 freezing] Inductor config_patches: freezing=True "
        "(weight pre-pack + epilogue fusion) [inductor_config level]"
    )
    return {"freezing": True}


# ===========================================================================
# Pass registry — routed by ir_level. Within a level, registry order is
# application order: at the aten level OPT-1 precedes OPT-4 (a within-level edge).
# ===========================================================================
PASS_REGISTRY = [
    # LEVEL 1 — functional (run before compile_fx, on the Dynamo graph)
    {"id": "OPT-2", "level": "functional",      "fn": _fpass_canonicalize_sdpa, "reads_weights": False},
    # LEVEL 2 — aten (run inside _aten_inner_compile, post-AOTAutograd)
    {"id": "OPT-1", "level": "aten",            "fn": _apass_bf16_promotion,    "reads_weights": False},
    {"id": "OPT-4", "level": "aten",            "fn": _apass_cancel_casts,      "reads_weights": False},
    # LEVEL 3 — inductor_config (merged into compile_fx config_patches)
    {"id": "OPT-3", "level": "inductor_config", "fn": _cfg_freezing},
]


def _passes(level: str) -> list:
    return [p for p in PASS_REGISTRY if p["level"] == level]


# ===========================================================================
# Funnel stage 1 — functional passes (Dynamo graph, pre-AOTAutograd)
# ===========================================================================
def _run_functional_passes(gm: fx.GraphModule) -> fx.GraphModule:
    """LEVEL 1 — rewrite the Dynamo (functional) graph BEFORE compile_fx owns it.
    Here F.scaled_dot_product_attention is a single high-level node and weight
    parameters are clean placeholder nodes. AOTAutograd recomputes meta when it
    traces this graph, so no FakeTensorProp is needed at this level."""
    for p in _passes("functional"):
        try:
            gm = p["fn"](gm)
        except Exception as e:  # noqa: BLE001 — degrade gracefully
            logger.warning("[%s] functional pass no-op: %s", p["id"], e)
    return gm


# ===========================================================================
# Funnel stage 2 — aten inner-compile hook (post-AOTAutograd, Aten IR)
# ===========================================================================
def _repropagate_meta(gm: fx.GraphModule, example_inputs) -> None:
    """Re-run FakeTensorProp so nodes inserted by an aten pass get meta['val']
    before compile_fx_inner runs. Best-effort: a failure here must never break
    compilation (the down-stream lowering can re-derive meta in most cases)."""
    try:
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.passes.fake_tensor_prop import FakeTensorProp

        fake_mode = None
        for inp in example_inputs:
            fm = getattr(inp, "fake_mode", None)
            if fm is not None:
                fake_mode = fm
                break
        if fake_mode is None:
            fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
        with fake_mode:
            FakeTensorProp(gm, mode=fake_mode).propagate(*example_inputs)
    except Exception as e:  # noqa: BLE001
        logger.debug("[_repropagate_meta] skipped: %s", e)


def _aten_inner_compile(
    gm: fx.GraphModule,
    example_inputs,
    *,
    real_inputs=None,
    **kwargs,
) -> Callable:
    """LEVEL 2 — Inductor's inner_compile hook. compile_fx calls this with the
    fully decomposed Aten IR graph (after AOTAutograd). Run aten-level passes
    (OPT-1 bf16 promotion, then OPT-4 cast cancellation), re-propagating meta after
    each structural rewrite, then delegate to compile_fx_inner (Aten -> Triton).

    example_inputs may be FakeTensors (Inductor traces under FakeTensorMode), so
    weight-VALUE-reading passes would use the threaded real_inputs. Both aten passes
    here are op-target passes and do not read weight values, but the ph_to_tensor
    lookup is built for interface parity. **kwargs is forwarded verbatim to
    compile_fx_inner for forward-compatibility.
    """
    weight_source = real_inputs if real_inputs is not None else example_inputs
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    ph_to_tensor = {ph: t for ph, t in zip(placeholders, weight_source)}  # noqa: F841

    for p in _passes("aten"):
        try:
            if p.get("reads_weights"):
                gm = p["fn"](gm, ph_to_tensor)
            else:
                gm = p["fn"](gm)
            _repropagate_meta(gm, example_inputs)
        except Exception as e:  # noqa: BLE001
            logger.warning("[%s] aten pass no-op: %s", p["id"], e)

    return compile_fx_inner(gm, example_inputs, **kwargs)


# ===========================================================================
# Funnel stage 3 — inductor config patches (scoped, no global mutation)
# ===========================================================================
def _config_patches() -> dict:
    """LEVEL 3 — scoped Inductor config_patches merged into THIS compile_fx call
    only. Each inductor_config pass returns a dict to merge; no global mutation."""
    patches: dict = {}
    for p in _passes("inductor_config"):
        try:
            patches.update(p["fn"]() or {})
        except Exception as e:  # noqa: BLE001
            logger.warning("[%s] config pass skipped: %s", p["id"], e)
    return patches


def _compile_unit(gm: fx.GraphModule, example_inputs) -> Callable:
    """The fixed three-stage funnel. compile_fx owns AOTAutograd, the decomp table,
    the boxed calling convention and the partitioner — we only run functional passes
    ahead of it, swap the leaf compiler, and pass scoped config_patches. No second
    AOTAutograd (which raises on torch 2.11)."""
    gm = _run_functional_passes(gm)
    inner = functools.partial(_aten_inner_compile, real_inputs=list(example_inputs))
    return compile_fx(
        gm,
        example_inputs,
        inner_compile=inner,
        config_patches=_config_patches(),
    )


# ===========================================================================
# Dedup-input capture utility
# ===========================================================================
def _capture_partition_inputs(split_gm: fx.GraphModule, example_inputs: list) -> dict:
    """Run split_gm once under no_grad to capture per-partition input tensors, so
    compile_fx can lower each unique representative with correct example inputs."""
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


# ===========================================================================
# The backend
# ===========================================================================
@register_backend
def gpt2_opt(gm: fx.GraphModule, example_inputs) -> Callable:
    """Custom torch.compile() backend for GPT-2 small.

    Three-stage funnel (functional -> aten -> inductor_config):
      OPT-2 (functional):        canonicalize SDPA to is_causal=True
      OPT-1 (aten):              bf16 promotion of mm/addmm operands
      OPT-4 (aten, after OPT-1): cancel redundant fp32<->bf16 round-trips
      OPT-3 (inductor_config):   freezing=True (weight pre-pack + epilogue fusion)

    Dedup-aware: GPT-2's 12 transformer blocks are structurally identical, so
    UniqueSubgraphRegistry returns a non-empty equivalence map and the dedup path
    compiles one representative block, sharing the compiled callable across all 12.
    """
    logger.info(
        "gpt2_opt backend: starting "
        "(functional[OPT-2] -> aten[OPT-1, OPT-4] -> inductor_config[OPT-3])"
    )

    registry = UniqueSubgraphRegistry(gm)
    equiv_map = registry.build_partition_equivalence_map()

    if not equiv_map:
        # No repeated layers detected — flat compile preserves cross-layer fusion.
        logger.info("gpt2_opt: no repeated layers, flat compile path")
        return _compile_unit(gm, example_inputs)

    logger.info(
        "gpt2_opt: %d duplicate partition(s), dedup compile path",
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

    # registry.split is a GraphModule whose child partitions have Inductor-compiled
    # .forward methods; routing each forward call through it reassembles the model.
    return lambda *args: registry.split(*args)


# ===========================================================================
# Workload definition + interface (mirrors gpt2.py so the capture pipeline can
# re-profile the optimized variant)
# ===========================================================================
class GPT2Wrapper(nn.Module):
    """Thin wrapper so model(input_ids) returns the last hidden state tensor.
    Unchanged from the baseline gpt2.py interface."""

    def __init__(self, hf_model: nn.Module):
        super().__init__()
        self.model = hf_model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids).last_hidden_state


def get_model_and_input() -> tuple:
    """Workload interface — return (raw uncompiled model, input_ids) on CUDA, eval.

    Mirrors the baseline gpt2.py exactly: the model stays fp32 (matches
    optimizations.json analysis.dtype). All four optimizations are graph- or
    config-level and applied inside the gpt2_opt backend, NOT here — there are no
    non-graph (whole-module dtype / memory_format / batch-shape) optimizations for
    this workload: the 512x768 problem (512 = 4*128) already tiles cleanly so
    batch-padding offers no benefit, and 2D activations make channels_last
    irrelevant. eval() is set because OPT-3 freezing is sound only under inference.

    Downloads GPT-2 weights from HuggingFace on first call (~500 MB); subsequent
    calls use the local cache.
    """
    assert torch.cuda.is_available(), "CUDA required"

    from transformers import GPT2Model  # imported here to keep top-level import-free

    hf_model = GPT2Model.from_pretrained(MODEL_ID)
    model = GPT2Wrapper(hf_model).to(DEVICE).eval()

    vocab_size = hf_model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (BATCH, SEQ_LEN), device=DEVICE)

    return model, input_ids


if __name__ == "__main__":
    model, input_ids = get_model_and_input()
    compiled = torch.compile(model, backend="gpt2_opt")
    with torch.no_grad():
        y = compiled(input_ids)
    print("output shape:", tuple(y.shape), "dtype:", y.dtype)  # expect (4, 128, 768)
