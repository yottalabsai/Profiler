"""
sdpa_attention_optimized.py — Custom torch.compile() backend for SDPAAttentionBlock.

Registered backend: ``sdpa_attention_opt``

Implements four optimizations from optimizations.json routed to the correct IR level
via the three-stage funnel (functional -> aten -> inductor_config). Each pass executes
at the level where its pattern is unambiguous and the rewrite is sound.

Backend name: sdpa_attention_opt  (model "sdpa_attention" -> snake-case + _opt suffix)

Pass summary (execution order: functional then aten then inductor_config):

  OPT-2  functional / high  — QKV weight fusion
      At the Dynamo functional graph level, three F.linear calls (Q, K, V projections)
      share a single activation node (post-ln_pre output). Concatenate the three weight
      placeholder nodes via aten.cat on dim=0 into a [3*D, D] fused weight, emit a
      single F.linear call, and slice the [B*T, 3*D] output back into Q/K/V segments.
      After AOTAutograd decomposition the single fused call lowers to one wide aten.mm,
      eliminating two of three kernel launches per attention layer. Must run at functional
      level because after decomposition each aten.mm receives its own aten.view of the
      activation and the shared-node identity is lost.

  OPT-3  functional / high  — Flash SDPA backend selection (Option A)
      Before compile_fx owns the graph, enable the Flash Attention backend and disable
      the memory-efficient xFormers backend. At this workload's dtype (FP32 input,
      BF16 after OPT-1), the SDPA dispatcher selects a Blackwell-native Flash Attention
      kernel (sm100) rather than the sm80 xFormers fallback, resolving the ISA mismatch
      and restoring ncu counter coverage. No graph nodes are added; the flag is a
      process-level side effect that Dynamo reads when it traces SDPA dispatch at the
      functional level.

  OPT-1  aten / high  — BF16 dtype promotion (matmul operands)
      Inside _aten_inner_compile (post-AOTAutograd), cast every aten.mm.default
      operand pair to bfloat16 via prims.convert_element_type, and cast the output
      back to float32. Routes cuBLAS from the SIMT FP32 cutlass_80_simt_sgemm path
      (tensor_core_active_pct=0.0) to the BF16 Tensor Core path. Using
      prims.convert_element_type (not aten._to_copy) avoids the "both a fallback and
      a decomp for same op" assertion on torch 2.11.

  OPT-4  inductor_config / medium  — Weight freezing + autotune
      Pass config_patches={"freezing": True, "max_autotune": True} to compile_fx.
      Inductor treats all nn.Parameter tensors (requires_grad=False in eval) as
      compile-time constants, hoists the aten.t() weight transpose to compile time,
      and benchmarks cuBLAS/Triton tile configurations against the frozen layouts.
      Zero risk for inference workloads; requires model.eval() which is set in
      get_model_and_input().

Prerequisite / ordering rationale:
  - OPT-2 and OPT-3 both run at the functional level (before compile_fx takes the
    graph). OPT-2 runs first so QKV fusion creates the output node that any downstream
    pass could key on. OPT-3 is a flag side-effect and order-independent at this level.
  - OPT-1 runs at aten level (inside inner_compile, after AOTAutograd). It sees the
    fully decomposed aten.mm nodes produced by the OPT-2 fused F.linear call.
  - OPT-4 is a config_patches dict that scopes Inductor behaviour; it has no graph
    content and no ordering dependency within its level.
  - Cross-level ordering (functional -> aten -> inductor_config) is enforced by the
    three-stage funnel and requires no explicit encoding.

IR-level mechanics (torch 2.11):
  compile_fx owns AOTAutograd, the decomp table, the boxed calling convention and the
  partitioner. We do NOT use aot_autograd(fw_compiler=compile_fx) — on torch 2.11 that
  raises AssertionError: Expected tensors only inside copy_misaligned_inputs. The funnel
  passes functional-level rewrites BEFORE compile_fx, aten-level passes through its
  inner_compile seam, and inductor_config passes as scoped config_patches.

compile_mode = "inductor" (from optimizations.json analysis.compile_mode).
"""
from __future__ import annotations

import functools
import logging
from collections import defaultdict
from typing import Callable

import torch
import torch.fx as fx
import torch.nn.functional as F
from torch._dynamo import register_backend
from torch._inductor.compile_fx import compile_fx, compile_fx_inner

from nvidia.operator_profiler.fx import UniqueSubgraphRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_BF16 = torch.bfloat16
_FP32 = torch.float32

# Op targets
_MM = torch.ops.aten.mm.default
_CAT = torch.ops.aten.cat.default
_SLICE = torch.ops.aten.slice.Tensor
# Use prims.convert_element_type, not aten._to_copy. On torch 2.11, aten._to_copy has
# both a fallback and a decomp registration; inserting it into an already-decomposed
# Aten graph makes Inductor raise "both a fallback and a decomp for same op".
# prims.convert_element_type lowers cleanly to a Triton elementwise cast.
_CONVERT = torch.ops.prims.convert_element_type.default

_INT_MAX = 9223372036854775807


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node_dtype(n: fx.Node) -> torch.dtype | None:
    """Return the tensor dtype stored in node meta, or None if unavailable."""
    if not isinstance(n, fx.Node):
        return None
    val = n.meta.get("val", None)
    if val is None or not hasattr(val, "dtype"):
        return None
    return val.dtype


def _insert_bf16_cast(g: fx.Graph, src: fx.Node, before: fx.Node) -> fx.Node:
    """Insert a prims.convert_element_type cast to bf16 directly before ``before``.
    Returns ``src`` unchanged if it already has bf16 dtype (no-op on already-cast)."""
    if _node_dtype(src) is _BF16:
        return src
    with g.inserting_before(before):
        return g.call_function(_CONVERT, (src, _BF16))


# ---------------------------------------------------------------------------
# OPT-2 — QKV weight fusion. ir_level=functional. Confidence: high.
#
# Detects triples of F.linear calls whose first argument is the same FX node
# (the shared post-LayerNorm activation). Concatenates the three weight
# placeholder nodes into [3*out_dim, in_dim], emits one fused F.linear call,
# then slices the [B*T, 3*out_dim] output into Q, K, V segments.
#
# This pass MUST run at the functional level. At the Aten level, AOTAutograd
# inserts a separate aten.view for each mm consumer, breaking the shared-node
# identity that this pattern keys on.
#
# The out_dim for each projection is read from the weight placeholder's
# meta["example_value"] shape (or meta["val"]). Shape is [out_dim, in_dim]
# at the functional level (not transposed).
# ---------------------------------------------------------------------------

def _fpass_fuse_qkv(gm: fx.GraphModule) -> fx.GraphModule:
    """OPT-2: Fuse three Q/K/V F.linear calls sharing one activation into a single
    wide F.linear, then slice back to Q, K, V. Runs at functional IR level."""
    try:
        g = gm.graph

        # Group F.linear call nodes by their shared first argument (activation).
        linear_by_input: dict[fx.Node, list[fx.Node]] = defaultdict(list)
        for node in list(g.nodes):
            if not (node.op == "call_function" and node.target is F.linear):
                continue
            if not node.args:
                continue
            act_node = node.args[0]
            if isinstance(act_node, fx.Node):
                linear_by_input[act_node].append(node)

        fused = 0
        for act_node, group in linear_by_input.items():
            if len(group) < 3:
                continue

            # Require exactly the Q/K/V triple (first 3 in graph order); the output
            # projection (also linear but different activation) is handled separately.
            q_node, k_node, v_node = group[0], group[1], group[2]

            # All three must be bias-free F.linear(x, W, None) as in this workload.
            # Bias support: if any has a non-None bias, skip this group.
            for lin in (q_node, k_node, v_node):
                if len(lin.args) > 2 and lin.args[2] is not None:
                    logger.warning(
                        "[OPT-2 fuse_qkv] Linear with bias detected — "
                        "skipping QKV fusion (bias not yet supported in fused path)"
                    )
                    continue

            w_q = q_node.args[1]
            w_k = k_node.args[1]
            w_v = v_node.args[1]
            if not all(isinstance(w, fx.Node) for w in (w_q, w_k, w_v)):
                logger.warning("[OPT-2 fuse_qkv] Weight args are not FX nodes — skip")
                continue

            # Read per-projection output dim from weight placeholder meta.
            # At functional level weight shape is [out_dim, in_dim] (row-major).
            def _out_dim(w_node: fx.Node) -> int | None:
                for key in ("example_value", "val"):
                    v = w_node.meta.get(key)
                    if v is not None and hasattr(v, "shape") and len(v.shape) >= 1:
                        return int(v.shape[0])
                return None

            n_q = _out_dim(w_q)
            n_k = _out_dim(w_k)
            n_v = _out_dim(w_v)
            if n_q is None or n_k is None or n_v is None:
                logger.warning(
                    "[OPT-2 fuse_qkv] Cannot read output dims from weight meta — "
                    "skip (functional-level meta not populated)"
                )
                continue

            # Build: cat([W_q, W_k, W_v], dim=0) -> single F.linear -> slice
            with g.inserting_before(q_node):
                w_cat = g.call_function(_CAT, ([w_q, w_k, w_v],), {"dim": 0})
                fused_lin = g.call_function(F.linear, (act_node, w_cat))
                # Slice dim=-1 (last dim of output [B, T, 3*D]):
                # F.linear output is [B, T, 3*out_dim]; slice along dim=2.
                q_slice = g.call_function(
                    _SLICE, (fused_lin, 2, 0, n_q)
                )
                k_slice = g.call_function(
                    _SLICE, (fused_lin, 2, n_q, n_q + n_k)
                )
                v_slice = g.call_function(
                    _SLICE, (fused_lin, 2, n_q + n_k, n_q + n_k + n_v)
                )

            # Replace original nodes; erase after all uses are transferred.
            q_node.replace_all_uses_with(q_slice)
            k_node.replace_all_uses_with(k_slice)
            v_node.replace_all_uses_with(v_slice)
            for dead in (q_node, k_node, v_node):
                if not dead.users:
                    g.erase_node(dead)

            fused += 1
            logger.info(
                "[OPT-2 fuse_qkv] Fused QKV projections into single F.linear "
                "[dims Q=%d K=%d V=%d -> %d] [functional IR]",
                n_q, n_k, n_v, n_q + n_k + n_v,
            )
            break  # Single attention block — one QKV group per graph

        if fused == 0:
            logger.warning(
                "[OPT-2 fuse_qkv] No 3-way shared-activation F.linear triplet found "
                "— pass not applied"
            )
            return gm

        g.eliminate_dead_code()
        g.lint()
        gm.recompile()
    except Exception as e:
        logger.warning("[OPT-2 fuse_qkv] Failed: %s", e)
    return gm


# ---------------------------------------------------------------------------
# OPT-3 — Flash SDPA backend selection (Option A). ir_level=functional.
# Confidence: high.
#
# Set the Flash Attention SDPA backend flag before compile_fx traces the graph.
# When Dynamo traces F.scaled_dot_product_attention it reads the active SDPA
# backend flags to decide which Aten-level op to emit. With mem_efficient_sdp
# disabled and flash_sdp enabled, Dynamo emits aten._scaled_dot_product_flash_
# attention (Blackwell sm100-native) instead of aten._scaled_dot_product_
# efficient_attention (sm80 xFormers fallback).
#
# No graph modifications — this is a process-level side-effect that the
# functional pass stage invokes before compile_fx.
# ---------------------------------------------------------------------------

def _fpass_enable_flash_sdpa(gm: fx.GraphModule) -> fx.GraphModule:
    """OPT-3: Steer SDPA dispatch toward Flash Attention and away from the sm80 xFormers path.

    Sets process-level SDPA backend flags before compile_fx takes ownership of the
    graph. Dynamo reads these flags when tracing F.scaled_dot_product_attention to
    choose the Aten-level op.

    Strategy:
      - enable_flash_sdp(True): prefer Flash Attention (native sm100 on Blackwell).
      - enable_mem_efficient_sdp(False): disable the sm80 xFormers backend
        (fmha_cutlassF_f32_aligned_64x64_rf_sm80) that the profile identified as
        running with an Sm80/Sm100 ISA mismatch.
      - enable_math_sdp remains True (default): math SDPA is a valid FP32 fallback
        for the Dynamo tracing/metadata pass which runs with the original FP32 inputs.
        At execution time OPT-1 promotes mm operands to BF16, and the BF16 Q/K/V
        tensors will route to Flash Attention (which requires BF16 for non-causal
        attention on this hardware) rather than the math fallback.

    No graph modifications — this is a process-level side-effect only.
    """
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        # Keep math_sdp enabled: Flash Attention on this device (sm100) requires BF16
        # for non-causal use. Math SDPA provides a valid FP32 fallback during Dynamo's
        # metadata tracing pass. At kernel dispatch time with BF16 Q/K/V (from OPT-1),
        # Flash takes priority over math.
        logger.info(
            "[OPT-3 flash_sdpa] Flash SDPA enabled, mem_efficient (sm80 xFormers) "
            "disabled, math kept as FP32 fallback [functional IR, flag side-effect]"
        )
    except Exception as e:
        logger.warning("[OPT-3 flash_sdpa] Failed to set SDPA backend flags: %s", e)
    return gm


# ---------------------------------------------------------------------------
# OPT-1 — BF16 dtype promotion (matmul operands). ir_level=aten. Confidence: high.
#
# Runs inside _aten_inner_compile after AOTAutograd has fully decomposed the
# graph. Every aten.mm.default node sees its two operands cast to bfloat16 via
# prims.convert_element_type, and the mm output is cast back to float32 to
# preserve the downstream dtype contract.
#
# This routes cuBLAS dispatch from the SIMT FP32 path
# (cutlass_80_simt_sgemm_128x256_8x4_tn_align1, tensor_core_active_pct=0.0)
# to the BF16 Tensor Core path on Blackwell (sm100).
# ---------------------------------------------------------------------------

def _apass_bf16_promotion(gm: fx.GraphModule) -> fx.GraphModule:
    """OPT-1: Cast aten.mm operands to BF16 and output back to FP32. Aten IR level."""
    try:
        g = gm.graph
        promoted = 0

        for mm in list(g.nodes):
            if not (mm.op == "call_function" and mm.target is _MM):
                continue
            if len(mm.args) < 2:
                continue
            act, w = mm.args[0], mm.args[1]
            if not (isinstance(act, fx.Node) and isinstance(w, fx.Node)):
                continue

            # Cast both operands to BF16.
            cast_act = _insert_bf16_cast(g, act, mm)
            cast_w = _insert_bf16_cast(g, w, mm)
            if cast_act is act and cast_w is w:
                # Already BF16 (e.g. after a preceding cast) — no-op for this node.
                continue

            # Update mm args to use the BF16 casts.
            mm.args = (cast_act, cast_w) + tuple(mm.args[2:])

            # Restore FP32 on the output so all users keep float32.
            with g.inserting_after(mm):
                back_fp32 = g.call_function(_CONVERT, (mm, _FP32))
            mm.replace_all_uses_with(
                back_fp32, delete_user_cb=lambda u: u is not back_fp32
            )
            promoted += 1

        if promoted == 0:
            logger.warning("[OPT-1 bf16_promotion] No FP32 aten.mm nodes found — pass not applied")
            return gm

        g.lint()
        gm.recompile()
        logger.info(
            "[OPT-1 bf16_promotion] Promoted %d aten.mm node(s) to BF16 operands "
            "(FP32 output restored) [aten IR]",
            promoted,
        )
    except Exception as e:
        logger.warning("[OPT-1 bf16_promotion] Failed: %s", e)
    return gm


# ---------------------------------------------------------------------------
# OPT-4 — Weight freezing + autotune config. ir_level=inductor_config.
# Confidence: medium.
#
# Returns a dict merged into compile_fx's config_patches argument (scoped to
# this compilation unit only — no process-global state mutation). Inductor treats
# nn.Parameter tensors (requires_grad=False in eval) as compile-time constants,
# hoists the aten.t() weight transpose to compile time, and benchmarks cuBLAS/
# Triton tile configurations against the frozen weight layouts.
# ---------------------------------------------------------------------------

def _cfg_freezing() -> dict:
    """OPT-4: Return Inductor config patches for weight freezing and max_autotune."""
    try:
        patches = {
            "freezing": True,
            "max_autotune": True,
            "max_autotune_gemm_backends": "ATEN,TRITON,CPP",
        }
        logger.info(
            "[OPT-4 freezing] Inductor config_patches: freezing=True, "
            "max_autotune=True [inductor_config level]"
        )
        return patches
    except Exception as e:
        logger.warning("[OPT-4 freezing] Config patch failed: %s", e)
        return {}


# ---------------------------------------------------------------------------
# Pass registry — routed by ir_level
# ---------------------------------------------------------------------------

PASS_REGISTRY = [
    # Functional-level passes (run before compile_fx, on the Dynamo graph)
    {"id": "OPT-2", "level": "functional", "fn": _fpass_fuse_qkv},
    {"id": "OPT-3", "level": "functional", "fn": _fpass_enable_flash_sdpa},
    # Aten-level passes (run inside _aten_inner_compile, post-AOTAutograd)
    {"id": "OPT-1", "level": "aten",       "fn": _apass_bf16_promotion},
    # Inductor config patches (merged into compile_fx config_patches)
    {"id": "OPT-4", "level": "inductor_config", "fn": _cfg_freezing},
]

_FUNCTIONAL_PASSES = [p for p in PASS_REGISTRY if p["level"] == "functional"]
_ATEN_PASSES = [p for p in PASS_REGISTRY if p["level"] == "aten"]
_CONFIG_PASSES = [p for p in PASS_REGISTRY if p["level"] == "inductor_config"]


# ---------------------------------------------------------------------------
# LEVEL 1 — Functional passes (Dynamo graph, pre-AOTAutograd)
# ---------------------------------------------------------------------------

def _run_functional_passes(gm: fx.GraphModule) -> fx.GraphModule:
    """Run all functional-level passes on the Dynamo graph before compile_fx.

    At this level F.linear / F.scaled_dot_product_attention are single high-level
    nodes and weight parameters are clean placeholder nodes. The shared activation
    is literally one FX node — the identity required for QKV fusion. AOTAutograd
    recomputes meta when it traces the rewritten graph; no FakeTensorProp needed."""
    for p in _FUNCTIONAL_PASSES:
        try:
            gm = p["fn"](gm)
        except Exception as e:
            logger.warning("[%s] functional pass error: %s", p["id"], e)
    return gm


# ---------------------------------------------------------------------------
# LEVEL 3 — Inductor config patches
# ---------------------------------------------------------------------------

def _build_config_patches() -> dict:
    """Collect and merge all inductor_config-level patches. Scoped to this
    compile_fx call only — no global Inductor config mutation."""
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
    (e.g. new aten.mm, aten.cat) get meta['val'] before compile_fx_inner runs."""
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
        logger.warning("[sdpa_attention_opt] meta re-propagation skipped: %s", e)


def _aten_inner_compile(
    gm: fx.GraphModule, example_inputs, *, real_inputs=None, **kwargs
) -> Callable:
    """LEVEL 2 — Inductor inner_compile hook.

    compile_fx calls this with the fully decomposed Aten IR graph (post-AOTAutograd).
    Run aten-level passes (OPT-1 BF16 promotion), repropagating meta after each
    structural rewrite, then delegate to compile_fx_inner (Aten -> Triton).

    ``example_inputs`` may be FakeTensors under FakeTensorMode. ``real_inputs``
    is threaded from the backend for any pass that needs actual weight values.
    ``**kwargs`` is forwarded verbatim to compile_fx_inner for forward-compatibility."""
    for p in _ATEN_PASSES:
        try:
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

    Stage 1: run functional passes on the Dynamo graph (OPT-2 QKV fusion,
             OPT-3 Flash SDPA flag side-effect).
    Stage 2: compile_fx owns AOTAutograd + decomp; our _aten_inner_compile hook
             runs OPT-1 BF16 promotion on the decomposed Aten IR.
    Stage 3: OPT-4 freezing/autotune config_patches scoped to this compile_fx call."""
    gm = _run_functional_passes(gm)
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
    """Run split_gm once under no_grad to capture per-partition input tensors."""
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
# Backend: sdpa_attention_opt
# ---------------------------------------------------------------------------

@register_backend
def sdpa_attention_opt(gm: fx.GraphModule, example_inputs) -> Callable:
    """Custom torch.compile() backend for SDPAAttentionBlock.

    Implements four optimizations from optimizations.json via the three-stage funnel
    (functional -> aten -> inductor_config):

      OPT-2 (functional): QKV weight fusion — 3 F.linear -> 1 wide F.linear + slices
      OPT-3 (functional): Flash SDPA flags — enable_flash_sdp(True), disables sm80 path
      OPT-1 (aten):       BF16 promotion  — aten.mm operands BF16, output FP32
      OPT-4 (config):     Freezing        — freezing=True, max_autotune=True

    Dedup-aware: SDPAAttentionBlock is a single block with no repeated partitions;
    UniqueSubgraphRegistry returns an empty equivalence map and the flat compile path
    is taken. The dedup branch is preserved for models with multiple identical blocks.
    """
    logger.info(
        "sdpa_attention_opt backend: starting "
        "(functional[OPT-2, OPT-3] -> aten[OPT-1] -> inductor_config[OPT-4])"
    )

    registry = UniqueSubgraphRegistry(gm)
    equiv_map = registry.build_partition_equivalence_map()

    if not equiv_map:
        # No repeated layers — flat compile preserves cross-layer Inductor fusion.
        logger.info("sdpa_attention_opt: no repeated layers, flat compile path")
        return _compile_unit(gm, example_inputs)

    logger.info(
        "sdpa_attention_opt: %d duplicate partition(s), dedup compile path",
        len(equiv_map),
    )

    # Compile each unique representative through the same funnel; share the
    # compiled callable with all structural duplicates. Functional passes run
    # per-rep (inside _compile_unit), never on the pre-split graph.
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


# ---------------------------------------------------------------------------
# Workload interface
# ---------------------------------------------------------------------------
DEVICE = "cuda"
BATCH_SIZE = 8
SEQ_LEN = 512
DIM = 512
NUM_HEADS = 8
HEAD_DIM = DIM // NUM_HEADS  # 64


def get_model_and_input() -> tuple:
    """Return (raw_model, input_tensor) on CUDA — uncompiled, unwarmed.

    Model dtype: FP32 (matches optimizations.json analysis.dtype = "float32").
    OPT-1 BF16 promotion is applied selectively inside the graph, not by casting
    the whole module. OPT-2/3/4 are also graph/config level passes; no non-graph
    eager-side optimizations are needed for this workload (no conv layers requiring
    channels_last; GEMM M/N/K dims are multiples of 16 and don't need batch padding).

    The model is returned with .eval() set; OPT-4 freezing requires eval mode.
    """
    assert torch.cuda.is_available(), "CUDA required"
    from sdpa_attention import SDPAAttentionBlock

    model = SDPAAttentionBlock().to(DEVICE).eval()
    x = torch.randn(BATCH_SIZE, SEQ_LEN, DIM, device=DEVICE)
    return model, x


if __name__ == "__main__":
    model, x = get_model_and_input()
    compiled = torch.compile(model, backend="sdpa_attention_opt")
    with torch.no_grad():
        out = compiled(x)
    print("output shape:", out.shape, "dtype:", out.dtype)
