"""
lstm_sequence_encoder_optimized.py — Custom torch.compile() backend for LSTMSequenceEncoder.

Registered backend: ``lstm_sequence_encoder_opt``

Implements the low-risk in-place optimization track from optimizations.json
(OPT-2 -> OPT-1 -> OPT-4), routed to the correct IR level via the three-stage funnel
(functional -> aten -> inductor_config). OPT-3 (cuDNN fused-RNN restore) is MUTUALLY
EXCLUSIVE with OPT-1/OPT-2 over the recurrent matmuls (strategist note) and is therefore
NOT applied; it is documented as a stub-only detection pass for completeness.

Backend name: lstm_sequence_encoder_opt
    (model "lstm_sequence_encoder" -> snake-case + _opt)

---------------------------------------------------------------------------
What the profile actually shows (profile.json, cross-validated):
---------------------------------------------------------------------------
The dominant cost (~95.6% of attributed time) is the unrolled stacked LSTM. Inductor's
decomposition of nn.LSTM produces, at the ATEN level:

  * 2x batched input-projection GEMMs  [[4096,256],[256,2048]] and [[4096,512],[512,2048]]
      (4096 = SEQ_LEN(128) * BATCH(32); these are W_ih @ x already hoisted across all
       timesteps by Inductor's own LSTM lowering — i.e. OPT-2's "precompute input
       projections" transform is ALREADY realized by the decomposition).
  * 512x per-timestep recurrent GEMMs  [[32,512],[512,2048]]  (W_hh @ h_{t-1};
      256 timesteps/layer * 2 layers). M=32 is tiny -> the SIMT FP32 path
      (cutlass_80_simt_sgemm_128x32) with ~8% occupancy and 0% Tensor-Core activity.
  * 1x classifier  aten::addmm  [[32,512],[512,10]].

Because the input-projection hoist (OPT-2) is already present in the decomposed graph,
the highest-leverage in-place change is OPT-1: promote every gate GEMM (mm/addmm) to
bf16 so cuBLAS selects a Blackwell tensorop kernel instead of the scalar SIMT path. The
recurrent GEMMs stay M=32 (genuine data dependence on h_{t-1}), but bf16 still engages
the Tensor Core MMA pipeline and halves operand byte traffic.

---------------------------------------------------------------------------
Pass summary (execution order enforced by the funnel: functional -> aten -> inductor_config):
---------------------------------------------------------------------------

  OPT-2  functional / high  — input-projection hoist (F.linear triplet fusion)
      ir_level=functional, match_target=F.linear. Hoists each layer's input-to-hidden
      projection out of the recurrence into one [seq*batch, in]x[in, 4h] GEMM. At the
      functional level nn.LSTM is a SINGLE opaque high-level node, so the generic
      F.linear-triplet matcher finds nothing to hoist in THIS workload — and the hoist is
      ALREADY performed by Inductor's nn.LSTM decomposition (the [4096,256]/[4096,512]
      batched input GEMMs in profile.json prove it). The pass therefore detects the
      shared-weight F.linear-per-timestep pattern (which only appears for a hand-written
      Python cell loop) and reports it when present; on nn.LSTM it logs that the hoist is
      already realized downstream and no-ops gracefully. Strategist order: OPT-2 first.

  OPT-1  aten / high  — bf16 / Tensor-Core promotion of the gate GEMMs
      ir_level=aten, match_target=torch.ops.aten.mm.default. After AOTAutograd decomposes
      nn.LSTM into per-timestep aten.mm and the classifier aten.addmm, cast both matmul
      operands to bfloat16 and cast the result back to float32, preserving the recurrent
      sigmoid/tanh/mul cell-state dtype contract. Inductor CSE folds the repeated constant
      weight casts (W_ih/W_hh are constant across all timesteps), so per-timestep overhead
      is one activation cast, dominated by the Tensor-Core GEMM speedup. Strategist order:
      OPT-1 after OPT-2 (the funnel runs functional before aten, so the hoisted/batched
      input GEMMs are bf16-promoted too). prerequisite_for: OPT-4.

  OPT-4  inductor_config / medium  — freezing + max_autotune
      ir_level=inductor_config, match_target=freezing. Scoped config_patches on this
      compile_fx call only: freeze the LSTM weight parameters as compile-time constants
      (hoist the _tn_ transpose, drop per-call weight guards) and autotune a GEMM config
      that avoids the per-GEMM cublasLt splitKreduce epilogue for these small-M shapes.
      With OPT-1's bf16 weights in place (functional->aten->config ordering enforced by the
      funnel), freezing additionally lets Inductor emit a pre-packed tensorop weight layout.
      eval() required.

  OPT-3  NOT APPLIED (mutually exclusive) — cuDNN fused-RNN restore
      ir_level=functional, match_target=F.scaled_dot_product_attention (the closest schema
      enum; the true target is the nn.LSTM / aten.lstm op site). Route A keeps nn.LSTM on
      the eager cuDNN fused path (elemWiseRNNcell + tensorop GEMM) instead of letting
      Inductor unroll it. This changes the execution BACKEND for the recurrent region
      rather than transforming kernels in place and is MUTUALLY EXCLUSIVE with OPT-1/OPT-2
      over the same recurrent matmuls. Per the strategist, the low-risk in-place track
      (OPT-2 -> OPT-1 -> OPT-4) is chosen; OPT-3 is registered as a detection-only stub
      that logs the available cuDNN path and never transforms the graph.

---------------------------------------------------------------------------
IR-level mechanics (torch 2.11):
---------------------------------------------------------------------------
compile_fx owns AOTAutograd, the decomp table, the boxed calling convention and the
partitioner. We do NOT use aot_autograd(fw_compiler=compile_fx) — on torch 2.11 that
raises AssertionError inside copy_misaligned_inputs. The funnel runs functional-level
passes BEFORE compile_fx, aten-level passes through its inner_compile seam, and
inductor_config passes as scoped config_patches (no global Inductor config mutation).

compile_mode = "inductor" (optimizations.json analysis.compile_mode).
dtype        = float32 baseline (analysis.dtype); OPT-1 promotes the GEMMs to bf16 in-graph.
"""
from __future__ import annotations

import functools
import logging
from collections import defaultdict
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
_MM_TARGET = torch.ops.aten.mm.default
_ADDMM_TARGET = torch.ops.aten.addmm.default
_GEMM_TARGETS = (_MM_TARGET, _ADDMM_TARGET)

# Dtype-cast op for the bf16 promotion. We use prims.convert_element_type.default — the
# canonical dtype-cast primitive AOTAutograd itself emits — rather than aten._to_copy:
# under freezing + max_autotune Inductor registers aten._to_copy BOTH as a decomposition
# AND as a fallback, raising "both a fallback and a decomp for same op: aten._to_copy"
# (InductorError). prims.convert_element_type lowers cleanly and is CSE-folded for the
# repeated constant weight casts.
_CONVERT_DTYPE = torch.ops.prims.convert_element_type.default

# Functional-level F.linear identity set (binds to a builtin; add name fallback).
_LINEAR_FNS = {torch.nn.functional.linear}
try:  # torch._C._nn.linear is the underlying builtin Dynamo may trace to
    _LINEAR_FNS.add(torch._C._nn.linear)  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - defensive
    pass


def _is_linear(node: fx.Node) -> bool:
    return (
        node.op == "call_function"
        and (node.target in _LINEAR_FNS or getattr(node.target, "__name__", "") == "linear")
    )


# ---------------------------------------------------------------------------
# OPT-2 — input-projection hoist (F.linear triplet fusion). ir_level=functional.
# Confidence: high. match_target=F.linear.
#
# Hoist the input-to-hidden gate projection out of the recurrence: replace per-timestep
# [batch,in]x[in,4h] F.linear calls that share the SAME weight node and consume independent
# slices of one input sequence with a single [seq*batch,in]x[in,4h] F.linear. This raises M
# from batch(32) to seq*batch(4096) and turns ~128 under-occupied launches/layer into one
# well-tiled GEMM.
#
# In THIS workload the model uses nn.LSTM, which is a SINGLE opaque high-level node at the
# functional level — there is no per-timestep F.linear to hoist here, and the hoist is
# ALREADY realized by Inductor's nn.LSTM decomposition (profile.json shows the batched
# [4096,256]/[4096,512] input GEMMs). The pass therefore matches the hand-written-cell
# pattern (shared-weight F.linear repeated across timesteps) and reports it when present;
# on nn.LSTM it logs that the hoist is already realized downstream and no-ops gracefully.
# ---------------------------------------------------------------------------

def _fpass_hoist_input_projection(gm: fx.GraphModule) -> fx.GraphModule:
    """OPT-2: hoist the per-timestep input projection into one batched F.linear.

    Functional IR level. Graceful no-op when the per-timestep F.linear pattern is absent
    (e.g. nn.LSTM, where Inductor's decomposition already batches the input projection)."""
    try:
        g = gm.graph
        # Group F.linear nodes by their shared weight node (args[1]). The hoistable input
        # projection is a weight reused across many timesteps whose activations are
        # independent slices of one input tensor.
        linears_by_weight: dict[fx.Node, list[fx.Node]] = defaultdict(list)
        for node in g.nodes:
            if _is_linear(node) and len(node.args) > 1 and isinstance(node.args[1], fx.Node):
                linears_by_weight[node.args[1]].append(node)

        candidates = {w: ls for w, ls in linears_by_weight.items() if len(ls) >= 3}
        if not candidates:
            logger.warning(
                "[OPT-2 hoist_input_projection] No shared-weight per-timestep F.linear "
                "pattern at functional level — nn.LSTM is a single opaque node here and "
                "the input-projection hoist is ALREADY realized by Inductor's LSTM "
                "decomposition (profile shows batched [4096,256]/[4096,512] input GEMMs); "
                "pass is a no-op"
            )
            return gm

        # A hand-written cell loop exposes the hoistable pattern. We do not attempt the full
        # structural hoist here (it requires recovering the per-timestep slice topology,
        # which is unsafe to infer generically); instead we report the opportunity. The
        # canonical fix is the module-level FastLSTM rewrite (see optimizations.json OPT-2
        # fx_steps). Inductor still batches independent input projections it can prove are
        # slice-of-one-tensor, so this remains a correctness-preserving no-op.
        for w, ls in candidates.items():
            logger.info(
                "[OPT-2 hoist_input_projection] Detected %d F.linear calls sharing weight "
                "'%s' — hoistable input projection. Prefer the module-level FastLSTM "
                "precompute rewrite; leaving graph unchanged (Inductor batches provable "
                "slice-of-one-tensor projections). [functional IR]",
                len(ls), w.name,
            )
        return gm
    except Exception as e:
        logger.warning("[OPT-2 hoist_input_projection] Failed: %s", e)
        return gm


# ---------------------------------------------------------------------------
# OPT-3 — cuDNN fused-RNN restore. ir_level=functional. NOT APPLIED (mutually exclusive).
# Detection-only stub: logs the available cuDNN fused path and never transforms the graph.
# ---------------------------------------------------------------------------

def _fpass_detect_cudnn_rnn_stub(gm: fx.GraphModule) -> fx.GraphModule:
    """OPT-3 (stub): detect an nn.LSTM / aten.lstm op site that could stay on the cuDNN
    fused RNN path. NEVER transforms the graph — mutually exclusive with OPT-1/OPT-2 over
    the recurrent matmuls; the low-risk in-place track is chosen instead."""
    try:
        for node in gm.graph.nodes:
            tname = getattr(node.target, "__name__", "")
            is_lstm = node.op == "call_function" and (
                "lstm" in str(node.target).lower() or tname == "lstm"
            )
            if is_lstm:
                logger.warning(
                    "[OPT-3 cudnn_rnn] nn.LSTM / aten.lstm op site detected. Route A (keep "
                    "the LSTM on the eager cuDNN fused RNN path) is MUTUALLY EXCLUSIVE with "
                    "OPT-1/OPT-2 over the recurrent matmuls — NOT applied. The low-risk "
                    "in-place track (OPT-2 -> OPT-1 -> OPT-4) is used instead. [stub]"
                )
                return gm
    except Exception as e:
        logger.warning("[OPT-3 cudnn_rnn] Failed: %s", e)
    return gm


# ---------------------------------------------------------------------------
# OPT-1 — bf16 / Tensor-Core promotion of the gate GEMMs. ir_level=aten. Confidence: high.
# match_target=torch.ops.aten.mm.default.
#
# Runs inside _aten_inner_compile after AOTAutograd has fully decomposed nn.LSTM into
# per-timestep aten.mm (recurrent + batched input projections) and the classifier
# aten.addmm. For every such node:
#   * insert prims.convert_element_type(.., bfloat16) on the GEMM operands
#       (mm: args 0,1 ; addmm: args 0,1,2 — bias + both matmul operands, because Inductor's
#        bias_addmm lowering requires bias and matmul operands to share a dtype; the addmm
#        output is cast back to fp32 below so precision is restored),
#   * cast the result back to float32 (prims.convert_element_type) to preserve the
#     recurrent-state dtype contract that downstream sigmoid/tanh/mul cell ops depend on.
# convert_element_type (not aten._to_copy) avoids Inductor's decomp/fallback collision
# under freezing + max_autotune. See _CONVERT_DTYPE note above.
# This is a pure op-target pass (does NOT read weight VALUES) so it needs no ph_to_tensor.
# Inductor CSE folds the repeated constant weight casts.
# ---------------------------------------------------------------------------

def _apass_bf16_promote_gemms(gm: fx.GraphModule) -> fx.GraphModule:
    """OPT-1: promote aten.mm / aten.addmm operands to bfloat16, cast result to float32.

    Aten IR level. Engages the Blackwell bf16 Tensor-Core MMA pipeline for the gate GEMMs
    (which the FP32 SIMT path leaves at 0% tensor-core utilization)."""
    try:
        g = gm.graph
        promoted = 0
        for node in list(g.nodes):
            if node.op != "call_function" or node.target not in _GEMM_TARGETS:
                continue

            if node.target is _MM_TARGET:
                # mm(mat1, mat2) — cast both operands.
                mat_idx = (0, 1)
            else:
                # addmm(bias, mat1, mat2) — cast ALL THREE operands to bf16. Inductor's
                # bias_addmm lowering calls torch.addmm(bias, mat1, mat2) directly and
                # requires bias and the matmul operands to share a dtype ("self and mat2
                # must have the same dtype"); leaving the bias fp32 raises at runtime. The
                # fp32 result cast below restores precision, so casting the bias is safe.
                mat_idx = (0, 1, 2)

            # Cast the GEMM operands to bf16 immediately before the GEMM.
            with g.inserting_before(node):
                for i in mat_idx:
                    src = node.args[i]
                    if not isinstance(src, fx.Node):
                        continue
                    cast = g.call_function(
                        _CONVERT_DTYPE,
                        args=(src, torch.bfloat16),
                    )
                    node.update_arg(i, cast)

            # Cast the bf16 GEMM result back to float32 to preserve the cell-state dtype
            # contract (sigmoid/tanh/mul downstream expect fp32).
            with g.inserting_after(node):
                out_f32 = g.call_function(
                    _CONVERT_DTYPE,
                    args=(node, torch.float32),
                )
            node.replace_all_uses_with(out_f32)
            # replace_all_uses_with also rewired out_f32's own input; restore it to `node`.
            out_f32.update_arg(0, node)
            promoted += 1

        if promoted == 0:
            logger.warning(
                "[OPT-1 bf16_promote] No aten.mm / aten.addmm nodes found — pass not applied"
            )
            return gm

        g.lint()
        gm.recompile()
        logger.info(
            "[OPT-1 bf16_promote] Promoted %d gate GEMM(s) (aten.mm/aten.addmm) to bf16 "
            "with fp32 result cast [aten IR] — engages Blackwell Tensor Cores",
            promoted,
        )
    except Exception as e:
        logger.warning("[OPT-1 bf16_promote] Failed: %s", e)
    return gm


# ---------------------------------------------------------------------------
# OPT-4 — freezing + max_autotune. ir_level=inductor_config. Confidence: medium.
# match_target=freezing. Returns a scoped config_patches dict (no graph surgery).
# ---------------------------------------------------------------------------

def _cfg_freeze_and_autotune() -> dict:
    """OPT-4: scoped Inductor config_patches.

    freezing      — treat LSTM weights as compile-time constants (fold the _tn_ transpose,
                    drop per-call weight guards); with OPT-1's bf16 weights, emit a
                    pre-packed tensorop weight layout.
    max_autotune  — let Inductor benchmark a GEMM config that avoids the per-GEMM cublasLt
                    splitKreduce epilogue for the small-M (32) recurrent shapes.
    Scoped to THIS compile_fx call only (no global torch._inductor.config mutation).
    eval() is required for freezing (handled in get_model_and_input)."""
    patches = {
        "freezing": True,
        "max_autotune": True,
        "max_autotune_gemm_backends": "ATEN,TRITON",
    }
    logger.info(
        "[OPT-4 freeze_autotune] Applying scoped config_patches: %s [inductor_config]",
        patches,
    )
    return patches


# ---------------------------------------------------------------------------
# Pass registry — routed by ir_level
# ---------------------------------------------------------------------------
# No pass in this workload reads weight VALUES (OPT-1 is a pure op-target dtype pass),
# so the ph_to_tensor lookup is built but unused — kept for funnel uniformity / future
# weight-value passes.
_WEIGHT_VALUE_PASSES: set[str] = set()

PASS_REGISTRY = [
    # LEVEL 1 — functional (Dynamo graph, before compile_fx).
    # Strategist order OPT-2 first; OPT-3 is a non-transforming stub.
    {"id": "OPT-2", "level": "functional", "fn": _fpass_hoist_input_projection},
    {"id": "OPT-3", "level": "functional", "fn": _fpass_detect_cudnn_rnn_stub},
    # LEVEL 2 — aten (inside compile_fx inner_compile hook, post-decomposition).
    {"id": "OPT-1", "level": "aten", "fn": _apass_bf16_promote_gemms},
    # LEVEL 3 — inductor_config (scoped config_patches on compile_fx).
    {"id": "OPT-4", "level": "inductor_config", "fn": _cfg_freeze_and_autotune},
]

_FUNCTIONAL_PASSES = [p for p in PASS_REGISTRY if p["level"] == "functional"]
_ATEN_PASSES = [p for p in PASS_REGISTRY if p["level"] == "aten"]
_CONFIG_PASSES = [p for p in PASS_REGISTRY if p["level"] == "inductor_config"]


def _reads_weight_values(p: dict) -> bool:
    return p["id"] in _WEIGHT_VALUE_PASSES


# ---------------------------------------------------------------------------
# LEVEL 1 — Functional passes (Dynamo graph, pre-AOTAutograd)
# ---------------------------------------------------------------------------

def _run_functional_passes(gm: fx.GraphModule) -> fx.GraphModule:
    """Run all functional-level passes on the Dynamo graph before compile_fx.

    At this level nn.LSTM / F.linear are single high-level nodes. AOTAutograd recomputes
    meta when it traces the rewritten graph, so no FakeTensorProp is needed here."""
    for p in _FUNCTIONAL_PASSES:
        try:
            gm = p["fn"](gm)
        except Exception as e:
            logger.warning("[%s] functional pass error: %s", p["id"], e)
    return gm


# ---------------------------------------------------------------------------
# LEVEL 2 — Aten-level passes (inside compile_fx inner_compile hook)
# ---------------------------------------------------------------------------

def _repropagate_meta(gm: fx.GraphModule, example_inputs) -> None:
    """Re-run FakeTensorProp after a structural rewrite so inserted nodes (the
    aten._to_copy casts) get meta['val'] before compile_fx_inner runs."""
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
                FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(*fake_inputs)
        else:
            FakeTensorProp(gm).propagate_dont_convert_inputs(*fake_inputs)
    except Exception as e:
        logger.warning("[lstm_sequence_encoder_opt] meta re-propagation skipped: %s", e)


def _aten_inner_compile(
    gm: fx.GraphModule, example_inputs, *, real_inputs=None, **kwargs
) -> Callable:
    """LEVEL 2 — Inductor inner_compile hook.

    compile_fx calls this with the fully decomposed Aten IR graph (post-AOTAutograd). Run
    aten-level passes (OPT-1 bf16 promotion), re-propagating meta after each structural
    rewrite, then delegate to compile_fx_inner (Aten -> Triton).

    ``example_inputs`` may be FakeTensors under FakeTensorMode; weight-VALUE-reading passes
    (none here) would use the threaded ``real_inputs``. ``**kwargs`` is forwarded verbatim
    for forward-compatibility."""
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
# LEVEL 3 — Inductor config patches
# ---------------------------------------------------------------------------

def _build_config_patches() -> dict:
    """Collect and merge all inductor_config-level patches (scoped to this compile_fx
    call only — no global Inductor config mutation)."""
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
# Three-stage funnel: functional -> (AOTAutograd decomposition) -> aten -> config
# ---------------------------------------------------------------------------

def _compile_unit(gm: fx.GraphModule, example_inputs) -> Callable:
    """Fixed three-stage funnel for one (sub)graph.

    Stage 1 (functional): OPT-2 input-projection hoist detection, OPT-3 cuDNN stub.
    Stage 2 (aten): compile_fx owns AOTAutograd + decomp; _aten_inner_compile runs OPT-1
                    bf16 promotion on the decomposed gate GEMMs.
    Stage 3 (inductor_config): OPT-4 freezing + max_autotune via scoped config_patches."""
    gm = _run_functional_passes(gm)
    inner = functools.partial(_aten_inner_compile, real_inputs=list(example_inputs))
    config_patches = _build_config_patches()
    return compile_fx(gm, example_inputs, inner_compile=inner, config_patches=config_patches)


# ---------------------------------------------------------------------------
# Partition input capture (dedup path)
# ---------------------------------------------------------------------------

def _capture_partition_inputs(
    split_gm: fx.GraphModule, example_inputs: list
) -> dict[str, list]:
    """Run split_gm once under no_grad to capture per-partition input tensors so each
    unique representative is compiled with correct (real-value) example inputs."""
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
# Backend: lstm_sequence_encoder_opt
# ---------------------------------------------------------------------------

@register_backend
def lstm_sequence_encoder_opt(gm: fx.GraphModule, example_inputs) -> Callable:
    """Custom torch.compile() backend for LSTMSequenceEncoder.

    Low-risk in-place track from optimizations.json (OPT-2 -> OPT-1 -> OPT-4) via the
    three-stage funnel (functional -> aten -> inductor_config):

      OPT-2 (functional): input-projection hoist detection (already realized by Inductor's
                          nn.LSTM decomposition; no-op on nn.LSTM)
      OPT-3 (functional): cuDNN fused-RNN restore — NOT applied (mutually exclusive); stub
      OPT-1 (aten):       bf16 / Tensor-Core promotion of the gate GEMMs (the real win)
      OPT-4 (inductor_config): freezing + max_autotune via scoped config_patches

    Dedup-aware: the model is a single nn.LSTM + classifier (no repeated structurally
    identical FX subgraphs at the partition level), so UniqueSubgraphRegistry returns an
    empty equivalence map -> flat compile path. The dedup branch is preserved for models
    with repeated identical blocks."""
    logger.info(
        "lstm_sequence_encoder_opt backend: starting "
        "(functional[OPT-2 hoist-detect, OPT-3 cuDNN stub] -> aten[OPT-1 bf16 promote] "
        "-> inductor_config[OPT-4 freezing + max_autotune])"
    )

    registry = UniqueSubgraphRegistry(gm)
    equiv_map = registry.build_partition_equivalence_map()

    if not equiv_map:
        logger.info("lstm_sequence_encoder_opt: no repeated layers, flat compile path")
        return _compile_unit(gm, example_inputs)

    logger.info(
        "lstm_sequence_encoder_opt: %d duplicate partition(s), dedup compile path",
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
BATCH_SIZE = 32
SEQ_LEN = 128
INPUT_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 2
NUM_CLASSES = 10


def get_model_and_input() -> tuple:
    """Return (raw_model, input_tensor) on CUDA — uncompiled, unwarmed.

    The model is the unmodified LSTMSequenceEncoder, returned with .eval() set — REQUIRED
    for OPT-4 freezing (incompatible with training) and for the inference-only bf16 GEMM
    promotion (OPT-1). dtype stays float32 at the module boundary (matches
    optimizations.json analysis.dtype); OPT-1 promotes only the gate-GEMM operands to bf16
    IN-GRAPH and casts results back to fp32, so the input/output dtype contract is
    unchanged and the recurrent cell-state precision is preserved.

    No non-graph optimization is applied here: OPT-2's input-projection hoist is already
    realized by Inductor's nn.LSTM decomposition, OPT-1 is an in-graph aten pass, and OPT-4
    is an inductor_config pass — none require a whole-module dtype/memory_format change."""
    assert torch.cuda.is_available(), "CUDA required"
    from lstm_sequence_encoder import LSTMSequenceEncoder

    model = LSTMSequenceEncoder().to(DEVICE).eval()
    x = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE, device=DEVICE)
    return model, x


if __name__ == "__main__":
    model, x = get_model_and_input()
    compiled = torch.compile(model, backend="lstm_sequence_encoder_opt")
    with torch.no_grad():
        out = compiled(x)
    print("output shape:", out.shape, "dtype:", out.dtype)
