"""
embedding_projection_optimized.py — Custom torch.compile() backend for EmbeddingProjection.

Registered backend: ``embedding_projection_opt``

Implements two optimizations from optimizations.json routed to the correct IR level
via the three-stage funnel (functional -> aten -> inductor_config). This workload has
no functional-level pass (no QKV-style shared-activation fusion is available — proj1,
proj2 and the logit projection form a sequential dependent chain), so the funnel runs
an aten-level pass and an inductor_config-level pass.

Backend name: embedding_projection_opt  (model "EmbeddingProjection" -> snake-case + _opt)

Pass summary (execution order: aten then inductor_config):

  OPT-1  aten / high  — BF16 dtype promotion (matmul + addmm operands)
      Inside _aten_inner_compile (post-AOTAutograd), cast both matrix operands of
      every aten.mm.default and aten.addmm.default node to bfloat16 via
      prims.convert_element_type, leave the addmm bias in fp32, then cast the GEMM
      result back to float32 to preserve the downstream dtype contract. This routes
      cuBLAS from the SIMT FP32 cutlass_80_simt_sgemm_* path (tensor_core_active_pct
      = 0.0 on every GEMM) to the BF16 Tensor Core path on Blackwell (sm100). The two
      wide logit GEMMs ([8192,512]x[512,32000]) are the single largest lever — bf16
      also halves their ~1.05 GB FP32 output write to ~512 MB. Using
      prims.convert_element_type (not aten._to_copy) avoids the "both a fallback and a
      decomp for same op" assertion on torch 2.11 against the already-decomposed graph.

  OPT-2  inductor_config / medium  — Weight freezing + autotune
      Pass config_patches={"freezing": True, "max_autotune": True,
      "max_autotune_gemm_backends": "ATEN,TRITON"} to compile_fx. Inductor treats the
      requires_grad=False eval-mode nn.Parameter tensors as compile-time constants,
      hoists the aten.t() weight transpose (the _tn_ suffix in the SGEMM kernel name) to
      compile time, and benchmarks cuBLAS/Triton tile/split-K configurations against the
      frozen bf16 weight layout. The unusual N=32000 logit GEMM is the prime autotune
      beneficiary. Zero risk for eval-mode inference; requires model.eval() which is set
      in get_model_and_input().

Prerequisite / ordering rationale:
  - OPT-1 (aten) is a prerequisite_for OPT-2 (inductor_config): freezing/autotune is most
    impactful once the GEMMs are bf16 tensorop kernels (more layout- and tile-sensitive
    tuning knobs than SIMT SGEMM). The cross-level ordering (aten before inductor_config)
    is enforced by the three-stage funnel and requires no explicit within-level encoding.

IR-level mechanics (torch 2.11):
  compile_fx owns AOTAutograd, the decomp table, the boxed calling convention and the
  partitioner. We do NOT use aot_autograd(fw_compiler=compile_fx) — on torch 2.11 that
  raises AssertionError: Expected tensors only inside copy_misaligned_inputs. The funnel
  passes functional-level rewrites BEFORE compile_fx (none here), aten-level passes through
  its inner_compile seam, and inductor_config passes as scoped config_patches.

compile_mode = "inductor" (from optimizations.json analysis.compile_mode).
"""
from __future__ import annotations

import functools
import logging
from typing import Callable

import torch
import torch.fx as fx
from torch._dynamo import register_backend
from torch._inductor.compile_fx import compile_fx, compile_fx_inner

from nvidia.operator_profiler.fx import UniqueSubgraphRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_BF16 = torch.bfloat16
_FP32 = torch.float32

# Op targets — the two GEMM-bearing decomposed aten ops in this workload.
_MM = torch.ops.aten.mm.default
_ADDMM = torch.ops.aten.addmm.default
_MM_TARGETS = (_MM, _ADDMM)

# Use prims.convert_element_type, not aten._to_copy. On torch 2.11, aten._to_copy has
# both a fallback and a decomp registration; inserting it into an already-decomposed
# Aten graph makes Inductor raise "both a fallback and a decomp for same op".
# prims.convert_element_type lowers cleanly to a Triton elementwise cast.
_CONVERT = torch.ops.prims.convert_element_type.default


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
# OPT-1 — BF16 dtype promotion (matmul + addmm operands). ir_level=aten.
# Confidence: high.
#
# Runs inside _aten_inner_compile after AOTAutograd has fully decomposed the
# graph. For each aten.mm.default node both operands (args 0, 1) are cast to
# bfloat16; for each aten.addmm.default node the two matrix operands (args 1, 2)
# are cast while the bias (arg 0) is left fp32. The GEMM output is cast back to
# float32 to preserve the downstream dtype contract.
#
# This routes cuBLAS dispatch from the SIMT FP32 path
# (cutlass_80_simt_sgemm_256x128_8x4_tn_align1, tensor_core_active_pct=0.0)
# to the BF16 Tensor Core path on Blackwell (sm100), and halves the wide logit
# GEMM's ~1.05 GB FP32 output write.
# ---------------------------------------------------------------------------

def _apass_bf16_promotion(gm: fx.GraphModule) -> fx.GraphModule:
    """OPT-1: Cast aten.mm / aten.addmm matrix operands to BF16 and output back to
    FP32. Aten IR level. addmm bias (args[0]) is left in fp32."""
    try:
        g = gm.graph
        promoted = 0

        for node in list(g.nodes):
            if not (node.op == "call_function" and node.target in _MM_TARGETS):
                continue

            if node.target is _ADDMM:
                # addmm(bias, mat1, mat2): cast mat1, mat2; leave bias fp32.
                mat_idx = (1, 2)
            else:
                # mm(mat1, mat2): cast both.
                mat_idx = (0, 1)

            if len(node.args) <= max(mat_idx):
                continue

            # Cast each matrix operand to BF16 (no-op if already BF16).
            new_args = list(node.args)
            changed = False
            non_node = False
            for i in mat_idx:
                src = node.args[i]
                if not isinstance(src, fx.Node):
                    non_node = True
                    break
                cast = _insert_bf16_cast(g, src, node)
                if cast is not src:
                    new_args[i] = cast
                    changed = True
            if non_node or not changed:
                # A matrix operand is a non-node constant, or both operands are
                # already BF16 — nothing to promote for this node.
                continue

            node.args = tuple(new_args)

            # Restore FP32 on the output so all downstream users keep float32.
            with g.inserting_after(node):
                back_fp32 = g.call_function(_CONVERT, (node, _FP32))
            node.replace_all_uses_with(
                back_fp32, delete_user_cb=lambda u: u is not back_fp32
            )
            promoted += 1

        if promoted == 0:
            logger.warning(
                "[OPT-1 bf16_promotion] No FP32 aten.mm/aten.addmm nodes found "
                "— pass not applied"
            )
            return gm

        g.lint()
        gm.recompile()
        logger.info(
            "[OPT-1 bf16_promotion] Promoted %d aten.mm/aten.addmm node(s) to BF16 "
            "operands (FP32 output restored, addmm bias kept FP32) [aten IR]",
            promoted,
        )
    except Exception as e:
        logger.warning("[OPT-1 bf16_promotion] Failed: %s", e)
    return gm


# ---------------------------------------------------------------------------
# OPT-2 — Weight freezing + autotune config. ir_level=inductor_config.
# Confidence: medium.
#
# Returns a dict merged into compile_fx's config_patches argument (scoped to
# this compilation unit only — no process-global state mutation). Inductor treats
# the eval-mode nn.Parameter tensors (requires_grad=False) as compile-time
# constants, hoists the aten.t() weight transpose to compile time, and benchmarks
# cuBLAS/Triton tile configurations against the frozen bf16 weight layouts.
# Most impactful after OPT-1 (bf16 tensorop kernels expose more tile knobs).
# ---------------------------------------------------------------------------

def _cfg_freezing() -> dict:
    """OPT-2: Return Inductor config patches for weight freezing and max_autotune."""
    try:
        patches = {
            "freezing": True,
            "max_autotune": True,
            "max_autotune_gemm_backends": "ATEN,TRITON",
        }
        logger.info(
            "[OPT-2 freezing] Inductor config_patches: freezing=True, "
            "max_autotune=True, gemm_backends=ATEN,TRITON [inductor_config level]"
        )
        return patches
    except Exception as e:
        logger.warning("[OPT-2 freezing] Config patch failed: %s", e)
        return {}


# ---------------------------------------------------------------------------
# Pass registry — routed by ir_level
# ---------------------------------------------------------------------------

PASS_REGISTRY = [
    # Aten-level passes (run inside _aten_inner_compile, post-AOTAutograd)
    {"id": "OPT-1", "level": "aten", "fn": _apass_bf16_promotion},
    # Inductor config patches (merged into compile_fx config_patches)
    {"id": "OPT-2", "level": "inductor_config", "fn": _cfg_freezing},
]

_FUNCTIONAL_PASSES = [p for p in PASS_REGISTRY if p["level"] == "functional"]
_ATEN_PASSES = [p for p in PASS_REGISTRY if p["level"] == "aten"]
_CONFIG_PASSES = [p for p in PASS_REGISTRY if p["level"] == "inductor_config"]


# ---------------------------------------------------------------------------
# LEVEL 1 — Functional passes (Dynamo graph, pre-AOTAutograd)
# ---------------------------------------------------------------------------

def _run_functional_passes(gm: fx.GraphModule) -> fx.GraphModule:
    """Run all functional-level passes on the Dynamo graph before compile_fx.

    This workload has no functional-level pass (no shared-activation GEMM fusion is
    available — the projections form a sequential dependent chain), so this is a no-op
    pass-through. The stage is kept so the funnel structure stays identical across
    examples and so a future functional pass can be added without restructuring."""
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
    (the new convert_element_type casts) get meta['val'] before compile_fx_inner runs."""
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
        logger.warning("[embedding_projection_opt] meta re-propagation skipped: %s", e)


def _aten_inner_compile(
    gm: fx.GraphModule, example_inputs, *, real_inputs=None, **kwargs
) -> Callable:
    """LEVEL 2 — Inductor inner_compile hook.

    compile_fx calls this with the fully decomposed Aten IR graph (post-AOTAutograd).
    Run aten-level passes (OPT-1 BF16 promotion), repropagating meta after each
    structural rewrite, then delegate to compile_fx_inner (Aten -> Triton).

    ``example_inputs`` may be FakeTensors under FakeTensorMode. ``real_inputs`` is
    threaded from the backend for any pass that needs actual weight values (OPT-1 is
    op-target only and does not read weight values, but the threading is kept for
    consistency with the canonical funnel). ``**kwargs`` is forwarded verbatim to
    compile_fx_inner for forward-compatibility."""
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

    Stage 1: run functional passes on the Dynamo graph (none for this workload).
    Stage 2: compile_fx owns AOTAutograd + decomp; our _aten_inner_compile hook
             runs OPT-1 BF16 promotion on the decomposed Aten IR.
    Stage 3: OPT-2 freezing/autotune config_patches scoped to this compile_fx call."""
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
# Backend: embedding_projection_opt
# ---------------------------------------------------------------------------

@register_backend
def embedding_projection_opt(gm: fx.GraphModule, example_inputs) -> Callable:
    """Custom torch.compile() backend for EmbeddingProjection.

    Implements two optimizations from optimizations.json via the three-stage funnel
    (functional -> aten -> inductor_config):

      OPT-1 (aten):   BF16 promotion — aten.mm / aten.addmm matrix operands BF16,
                      output FP32, addmm bias kept FP32
      OPT-2 (config): Freezing       — freezing=True, max_autotune=True

    Dedup-aware: EmbeddingProjection is a single linear chain with no repeated
    partitions; UniqueSubgraphRegistry returns an empty equivalence map and the flat
    compile path is taken (flat compile also preserves cross-op Inductor fusion of the
    embedding/LayerNorm/GELU tails). The dedup branch is preserved for models with
    multiple identical blocks.
    """
    logger.info(
        "embedding_projection_opt backend: starting "
        "(aten[OPT-1] -> inductor_config[OPT-2])"
    )

    registry = UniqueSubgraphRegistry(gm)
    equiv_map = registry.build_partition_equivalence_map()

    if not equiv_map:
        # No repeated layers — flat compile preserves cross-op Inductor fusion.
        logger.info("embedding_projection_opt: no repeated layers, flat compile path")
        return _compile_unit(gm, example_inputs)

    logger.info(
        "embedding_projection_opt: %d duplicate partition(s), dedup compile path",
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
BATCH_SIZE = 64
SEQ_LEN = 128
VOCAB_SIZE = 32_000
DIM = 512
DIM_FF = 2048


def get_model_and_input() -> tuple:
    """Return (raw_model, input_tensor) on CUDA — uncompiled, unwarmed.

    Model dtype: FP32 (matches optimizations.json analysis.dtype = "float32").
    OPT-1 BF16 promotion is applied selectively inside the graph (on GEMM operands
    only), not by casting the whole module — so the module and its input stay FP32.
    OPT-2 freezing/autotune is a config-level pass; no non-graph eager-side
    optimization is needed (no conv layers requiring channels_last; the GEMM M/N/K
    dims are multiples of 16 and need no batch padding).

    The model is returned with .eval() set; OPT-2 freezing requires eval mode.
    Input is integer token IDs in [0, VOCAB_SIZE).
    """
    assert torch.cuda.is_available(), "CUDA required"
    from embedding_projection import EmbeddingProjection

    model = EmbeddingProjection().to(DEVICE).eval()
    token_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    return model, token_ids


if __name__ == "__main__":
    model, token_ids = get_model_and_input()
    compiled = torch.compile(model, backend="embedding_projection_opt")
    with torch.no_grad():
        out = compiled(token_ids)
    print("output shape:", out.shape, "dtype:", out.dtype)
