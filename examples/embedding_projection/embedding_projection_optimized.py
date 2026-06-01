"""
embedding_projection_optimized.py — Custom torch.compile() backend for the
embedding-lookup + projection-head workload.

Registered backend: ``embedding_projection_opt``

Implements the three optimizations from optimizations.json, each routed to the IR
level where its pattern is unambiguous and the rewrite is sound, via the fixed
three-stage funnel (functional -> aten -> inductor_config):

  OPT-1  inductor_config (also a process-global flag) / high — Enable TF32
      Set torch.backends.cuda.matmul.allow_tf32 / cudnn.allow_tf32 at backend entry
      and request TF32 in the scoped Inductor config_patches. The profile shows every
      GEMM dispatching to cutlass_80_simt_sgemm_*_align1 (tensor_core_active_pct=0.0)
      because TF32 was disabled, so cuBLAS/CUTLASS had no FP32-via-TF32 Tensor-Core
      path. TF32 keeps FP32 storage/semantics (no graph dtype change) but lets the
      GEMM lowering pick the TF32 Tensor-Core kernel. Lowest-risk, highest-leverage:
      touches all six GEMMs with no graph mutation. NOTE: where OPT-2 promotes a node
      to BF16, that node runs on the BF16 HMMA path and TF32 becomes the safe fallback
      for any GEMM left in FP32.

  OPT-2  aten / medium — BF16 dtype promotion (aten.mm / aten.addmm operands)
      Inside _aten_inner_compile (post-AOTAutograd), cast both matmul operands of every
      aten.mm.default AND aten.addmm.default node (and the addmm bias) to bfloat16 via
      prims.convert_element_type with FP32 accumulate, then cast the GEMM result back to
      FP32 so downstream FP32 consumers keep their dtype contract. Routes the two dominant
      [8192,512]x[512,32000] logit GEMMs (85.4% of attributed time) and the MLP GEMMs from
      the SIMT FP32 path to the Blackwell BF16 Tensor-Core (HMMA) path, and halves the
      ~1 GB FP32 logit-output write. LayerNorm/GELU reductions stay FP32. medium confidence:
      bf16 changes numerics on a 32000-way logit head — validate top-k/argmax agreement.

  OPT-3  inductor_config / medium — Weight freezing + max_autotune_gemm
      Return config_patches={"freezing": True, "max_autotune_gemm": True, ...} merged into
      THIS compile_fx call only (scoped — no global config mutation). Inductor treats the
      requires_grad=False eval-mode projection weights as compile-time constants,
      constant-folds and re-lays-out (pre-transposes/aligns) them so the GEMM autotuner can
      select an aligned (align8) Tensor-Core template instead of the align1 SIMT fallback.
      Requires an inference/eval graph (set in get_model_and_input()).

Prerequisite / ordering rationale:
  - OPT-2 is prerequisite_for OPT-3: freezing materializes the constant weight buffer at
    the runtime dtype, so the weight must already be promoted to BF16 before it is frozen.
    OPT-2 is aten (funnel stage 2) and OPT-3 is inductor_config (funnel stage 3), so the
    funnel level ordering satisfies this prerequisite automatically — no within-level
    sequencing needed.
  - OPT-1 (TF32) is set as a process flag at backend entry AND requested in config_patches;
    it is harmless alongside OPT-2 (BF16 HMMA supersedes TF32 on promoted nodes; TF32 is the
    fallback for any GEMM left FP32).

IR-level mechanics (torch 2.11):
  compile_fx owns AOTAutograd, the decomp table, the boxed calling convention and the
  partitioner. We do NOT use aot_autograd(fw_compiler=compile_fx) — on torch 2.11 that
  raises AssertionError: Expected tensors only inside copy_misaligned_inputs. The funnel
  passes functional rewrites BEFORE compile_fx (none here), aten passes through the
  inner_compile seam, and inductor_config passes as scoped config_patches.

Dedup-aware: this workload is a single linear forward path with no repeated layer
structure, so UniqueSubgraphRegistry finds no duplicate partitions and the flat compile
path is taken (preserving cross-op Inductor fusion). The dedup branch is retained for
interface uniformity.

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
from torch._inductor.compile_fx import compile_fx, compile_fx_inner

from nvidia.operator_profiler.fx import UniqueSubgraphRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_BF16 = torch.bfloat16
_FP32 = torch.float32

# Op targets — the decomposed GEMM primitives produced by AOTAutograd for nn.Linear.
#   nn.Linear(bias=True)  -> aten.addmm.default(bias, x, t(weight))
#   nn.Linear(bias=False) -> aten.mm.default(x, t(weight))   (the logit head)
_MM = torch.ops.aten.mm.default
_ADDMM = torch.ops.aten.addmm.default

# Use prims.convert_element_type, not aten._to_copy. On torch 2.11, aten._to_copy has
# both a fallback and a decomp registration; inserting it into an already-decomposed
# Aten graph makes Inductor raise "both a fallback and a decomp for same op".
# prims.convert_element_type lowers cleanly to a Triton elementwise cast.
_CONVERT = torch.ops.prims.convert_element_type.default


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node_dtype(n) -> "torch.dtype | None":
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
# OPT-1 — Enable TF32. ir_level=inductor_config (+ process flag). Confidence: high.
#
# Two halves:
#   (a) Process-global math-mode flags set once at backend entry (_enable_tf32_flags):
#       torch.backends.cuda.matmul.allow_tf32 / torch.backends.cudnn.allow_tf32. These
#       are the canonical switches the GEMM lowering honors to route FP32 GEMMs onto the
#       TF32 Tensor-Core kernel instead of the cutlass_80_simt_sgemm_*_align1 path.
#   (b) A config-patch contribution (_cfg_tf32) so the policy is recorded on this
#       compile_fx call. Inductor reads the backends flags for cuBLAS path selection;
#       this keeps the funnel's config stage explicit even when the patch is empty.
# No graph node surgery — this is a lowering-policy decision Inductor owns.
# ---------------------------------------------------------------------------

def _enable_tf32_flags() -> None:
    """OPT-1 (a): Set the process-global TF32 math-mode flags. Idempotent / cheap."""
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Explicit fp32-matmul precision policy (torch >= 2.x): 'high' == TF32 reduced.
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        logger.info(
            "[OPT-1 tf32] Enabled TF32: cuda.matmul.allow_tf32=True, "
            "cudnn.allow_tf32=True, float32_matmul_precision='high' "
            "[inductor_config / process flag]"
        )
    except Exception as e:
        logger.warning("[OPT-1 tf32] Failed to set TF32 flags: %s", e)


def _cfg_tf32() -> dict:
    """OPT-1 (b): Inductor config patch recording the TF32 lowering policy.

    The effective TF32 switch is the torch.backends flag set in _enable_tf32_flags();
    Inductor reads that for cuBLAS GEMM path selection. No extra config keys are required,
    so this returns an empty (but logged) patch — the funnel's config stage stays explicit.
    """
    try:
        logger.info(
            "[OPT-1 tf32] Config stage: TF32 honored via torch.backends flags "
            "(no extra Inductor config key required) [inductor_config level]"
        )
        return {}
    except Exception as e:
        logger.warning("[OPT-1 tf32] Config patch failed: %s", e)
        return {}


# ---------------------------------------------------------------------------
# OPT-2 — BF16 dtype promotion (aten.mm / aten.addmm operands). ir_level=aten. medium.
#
# Runs inside _aten_inner_compile after AOTAutograd decomposition. Casts both matmul
# operands of every aten.mm.default and aten.addmm.default node (and the addmm bias) to
# bfloat16 and casts the GEMM output back to float32. The mm branch covers the bias-free
# logit head (the two dominant [8192,512]x[512,32000] GEMMs); the addmm branch covers the
# two MLP projections (proj1 up, proj2 down). Routes cuBLAS from the SIMT FP32
# cutlass_80_simt_sgemm path (tensor_core_active_pct=0.0) to the Blackwell BF16 Tensor-Core
# (HMMA) path and halves the ~1 GB FP32 logit-output write. LayerNorm/GELU stay FP32.
# Confidence medium — bf16 changes numerics on the 32000-way logit head; validate argmax.
# ---------------------------------------------------------------------------

def _apass_bf16_promotion(gm: fx.GraphModule) -> fx.GraphModule:
    """OPT-2: Cast aten.mm/aten.addmm matmul operands to BF16; restore FP32 output.

    Aten IR level (post-AOTAutograd). For addmm(bias, a, b) the matmul operands are a
    (arg1) and b (arg2) and the bias is arg0; all three are promoted to BF16 so the fused
    GEMM+bias runs on Tensor Cores with FP32 accumulate, then the result is down-cast to
    FP32 for downstream consumers. medium confidence -> include the matched-guard.
    """
    try:
        g = gm.graph
        promoted = 0

        for node in list(g.nodes):
            if node.op != "call_function":
                continue

            if node.target is _MM:
                if len(node.args) < 2:
                    continue
                a, b = node.args[0], node.args[1]
                if not (isinstance(a, fx.Node) and isinstance(b, fx.Node)):
                    continue
                a16 = _insert_bf16_cast(g, a, node)
                b16 = _insert_bf16_cast(g, b, node)
                if a16 is a and b16 is b:
                    continue  # already BF16
                node.args = (a16, b16) + tuple(node.args[2:])

            elif node.target is _ADDMM:
                if len(node.args) < 3:
                    continue
                bias, a, b = node.args[0], node.args[1], node.args[2]
                if not all(isinstance(x, fx.Node) for x in (bias, a, b)):
                    continue
                bias16 = _insert_bf16_cast(g, bias, node)
                a16 = _insert_bf16_cast(g, a, node)
                b16 = _insert_bf16_cast(g, b, node)
                if bias16 is bias and a16 is a and b16 is b:
                    continue  # already BF16
                node.args = (bias16, a16, b16) + tuple(node.args[3:])

            else:
                continue

            # Restore FP32 on the output so all downstream users keep float32.
            with g.inserting_after(node):
                back_fp32 = g.call_function(_CONVERT, (node, _FP32))
            node.replace_all_uses_with(
                back_fp32, delete_user_cb=lambda u: u is not back_fp32
            )
            promoted += 1

        if promoted == 0:
            logger.warning(
                "[OPT-2 bf16_promotion] No aten.mm/aten.addmm nodes found — "
                "pass not applied"
            )
            return gm

        g.lint()
        gm.recompile()
        logger.info(
            "[OPT-2 bf16_promotion] Promoted %d aten.mm/addmm node(s) to BF16 operands "
            "(FP32 accumulate, FP32 output restored) [aten IR]",
            promoted,
        )
    except Exception as e:
        logger.warning("[OPT-2 bf16_promotion] Failed: %s", e)
    return gm


# ---------------------------------------------------------------------------
# OPT-3 — Weight freezing + max_autotune_gemm. ir_level=inductor_config. medium.
#
# Returns a dict merged into compile_fx's config_patches (scoped to this compilation
# unit — no process-global Inductor config mutation). Inductor treats requires_grad=False
# parameters (eval mode) as compile-time constants: constant-folds and re-lays-out
# (pre-transposes/aligns) the frozen projection weights so the GEMM autotuner can pick an
# aligned (align8) Tensor-Core template over the align1 SIMT fallback. Requires eval mode
# (set in get_model_and_input()). Runs after OPT-2 by funnel level ordering, so the frozen
# weight is materialized at BF16.
# ---------------------------------------------------------------------------

def _cfg_freezing() -> dict:
    """OPT-3: Return Inductor config patches for weight freezing and GEMM autotuning."""
    try:
        patches = {
            "freezing": True,
            "max_autotune_gemm": True,
            "max_autotune": True,
            "max_autotune_gemm_backends": "ATEN,TRITON",
        }
        logger.info(
            "[OPT-3 freezing] Inductor config_patches: freezing=True, "
            "max_autotune_gemm=True, max_autotune=True, "
            "max_autotune_gemm_backends=ATEN,TRITON [inductor_config level]"
        )
        return patches
    except Exception as e:
        logger.warning("[OPT-3 freezing] Config patch failed: %s", e)
        return {}


# ---------------------------------------------------------------------------
# Pass registry — routed by ir_level
# ---------------------------------------------------------------------------

PASS_REGISTRY = [
    # Aten-level passes (run inside _aten_inner_compile, post-AOTAutograd)
    {"id": "OPT-2", "level": "aten",            "fn": _apass_bf16_promotion},
    # Inductor config patches (merged into compile_fx config_patches)
    {"id": "OPT-1", "level": "inductor_config", "fn": _cfg_tf32},
    {"id": "OPT-3", "level": "inductor_config", "fn": _cfg_freezing},
]

_FUNCTIONAL_PASSES = [p for p in PASS_REGISTRY if p["level"] == "functional"]
_ATEN_PASSES = [p for p in PASS_REGISTRY if p["level"] == "aten"]
_CONFIG_PASSES = [p for p in PASS_REGISTRY if p["level"] == "inductor_config"]


# ---------------------------------------------------------------------------
# LEVEL 1 — Functional passes (Dynamo graph, pre-AOTAutograd). None for this model.
# ---------------------------------------------------------------------------

def _run_functional_passes(gm: fx.GraphModule) -> fx.GraphModule:
    """Run all functional-level passes on the Dynamo graph before compile_fx.

    No functional-level passes are defined for this workload (no fusion / SDPA
    formation applies). Retained for funnel uniformity."""
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
    """Collect and merge all inductor_config-level patches. Scoped to this compile_fx
    call only — no global Inductor config mutation."""
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
    Run aten-level passes (OPT-2 BF16 promotion), repropagating meta after each
    structural rewrite, then delegate to compile_fx_inner (Aten -> Triton).

    ``example_inputs`` may be FakeTensors under FakeTensorMode. ``real_inputs`` is threaded
    from the backend for any pass needing actual weight values (none here — OPT-2 is an
    op-target pass). ``**kwargs`` is forwarded verbatim for forward-compatibility."""
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

    Stage 1: functional passes on the Dynamo graph (none for this model).
    Stage 2: compile_fx owns AOTAutograd + decomp; _aten_inner_compile runs OPT-2 BF16
             promotion on the decomposed Aten IR.
    Stage 3: OPT-1 TF32 + OPT-3 freezing/autotune config_patches scoped to this compile_fx
             call. (OPT-1's process-global flag is set once at backend entry.)"""
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
) -> dict:
    """Run split_gm once under no_grad to capture per-partition input tensors."""
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


# ---------------------------------------------------------------------------
# Backend: embedding_projection_opt
# ---------------------------------------------------------------------------

@register_backend
def embedding_projection_opt(gm: fx.GraphModule, example_inputs) -> Callable:
    """Custom torch.compile() backend for the embedding-projection workload.

    Implements the optimizations from optimizations.json via the three-stage funnel
    (functional -> aten -> inductor_config):

      OPT-1 (config + flag): Enable TF32 — cuda.matmul.allow_tf32=True (process flag)
      OPT-2 (aten):          BF16 promotion — aten.mm + aten.addmm operands BF16, FP32 out
      OPT-3 (config):        Freezing + autotune — freezing=True, max_autotune_gemm=True

    OPT-2 is prerequisite_for OPT-3 (frozen weight must be BF16-materialized); the funnel
    level ordering (aten before inductor_config) satisfies this automatically.

    Dedup-aware: this workload is a single linear forward path with no repeated layer
    structure, so UniqueSubgraphRegistry finds no duplicates and the flat compile path is
    taken (preserving cross-op Inductor fusion). The dedup branch is retained for
    interface uniformity.
    """
    logger.info(
        "embedding_projection_opt backend: starting "
        "(aten[OPT-2] -> inductor_config[OPT-1, OPT-3]; OPT-1 TF32 flag at entry)"
    )

    # OPT-1 (a): set the process-global TF32 math-mode flags before any compile.
    _enable_tf32_flags()

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

    # Compile each unique representative through the same funnel; share the compiled
    # callable with all structural duplicates.
    partition_inputs = _capture_partition_inputs(registry.split, example_inputs)
    for rep_name, rep_mod in registry.unique_reps:
        inputs = partition_inputs.get(rep_name, example_inputs)
        compiled = _compile_unit(rep_mod, inputs)
        rep_mod.forward = compiled
        for _, dup_mod in registry.duplicates_of(rep_name):
            dup_mod.forward = compiled

    return lambda *args: registry.split(*args)


# ---------------------------------------------------------------------------
# Workload — unchanged architecture from embedding_projection.py so the capture
# pipeline can re-profile the optimized backend.
# ---------------------------------------------------------------------------
DEVICE     = "cuda"
BATCH_SIZE = 64
SEQ_LEN    = 128
VOCAB_SIZE = 32_000
DIM        = 512
DIM_FF     = 2048


class EmbeddingProjection(nn.Module):
    """Token embedding lookup + two-layer projection + logit head."""

    def __init__(self):
        super().__init__()
        self.embed   = nn.Embedding(VOCAB_SIZE, DIM)
        self.ln      = nn.LayerNorm(DIM)
        self.proj1   = nn.Linear(DIM,    DIM_FF, bias=True)
        self.proj2   = nn.Linear(DIM_FF, DIM,    bias=True)
        self.logits  = nn.Linear(DIM, VOCAB_SIZE, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(token_ids)          # (B, T, DIM)
        x = self.ln(x)
        x = F.gelu(self.proj1(x))          # (B, T, DIM_FF)
        x = self.proj2(x)                  # (B, T, DIM)
        return self.logits(x)              # (B, T, VOCAB_SIZE)


def get_model_and_input() -> tuple:
    """Workload interface — return (uncompiled model on CUDA, int64 token_ids on CUDA).

    Model dtype: FP32 (matches optimizations.json analysis.dtype = "float32"). OPT-2 BF16
    promotion is applied selectively inside the graph (GEMM operands only), not by casting
    the whole module, so LayerNorm/GELU stay FP32. OPT-1 (TF32) and OPT-3 (freezing) are
    flag/config level. No non-graph eager-side optimizations are needed: there are no conv
    layers (no channels_last), and the GEMM M/N/K dims (M=8192, K/N in {512,2048,32000})
    are all multiples of 16, so no batch padding is required.

    The model is returned with .eval() set; OPT-3 freezing requires eval/inference mode.
    Token IDs stay int64 (embedding indices); the embedding output feeds the BF16-promoted
    GEMMs.
    """
    assert torch.cuda.is_available(), "CUDA required"
    model     = EmbeddingProjection().to(DEVICE).eval()
    token_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    return model, token_ids


if __name__ == "__main__":
    model, token_ids = get_model_and_input()
    compiled = torch.compile(model, backend="embedding_projection_opt")
    with torch.no_grad():
        y = compiled(token_ids)
    # expect (64, 128, 32000) float32
    print(f"Output shape: {y.shape} dtype: {y.dtype}")
