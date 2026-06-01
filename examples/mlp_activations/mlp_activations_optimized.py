"""
mlp_activations_optimized.py — Custom torch.compile() backend for MLPActivations.

Registered backend: ``mlp_activations_opt``

Implements the three optimizations from optimizations.json routed to the correct
IR level via the three-stage funnel (functional -> aten -> inductor_config). This
workload is a strictly sequential MLP (each Linear consumes the previous
activation), so there is NO parallel-branch (QKV-style) fusion and NO SDPA; the
functional level has no pass to run here.

Backend name: mlp_activations_opt  (model "mlp_activations" -> snake-case + _opt)

Pass summary (execution order: aten then inductor_config):

  OPT-1  aten / high  — BF16 dtype promotion (matmul operands)
      Inside _aten_inner_compile (post-AOTAutograd), cast the matmul operands of every
      aten.mm.default AND aten.addmm.default node to bfloat16 via
      prims.convert_element_type, computing the (add)mm in bf16, and cast the result
      back to float32 to preserve the downstream dtype contract. Eval-mode nn.Linear
      layers lower to aten.addmm.default (bias + matmul fused), so addmm — not mm — is
      the node that actually appears in this MLP; promoting it is the dominant lever.
      At execution this routes cuBLAS off the SIMT FP32 SGEMM path
      (cutlass_80_simt_sgemm_*_tn_align1, tensor_core_active_pct=0.0) onto a Blackwell
      BF16 Tensor Core GEMM path, lowering register pressure and raising occupancy.
      Using prims.convert_element_type (not aten._to_copy) avoids the "both a fallback
      and a decomp for same op" assertion on torch 2.11.

  OPT-2  inductor_config / medium  — max_autotune + epilogue fusion
      Pass config_patches={"max_autotune": True, "max_autotune_gemm_backends":
      "ATEN,TRITON", "epilogue_fusion": True} to compile_fx. Lets Inductor emit Triton
      GEMM templates that fuse the bias-add + ReLU/GELU/SiLU/Tanh pointwise epilogue
      into the GEMM (removing the separate triton_poi_fused_addmm_* kernels and their
      intermediate DRAM round-trips) and autotune tile/split-K per shape, repairing the
      ~6% sm_throughput on the skinny 2048->512 projections.

  OPT-3  inductor_config / medium  — weight freezing
      Pass config_patches={"freezing": True} to compile_fx. Inductor treats the MLP
      weights (requires_grad=False in eval) as compile-time constants, pre-transposes
      them to the GEMM-preferred layout (hoisting the runtime _tn_ transpose to compile
      time), drops per-call weight guards, and exposes a pre-packed BF16 weight layout
      to OPT-2's autotuner. Requires model.eval(), which get_model_and_input() sets.

Prerequisite / ordering rationale:
  - OPT-1 (aten) is a prerequisite of OPT-2 and OPT-3 by pipeline-level ordering:
    the funnel runs the aten BF16 promotion before Inductor lowering, so the
    BF16 Tensor Core templates (where autotune + epilogue fusion + frozen pre-packed
    layouts pay off most) are what the inductor_config passes operate on.
  - OPT-2 and OPT-3 are both inductor_config with no within-level ordering; their
    patches are merged into a single config dict for one compile_fx call.
  - Cross-level ordering (aten -> inductor_config) is enforced by the funnel and
    requires no explicit encoding.

IR-level mechanics (torch 2.11):
  compile_fx owns AOTAutograd, the decomp table, the boxed calling convention and the
  partitioner. We do NOT use aot_autograd(fw_compiler=compile_fx) — on torch 2.11 that
  raises AssertionError: Expected tensors only inside copy_misaligned_inputs. The funnel
  passes functional-level rewrites BEFORE compile_fx (none here), aten-level passes
  through its inner_compile seam, and inductor_config passes as scoped config_patches.

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
# OPT-1 — BF16 dtype promotion (matmul operands). ir_level=aten. Confidence: high.
#
# Runs inside _aten_inner_compile after AOTAutograd has fully decomposed the
# graph. Every aten.mm.default and aten.addmm.default node sees its matmul
# operands cast to bfloat16 via prims.convert_element_type, the (add)mm is
# computed in bf16, and its output is cast back to float32 to preserve the
# downstream dtype contract.
#
# Eval-mode nn.Linear layers lower to aten.addmm.default(bias, mat1, mat2) — bias
# + matmul fused — so addmm is the node that actually carries the GEMM in this MLP.
# For addmm, all three operands (bias, mat1, mat2) are cast to bf16 so the addmm
# dtype is internally consistent, then the bf16 result is cast back to fp32.
#
# This routes cuBLAS dispatch from the SIMT FP32 path
# (cutlass_80_simt_sgemm_128x256_8x4_tn_align1, tensor_core_active_pct=0.0)
# to the BF16 Tensor Core path on Blackwell (sm100).
# ---------------------------------------------------------------------------

def _promote_node_to_bf16(g: fx.Graph, node: fx.Node, operand_idxs) -> bool:
    """Cast the given operand positions of ``node`` to BF16 and cast the node's
    output back to FP32. Returns True if the node was promoted, False if all the
    targeted operands were already BF16 (no-op for this node).

    The output cast-back is inserted via replace_all_uses_with with a guard so the
    newly-inserted convert (the only legitimate consumer of the raw bf16 result) is
    not itself rewired back onto its own input.
    """
    new_args = list(node.args)
    any_cast = False
    for i in operand_idxs:
        operand = new_args[i]
        if not isinstance(operand, fx.Node):
            continue
        cast = _insert_bf16_cast(g, operand, node)
        if cast is not operand:
            new_args[i] = cast
            any_cast = True
    if not any_cast:
        # Every targeted operand was already BF16 — leave the node untouched.
        return False

    node.args = tuple(new_args)

    # Restore FP32 on the output so all downstream users keep float32.
    with g.inserting_after(node):
        back_fp32 = g.call_function(_CONVERT, (node, _FP32))
    node.replace_all_uses_with(
        back_fp32, delete_user_cb=lambda u: u is not back_fp32
    )
    return True


def _apass_bf16_promotion(gm: fx.GraphModule) -> fx.GraphModule:
    """OPT-1: Cast aten.mm / aten.addmm matmul operands to BF16, output back to FP32.

    Aten IR level. High confidence: assume the pattern exists (this is an MLP of
    eval-mode Linear layers, every one of which decomposes to aten.addmm.default —
    bias + matmul fused). aten.mm.default is handled too (harmless if absent). An
    exception is a real error and is logged with a warning so compilation can still
    fall through to compile_fx_inner.

    addmm(bias, mat1, mat2): bias is args[0], the matmul operands are args[1] (mat1)
    and args[2] (mat2). All three are cast to BF16 so the addmm dtype is internally
    consistent, then the BF16 result is cast back to FP32.
    mm(mat1, mat2): args[0] and args[1] are the matmul operands.
    """
    try:
        g = gm.graph
        mm_promoted = 0
        addmm_promoted = 0

        for node in list(g.nodes):
            if node.op != "call_function":
                continue
            if node.target is _MM:
                if len(node.args) < 2:
                    continue
                if _promote_node_to_bf16(g, node, (0, 1)):
                    mm_promoted += 1
            elif node.target is _ADDMM:
                if len(node.args) < 3:
                    continue
                # Cast bias (0), mat1 (1) and mat2 (2): the whole addmm runs in BF16,
                # then the result is converted back to FP32 by _promote_node_to_bf16.
                if _promote_node_to_bf16(g, node, (0, 1, 2)):
                    addmm_promoted += 1

        if mm_promoted == 0 and addmm_promoted == 0:
            logger.warning(
                "[OPT-1 bf16_promotion] No FP32 aten.mm / aten.addmm nodes found "
                "— pass not applied"
            )
            return gm

        g.lint()
        gm.recompile()
        logger.info(
            "[OPT-1 bf16_promotion] Promoted %d aten.mm + %d aten.addmm node(s) to "
            "BF16 operands (FP32 output restored) [aten IR]",
            mm_promoted,
            addmm_promoted,
        )
    except Exception as e:
        logger.warning("[OPT-1 bf16_promotion] Failed: %s", e)
    return gm


# ---------------------------------------------------------------------------
# OPT-2 — max_autotune + epilogue fusion. ir_level=inductor_config. Confidence: medium.
#
# Returns a dict merged into compile_fx's config_patches (scoped to this compile
# unit only — no process-global Inductor config mutation). Inductor benchmarks
# Triton GEMM templates that fuse the bias-add + activation epilogue into the GEMM
# and autotunes tile/split-K per shape for the small-M (batch=256) projections.
# ---------------------------------------------------------------------------

def _cfg_max_autotune() -> dict:
    """OPT-2: Inductor config patches enabling max_autotune + epilogue fusion."""
    try:
        patches = {
            "max_autotune": True,
            "max_autotune_gemm_backends": "ATEN,TRITON",
            "epilogue_fusion": True,
        }
        logger.info(
            "[OPT-2 max_autotune] Inductor config_patches: max_autotune=True, "
            "max_autotune_gemm_backends='ATEN,TRITON', epilogue_fusion=True "
            "[inductor_config level]"
        )
        return patches
    except Exception as e:
        logger.warning("[OPT-2 max_autotune] Config patch failed: %s", e)
        return {}


# ---------------------------------------------------------------------------
# OPT-3 — Weight freezing. ir_level=inductor_config. Confidence: medium.
#
# Returns a dict merged into compile_fx's config_patches. Inductor constant-folds
# the requires_grad=False MLP weights, pre-transposes them to the GEMM-preferred
# layout (hoisting the runtime _tn_ transpose to compile time), and drops per-call
# weight guards. Requires model.eval() (set in get_model_and_input()).
# ---------------------------------------------------------------------------

def _cfg_freezing() -> dict:
    """OPT-3: Inductor config patch enabling weight freezing for the eval model."""
    try:
        patches = {"freezing": True}
        logger.info(
            "[OPT-3 freezing] Inductor config_patches: freezing=True "
            "[inductor_config level]"
        )
        return patches
    except Exception as e:
        logger.warning("[OPT-3 freezing] Config patch failed: %s", e)
        return {}


# ---------------------------------------------------------------------------
# Pass registry — routed by ir_level
# ---------------------------------------------------------------------------

PASS_REGISTRY = [
    # No functional-level passes: this is a sequential MLP with no shared-activation
    # branch fusion (no QKV) and no SDPA. The functional stage is a no-op here.
    # Aten-level passes (run inside _aten_inner_compile, post-AOTAutograd)
    {"id": "OPT-1", "level": "aten", "fn": _apass_bf16_promotion},
    # Inductor config patches (merged into compile_fx config_patches)
    {"id": "OPT-2", "level": "inductor_config", "fn": _cfg_max_autotune},
    {"id": "OPT-3", "level": "inductor_config", "fn": _cfg_freezing},
]

_FUNCTIONAL_PASSES = [p for p in PASS_REGISTRY if p["level"] == "functional"]
_ATEN_PASSES = [p for p in PASS_REGISTRY if p["level"] == "aten"]
_CONFIG_PASSES = [p for p in PASS_REGISTRY if p["level"] == "inductor_config"]


# ---------------------------------------------------------------------------
# LEVEL 1 — Functional passes (Dynamo graph, pre-AOTAutograd)
# ---------------------------------------------------------------------------

def _run_functional_passes(gm: fx.GraphModule) -> fx.GraphModule:
    """Run all functional-level passes on the Dynamo graph before compile_fx.

    No functional passes are registered for this MLP (no QKV fusion / SDPA), so
    this is a structural no-op kept for funnel uniformity and forward extension."""
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
    (the new convert_element_type casts) get meta['val'] before compile_fx_inner."""
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
        logger.warning("[mlp_activations_opt] meta re-propagation skipped: %s", e)


def _aten_inner_compile(
    gm: fx.GraphModule, example_inputs, *, real_inputs=None, **kwargs
) -> Callable:
    """LEVEL 2 — Inductor inner_compile hook.

    compile_fx calls this with the fully decomposed Aten IR graph (post-AOTAutograd).
    Run aten-level passes (OPT-1 BF16 promotion), repropagating meta after each
    structural rewrite, then delegate to compile_fx_inner (Aten -> Triton).

    ``example_inputs`` may be FakeTensors under FakeTensorMode. ``real_inputs`` is
    threaded from the backend for any pass that needs actual weight values (OPT-1 is
    an op-target pass and does not read weight values, so it does not use it here).
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

    Stage 1: functional passes (no-op for this MLP).
    Stage 2: compile_fx owns AOTAutograd + decomp; our _aten_inner_compile hook
             runs OPT-1 BF16 promotion on the decomposed Aten IR.
    Stage 3: OPT-2 max_autotune + OPT-3 freezing config_patches, scoped to this
             compile_fx call."""
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
# Backend: mlp_activations_opt
# ---------------------------------------------------------------------------

@register_backend
def mlp_activations_opt(gm: fx.GraphModule, example_inputs) -> Callable:
    """Custom torch.compile() backend for MLPActivations.

    Implements three optimizations from optimizations.json via the three-stage funnel
    (functional -> aten -> inductor_config):

      OPT-1 (aten):   BF16 promotion  — aten.mm operands BF16, output FP32
      OPT-2 (config): max_autotune    — Triton GEMM templates + epilogue fusion
      OPT-3 (config): freezing        — constant-fold + pre-transpose eval weights

    Dedup-aware: the MLP is a strictly sequential stack with no repeated partitions;
    UniqueSubgraphRegistry returns an empty equivalence map and the flat compile path
    is taken (which also preserves cross-layer Inductor fusion). The dedup branch is
    preserved for models with multiple identical blocks.
    """
    logger.info(
        "mlp_activations_opt backend: starting "
        "(functional[none] -> aten[OPT-1] -> inductor_config[OPT-2, OPT-3])"
    )

    registry = UniqueSubgraphRegistry(gm)
    equiv_map = registry.build_partition_equivalence_map()

    if not equiv_map:
        # No repeated layers — flat compile preserves cross-layer Inductor fusion.
        logger.info("mlp_activations_opt: no repeated layers, flat compile path")
        return _compile_unit(gm, example_inputs)

    logger.info(
        "mlp_activations_opt: %d duplicate partition(s), dedup compile path",
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
BATCH_SIZE = 256
DIM_IN = 512
DIM_HIDDEN = 2048
DIM_OUT = 512


class MLPActivations(nn.Module):
    """Four-layer MLP with heterogeneous activations (ReLU/GELU/SiLU/Tanh).

    Identical architecture to the baseline mlp_activations.py. All graph-level
    optimizations (OPT-1 BF16 promotion) and config-level optimizations (OPT-2
    autotune, OPT-3 freezing) are applied by the mlp_activations_opt backend at
    compile time, not by modifying the module here.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(DIM_IN, DIM_HIDDEN, bias=True)
        self.fc2 = nn.Linear(DIM_HIDDEN, DIM_HIDDEN, bias=True)
        self.fc3 = nn.Linear(DIM_HIDDEN, DIM_HIDDEN, bias=True)
        self.fc4 = nn.Linear(DIM_HIDDEN, DIM_OUT, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = F.silu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x


def get_model_and_input() -> tuple:
    """Return (raw_model, input_tensor) on CUDA — uncompiled, unwarmed.

    Model dtype: FP32 (matches optimizations.json analysis.dtype = "float32").
    OPT-1 BF16 promotion is applied selectively inside the graph at the aten level,
    not by casting the whole module, to preserve the FP32 input/output dtype contract.

    No non-graph (eager-side) optimizations are required for this workload: there are
    no conv layers needing channels_last, and the GEMM M/N/K dims (256, 512, 2048) are
    multiples of 16, so no batch padding is needed.

    The model is returned with .eval() set; OPT-3 freezing requires eval mode.
    """
    assert torch.cuda.is_available(), "CUDA required"
    model = MLPActivations().to(DEVICE).eval()
    x = torch.randn(BATCH_SIZE, DIM_IN, device=DEVICE)
    return model, x


if __name__ == "__main__":
    model, x = get_model_and_input()
    compiled = torch.compile(model, backend="mlp_activations_opt")
    with torch.no_grad():
        out = compiled(x)
    print("output shape:", out.shape, "dtype:", out.dtype)
