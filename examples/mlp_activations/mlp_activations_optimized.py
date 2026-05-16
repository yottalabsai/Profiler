"""
mlp_activations_optimized.py — MLPActivations with custom torch.compile() backend.

Implements five operator-level optimizations derived from ncu profiling:

  OPT-1 (high confidence)   : BF16 dtype promotion — applied in get_model_and_input()
                              before torch.compile. Every cuBLAS 'Kernel2' SIMT SGEMM
                              (0% Tensor Core, 200-210 regs/thread, 8-17% occupancy)
                              is replaced by a BF16 HMMA Tensor Core kernel with
                              ~32-64 regs/thread and full wave coverage. Expected
                              ~49.2% total latency reduction.

  OPT-2 (medium confidence) : max_autotune GEMM tuning — applied in get_model_and_input()
                              via torch._inductor.config before torch.compile. Inductor
                              benchmarks multiple Triton tile configs for each unique
                              GEMM shape. For [256x2048 @ 2048x512] (wave-starved at
                              eligible_cycles_pct=12%), a tuned split-K config raises
                              SM coverage significantly. Requires OPT-1 (BF16) to tune
                              at the correct dtype. Expected additional ~5% reduction.

  OPT-3 (medium confidence) : TF32 enable — ALTERNATIVE to OPT-1. Keeps FP32 external
                              dtype while routing cuBLAS through the HMMA TC path.
                              Expected ~35.2% reduction. NOT applied when OPT-1 is
                              active (they are mutually exclusive). Implemented as a
                              runtime flag in get_model_and_input() comment only.

  OPT-4 (medium confidence) : reduce-overhead CUDA graphs — compile mode knob that
                              captures the full 12-kernel forward pass as a CUDA graph,
                              eliminating ~60 us of per-kernel cuLaunchKernel CPU
                              overhead. Statically-shaped model (batch=256) is an ideal
                              candidate. Expected additional ~3.4% reduction.

  OPT-5 (low confidence)    : Batch padding for output-projection GEMMs [256x2048 @
                              2048x512] — FX stub only. Pads M from 256 to 384 to
                              improve SM wave coverage from 2.1 to 3.1 blocks/SM.
                              Requires post-lowering shape metadata to identify target
                              nodes; detection only at pre-Inductor level; never applies
                              the transform. Expected additional ~2.4% reduction when
                              fully implemented.

Pass ordering (per optimizations.json prerequisite_for and Rule 6):
  OPT-1 (non-graph, get_model_and_input) →
  OPT-2 (non-graph, torch._inductor.config before compile) →
  OPT-4 (non-graph, compile mode) →
  OPT-5 (FX stub, detect-only, never transforms) →
  compile_fx

Backend name: mlp_activations_opt

Architecture notes:
  - MLPActivations is a flat four-layer MLP with no repeated block structure.
    UniqueSubgraphRegistry.build_partition_equivalence_map() returns empty dict.
    The flat compile path is always taken.
  - All substantive optimizations are non-graph (dtype, compile config, compile mode).
    The FX backend is structurally complete but thin: it registers the backend,
    handles dedup detection, and delegates to compile_fx.
  - OPT-5 is a low-confidence stub: it logs detection information but never mutates
    the graph. Full implementation requires post-Inductor shape metadata.

To profile:
    PYTHONPATH=/home/ubuntu/Profiler nsys profile \\
        --trace=cuda,nvtx \\
        --output=examples/mlp_activations/profiler_output/mlp_activations_opt \\
        --force-overwrite=true \\
        python3 nvidia/scripts/run_workload.py \\
            --workload examples/mlp_activations/mlp_activations_optimized.py \\
            --compile-backend mlp_activations_opt \\
            --warmup-iters 2 --measure-iters 2 \\
            --output-prefix examples/mlp_activations/profiler_output/mlp_activations_opt \\
            --inductor-debug-dir examples/mlp_activations/profiler_output/mlp_activations_opt_inductor_debug
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Callable

import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
from torch._dynamo import register_backend
from torch._inductor.compile_fx import compile_fx  # function, not module

from nvidia.operator_profiler.fx import UniqueSubgraphRegistry, FxPassRunner

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model and workload constants (mirrored from baseline)
# ---------------------------------------------------------------------------
DEVICE     = "cuda"
BATCH_SIZE = 256
DIM_IN     = 512
DIM_HIDDEN = 2048
DIM_OUT    = 512


# ===========================================================================
# Optimized model class
# ===========================================================================

class MLPActivations(nn.Module):
    """
    Four-layer MLP with heterogeneous activations — identical architecture to baseline.

    OPT-1 (BF16) and OPT-2 (max_autotune) are applied externally in
    get_model_and_input(). This class retains the same forward logic so that
    Dynamo traces the same graph structure, allowing the FX backend to be
    validated against the baseline.

    Layer 1: Linear(512  -> 2048) + ReLU
    Layer 2: Linear(2048 -> 2048) + GELU
    Layer 3: Linear(2048 -> 2048) + SiLU
    Layer 4: Linear(2048 -> 512)  + Tanh
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(DIM_IN,     DIM_HIDDEN, bias=True)
        self.fc2 = nn.Linear(DIM_HIDDEN, DIM_HIDDEN, bias=True)
        self.fc3 = nn.Linear(DIM_HIDDEN, DIM_HIDDEN, bias=True)
        self.fc4 = nn.Linear(DIM_HIDDEN, DIM_OUT,    bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = F.silu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x


# ===========================================================================
# OPT-5 (low confidence): Batch-padding stub for output-projection GEMMs
# ===========================================================================
# Classification: Low-confidence FX stub.
#   - Target: aten::mm nodes for [256x2048] @ [2048x512] shapes (op_id=9,19).
#   - Problem: M=256 yields 176 thread blocks → 2.1 blocks/SM on 84 SMs.
#     eligible_cycles_pct=12% → warp scheduler stalled 88% of cycles.
#   - Proposed fix: pad M to 384 (next multiple of 128) before mm, slice
#     output back to [256, 512]. Raises SM coverage to ~3.1 blocks/SM.
#   - Why stub: shape detection requires node.meta['tensor_meta'], which is
#     only populated after Inductor's shape propagation pass. At the pre-
#     Inductor level (where @register_backend receives the graph), tensor_meta
#     is absent on most nodes. The pass is detection-only and never transforms
#     the graph.
#   - Full implementation requires a post-Inductor FX pass hook, which is
#     outside the @register_backend scope. See optimizations.json OPT-5 for
#     the complete aten.constant_pad_nd + aten.slice surgery.
# ===========================================================================

def _stub_pass_pad_output_projections(gm: fx.GraphModule) -> fx.GraphModule:
    """
    Low-confidence stub: detect output-projection GEMM candidates for M-padding.

    Logs detection information but NEVER mutates the graph. The actual pad/slice
    surgery requires post-Inductor shape metadata (node.meta['tensor_meta']) that
    is absent at the pre-Inductor IR level.

    Always returns gm unchanged.
    """
    # Detection only — look for mm or linear nodes whose shape metadata is
    # available and matches the [256x2048 @ 2048x512] output projection pattern.
    candidates_found = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function":
            continue
        if node.target not in (F.linear, torch.ops.aten.mm.default):
            continue
        # Attempt shape lookup from meta (may be absent at pre-Inductor level)
        tm = node.meta.get("tensor_meta")
        if tm is None:
            continue
        try:
            out_shape = list(tm.shape)
            if out_shape == [BATCH_SIZE, DIM_OUT]:
                candidates_found += 1
                logger.warning(
                    "[_stub_pass_pad_output_projections] Candidate output-projection "
                    "node '%s' detected (shape %s). Full M-padding transform requires "
                    "post-Inductor shape metadata — stub does NOT apply the transform.",
                    node.name, out_shape,
                )
        except Exception:
            pass

    if candidates_found == 0:
        logger.warning(
            "[_stub_pass_pad_output_projections] OPT-5: No output-projection "
            "candidates detected at pre-Inductor IR level (tensor_meta absent). "
            "This is expected — shape metadata is only available post-lowering."
        )

    # ALWAYS return gm unchanged — this is a stub, never a transform.
    return gm


# ===========================================================================
# Utility: capture per-partition actual input tensors (dedup path)
# ===========================================================================

def _capture_partition_inputs(
    split_gm: fx.GraphModule,
    example_inputs: list,
) -> dict[str, list]:
    """
    Run split_gm once via forward-pre hooks to capture the actual input tensors
    seen by each partition submodule.

    Required for any pass that reads weight tensor values from partition_inputs
    (Dynamo lifts all nn.Module parameters to placeholder nodes, so weight values
    flow in as function arguments at runtime).
    """
    captured: dict[str, list] = {}
    hooks = []
    for name, submod in split_gm.named_children():
        if isinstance(submod, fx.GraphModule):
            def _hook(mod, args, _name=name):
                captured[_name] = list(args)
            hooks.append(submod.register_forward_pre_hook(_hook))
    with torch.no_grad():
        split_gm(*example_inputs)
    for h in hooks:
        h.remove()
    return captured


# ===========================================================================
# Backend Registration
# ===========================================================================

@register_backend
def mlp_activations_opt(gm: fx.GraphModule, example_inputs: list) -> Callable:
    """
    Custom torch.compile() backend for MLPActivations.

    OPT-1 (BF16 promotion) and OPT-2 (max_autotune config) are applied in
    get_model_and_input() before torch.compile is called. These are non-graph
    optimizations invisible to the FX IR.

    OPT-5 (batch padding) is a low-confidence detection stub applied here —
    it inspects the graph but never mutates it. Full implementation is out of
    scope for this backend level.

    Dedup awareness (Rule 10):
      MLPActivations is a flat 4-layer MLP with no repeated block structure.
      UniqueSubgraphRegistry finds no equivalent partitions; the flat compile
      path is always taken. The dedup path is included for structural
      consistency in case this backend is reused with a deeper variant.
    """
    logger.info("mlp_activations_opt backend: starting")

    registry = UniqueSubgraphRegistry(gm)
    equiv_map = registry.build_partition_equivalence_map()

    if not equiv_map:
        # ------------------------------------------------------------------
        # Flat compile path — no repeated layers (expected for this MLP).
        # Apply low-confidence stub pass, then delegate to Inductor.
        # Preserves any cross-layer fusion Inductor may discover across the
        # flat activation sequence.
        # ------------------------------------------------------------------
        logger.info(
            "mlp_activations_opt: no repeated layers detected — flat compile path"
        )

        # OPT-5 stub: detect-only, never mutates the graph
        gm = _stub_pass_pad_output_projections(gm)

        logger.info("mlp_activations_opt: delegating to Inductor")
        return compile_fx(gm, example_inputs)

    # -----------------------------------------------------------------------
    # Dedup compile path — for deeper repeated-block variants of this model.
    # Apply stub passes to each unique representative; share compiled callable
    # with structural duplicates.
    # -----------------------------------------------------------------------
    logger.info(
        "mlp_activations_opt: %d duplicate partition(s) detected — dedup path",
        len(equiv_map),
    )

    partition_inputs = _capture_partition_inputs(registry.split, example_inputs)

    for rep_name, rep_mod in registry.unique_reps:
        rep_inputs = partition_inputs.get(rep_name, example_inputs)

        # OPT-5 stub (detect-only) on each unique rep
        _stub_pass_pad_output_projections(rep_mod)

        try:
            compiled = compile_fx(rep_mod, rep_inputs)
        except Exception as e:
            logger.warning(
                "mlp_activations_opt: compile_fx failed for partition '%s' (%s) "
                "— falling back to eager forward",
                rep_name, e,
            )
            compiled = rep_mod.forward

        rep_mod.forward = compiled

        for dup_name, dup_mod in registry.duplicates_of(rep_name):
            dup_mod.forward = compiled
            logger.info(
                "mlp_activations_opt: shared compiled callable %s → %s",
                rep_name, dup_name,
            )

    logger.info("mlp_activations_opt: backend assembly complete")
    return lambda *args: registry.split(*args)


# ===========================================================================
# Workload Interface
# ===========================================================================

def get_model_and_input() -> tuple:
    """
    Workload interface for run_workload.py.

    Applies non-graph optimizations before returning:

      OPT-1 (BF16 dtype promotion, high confidence):
        Casts all model parameters to bfloat16 before Dynamo traces the model.
        Forces cuBLAS to select the HMMA Tensor Core path (BF16-native) for
        every aten::mm and aten::addmm node, replacing the 'Kernel2' SIMT SGEMM
        path (0% Tensor Core, 200-210 regs/thread) with a TC kernel at ~32-64
        regs/thread. The fused Triton activation kernels (relu/gelu/silu/tanh)
        also run natively in BF16.
        Expected ~49.2% latency reduction.
        Applied idempotently — skipped if parameters already in BF16.

      OPT-2 (max_autotune GEMM tuning, medium confidence):
        Sets torch._inductor.config.max_autotune = True and
        max_autotune_gemm = True before torch.compile is called by the caller.
        Inductor benchmarks multiple Triton tile configs for each unique GEMM
        shape; for [256x2048 @ 2048x512] (eligible_cycles_pct=12%), a tuned
        split-K config raises SM coverage from 2.1 to 3.1+ blocks/SM.
        Must be applied at same dtype as OPT-1 (BF16) per global_notes.
        Expected additional ~5% reduction.

      OPT-3 (TF32 enable) is the ALTERNATIVE to OPT-1 and is NOT applied here.
        If BF16 precision is unacceptable, replace OPT-1 block with:
          torch.backends.cuda.matmul.allow_tf32 = True
          torch.backends.cudnn.allow_tf32 = True
        Per global_notes: "Apply OPT-1 (BF16) for maximum throughput gain.
        Apply OPT-3 (TF32) as a less invasive alternative. Do not stack both."

      OPT-4 (reduce-overhead CUDA graphs, medium confidence):
        The compile mode is configured here via the COMPILE_MODE constant.
        When torch.compile(model, backend='mlp_activations_opt', mode=...) is
        called by the user or run_workload.py, the mode is passed through
        torch.compile → Dynamo → this backend. Setting COMPILE_MODE='reduce-
        overhead' on the torch.compile call site enables CUDA graph capture.
        NOTE: run_workload.py accepts --compile-mode; to activate OPT-4 pass
        --compile-mode reduce-overhead at the CLI. The default here is
        'max-autotune' (OPT-2), which combined with OPT-1 is the recommended
        single invocation per global_notes OPT-2 DEPENDENCY note.

    OPT-5 (batch padding) is applied in the mlp_activations_opt backend at
    FX graph level (stub only — detect but never transform).

    Returns:
        model : MLPActivations in eval mode on CUDA, parameters in BF16
        x     : FloatTensor of shape (BATCH_SIZE, DIM_IN) = (256, 512) in BF16 on CUDA
    """
    assert torch.cuda.is_available(), "CUDA required"

    model = MLPActivations().to(DEVICE).eval()
    x = torch.randn(BATCH_SIZE, DIM_IN, device=DEVICE)

    # ------------------------------------------------------------------
    # OPT-2: Configure Inductor max_autotune BEFORE any torch.compile call.
    # Must set inductor config at module load / get_model_and_input() time
    # so the config is active when torch.compile triggers compilation.
    # Apply at BF16 dtype (OPT-1 must precede autotuning per global_notes).
    # ------------------------------------------------------------------
    try:
        import torch._inductor.config as _inductor_cfg
        _inductor_cfg.max_autotune = True
        _inductor_cfg.max_autotune_gemm = True
        # Use both Triton and CUTLASS backends for maximum search coverage
        _inductor_cfg.max_autotune_gemm_backends = "TRITON,CUTLASS"
        logger.info(
            "get_model_and_input: OPT-2 applied — max_autotune=True, "
            "max_autotune_gemm=True, backends=TRITON,CUTLASS"
        )
    except Exception as e:
        logger.warning(
            "get_model_and_input: OPT-2 failed to configure max_autotune (%s) "
            "— continuing without GEMM autotuning",
            e,
        )

    # ------------------------------------------------------------------
    # OPT-1: BF16 dtype promotion (non-graph, must precede torch.compile)
    # Cast model parameters first, then input tensor.
    # ------------------------------------------------------------------
    if next(model.parameters()).dtype != torch.bfloat16:
        model = model.to(torch.bfloat16)
        logger.info(
            "get_model_and_input: OPT-1 applied — model cast to bfloat16"
        )

    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
        logger.info(
            "get_model_and_input: OPT-1 applied — input cast to bfloat16"
        )

    logger.info(
        "get_model_and_input: model dtype=%s, input shape=%s, input dtype=%s",
        next(model.parameters()).dtype,
        tuple(x.shape),
        x.dtype,
    )

    return model, x


# ===========================================================================
# CLI entry point
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model, x = get_model_and_input()

    # OPT-4 (reduce-overhead) can be activated by changing the mode here:
    #   mode="reduce-overhead"  — CUDA graph capture (~3.4% additional gain)
    #   mode="max-autotune"     — Inductor GEMM tuning (OPT-2, ~5% gain)
    # The two are mutually exclusive: reduce-overhead does not autotune.
    compiled_model = torch.compile(model, backend="mlp_activations_opt", mode="max-autotune")

    with torch.no_grad():
        y = compiled_model(x)

    print(f"Output shape : {y.shape}")   # expect torch.Size([256, 512])
    print(f"Output dtype : {y.dtype}")   # expect torch.bfloat16
