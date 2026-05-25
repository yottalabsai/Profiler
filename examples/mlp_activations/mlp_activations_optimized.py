"""
mlp_activations_optimized.py — Custom torch.compile() backend for MLPActivations.

Implements five profiling-guided optimizations from optimizations.json:

  OPT-1 (HIGH, priority 1)   — TF32/BF16 dtype promotion
                               Stage 1: torch.set_float32_matmul_precision('high')
                               and allow_tf32 flags set at module load (non-graph).
                               Stage 2: FX pass that casts F.linear input, weight,
                               and bias to bfloat16 before each mm/addmm, and
                               restores float32 on the output.
                               Dominant fix: Tensor Cores were 0% active on all
                               GEMM kernels — cuBLAS was routing to SIMT FP32
                               scalar path (Kernel2) exclusively.

  OPT-2 (HIGH, priority 2)   — Eliminate splitKreduce overhead
                               Set torch._inductor.config.coordinate_descent_tuning
                               = True so the autotuner finds non-splitK tile configs
                               for the M=256 tall-skinny shapes [256x512]x[512x2048]
                               and [256x2048]x[2048x512].
                               Implemented as Inductor config in the backend.

  OPT-3 (MEDIUM, priority 3) — Activation epilogue fusion
                               Set torch._inductor.config.epilogue_fusion = True and
                               epilogue_fusion_first_threshold = 1000 to fold
                               relu/gelu/silu/tanh into the GEMM epilogue tile,
                               eliminating separate pointwise kernel launches.
                               Implemented as Inductor config in the backend.

  OPT-4 (MEDIUM, priority 4) — Batch repeated GEMMs into bmm
                               FX pass that detects pairs of F.linear nodes sharing
                               the same weight placeholder and replaces them with a
                               single bmm on a stacked input, eliminating duplicate
                               kernel launches for identical-weight GEMMs.
                               Operates at the F.linear level (pre-Inductor).

  OPT-5 (LOW, priority 5)    — Register pressure via max_autotune_gemm
                               Set torch._inductor.config.max_autotune_gemm = True
                               to let Inductor select lower-register-pressure Triton
                               GEMM templates. Stub: depends on OPT-1 outcome.

Backend registration name: mlp_activations_opt
Prerequisite order: OPT-1 → OPT-2, OPT-1 → OPT-3 (per optimizations.json).
OPT-4 and OPT-5 are independent.
"""
from __future__ import annotations

import logging
import operator
from collections import defaultdict
from typing import Callable

import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
from torch._dynamo import register_backend
from torch._inductor.compile_fx import compile_fx  # function, not module

from nvidia.operator_profiler.fx import UniqueSubgraphRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constant: backend name
# ---------------------------------------------------------------------------
BACKEND_NAME = "mlp_activations_opt"

# ---------------------------------------------------------------------------
# Module-load-time side effects for OPT-1 Stage 1
# Must be set BEFORE torch.compile traces the model.
# ---------------------------------------------------------------------------

# OPT-1 Stage 1: route all FP32 GEMMs through TF32 Tensor Core path.
# 'high' maps to TF32; 'medium' would select BF16 accumulation in some backends.
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
logger.info(
    "mlp_activations_opt: set_float32_matmul_precision('high'), allow_tf32=True "
    "at module load [OPT-1 Stage 1]"
)


# ---------------------------------------------------------------------------
# Model definition (verbatim copy so this file is self-contained)
# ---------------------------------------------------------------------------

DEVICE     = "cuda"
BATCH_SIZE = 256
DIM_IN     = 512
DIM_HIDDEN = 2048
DIM_OUT    = 512


class MLPActivations(nn.Module):
    """
    Four-layer MLP with heterogeneous activations.

    Layer 1: Linear + ReLU
    Layer 2: Linear + GELU
    Layer 3: Linear + SiLU
    Layer 4: Linear + Tanh
    """
    def __init__(self):
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


# ---------------------------------------------------------------------------
# FX Pass: OPT-1 Stage 2 — BF16 promotion on F.linear nodes (high confidence)
# ---------------------------------------------------------------------------

def _pass_promote_linear_to_bf16(gm: fx.GraphModule) -> fx.GraphModule:
    """
    OPT-1 Stage 2 — insert bfloat16 casts around every F.linear node.

    At pre-Inductor level, nn.Linear lowers to call_function with
    target=torch.nn.functional.linear, args=(input, weight, bias).

    Strategy:
      - Cast input, weight (and bias if present) to bfloat16 before the linear.
      - Cast the output back to float32 so downstream activations and the next
        linear's input-cast remain dtype-consistent.

    This forces cuBLAS/Inductor to select a Tensor Core (BF16) GEMM algorithm
    on Blackwell, replacing the SIMT Kernel2 path that had 0% Tensor Core
    activity in the baseline profile.
    """
    try:
        matched = False
        graph = gm.graph
        for node in list(graph.nodes):
            if not (node.op == "call_function" and node.target is F.linear):
                continue
            matched = True

            inp_node    = node.args[0]
            weight_node = node.args[1]
            bias_node   = node.args[2] if len(node.args) > 2 else None

            with graph.inserting_before(node):
                cast_inp = graph.call_function(
                    torch.ops.aten.to.dtype,
                    args=(inp_node, torch.bfloat16),
                )
                cast_w = graph.call_function(
                    torch.ops.aten.to.dtype,
                    args=(weight_node, torch.bfloat16),
                )

            new_args: tuple
            if bias_node is not None:
                with graph.inserting_before(node):
                    cast_bias = graph.call_function(
                        torch.ops.aten.to.dtype,
                        args=(bias_node, torch.bfloat16),
                    )
                new_args = (cast_inp, cast_w, cast_bias)
            else:
                new_args = (cast_inp, cast_w)

            node.args = new_args

            # Restore float32 on the output to preserve downstream dtype contract
            with graph.inserting_after(node):
                cast_out = graph.call_function(
                    torch.ops.aten.to.dtype,
                    args=(node, torch.float32),
                )
            node.replace_all_uses_with(cast_out)
            # Fix the self-reference broken by replace_all_uses_with
            cast_out.args = (node,) + cast_out.args[1:]

        if not matched:
            logger.warning(
                "[_pass_promote_linear_to_bf16] No F.linear nodes found — pass not applied"
            )
            return gm

        graph.lint()
        gm.recompile()
        logger.info(
            "[_pass_promote_linear_to_bf16] Applied BF16 casts to all F.linear nodes [OPT-1]"
        )

    except Exception as exc:
        logger.warning("[_pass_promote_linear_to_bf16] Failed: %s", exc)

    return gm


# ---------------------------------------------------------------------------
# FX Pass: OPT-4 — Batch repeated GEMMs sharing the same weight into bmm
#                  (medium confidence)
# ---------------------------------------------------------------------------

def _pass_fuse_repeated_mm_to_bmm(gm: fx.GraphModule) -> fx.GraphModule:
    """
    OPT-4 — replace pairs of F.linear nodes that share a weight placeholder
    with a single bmm on a stacked input batch.

    Pattern: two call_function(F.linear, args=(x_i, w, bias)) nodes where
    w (node.args[1]) is the identical FX node — i.e. the same weight placeholder
    is consumed by two separate linear calls.

    Replacement:
      stack_input = torch.stack([x_0, x_1], dim=0)   # [2, M, K]
      w_expanded  = w.unsqueeze(0).expand(2, -1, -1)  # [2, N, K]
      bmm_out     = torch.bmm(stack_input, w_expanded.transpose(-1, -2))  # [2, M, N]
      # ... then re-add bias and select slices for each original user

    Note: At the pre-Inductor level this MLP has distinct weight parameters for
    each of the four layers, so no weights are actually shared. The pass detects
    this gracefully and logs a warning rather than transforming the graph.
    The implementation is correct for workloads where weight sharing does occur
    (e.g. tied embeddings or repeated blocks with shared projection weights).

    Bias handling: F.linear adds bias after the matmul. For the batched case,
    we add the bias to each bmm slice before returning the per-original output.
    """
    try:
        matched = False
        graph = gm.graph
        nodes_snapshot = list(graph.nodes)

        # Group F.linear nodes by their weight argument (node identity = same tensor)
        weight_to_linears: dict = defaultdict(list)
        for node in nodes_snapshot:
            if node.op == "call_function" and node.target is F.linear:
                weight_arg = node.args[1]
                weight_to_linears[weight_arg].append(node)

        for weight_node, linear_nodes in weight_to_linears.items():
            if len(linear_nodes) < 2:
                continue

            # Only fuse pairs (extend to N in a loop if needed)
            n1, n2 = linear_nodes[0], linear_nodes[1]
            bias_n1 = n1.args[2] if len(n1.args) > 2 else None
            bias_n2 = n2.args[2] if len(n2.args) > 2 else None
            inp_n1  = n1.args[0]
            inp_n2  = n2.args[0]

            # n1 must appear before n2 in topological order (already guaranteed
            # by iterating nodes_snapshot in graph order)
            matched = True

            with graph.inserting_before(n1):
                # Stack inputs along batch dim: [2, M, K]
                stack_node = graph.call_function(
                    torch.stack,
                    args=([inp_n1, inp_n2],),
                    kwargs={"dim": 0},
                )
                # Expand weight: [N, K] -> [1, N, K] -> [2, N, K]
                w_unsqueezed = graph.call_function(
                    torch.ops.aten.unsqueeze.default,
                    args=(weight_node, 0),
                )
                w_expanded = graph.call_function(
                    torch.ops.aten.expand.default,
                    args=(w_unsqueezed, [2, -1, -1]),
                )
                # Transpose weight for mm: [2, N, K] -> [2, K, N]
                w_transposed = graph.call_function(
                    torch.ops.aten.transpose.int,
                    args=(w_expanded, -1, -2),
                )
                # Batched matmul: [2, M, K] @ [2, K, N] -> [2, M, N]
                bmm_node = graph.call_function(
                    torch.ops.aten.bmm.default,
                    args=(stack_node, w_transposed),
                )
                # Select slice 0 and 1 for the two original linears
                slice0 = graph.call_function(
                    torch.ops.aten.select.int,
                    args=(bmm_node, 0, 0),
                )
                slice1 = graph.call_function(
                    torch.ops.aten.select.int,
                    args=(bmm_node, 0, 1),
                )
                # Re-add bias if present (F.linear semantics: output + bias)
                out0: fx.Node = slice0
                if bias_n1 is not None:
                    out0 = graph.call_function(
                        torch.ops.aten.add.Tensor,
                        args=(slice0, bias_n1),
                    )
                out1: fx.Node = slice1
                if bias_n2 is not None:
                    out1 = graph.call_function(
                        torch.ops.aten.add.Tensor,
                        args=(slice1, bias_n2),
                    )

            n1.replace_all_uses_with(out0)
            n2.replace_all_uses_with(out1)

            logger.info(
                "[_pass_fuse_repeated_mm_to_bmm] Fused %s + %s "
                "sharing weight %s into bmm [OPT-4]",
                n1.name, n2.name, weight_node.name,
            )

        if not matched:
            logger.warning(
                "[_pass_fuse_repeated_mm_to_bmm] No F.linear pairs share a weight node "
                "— pass not applied (distinct weights per layer, as expected for this MLP) "
                "[OPT-4]"
            )
            return gm

        graph.eliminate_dead_code()
        graph.lint()
        gm.recompile()
        logger.info("[_pass_fuse_repeated_mm_to_bmm] Applied [OPT-4]")

    except Exception as exc:
        logger.warning("[_pass_fuse_repeated_mm_to_bmm] Failed: %s", exc)

    return gm


# ---------------------------------------------------------------------------
# Utility: capture partition inputs (needed for dedup path)
# ---------------------------------------------------------------------------

def _capture_partition_inputs(
    split_gm: fx.GraphModule,
    example_inputs: list,
) -> dict[str, list]:
    """Capture actual input tensors for each partition by running split_gm once."""
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
    """
    Custom torch.compile() backend for MLPActivations.

    Pass application order (per prerequisite_for constraints in optimizations.json):
      1. OPT-4 — batch repeated GEMMs to bmm (before BF16 pass to avoid
                 dtype mismatches in the pattern-matching check)
      2. OPT-1 Stage 2 — BF16 cast on all F.linear nodes
      3. OPT-2 — coordinate_descent_tuning (Inductor config; no graph change)
      4. OPT-3 — epilogue_fusion (Inductor config; no graph change)
      5. OPT-5 — max_autotune_gemm (Inductor config; stub)
      6. Delegate to compile_fx

    OPT-1 Stage 1 (TF32 global flags) is applied at module-load time above.
    """
    logger.info("mlp_activations_opt backend: starting FX pass pipeline")

    # Build dedup registry to detect any repeated subgraph structure.
    # MLPActivations has four distinct layers — no structural duplicates expected.
    # The flat-compile path is the primary path.
    registry = UniqueSubgraphRegistry(gm)
    equiv_map = registry.build_partition_equivalence_map()

    if not equiv_map:
        # Flat path — no repeated partitions. Apply passes directly to full graph.
        logger.info("mlp_activations_opt: no repeated partitions — flat compile path")

        # OPT-4: batch repeated GEMMs (run before BF16 to keep pattern detection clean)
        gm = _pass_fuse_repeated_mm_to_bmm(gm)

        # OPT-1 Stage 2: BF16 cast on all F.linear nodes
        gm = _pass_promote_linear_to_bf16(gm)

        # OPT-2 + OPT-3 + OPT-5: Inductor config directives
        _apply_inductor_config()

        logger.info("mlp_activations_opt: delegating to Inductor compile_fx")
        return compile_fx(gm, example_inputs)

    # Dedup path — unexpected for MLPActivations, included for robustness
    logger.info(
        "mlp_activations_opt: %d duplicate partition(s) detected — dedup path",
        len(equiv_map),
    )

    for rep_name, rep_mod in registry.unique_reps:
        _pass_fuse_repeated_mm_to_bmm(rep_mod)
        _pass_promote_linear_to_bf16(rep_mod)
        for _, dup_mod in registry.duplicates_of(rep_name):
            _pass_fuse_repeated_mm_to_bmm(dup_mod)
            _pass_promote_linear_to_bf16(dup_mod)

    _apply_inductor_config()

    partition_inputs = _capture_partition_inputs(registry.split, example_inputs)

    for rep_name, rep_mod in registry.unique_reps:
        inputs = partition_inputs.get(rep_name, example_inputs)
        compiled = compile_fx(rep_mod, inputs)
        rep_mod.forward = compiled
        for _, dup_mod in registry.duplicates_of(rep_name):
            dup_mod.forward = compiled

    return lambda *args: registry.split(*args)


def _apply_inductor_config() -> None:
    """
    Apply Inductor config directives for OPT-2, OPT-3, and OPT-5.

    OPT-2: coordinate_descent_tuning — lets Inductor's autotuner search
    non-splitK tile configs for M=256 tall-skinny GEMMs.

    OPT-3: epilogue_fusion — folds activation functions (relu/gelu/silu/tanh)
    into the GEMM output tile write, eliminating separate pointwise kernels.

    OPT-5: max_autotune_gemm — stub; enables Inductor to select lower-register
    Triton GEMM templates. Depends on OPT-1 switching to TC-path first.
    """
    try:
        import torch._inductor.config as inductor_config

        # OPT-2: coordinate descent tile-size search (eliminates splitKreduce)
        if not getattr(inductor_config, "coordinate_descent_tuning", False):
            inductor_config.coordinate_descent_tuning = True
            logger.info(
                "mlp_activations_opt: coordinate_descent_tuning=True [OPT-2]"
            )
        if hasattr(inductor_config, "coordinate_descent_n_steps"):
            inductor_config.coordinate_descent_n_steps = 10

        # OPT-3: epilogue fusion for activation folding into GEMM
        if not getattr(inductor_config, "epilogue_fusion", True):
            inductor_config.epilogue_fusion = True
            logger.info("mlp_activations_opt: epilogue_fusion=True [OPT-3]")
        else:
            # Already True by default in recent Inductor; log confirmation
            logger.info(
                "mlp_activations_opt: epilogue_fusion already enabled [OPT-3]"
            )
        if hasattr(inductor_config, "epilogue_fusion_first_threshold"):
            inductor_config.epilogue_fusion_first_threshold = 1000
            logger.info(
                "mlp_activations_opt: epilogue_fusion_first_threshold=1000 [OPT-3]"
            )

        # OPT-5: max_autotune_gemm — enable Triton GEMM template autotuning
        if not getattr(inductor_config, "max_autotune_gemm", False):
            inductor_config.max_autotune_gemm = True
            logger.info("mlp_activations_opt: max_autotune_gemm=True [OPT-5 stub]")
        if hasattr(inductor_config, "max_autotune_gemm_backends"):
            inductor_config.max_autotune_gemm_backends = "TRITON,ATEN"

    except Exception as exc:
        logger.warning("mlp_activations_opt: Inductor config setup failed: %s", exc)


# ---------------------------------------------------------------------------
# Workload interface
# ---------------------------------------------------------------------------

def get_model_and_input() -> tuple:
    """
    Return (uncompiled_model, input_tensor) ready for torch.compile.

    Non-graph optimizations applied here:
    - OPT-1 Stage 1: TF32 global flags are set at module-load time above.
      No model-level dtype conversion is done here; BF16 casts are injected
      by the backend FX pass so the public interface stays float32.

    No channels_last or batch-padding required: MLPActivations uses 2-D
    input [BATCH_SIZE, DIM_IN] — channels_last only applies to 4-D NCHW tensors.
    """
    assert torch.cuda.is_available(), "CUDA required"
    model = MLPActivations().to(DEVICE).eval()
    x     = torch.randn(BATCH_SIZE, DIM_IN, device=DEVICE)
    return model, x


if __name__ == "__main__":
    model, x = get_model_and_input()
    compiled = torch.compile(model, backend=BACKEND_NAME)
    with torch.no_grad():
        out = compiled(x)
    print(f"Output shape: {out.shape}, dtype: {out.dtype}")
