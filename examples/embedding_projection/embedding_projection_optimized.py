"""
embedding_projection_optimized.py — EmbeddingProjection with custom torch.compile() backend.

Implements the following operator-level optimizations derived from ncu profiling:

  1. OPT-1 (BF16 dtype promotion, high confidence) — Casts model parameters to
     bfloat16 before torch.compile() so Dynamo traces a monomorphic BF16 graph.
     Routes all GEMMs from ampere_sgemm_128x64_tn (FP32 SIMT, 0% Tensor Core)
     to sm80_xmma_gemm_bf16bf16 (HMMA Tensor Core, ~312 TFLOPS peak on A100).
     Token IDs remain int64 (embedding lookup requires integer indices).

  2. OPT-2 (batch logit projection, medium confidence) — FX pass stub.
     The optimizations.json describes 6 independent logit projection calls
     sharing one weight matrix; the baseline model makes a single call, so this
     pass implements detection logic only and logs when the multi-call pattern
     is found. Applied as a manual per-rep pass in the backend.

  3. OPT-3 (pre-transposed weights, medium confidence) — FX pass that detects
     aten.t(get_attr) → aten.mm / aten.addmm patterns, pre-transposes the weight
     tensor into a contiguous buffer, and eliminates the runtime transpose. This
     switches cuBLAS from TN to NN GEMM mode on eligible large weight matrices
     (numel * elem_size >= 1 MB). Applied as a manual per-rep pass in the backend.

  4. OPT-4 (max-autotune, medium confidence) — Enables Inductor's cuBLAS
     algorithm search and Triton GEMM autotuning via config_patches to compile_fx.
     Most effective for the large logit projection ([8192,512] x [512,32000] BF16).

Application order (from optimizations.json dependency_dag):
  OPT-1 → OPT-3 → OPT-2 → compile with max-autotune (OPT-4)

Backend name: embedding_projection_opt

To profile with optimizations:
    python nvidia/scripts/run_workload.py \\
        --workload examples/embedding_projection/embedding_projection_optimized.py \\
        --compile-backend embedding_projection_opt \\
        --output-prefix runs/embedding_projection_opt \\
        --warmup-iters 3 --measure-iters 10
"""
from __future__ import annotations

import logging
from typing import Callable

import torch
import torch.fx as fx
import torch.nn as nn
from torch._dynamo import register_backend
from torch._inductor.compile_fx import compile_fx  # import the function, not the module

from nvidia.operator_profiler.fx import UniqueSubgraphRegistry, FxPassRunner

# ---------------------------------------------------------------------------
# Re-export baseline model classes so this file is a self-contained workload
# ---------------------------------------------------------------------------
from examples.embedding_projection.embedding_projection import (
    EmbeddingProjection,
    get_model_and_input as _baseline_get_model_and_input,
    DEVICE,
    BATCH_SIZE,
    SEQ_LEN,
    VOCAB_SIZE,
    DIM,
    DIM_FF,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Non-graph configuration (set before torch.compile)
# ---------------------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True  # belt-and-suspenders for BF16 path


# ============================================================================
# OPT-2: Batch logit projection — detection stub
# ============================================================================
# Classification: Manual per-rep stub.
#   The baseline EmbeddingProjection calls self.logits once per forward pass.
#   The profile mentions 6 independent logit calls in a multi-layer variant.
#   This stub detects if >= 2 aten.mm / aten.addmm nodes share the same weight
#   get_attr (same logit matrix). If found, it logs a warning; it does not
#   restructure the graph because the single-call baseline does not exhibit the
#   pattern and the multi-call variant is structurally different from the
#   workload in this file.
# ============================================================================

def _pass_batch_logit_projection_stub(gm: fx.GraphModule) -> fx.GraphModule:
    """
    Detection stub for batched logit projection (OPT-2).

    Pattern: multiple aten.mm(x_i, t(W_logit)) or aten.addmm(b, x_i, t(W_logit))
    nodes that share the exact same weight get_attr node (W_logit).

    If the pattern is detected on the graph passed to this backend (e.g. in a
    multi-layer variant of the model), log a warning explaining what
    infrastructure is needed and return the graph unchanged.

    Returns gm unchanged in all cases.
    """
    try:
        # Map weight_node -> [mm_nodes that use it as the weight argument]
        weight_to_mms: dict[fx.Node, list[fx.Node]] = {}

        for node in list(gm.graph.nodes):
            if node.op != "call_function":
                continue

            if node.target is torch.ops.aten.mm.default and len(node.args) >= 2:
                # mm(x, weight_or_t_node)
                weight_arg = node.args[1]
            elif node.target is torch.ops.aten.addmm.default and len(node.args) >= 3:
                # addmm(bias, x, weight_or_t_node)
                weight_arg = node.args[2]
            else:
                continue

            # The weight arg may be a direct get_attr or wrapped in aten.t()
            if weight_arg.op == "get_attr":
                key = weight_arg
            elif (
                weight_arg.op == "call_function"
                and weight_arg.target is torch.ops.aten.t.default
                and len(weight_arg.args) >= 1
                and weight_arg.args[0].op == "get_attr"
            ):
                key = weight_arg.args[0]
            else:
                continue

            weight_to_mms.setdefault(key, []).append(node)

        # Report if any weight is used in 2+ independent mm nodes
        for weight_node, mm_nodes in weight_to_mms.items():
            if len(mm_nodes) >= 2:
                logger.warning(
                    "[_pass_batch_logit_projection_stub] Detected %d independent mm "
                    "ops sharing weight '%s'. Batching into a single GEMM would reduce "
                    "kernel launches from %d to 1, but is NOT applied here — requires "
                    "graph-level concatenation of activation inputs and splitting of "
                    "outputs, which is model-specific. Implement OPT-2 as a model-level "
                    "rewrite in forward() for production use.",
                    len(mm_nodes),
                    weight_node.target,
                    len(mm_nodes),
                )
    except Exception as e:
        logger.warning("[_pass_batch_logit_projection_stub] Detection failed: %s", e)
    return gm  # always returned unchanged


# ============================================================================
# OPT-3: Pre-transposed weights — manual per-rep FX pass
# ============================================================================
# Classification: Manual per-rep pass.
#   Requires register_buffer (cannot use replace_pattern).
#   Targets aten.t() nodes wrapping get_attr in the Inductor-lowered graph.
#   Only applied to weight matrices with numel * element_size >= 1 MB.
#
# Pattern (post-Dynamo Inductor-lowered graph):
#   aten.addmm(bias, x, aten.t(get_attr('weight')))  — proj1, proj2 (bias-bearing)
#   aten.mm(x, aten.t(get_attr('weight')))            — logits (bias=None in Linear)
# ============================================================================

def _pass_pretranspose_weights(gm: fx.GraphModule) -> fx.GraphModule:
    """
    Detect aten.t(get_attr) nodes feeding aten.mm or aten.addmm.

    For each matching node whose weight tensor is >= 1 MB (BF16 after OPT-1):
      1. Pre-transposes the weight in-place to a contiguous buffer.
      2. Registers it as a new buffer on gm.
      3. Inserts a get_attr node pointing to the buffer.
      4. Patches the mm / addmm args to use the transposed buffer directly
         (no aten.t() in the hot path).
      5. Erases the orphaned aten.t() node.

    Wrapped in try/except; returns gm unchanged on any failure.
    """
    try:
        rewritten = 0
        nodes_snapshot = list(gm.graph.nodes)  # snapshot — never iterate live

        for node in nodes_snapshot:
            if node.op != "call_function":
                continue

            target = node.target

            if target is torch.ops.aten.mm.default and len(node.args) >= 2:
                x_node = node.args[0]
                t_node = node.args[1]
                bias_node = None
                is_addmm = False
            elif target is torch.ops.aten.addmm.default and len(node.args) >= 3:
                bias_node = node.args[0]
                x_node = node.args[1]
                t_node = node.args[2]
                is_addmm = True
            else:
                continue

            # t_node must be aten.t() wrapping a get_attr (parameter node)
            if not (
                t_node.op == "call_function"
                and t_node.target is torch.ops.aten.t.default
                and len(t_node.args) >= 1
            ):
                continue
            weight_node = t_node.args[0]
            if weight_node.op != "get_attr":
                continue

            # Retrieve the actual parameter tensor
            try:
                weight_tensor = gm.get_parameter(weight_node.target)
            except AttributeError:
                try:
                    weight_tensor = gm.get_buffer(weight_node.target)
                except AttributeError:
                    logger.warning(
                        "[_pass_pretranspose_weights] Cannot retrieve tensor for '%s' "
                        "— skipping this node",
                        weight_node.target,
                    )
                    continue

            # Only pre-transpose weights >= 1 MB (in current dtype after OPT-1)
            size_bytes = weight_tensor.numel() * weight_tensor.element_size()
            if size_bytes < 1024 * 1024:
                logger.info(
                    "[_pass_pretranspose_weights] Weight '%s' is %.1f KB < 1 MB "
                    "threshold — skipping",
                    weight_node.target,
                    size_bytes / 1024,
                )
                continue

            # Build buffer name; skip if already registered (weight shared across mm nodes)
            buf_name = weight_node.target.replace(".", "_") + "_T"
            if not hasattr(gm, buf_name):
                W_T = weight_tensor.t().contiguous()
                gm.register_buffer(buf_name, nn.Parameter(W_T, requires_grad=False))
                logger.info(
                    "[_pass_pretranspose_weights] Registered buffer '%s' "
                    "(shape=%s, dtype=%s, %.1f MB)",
                    buf_name,
                    list(W_T.shape),
                    W_T.dtype,
                    W_T.numel() * W_T.element_size() / (1024 * 1024),
                )

            # Insert get_attr node for the transposed buffer before the mm/addmm node
            with gm.graph.inserting_before(node):
                new_weight_node = gm.graph.get_attr(buf_name)
                new_weight_node.meta = {}

            # Patch mm / addmm args — the buffer is already transposed, no aten.t()
            if is_addmm:
                node.args = (bias_node, x_node, new_weight_node)
            else:
                node.args = (x_node, new_weight_node)

            # Erase orphaned aten.t() node only if no other users remain
            if len(t_node.users) == 0:
                gm.graph.erase_node(t_node)

            rewritten += 1

        if rewritten > 0:
            gm.graph.lint()
            gm.recompile()
            logger.info(
                "[_pass_pretranspose_weights] Applied — pre-transposed %d weight(s). "
                "cuBLAS switches from TN to NN GEMM mode on eligible kernels.",
                rewritten,
            )
        else:
            logger.warning(
                "[_pass_pretranspose_weights] Pattern aten.t(get_attr) not found — "
                "pass not applied. This is expected if running on the pre-Inductor "
                "graph; the pass targets the post-Dynamo Inductor-lowered graph."
            )
    except Exception as e:
        logger.warning("[_pass_pretranspose_weights] Failed: %s", e)
    return gm


# ============================================================================
# Utility: capture per-partition actual input tensors
# ============================================================================

def _capture_partition_inputs(
    split_gm: fx.GraphModule,
    example_inputs: list,
) -> dict[str, list]:
    """
    Run split_gm once to capture actual per-partition input tensors via hooks.

    Returns a dict mapping partition submodule name -> list of input tensors.
    This is necessary because each partition may see different shapes/dtypes
    than the top-level example_inputs (e.g. embedding outputs vs. token ids).
    """
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


# ============================================================================
# Backend Registration
# ============================================================================

@register_backend
def embedding_projection_opt(gm: fx.GraphModule, example_inputs) -> Callable:
    """
    Custom torch.compile() backend for EmbeddingProjection.

    Pass order (per optimizations.json dependency_dag and Rule 6):
      1. OPT-3 pre-transpose (manual per-rep) — applied to the Inductor-lowered
         graph so aten.t() nodes are visible. Eliminates runtime weight transpose,
         switches cuBLAS from TN to NN GEMM mode.
      2. OPT-2 detection stub (manual per-rep) — detects multi-logit-projection
         pattern; logs warning if found; does not modify the graph.
      3. OPT-4 max-autotune — forwarded as config_patches to compile_fx, enabling
         Inductor's cuBLAS algorithm search and Triton GEMM tile autotuning.

    Dedup awareness (Rule 10):
      EmbeddingProjection has no repeated layer structure. The flat path is taken;
      all passes are applied to the full flat graph and the result is compiled
      with compile_fx. The dedup path is included for correctness if this backend
      is reused with a layered variant.
    """
    logger.info("embedding_projection_opt backend: starting")

    # OPT-4: max-autotune options forwarded to Inductor
    inductor_options = {"max_autotune": True}

    registry = UniqueSubgraphRegistry(gm)
    equiv_map = registry.build_partition_equivalence_map()

    if not equiv_map:
        # ------------------------------------------------------------------
        # Flat compile path — EmbeddingProjection has no repeated layers.
        # Apply manual passes directly to the full flat graph, then hand off
        # to Inductor. Preserves cross-layer operator fusion.
        # ------------------------------------------------------------------
        logger.info(
            "embedding_projection_opt: no repeated layers detected — flat compile path"
        )

        # OPT-3: pre-transpose eligible weights
        gm = _pass_pretranspose_weights(gm)

        # OPT-2: detection stub (logs if multi-logit pattern found)
        gm = _pass_batch_logit_projection_stub(gm)

        logger.info(
            "embedding_projection_opt: delegating to Inductor (max_autotune=True)"
        )
        return compile_fx(gm, example_inputs, config_patches=inductor_options)

    # -----------------------------------------------------------------------
    # Dedup compile path — for future layered variants of this model.
    # Apply passes to each unique representative partition; share compiled
    # callable with all structural duplicates.
    # -----------------------------------------------------------------------
    logger.info(
        "embedding_projection_opt: %d duplicate partition(s) detected — dedup path",
        len(equiv_map),
    )

    # Capture actual per-partition inputs (shapes may differ from top-level inputs)
    partition_inputs = _capture_partition_inputs(registry.split, example_inputs)

    for rep_name, rep_mod in registry.unique_reps:
        # OPT-3: pre-transpose — manual pass, applied per unique representative
        _pass_pretranspose_weights(rep_mod)

        # OPT-2: detection stub
        _pass_batch_logit_projection_stub(rep_mod)

        # OPT-4: compile unique rep with max-autotune
        rep_inputs = partition_inputs.get(rep_name, example_inputs)
        try:
            compiled = compile_fx(
                rep_mod, rep_inputs, config_patches=inductor_options
            )
        except Exception as e:
            logger.warning(
                "embedding_projection_opt: compile_fx failed for partition '%s' "
                "(%s) — falling back to eager forward",
                rep_name,
                e,
            )
            compiled = rep_mod.forward

        # Patch representative's forward and share with duplicates
        rep_mod.forward = compiled
        for dup_name, dup_mod in registry.duplicates_of(rep_name):
            dup_mod.forward = compiled
            logger.info(
                "embedding_projection_opt: shared compiled callable %s → %s",
                rep_name,
                dup_name,
            )

    logger.info("embedding_projection_opt: backend assembly complete")
    return lambda *args: registry.split(*args)


# ============================================================================
# Workload Interface
# ============================================================================

def get_model_and_input() -> tuple:
    """
    Workload interface for run_workload.py.

    Applies non-graph optimizations in dependency order before returning:

      Step 1 — OPT-1 (BF16 dtype promotion, high confidence):
        Casts model.parameters() to bfloat16 so Dynamo traces a monomorphic
        BF16 graph. All GEMMs route to HMMA Tensor Core path on Ampere.
        Token IDs remain int64 — embedding lookup requires integer indices.
        Check is applied idempotently (skipped if already BF16).

    OPT-4 (max-autotune) is applied at the compile_fx call site inside the
    backend above, not here. The caller (run_workload.py or __main__) must use
    backend='embedding_projection_opt'.

    Returns:
        model : EmbeddingProjection in eval mode on CUDA, parameters in BF16
        token_ids : LongTensor of shape (BATCH_SIZE, SEQ_LEN) on CUDA
    """
    assert torch.cuda.is_available(), "CUDA required"

    model, token_ids = _baseline_get_model_and_input()

    # ------------------------------------------------------------------
    # Step 1: OPT-1 — BF16 dtype promotion
    # Cast before torch.compile() so Dynamo sees a monomorphic BF16 graph.
    # token_ids is intentionally NOT cast — embedding requires int indices.
    # ------------------------------------------------------------------
    if next(model.parameters()).dtype != torch.bfloat16:
        logger.info("get_model_and_input: applying OPT-1 (BF16 dtype promotion)")
        model = model.to(torch.bfloat16)
        logger.info(
            "get_model_and_input: OPT-1 applied — model dtype is now %s",
            next(model.parameters()).dtype,
        )
    else:
        logger.info("get_model_and_input: model already BF16 — OPT-1 skipped")

    # Verify token_ids are integer (embedding constraint)
    assert token_ids.dtype in (torch.long, torch.int32, torch.int64), (
        f"token_ids must be integer tensor, got {token_ids.dtype}"
    )

    logger.info(
        "get_model_and_input: ready — model dtype=%s, input shape=%s, input dtype=%s",
        next(model.parameters()).dtype,
        list(token_ids.shape),
        token_ids.dtype,
    )
    return model, token_ids


# ============================================================================
# CLI entry point
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model, token_ids = get_model_and_input()

    compiled_model = torch.compile(model, backend="embedding_projection_opt")

    with torch.no_grad():
        y = compiled_model(token_ids)

    print(f"Output shape : {y.shape}")   # expect torch.Size([64, 128, 32000])
    print(f"Output dtype : {y.dtype}")   # expect torch.bfloat16
