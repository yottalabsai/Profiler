"""
embedding_projection_optimized.py — EmbeddingProjection with custom torch.compile() backend.

Implements 5 operator-level optimizations via FX graph passes derived from
ncu profiling of the baseline EmbeddingProjection workload:

  1. BF16 dtype cast (OPT-1/2) — forces cuBLAS onto the HMMA tensor-core path,
     eliminating Kernel2 (FP32 FFMA, 212 regs/thread, 16.7% occupancy).
  2. pass_insert_bf16_casts — FX pass: inserts aten.to(torch.bfloat16) on both
     inputs to every aten::mm node so inductor/max-autotune emits Triton HMMA GEMMs.
  3. pass_batch_sequential_mm — FX pass: detects groups of ≥2 aten::mm nodes
     sharing the same weight tensor and fuses them into a single batched mm,
     eliminating per-launch host-side dispatch overhead (OPT-3).
  4. pass_propagate_bf16_pointwise — FX pass: ensures pointwise fused kernels
     (addmm+gelu) inherit BF16 dtype from upstream mm outputs, halving DRAM
     traffic on the bandwidth-bound triton_poi_fused_addmm_gelu_view_1 (OPT-4).
  5. pass_detect_embedding_quant — stub pass: detects the embedding+layer_norm
     fused kernel and logs a recommendation for INT8 weight quantization (OPT-5,
     low-confidence; requires a custom dequant Triton kernel).

Expected aggregate latency improvement vs. baseline (FP32, Kernel2):
  - OPT-1/2: 4–8× reduction on aten::mm nodes (85.5% + 6.2% of wall time)
  - OPT-3:   30–50 µs dispatch elimination
  - OPT-4:   ~50% reduction on addmm+gelu kernel (8.2% of wall time)
  - OPT-5:   ~4 µs (deprioritized)

To profile with optimizations:
    python scripts/run_workload.py \\
        --workload scripts/embedding_projection_optimized.py \\
        --compile-backend transformer_opt

To profile baseline for comparison:
    python scripts/run_workload.py \\
        --workload scripts/embedding_projection.py \\
        --compile-mode inductor
"""
from __future__ import annotations

import logging
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
from torch._dynamo import register_backend
from torch._inductor.compile_fx import compile_fx

# ---------------------------------------------------------------------------
# Baseline workload import — must be on PYTHONPATH
# ---------------------------------------------------------------------------
from embedding_projection import (
    get_model_and_input as _get_baseline_model_and_input,
    EmbeddingProjection,
    DEVICE,
    BATCH_SIZE,
    SEQ_LEN,
    VOCAB_SIZE,
    DIM,
    DIM_FF,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ============================================================================
# FX Graph Passes
# ============================================================================

def pass_insert_bf16_casts(gm: fx.GraphModule) -> fx.GraphModule:
    """
    OPT-1 / OPT-2: Insert aten.to(torch.bfloat16) on both inputs of every
    aten::mm node.

    Pattern:
        %a = ...
        %b = ...
        %mm = aten.mm.default(%a, %b)

    Transformation:
        %a_bf16  = aten._to_copy.default(%a,  dtype=torch.bfloat16)
        %b_bf16  = aten._to_copy.default(%b,  dtype=torch.bfloat16)
        %mm_bf16 = aten.mm.default(%a_bf16, %b_bf16)

    Effect:
        Replaces cuBLAS Kernel2 (FP32 FFMA, 212 regs/thread, 16.7% occupancy,
        tensor_core_active=0%) with a Triton HMMA GEMM (~80 regs/thread,
        ≥50% occupancy, full tensor-core utilisation).
        Applies to op_ids 10,20,30,40,50,60,70,80,90,100 (32000-wide) and
        7,17,27,37,47,57,67,77,87,97 (2048-wide).
    """
    mm_targets = {
        torch.ops.aten.mm.default,
        torch.ops.aten.addmm.default,
    }

    try:
        cast_count = 0
        for node in list(gm.graph.nodes):
            if node.op != "call_function" or node.target not in mm_targets:
                continue

            # addmm(bias, input, weight) — skip bias, cast args[1] and args[2]
            # mm(input, weight)          — cast args[0] and args[1]
            if node.target == torch.ops.aten.addmm.default:
                cast_indices = [1, 2]
            else:
                cast_indices = [0, 1]

            new_args = list(node.args)
            for idx in cast_indices:
                orig = new_args[idx]
                with gm.graph.inserting_before(node):
                    cast_node = gm.graph.call_function(
                        torch.ops.aten._to_copy.default,
                        args=(orig,),
                        kwargs={"dtype": torch.bfloat16},
                    )
                    cast_node.name = f"{orig.name}_bf16" if hasattr(orig, "name") else f"cast_bf16_{idx}"
                new_args[idx] = cast_node
                cast_count += 1

            node.args = tuple(new_args)

        if cast_count:
            logger.info(f"pass_insert_bf16_casts: inserted {cast_count} BF16 cast nodes")
        else:
            logger.warning("pass_insert_bf16_casts: no mm/addmm nodes found — no casts inserted")

        gm.graph.lint()
        gm.recompile()

    except Exception as e:
        logger.warning(f"pass_insert_bf16_casts failed: {e!r} — skipping pass")

    return gm


def pass_batch_sequential_mm(gm: fx.GraphModule) -> fx.GraphModule:
    """
    OPT-3: Batch groups of ≥2 sequential aten::mm nodes that share the same
    weight tensor into a single batched mm, eliminating per-dispatch host
    overhead (~5 µs × 9 = ~45 µs saved per 10-layer forward pass).

    Pattern (N instances):
        %mm_0 = aten.mm.default(%x_0, %W)
        %mm_1 = aten.mm.default(%x_1, %W)
        ...
        %mm_N = aten.mm.default(%x_N, %W)

    Transformation:
        %stacked   = aten.stack.default([%x_0, ..., %x_N], 0)   # [N, M, K]
        %batched   = aten.bmm.default(%stacked, %W_expanded)     # [N, M, out]
        %unbinded  = aten.unbind.int(%batched, 0)                # N tensors

    Note: Only valid when all activations have identical shape and the weight
    is the same tensor (e.g. tied output projection or per-layer inference
    over a shared weight). The pass verifies this by checking node identity
    on the weight argument. If shapes differ or there are fewer than 2
    qualifying mm nodes per weight, the pass is skipped for that group.

    Confidence: medium — pattern-matched against op_ids in OPTIMIZATIONS.json;
    batching is not applicable if weights differ across layers.
    """
    try:
        # Build weight_node → [mm_nodes] map
        weight_to_mms: dict[fx.Node, list[fx.Node]] = {}
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.mm.default:
                if len(node.args) >= 2:
                    weight_node = node.args[1]
                    weight_to_mms.setdefault(weight_node, []).append(node)

        fused_groups = 0
        for weight_node, mm_nodes in weight_to_mms.items():
            if len(mm_nodes) < 2:
                continue

            # Verify all activation shapes are identical (check meta if available)
            act_nodes = [n.args[0] for n in mm_nodes]
            shapes = []
            for a in act_nodes:
                if hasattr(a, "meta") and "val" in a.meta:
                    shapes.append(tuple(a.meta["val"].shape))
                else:
                    shapes.append(None)

            if any(s is None for s in shapes):
                logger.info(
                    f"pass_batch_sequential_mm: cannot verify shapes for weight "
                    f"{weight_node.name} — skipping group of {len(mm_nodes)}"
                )
                continue

            if len(set(shapes)) > 1:
                logger.info(
                    f"pass_batch_sequential_mm: heterogeneous activation shapes for "
                    f"weight {weight_node.name} — skipping"
                )
                continue

            # Insert stack + bmm + unbind after last mm node in group
            last_mm = mm_nodes[-1]
            with gm.graph.inserting_after(last_mm):
                stack_node = gm.graph.call_function(
                    torch.ops.aten.stack.default,
                    args=(act_nodes, 0),
                )
                stack_node.name = f"stacked_acts_{weight_node.name}"

                # Expand weight [K, out] → [N, K, out] for bmm
                n = len(mm_nodes)
                expand_node = gm.graph.call_function(
                    torch.ops.aten.unsqueeze.default,
                    args=(weight_node, 0),
                )
                expand_node.name = f"unsqueeze_w_{weight_node.name}"

                expand2_node = gm.graph.call_function(
                    torch.ops.aten.expand.default,
                    args=(expand_node, [n, -1, -1]),
                )
                expand2_node.name = f"expand_w_{weight_node.name}"

                bmm_node = gm.graph.call_function(
                    torch.ops.aten.bmm.default,
                    args=(stack_node, expand2_node),
                )
                bmm_node.name = f"batched_mm_{weight_node.name}"

                unbind_node = gm.graph.call_function(
                    torch.ops.aten.unbind.int,
                    args=(bmm_node, 0),
                )
                unbind_node.name = f"unbinded_{weight_node.name}"

            # Replace each original mm output with the corresponding unbind slice
            for i, mm_node in enumerate(mm_nodes):
                with gm.graph.inserting_after(unbind_node):
                    get_item = gm.graph.call_function(
                        operator_getitem,
                        args=(unbind_node, i),
                    )
                    get_item.name = f"unbind_item_{i}_{weight_node.name}"
                mm_node.replace_all_uses_with(get_item)
                gm.graph.erase_node(mm_node)

            fused_groups += 1
            logger.info(
                f"pass_batch_sequential_mm: fused {n} mm nodes sharing weight "
                f"'{weight_node.name}' into single bmm"
            )

        if not fused_groups:
            logger.info(
                "pass_batch_sequential_mm: no eligible mm groups found "
                "(requires ≥2 mm nodes with identical weight and activation shape)"
            )

        gm.graph.lint()
        gm.recompile()

    except Exception as e:
        logger.warning(f"pass_batch_sequential_mm failed: {e!r} — skipping pass")

    return gm


def pass_propagate_bf16_pointwise(gm: fx.GraphModule) -> fx.GraphModule:
    """
    OPT-4: Ensure pointwise fused kernels downstream of BF16 mm outputs carry
    BF16 loads/stores rather than promoting back to FP32.

    This pass inserts explicit aten.to(torch.bfloat16) casts on the outputs of
    any aten.add / aten.gelu / aten.relu nodes whose inputs are already BF16
    cast nodes (identified by name suffix '_bf16' inserted by
    pass_insert_bf16_casts). inductor then regenerates the fused
    triton_poi_fused_addmm_gelu_view_1 with BF16 loads/stores, halving DRAM
    traffic from ~67 MB to ~33 MB per 10 invocations.

    This is a best-effort structural pass; inductor's dtype propagation usually
    handles this automatically when the upstream mm is BF16. The pass provides
    an explicit nudge in case inductor widens back to FP32 internally.

    Confidence: medium — free side-effect of OPT-1/2 in most cases.
    """
    pointwise_targets = {
        torch.ops.aten.add.Tensor,
        torch.ops.aten.gelu.default,
        torch.ops.aten.relu.default,
        torch.ops.aten.silu.default,
    }

    try:
        nudge_count = 0
        for node in list(gm.graph.nodes):
            if node.op != "call_function" or node.target not in pointwise_targets:
                continue

            # Check if any direct input came from a BF16 cast node
            has_bf16_input = any(
                hasattr(inp, "name") and "_bf16" in inp.name
                for inp in node.all_input_nodes
            )
            if not has_bf16_input:
                continue

            # Insert a BF16 cast on the output of this node
            with gm.graph.inserting_after(node):
                cast_node = gm.graph.call_function(
                    torch.ops.aten._to_copy.default,
                    args=(node,),
                    kwargs={"dtype": torch.bfloat16},
                )
                cast_node.name = f"{node.name}_bf16_out"

            node.replace_all_uses_with(cast_node)
            # Restore: cast_node's input is still node, not itself
            cast_node.args = (node,)
            nudge_count += 1

        if nudge_count:
            logger.info(f"pass_propagate_bf16_pointwise: nudged {nudge_count} pointwise nodes to BF16 output")
        else:
            logger.info("pass_propagate_bf16_pointwise: no BF16-input pointwise nodes found — likely handled by inductor")

        gm.graph.lint()
        gm.recompile()

    except Exception as e:
        logger.warning(f"pass_propagate_bf16_pointwise failed: {e!r} — skipping pass")

    return gm


def pass_detect_embedding_quant(gm: fx.GraphModule) -> fx.GraphModule:
    """
    OPT-5 (stub): Detect the aten::embedding + layer_norm fused pattern and
    log an INT8 quantization recommendation.

    The profiled kernel (triton_red_fused_embedding_native_layer_norm_0) shows
    lts__t_sector_hit_rate=11.2% — near-zero L2 reuse from random scatter-reads
    into the 32000×512 embedding table. INT8 quantization would halve DRAM
    reads for the weight table.

    This pass does NOT apply the transformation because INT8 embedding lookup
    requires a custom dequantize Triton kernel registered via torch.library.
    It detects the pattern and emits a structured recommendation.

    Confidence: low — 0.2% of total time (87.8 µs); deprioritize unless
    memory-capacity-constrained.
    """
    try:
        embed_nodes = [
            n for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target == torch.ops.aten.embedding.default
        ]
        ln_nodes = [
            n for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target in {
                torch.ops.aten.native_layer_norm.default,
                torch.ops.aten.layer_norm.default,
            }
        ]

        if embed_nodes:
            logger.warning(
                "pass_detect_embedding_quant [STUB]: Detected %d aten::embedding node(s). "
                "INT8 embedding weight quantization would reduce DRAM reads by ~50%% "
                "(lts__t_sector_hit_rate=11.2%%). "
                "TODO: implement custom torch.ops dequant kernel via torch.library and "
                "re-fuse embedding+dequant+layer_norm through inductor fx_passes.fuse. "
                "Deprioritize unless memory-capacity-constrained (0.2%% of wall time).",
                len(embed_nodes),
            )
        else:
            logger.info("pass_detect_embedding_quant: no aten::embedding nodes found in graph")

        if embed_nodes and ln_nodes:
            # Check for embedding → layer_norm data flow
            embed_outputs = {n for n in embed_nodes}
            for ln in ln_nodes:
                for inp in ln.all_input_nodes:
                    if inp in embed_outputs:
                        logger.warning(
                            "pass_detect_embedding_quant [STUB]: Found embedding→layer_norm "
                            "chain — this is the triton_red_fused_embedding_native_layer_norm_0 "
                            "kernel. INT8 quantization applicable here."
                        )
                        break

    except Exception as e:
        logger.warning(f"pass_detect_embedding_quant failed: {e!r} — skipping pass")

    return gm


# operator.getitem used for unbind slicing
try:
    from operator import getitem as operator_getitem
except ImportError:
    operator_getitem = lambda obj, idx: obj[idx]  # noqa: E731


# ============================================================================
# Backend Registration
# ============================================================================

@register_backend
def transformer_opt(gm: fx.GraphModule, example_inputs) -> Callable:
    """
    Custom torch.compile() backend for EmbeddingProjection.

    Applies the following passes in order before delegating to inductor
    max-autotune:

      1. pass_insert_bf16_casts         — HIGH confidence
         Insert BF16 casts on both mm inputs to activate HMMA tensor-core path.
         Must run FIRST so downstream passes see BF16-typed nodes.

      2. pass_propagate_bf16_pointwise  — MEDIUM confidence
         Propagate BF16 through pointwise fused kernels (addmm+gelu).
         Must run AFTER pass_insert_bf16_casts so '_bf16' name hints are present.

      3. pass_batch_sequential_mm       — MEDIUM confidence
         Fuse groups of sequential mm dispatches sharing the same weight.
         Runs after dtype passes so batched mm also benefits from BF16.

      4. pass_detect_embedding_quant    — LOW confidence (stub)
         Detection-only; logs INT8 quantization recommendation.
         Order is irrelevant, placed last to avoid cluttering logs.
    """
    logger.info("transformer_opt backend: starting FX graph passes")

    gm = pass_insert_bf16_casts(gm)
    gm = pass_propagate_bf16_pointwise(gm)
    gm = pass_batch_sequential_mm(gm)
    gm = pass_detect_embedding_quant(gm)

    logger.info("transformer_opt backend: all passes complete — delegating to inductor max-autotune")

    # Delegate to inductor with max-autotune to replace Kernel2 with profiled
    # Triton HMMA GEMMs and cache autotuned tile configs.
    import torch._inductor.config as inductor_cfg
    # Allow cached autotuned configs (OPT-2: tile config annotation)
    inductor_cfg.force_disable_caches = False

    return compile_fx(gm, example_inputs, config_patches={"max_autotune": True})


# ============================================================================
# Workload Interface
# ============================================================================

def get_model_and_input() -> tuple:
    """
    Workload interface for run_workload.py.

    Starts from the baseline get_model_and_input() and applies complementary
    optimizations that operate outside the FX graph:

      - BF16 dtype cast (OPT-1/2): applied here so that (a) the embedding
        lookup and layer_norm also run in BF16, and (b) inductor can infer
        correct input dtypes when lowering the BF16-cast FX graph. The FX
        pass pass_insert_bf16_casts provides an additional in-graph cast as a
        belt-and-suspenders measure for any operator that receives FP32 inputs
        after tracing.

    Note: token_ids remain int64 (embedding indices are always integer).
    """
    assert torch.cuda.is_available(), "CUDA required"

    model, token_ids = _get_baseline_model_and_input()

    # Apply BF16 only if baseline has not already cast
    if next(model.parameters()).dtype != torch.bfloat16:
        logger.info("get_model_and_input: casting model to BF16 (OPT-1/2)")
        model = model.to(torch.bfloat16)
    else:
        logger.info("get_model_and_input: model already BF16 — skipping cast")

    # token_ids must remain integer; no dtype change needed
    assert token_ids.dtype in (torch.int32, torch.int64), (
        f"Unexpected token_ids dtype: {token_ids.dtype}"
    )

    return model, token_ids


if __name__ == "__main__":
    import torch

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    m, token_ids = get_model_and_input()
    compiled = torch.compile(m, backend="transformer_opt")

    with torch.no_grad():
        # Warmup
        _ = compiled(token_ids)
        torch.cuda.synchronize()
        # Measure
        y = compiled(token_ids)
        torch.cuda.synchronize()

    print(f"✓ Output shape:  {y.shape}")
    print(f"✓ Output dtype:  {y.dtype}")
    print(f"✓ Model dtype:   {next(m.parameters()).dtype}")
    print(f"✓ Backend registered: {'transformer_opt' in torch._dynamo.list_backends()}")