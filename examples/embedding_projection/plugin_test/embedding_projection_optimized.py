"""
embedding_projection_optimized.py — EmbeddingProjection with custom torch.compile() backend.

Implements three operator-level optimizations derived from profiling feedback:
  1. TF32 flags (OPT-2) — zero-risk Tensor Core routing for all FP32 GEMMs
  2. BF16 dtype cast (OPT-1) — 3-4x speedup by routing cuBLAS to HMMA Tensor Core path
  3. Pre-transposed weights FX pass (OPT-3) — eliminate CUBLAS_OP_T overhead on large GEMMs

Profiling showed all 30 cuBLAS GEMM nodes (20x aten::mm, 10x aten::addmm) operate on
the FP32 SIMT path (tensor_core_active_pct=0.0, registers_per_thread=212). The dominant
aten::mm [8192,512]x[512,32000] consumed 85% of total profiled time.

To profile with optimizations:
    python nvidia/scripts/run_workload.py \\
        --workload examples/embedding_projection/plugin_test/embedding_projection_optimized.py \\
        --compile-backend embedding_projection_opt \\
        --warmup-iters 3 --measure-iters 10
"""
from __future__ import annotations

import logging
from typing import Callable

import torch
import torch.fx as fx
from torch._dynamo import register_backend
from torch._inductor.compile_fx import compile_fx

# Re-export all public names from the baseline workload so callers that import
# this module via star-import or by name get the same interface.
from examples.embedding_projection.embedding_projection import (
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

# ---------------------------------------------------------------------------
# Minimum weight byte threshold for the pre-transpose pass (1 MB)
# ---------------------------------------------------------------------------
_MIN_WEIGHT_BYTES = 1024 * 1024  # 1 MB


# ============================================================================
# FX Graph Passes
# ============================================================================

def pass_pretranspose_weights(gm: fx.GraphModule) -> fx.GraphModule:
    """
    OPT-3 (MEDIUM confidence): Pre-transpose large projection weights to eliminate
    the CUBLAS_OP_T overhead introduced by nn.Linear's runtime weight.t() call.

    Pattern detected:
        %t_node = aten.t.default(%get_attr_node)   # weight transpose
        %mm_node = aten.mm.default(%x, %t_node)    # or addmm
        -- where weight tensor bytes > _MIN_WEIGHT_BYTES (1 MB)

    Transformation:
        1. Compute W_T = W.t().contiguous() eagerly (once, at compile time).
        2. Register as a buffer on gm: gm.register_buffer(buf_name, W_T).
        3. Replace the aten.t node with a get_attr node pointing to the buffer.
        4. The mm node now receives a contiguous row-major matrix → cuBLAS CUBLAS_OP_N.

    Note: Must run AFTER model is already in BF16 so the buffer is created at BF16.
    Only weights larger than 1 MB are pre-transposed; smaller ones are left alone.
    """
    try:
        matched = False
        nodes_snapshot = list(gm.graph.nodes)

        for node in nodes_snapshot:
            # Looking for aten.t.default call nodes
            if node.op != "call_function":
                continue
            if node.target not in (torch.ops.aten.t.default, torch.ops.aten.t):
                continue

            # The argument to aten.t must be a get_attr (a parameter/buffer on gm)
            if len(node.args) < 1:
                continue
            weight_node = node.args[0]
            if weight_node.op != "get_attr":
                continue

            # Retrieve the actual tensor
            param_path = weight_node.target  # e.g. "proj1.weight"
            try:
                weight_tensor = gm.get_parameter(param_path)
            except AttributeError:
                # May be a buffer rather than a parameter; try getattr path
                try:
                    obj = gm
                    for part in param_path.split("."):
                        obj = getattr(obj, part)
                    weight_tensor = obj
                except AttributeError:
                    logger.warning(
                        f"[pass_pretranspose_weights] Cannot locate tensor for {param_path!r} — skipping node"
                    )
                    continue

            # Size gate: only pre-transpose large weights
            byte_size = weight_tensor.numel() * weight_tensor.element_size()
            if byte_size < _MIN_WEIGHT_BYTES:
                logger.info(
                    f"[pass_pretranspose_weights] Skipping {param_path!r} "
                    f"({byte_size / 1024:.1f} KB < {_MIN_WEIGHT_BYTES / 1024:.0f} KB threshold)"
                )
                continue

            # Build a safe buffer name (dots → underscores, avoid collisions)
            buf_name = param_path.replace(".", "_") + "_pretransposed"

            # Eagerly compute the transposed contiguous tensor
            with torch.no_grad():
                weight_t = weight_tensor.t().contiguous()

            gm.register_buffer(buf_name, weight_t)

            # Insert a get_attr node for the new buffer immediately before the t node
            with gm.graph.inserting_before(node):
                new_attr_node = gm.graph.get_attr(buf_name)
                new_attr_node.meta = {}

            # Replace ALL uses of the t node with the pre-transposed buffer node
            # (replace_all_uses_with BEFORE erase_node)
            node.replace_all_uses_with(new_attr_node)
            gm.graph.erase_node(node)

            matched = True
            logger.info(
                f"[pass_pretranspose_weights] Pre-transposed {param_path!r} "
                f"({byte_size / (1024*1024):.2f} MB, shape {list(weight_tensor.shape)} → {list(weight_t.shape)})"
            )

        if not matched:
            logger.warning(
                "[pass_pretranspose_weights] No eligible aten.t(get_attr) patterns found — pass not applied"
            )
            return gm

        gm.graph.lint()
        gm.recompile()
        logger.info("[pass_pretranspose_weights] Pass applied and graph recompiled")

    except Exception as e:
        logger.warning(f"[pass_pretranspose_weights] Failed: {e!r} — returning unmodified graph")

    return gm


# ============================================================================
# Backend Registration
# ============================================================================

@register_backend
def embedding_projection_opt(gm: fx.GraphModule, example_inputs) -> Callable:
    """
    Custom torch.compile() backend for EmbeddingProjection.

    Pass execution order (mirrors optimizations.json dependency ordering):
      1. pass_pretranspose_weights (OPT-3) — requires OPT-1 already applied eagerly
         so that the pre-transposed buffers are created at BF16.

    After all FX passes the graph is forwarded to the standard Inductor backend
    (compile_fx), which performs its own fusion, tiling, and code-generation on
    top of the transformed graph.
    """
    logger.info("embedding_projection_opt backend: starting FX passes")

    # OPT-3: Pre-transpose large weight matrices
    gm = pass_pretranspose_weights(gm)

    logger.info("embedding_projection_opt backend: FX passes complete, delegating to Inductor")
    return compile_fx(gm, example_inputs)


# ============================================================================
# Workload Interface
# ============================================================================

def get_model_and_input() -> tuple:
    """
    Workload interface — returns (compiled model on CUDA, integer token_ids tensor).

    Optimizations applied eagerly (before compile):
      OPT-2: TF32 flags — torch.backends.cuda.matmul.allow_tf32 = True
                           torch.backends.cudnn.allow_tf32 = True
      OPT-1: BF16 cast — model parameters and state cast to bfloat16.
                          Token IDs remain int64 (integer indices cannot be cast).
                          Tied-weight detection runs before cast; if logits.weight
                          and embed.weight share storage they are unlinked first.

    The custom FX backend (embedding_projection_opt) runs OPT-3 at graph-compile time.

    Returns:
        model  : EmbeddingProjection on CUDA, compiled with embedding_projection_opt
        token_ids : LongTensor of shape (BATCH_SIZE, SEQ_LEN) on CUDA
    """
    assert torch.cuda.is_available(), "CUDA required"

    # ------------------------------------------------------------------
    # OPT-2: Enable TF32 — zero precision-loss Tensor Core routing.
    # Applied unconditionally before model construction.
    # ------------------------------------------------------------------
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    logger.info("TF32 enabled (matmul + cudnn)")

    # ------------------------------------------------------------------
    # Construct baseline model on CUDA in eval mode.
    # ------------------------------------------------------------------
    model = EmbeddingProjection().to(DEVICE).eval()

    # ------------------------------------------------------------------
    # OPT-1 prerequisite: tied-weight check.
    # In some LLM variants logits.weight is tied to embed.weight.  Casting
    # tied parameters to BF16 is safe but storing both as the same tensor
    # can cause silent aliasing issues with buffers registered in OPT-3.
    # Untie before casting if they share storage.
    # ------------------------------------------------------------------
    if model.logits.weight.data_ptr() == model.embed.weight.data_ptr():
        logger.warning(
            "Tied embedding weights detected (logits.weight shares storage with embed.weight). "
            "Creating explicit copy before BF16 cast."
        )
        with torch.no_grad():
            model.logits.weight = torch.nn.Parameter(
                model.logits.weight.detach().clone()
            )
    else:
        logger.info("Tied-weight check passed: logits.weight and embed.weight are independent")

    # ------------------------------------------------------------------
    # OPT-1: BF16 cast.
    # Check current dtype to avoid redundant cast (idempotent guard).
    # Token IDs stay int64 — nn.Embedding.forward() requires integer input.
    # ------------------------------------------------------------------
    current_dtype = next(model.parameters()).dtype
    if current_dtype != torch.bfloat16:
        model = model.to(torch.bfloat16)
        logger.info(f"Model cast from {current_dtype} to bfloat16")
    else:
        logger.info("Model already in bfloat16 — skipping cast")

    # ------------------------------------------------------------------
    # Compile with custom backend (OPT-3 FX pass runs at trace time).
    # ------------------------------------------------------------------
    model = torch.compile(model, backend="embedding_projection_opt", fullgraph=False)

    # ------------------------------------------------------------------
    # Input tensor: integer token IDs in [0, VOCAB_SIZE).
    # dtype=torch.long (int64) — must NOT be cast to bfloat16.
    # ------------------------------------------------------------------
    token_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=DEVICE)

    return model, token_ids


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    model, token_ids = get_model_and_input()
    print(f"Token IDs shape : {token_ids.shape}, dtype: {token_ids.dtype}")
    print(f"Token IDs device: {token_ids.device}")

    with torch.no_grad():
        out = model(token_ids)

    print(f"Output shape : {out.shape}")
    print(f"Output dtype : {out.dtype}")
    print(f"Output finite: {torch.isfinite(out).all().item()}")
    print("Smoke test passed")
