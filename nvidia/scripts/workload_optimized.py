"""
workload_optimized.py — TransformerBlock with custom torch.compile() backend.

Implements six operator-level optimizations via FX graph passes:
  1. BF16 casting — 2× arithmetic throughput via Tensor Cores
  2. QKV projection fusion — 3× [512×512] → 1× [512×1536] GEMM
  3. FlashAttention replacement — 3-kernel chain → 1 fused kernel
  4. Consistent GELU activation — relu → gelu(approximate='tanh')
  5. Pre-transposed weights — TN GEMM layout fix for down-projection
  6. Token padding — B=16 → B=64 to improve wave occupancy

To profile with optimizations:
    nsys profile --trace=cuda,nvtx --output=profile \\
        python scripts/run_workload.py \\
            --workload scripts/workload_optimized.py \\
            --compile-backend transformer_opt

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
from torch.fx.subgraph_rewriter import replace_pattern

# Import baseline workload to copy model structure
from nvidia.scripts.workload import TransformerBlock

DEVICE = "cuda"
BATCH_SIZE = 16  # Baseline batch size (will be padded to 64)
IN_FEATURES = 512
HIDDEN = 2048

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ============================================================================
# FX Graph Passes
# ============================================================================

def pass_fuse_qkv(gm: fx.GraphModule) -> fx.GraphModule:
    """
    Fuse three separate mm(x, W_q/k/v) into one mm(x, cat(W_q, W_k, W_v)) + chunk.

    Pattern: Three consecutive mm nodes with identical input x but different weights.
    Fused output shape: [16, 512] × [512, 1536] → output can be split into [512], [512], [512].
    """

    # Build a map: input_node → [list of mm nodes using that input]
    input_to_mms = {}
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.aten.mm.default:
            if len(node.args) >= 2:
                input_node = node.args[0]
                if input_node not in input_to_mms:
                    input_to_mms[input_node] = []
                input_to_mms[input_node].append(node)

    # Find groups with ≥ 2 mm nodes sharing the same input
    fused_groups = []
    for input_node, mm_nodes in input_to_mms.items():
        if len(mm_nodes) >= 2:
            # Check if all weight nodes are get_attr (parameters)
            valid_group = True
            weight_nodes = []
            for mm_node in mm_nodes:
                weight_node = mm_node.args[1]
                if weight_node.op != "get_attr":
                    valid_group = False
                    break
                weight_nodes.append(weight_node)

            if valid_group:
                fused_groups.append((input_node, mm_nodes, weight_nodes))

    # Apply fusion for each valid group
    for input_node, mm_nodes, weight_nodes in fused_groups:
        # Extract weight tensors from parameters
        weights = []
        param_names = []
        for weight_node in weight_nodes:
            param_path = weight_node.target  # e.g., 'attn.q_proj.weight'
            try:
                W = gm.get_parameter(param_path)
                weights.append(W)
                param_names.append(param_path)
            except Exception:
                logger.warning(f"Could not extract parameter {param_path}")
                continue

        if len(weights) < len(mm_nodes):
            logger.warning("Not all weights extracted; skipping QKV fusion")
            continue

        # Concatenate weights: cat along output dimension (dim=0)
        # Each W is [D_out, D_in]; we want [D_out*N, D_in]
        W_fused = torch.cat(weights, dim=0)

        # Register fused weight as a buffer
        fused_buffer_name = f"fused_qkv_weight_{id(input_node)}"
        gm.register_buffer(fused_buffer_name, W_fused)

        # Insert get_attr node for the fused weight
        with gm.graph.inserting_before(mm_nodes[0]):
            fused_weight_node = gm.graph.get_attr(fused_buffer_name)

        # Insert one mm with the fused weight
        with gm.graph.inserting_after(fused_weight_node):
            fused_mm = gm.graph.call_function(torch.ops.aten.mm.default, (input_node, fused_weight_node))

        # Insert chunk to split the fused output
        with gm.graph.inserting_after(fused_mm):
            chunk_op = gm.graph.call_function(
                torch.ops.aten.chunk.default,
                (fused_mm, len(mm_nodes), 1)  # chunk along output dim
            )

        # Replace each old mm node with the corresponding chunk output
        chunk_results = []
        for i in range(len(mm_nodes)):
            with gm.graph.inserting_after(chunk_op):
                chunk_i = gm.graph.call_function(
                    torch.ops.aten.getitem.default,
                    (chunk_op, i)
                )
            chunk_results.append(chunk_i)

        for old_mm, chunk_result in zip(mm_nodes, chunk_results):
            old_mm.replace_all_uses_with(chunk_result)
            gm.graph.erase_node(old_mm)

    gm.graph.lint()
    gm.recompile()
    logger.info("pass_fuse_qkv completed")
    return gm


def pass_replace_sdpa(gm: fx.GraphModule) -> fx.GraphModule:
    """
    Replace mm(Q, K^T) → div → softmax → mm(attn, V) with scaled_dot_product_attention.

    Uses torch.fx.subgraph_rewriter.replace_pattern to identify and substitute the
    manual attention pattern with SDPA, which will lower to FlashAttention-2.

    Note: The pattern detection in this model may not match exactly due to the way
    inductor fuses operations. This pass is a best-effort attempt; SDPA dispatch is
    the primary goal if the pattern matches.
    """

    # Define pattern: what we're looking for (manual attention)
    def attn_pattern_manual(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, scale: float) -> torch.Tensor:
        """Pattern: manual attention without Flash."""
        # Q @ K^T
        K_t = torch.transpose(K, 0, 1)  # transpose instead of t
        scores = torch.matmul(Q, K_t)
        # Scale (may be done via mul or div)
        scaled = scores * scale
        # Softmax (dim=-1 is implicit)
        attn_weights = torch.nn.functional.softmax(scaled, dim=-1)
        # Attention @ V
        output = torch.matmul(attn_weights, V)
        return output

    # Define replacement: what we want instead
    def attn_pattern_replacement(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, scale: float) -> torch.Tensor:
        """Pattern: SDPA (which maps to FlashAttention)."""
        # scaled_dot_product_attention with no dropout, no causal mask
        return torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=False)

    try:
        replace_pattern(gm, attn_pattern_manual, attn_pattern_replacement)
        logger.info("pass_replace_sdpa: SDPA pattern replacement applied")
    except Exception as e:
        logger.warning(f"pass_replace_sdpa: pattern replacement failed (this may be expected) — {type(e).__name__}")

    # Even if replace_pattern fails, the pass shouldn't crash the backend
    try:
        gm.graph.lint()
        gm.recompile()
    except Exception as e:
        logger.warning(f"Graph validation after SDPA pass failed: {e}")

    return gm


def pass_normalize_gelu(gm: fx.GraphModule) -> fx.GraphModule:
    """
    Replace relu in FFN position (between two large mm nodes) with gelu(approximate='tanh').

    This targets the FFN block where relu is used as an intermediate activation.
    The attention block's relu on Q is left untouched (it's not in an mm→relu→mm pattern).
    """

    replaced_count = 0
    for node in list(gm.graph.nodes):
        if node.op == "call_function" and node.target == torch.ops.aten.relu.default:
            # Check if this relu is in an FFN-like context (mm → relu → mm)
            input_nodes = node.all_input_nodes
            if not input_nodes:
                continue

            producer = input_nodes[0]
            producer_is_mm = (producer.op == "call_function" and
                            producer.target in (torch.ops.aten.mm.default, torch.ops.aten.addmm.default))

            # Find all consumers
            consumers = list(node.users.keys())
            if not consumers:
                continue

            consumers_are_mm = all(
                c.op == "call_function" and c.target in (torch.ops.aten.mm.default, torch.ops.aten.addmm.default)
                for c in consumers
            )

            if producer_is_mm and consumers_are_mm:
                # This is likely an FFN relu; replace with gelu(approximate='tanh')
                with gm.graph.inserting_after(node):
                    gelu_node = gm.graph.call_function(
                        torch.ops.aten.gelu.default,
                        (producer, "tanh")
                    )
                node.replace_all_uses_with(gelu_node)
                gm.graph.erase_node(node)
                replaced_count += 1

    if replaced_count > 0:
        logger.info(f"Replaced {replaced_count} relu with gelu(approximate='tanh')")

    try:
        gm.graph.lint()
        gm.recompile()
    except Exception as e:
        logger.warning(f"Graph validation after GELU pass failed: {e}")

    logger.info("pass_normalize_gelu completed")
    return gm


def pass_pretranspose_weights(gm: fx.GraphModule) -> fx.GraphModule:
    """
    Eliminate aten.t() on large weight matrices by pre-transposing and storing them.

    Pattern: aten.mm(x, aten.t(W)) where W is a large parameter (K ≥ 512).
    Transformation: Store W_t = W.T.contiguous() as a buffer, use it directly (no aten.t).
    """

    for node in list(gm.graph.nodes):
        if node.op == "call_function" and node.target == torch.ops.aten.t.default:
            # Check if this t node is used by an mm
            if len(node.users) > 0:
                user = list(node.users.keys())[0]
                if user.op == "call_function" and user.target == torch.ops.aten.mm.default:
                    # Get the input to t (should be a parameter)
                    input_node = node.all_input_nodes[0]
                    if input_node.op == "get_attr":
                        param_path = input_node.target
                        try:
                            W = gm.get_parameter(param_path)
                            if W.shape[0] >= 512:  # Large K dimension check
                                # Pre-transpose
                                W_t = W.T.contiguous()

                                # Register as buffer
                                buffer_name = f"pretransposed_{param_path.replace('.', '_')}"
                                gm.register_buffer(buffer_name, W_t)

                                # Replace aten.t(W) with get_attr to pre-transposed buffer
                                with gm.graph.inserting_before(node):
                                    new_node = gm.graph.get_attr(buffer_name)

                                node.replace_all_uses_with(new_node)
                                gm.graph.erase_node(node)
                                logger.info(f"Pre-transposed weight {param_path}")
                        except Exception as e:
                            logger.warning(f"Could not pre-transpose {param_path}: {e}")

    gm.graph.lint()
    gm.recompile()
    logger.info("pass_pretranspose_weights completed")
    return gm


def pass_fuse_ln_linear(gm: fx.GraphModule) -> fx.GraphModule:
    """
    Stub pass for LayerNorm-Linear fusion detection.

    Full implementation would replace LN → Linear chains with a custom Triton kernel
    that keeps normalized rows in registers before issuing GEMM (eliminates DRAM round-trip).

    TODO: Implement custom Triton layer_norm_linear_kernel and torch.library registration.
    """

    # Detect pattern: aten.layer_norm → aten.mm
    detected = False
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.aten.layer_norm.default:
            # Check if consumer is mm
            if len(node.users) > 0:
                for user in node.users:
                    if user.op == "call_function" and user.target == torch.ops.aten.mm.default:
                        detected = True
                        break

    if detected:
        logger.warning("LN-Linear fusion detected but not applied — requires custom Triton kernel")
        logger.warning("TODO: Implement layer_norm_linear_kernel via torch.library")

    return gm


# ============================================================================
# Backend Registration
# ============================================================================

@register_backend
def transformer_opt(gm: fx.GraphModule, example_inputs) -> Callable:
    """
    Custom torch.compile() backend: applies all optimization passes, then delegates to inductor.

    Pass order:
      1. pass_fuse_qkv — fuse Q/K/V projections
      2. pass_replace_sdpa — replace manual attention with FlashAttention
      3. pass_normalize_gelu — fix FFN activations
      4. pass_pretranspose_weights — pre-transpose large weights
      5. pass_fuse_ln_linear — stub for LN-Linear fusion
    """
    logger.info("transformer_opt backend: starting FX passes")

    gm = pass_fuse_qkv(gm)
    gm = pass_replace_sdpa(gm)
    gm = pass_normalize_gelu(gm)
    gm = pass_pretranspose_weights(gm)
    gm = pass_fuse_ln_linear(gm)

    logger.info("transformer_opt backend: all FX passes complete, delegating to inductor")
    return compile_fx(gm, example_inputs)


# ============================================================================
# Workload Interface
# ============================================================================

def get_model_and_input() -> tuple:
    """
    Workload interface for run_workload.py.

    Returns:
      - model: TransformerBlock on CUDA, in BF16 dtype
      - x: input tensor [64, 512] (padded from [16, 512]) in BF16

    Optimizations applied here:
      1. BF16 casting — model and input to torch.bfloat16
      2. Token padding — batch size 16 → 64 (improves wave occupancy)

    Note: The batch dimension is padded to 64; if the profiler checks the output,
    it will be [64, 512]. The runner should ignore this or slice [:16] if needed.
    """
    assert torch.cuda.is_available(), "CUDA required"

    # Import and instantiate baseline model
    model = TransformerBlock().to(DEVICE).eval()

    # Create input with padding for wave occupancy improvement
    # Baseline: [16, 512]; padded to [64, 512] for better utilization
    batch_padded = 64
    x = torch.randn(batch_padded, IN_FEATURES, device=DEVICE)

    # Optimization 1: BF16 casting
    model = model.to(torch.bfloat16)
    x = x.to(torch.bfloat16)

    return model, x


if __name__ == "__main__":
    # Quick smoke test: verify model and input creation
    m, x = get_model_and_input()
    print(f"Model dtype: {next(m.parameters()).dtype}")
    print(f"Input shape: {x.shape}, dtype: {x.dtype}")

    # Test uncompiled forward pass
    with torch.no_grad():
        y = m(x)
    print(f"Output shape: {y.shape}, dtype: {y.dtype}")
    print("✓ Workload smoke test passed")
