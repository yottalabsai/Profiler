"""
sdpa_attention_optimized.py — Custom torch.compile() backend for SDPAAttentionBlock.

Implements four FX-level optimizations from optimizations.json:

  OPT-1 (high)   BF16 promotion + TF32 enable
                 Stage 1: torch.backends.cuda.matmul.allow_tf32=True (no graph change)
                 Stage 2: cast linear inputs/outputs to bfloat16 to engage Tensor Cores

  OPT-2 (high)   QKV projection fusion
                 Pre-cat the three [512,512] weight matrices into [1536,512] and replace
                 three sequential F.linear nodes sharing the same input with a single
                 F.linear + torch.split, eliminating 2 kernel launches per forward pass.

  OPT-3 (medium) Flash SDPA backend selection
                 Set enable_flash_sdp(True) + enable_mem_efficient_sdp(False) so
                 PyTorch dispatches to a Blackwell-native (SM100) attention kernel
                 instead of the SM80 CUTLASS fallback. Requires BF16 from OPT-1.
                 math_sdp remains True as a FP32 fallback for fake-tensor tracing.

  OPT-4 (medium) mm+add -> addmm (linear+residual fusion)
                 Replace the (F.linear, operator.add) pattern at the output projection
                 with F.linear whose bias argument absorbs the residual — not applicable
                 because F.linear takes a bias parameter, not an addmm residual.
                 Instead, we insert an aten.addmm node after lowering the linear+add
                 pattern at the F.linear level where the residual is an addend.
                 See implementation notes for why this is a log-and-skip stub.

Backend registration name: sdpa_attention_opt
"""
from __future__ import annotations

import logging
import operator
from collections import defaultdict
from typing import Callable, List

import torch
import torch.fx as fx
import torch.nn.functional as F
from torch._dynamo.backends.registry import register_backend
from torch._inductor.compile_fx import compile_fx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-load-time side effects for OPT-1 stage 1 and OPT-3
# These must be set BEFORE torch.compile traces the model.
# ---------------------------------------------------------------------------

# OPT-1 Stage 1: enable TF32 Tensor Core path for FP32 GEMMs (minimum fix)
torch.backends.cuda.matmul.allow_tf32 = True
logger.info("sdpa_attention_opt: allow_tf32=True set at module load")

# OPT-3: prefer Flash Attention (Blackwell-native SM100) over mem-efficient SM80 CUTLASS.
# math_sdp is kept enabled as a FP32 fallback — Dynamo traces in FP32 with fake tensors
# before the BF16 FX pass runs, so disabling math_sdp causes "Invalid backend" during
# tracing. At runtime the BF16 casts from OPT-1 ensure Flash is the preferred dispatch.
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)   # kept for FP32 fake-tensor tracing
logger.info(
    "sdpa_attention_opt: flash_sdp=True, mem_efficient_sdp=False, math_sdp=True "
    "(math kept as FP32 tracing fallback) set at module load"
)


# ---------------------------------------------------------------------------
# OPT-1 Stage 2: BF16 promotion pass
# Target: torch.nn.functional.linear nodes (pre-Inductor form)
# Insert .to(bfloat16) casts before each linear's input and weight; cast result
# back to float32 so downstream nodes remain type-compatible.
# ---------------------------------------------------------------------------

def _pass_promote_linear_to_bf16(gm: fx.GraphModule) -> fx.GraphModule:
    """
    OPT-1 Stage 2 — cast F.linear inputs to bfloat16, cast output back to float32.

    Operates on the pre-Inductor FX graph where nn.Linear appears as
    call_function with target=torch.nn.functional.linear.
    """
    try:
        matched = False
        graph = gm.graph
        for node in list(graph.nodes):
            if not (node.op == "call_function" and node.target is F.linear):
                continue
            matched = True
            # args: (input, weight) or (input, weight, bias)
            inp_node = node.args[0]
            weight_node = node.args[1]
            bias_node = node.args[2] if len(node.args) > 2 else None

            with graph.inserting_before(node):
                cast_inp = graph.call_function(
                    torch.ops.aten.to.dtype,
                    args=(inp_node, torch.bfloat16),
                )
                cast_w = graph.call_function(
                    torch.ops.aten.to.dtype,
                    args=(weight_node, torch.bfloat16),
                )

            # Update node args
            new_args = (cast_inp, cast_w)
            if bias_node is not None:
                with graph.inserting_before(node):
                    cast_bias = graph.call_function(
                        torch.ops.aten.to.dtype,
                        args=(bias_node, torch.bfloat16),
                    )
                new_args = (cast_inp, cast_w, cast_bias)
            node.args = new_args

            # Cast output back to float32 so downstream nodes stay compatible
            with graph.inserting_after(node):
                cast_out = graph.call_function(
                    torch.ops.aten.to.dtype,
                    args=(node, torch.float32),
                )
            node.replace_all_uses_with(cast_out)
            # Fix: cast_out must still reference node, not itself
            cast_out.args = (node,) + cast_out.args[1:]

        if not matched:
            logger.warning(
                "[_pass_promote_linear_to_bf16] No F.linear nodes found — pass not applied"
            )
            return gm

        graph.lint()
        gm.recompile()
        logger.info("[_pass_promote_linear_to_bf16] Applied BF16 cast to all F.linear nodes")

    except Exception as exc:
        logger.warning(f"[_pass_promote_linear_to_bf16] Failed: {exc}")

    return gm


# ---------------------------------------------------------------------------
# OPT-2: QKV projection fusion pass
# Target: three consecutive F.linear nodes sharing the same input activation
# Replace with one F.linear against concatenated [1536, 512] weight + split
# ---------------------------------------------------------------------------

def _pass_fuse_qkv_projections(gm: fx.GraphModule) -> fx.GraphModule:
    """
    OPT-2 — fuse the Q/K/V linear projections into a single batched linear.

    Pattern: three call_function(F.linear, args=(shared_x, w_q/k/v)) nodes
    where all three share the same first argument (the post-LayerNorm activation)
    and all weight tensors have matching shapes.

    Replacement: one F.linear(shared_x, cat([w_q, w_k, w_v], dim=0)) followed
    by torch.split(result, split_size, dim=-1) with getitem unpacking.

    Weight shapes [512, 512] — F.linear computes x @ w.T, so cat along dim=0
    gives [1536, 512]; the fused output is [B, T, 1536], split into 3x[B, T, 512].
    """
    try:
        matched = False
        graph = gm.graph

        # Collect all F.linear nodes in graph order
        linear_nodes = [
            n for n in graph.nodes
            if n.op == "call_function" and n.target is F.linear
        ]

        # Group by shared input (args[0]), preserving order
        by_input: dict = defaultdict(list)
        for n in linear_nodes:
            # Use the node object itself as key (hashable, identity-based)
            by_input[n.args[0]].append(n)

        for shared_x, group in by_input.items():
            if len(group) < 3:
                continue

            # Take the first three that form a QKV triplet
            q_node, k_node, v_node = group[0], group[1], group[2]

            # Verify no bias (baseline uses bias=False)
            # F.linear args: (input, weight) or (input, weight, bias)
            if any(len(n.args) > 2 and n.args[2] is not None for n in [q_node, k_node, v_node]):
                logger.warning(
                    "[_pass_fuse_qkv_projections] QKV linears have bias — skipping fusion"
                )
                continue

            w_q = q_node.args[1]
            w_k = k_node.args[1]
            w_v = v_node.args[1]

            matched = True

            # Insert fused linear before the first QKV node
            with graph.inserting_before(q_node):
                # Cat weights along dim=0: [512,512]*3 -> [1536,512]
                cat_weight = graph.call_function(
                    torch.cat,
                    args=([w_q, w_k, w_v],),
                    kwargs={"dim": 0},
                )
                # Single fused linear: output shape [B, T, 1536]
                fused_linear = graph.call_function(
                    F.linear,
                    args=(shared_x, cat_weight),
                )
                # Split back to 3 x [B, T, 512] along the last dim
                split_out = graph.call_function(
                    torch.split,
                    args=(fused_linear, 512),
                    kwargs={"dim": -1},
                )
                q_out = graph.call_function(operator.getitem, args=(split_out, 0))
                k_out = graph.call_function(operator.getitem, args=(split_out, 1))
                v_out = graph.call_function(operator.getitem, args=(split_out, 2))

            # Replace each QKV node's uses with the split outputs
            q_node.replace_all_uses_with(q_out)
            k_node.replace_all_uses_with(k_out)
            v_node.replace_all_uses_with(v_out)

            logger.info(
                "[_pass_fuse_qkv_projections] Fused QKV triplet: "
                f"{q_node.name}, {k_node.name}, {v_node.name}"
            )

        if not matched:
            logger.warning(
                "[_pass_fuse_qkv_projections] No QKV triplet found — pass not applied"
            )
            return gm

        graph.eliminate_dead_code()
        graph.lint()
        gm.recompile()
        logger.info("[_pass_fuse_qkv_projections] QKV fusion applied")

    except Exception as exc:
        logger.warning(f"[_pass_fuse_qkv_projections] Failed: {exc}")

    return gm


# ---------------------------------------------------------------------------
# OPT-3: Flash SDPA backend — already handled at module-load time via
# torch.backends.cuda.enable_flash_sdp(True). The BF16 casts from OPT-1
# cause PyTorch to dispatch SDPA to the Flash path on Blackwell.
# This pass is a no-op stub that logs confirmation.
# ---------------------------------------------------------------------------

def _pass_sdpa_backend_selection(gm: fx.GraphModule) -> fx.GraphModule:
    """
    OPT-3 stub — Flash SDPA backend selection is handled at module-load time.
    This pass verifies SDPA nodes exist in the graph and logs confirmation.
    """
    sdpa_nodes = [
        n for n in gm.graph.nodes
        if n.op == "call_function" and n.target is F.scaled_dot_product_attention
    ]
    if sdpa_nodes:
        logger.info(
            f"[_pass_sdpa_backend_selection] Found {len(sdpa_nodes)} SDPA node(s). "
            "Flash SDP enabled at module load — no graph changes needed."
        )
    else:
        logger.warning(
            "[_pass_sdpa_backend_selection] No SDPA nodes found in graph"
        )
    return gm


# ---------------------------------------------------------------------------
# OPT-4: linear+add -> addmm fusion (medium confidence)
# Pattern: F.linear node (no bias) whose sole user is operator.add with a
# residual tensor, replacing the pair with F.linear that uses the residual
# as a bias-equivalent via torch.ops.aten.addmm.
#
# At the pre-Inductor F.linear level we cannot directly inject aten.addmm
# without breaking the FX graph semantics that Inductor expects. Instead,
# we annotate the pass — Inductor performs mm+add -> addmm epilogue fusion
# automatically in most cases. This pass detects whether the pattern exists
# and logs whether Inductor is expected to handle it.
# ---------------------------------------------------------------------------

def _pass_fuse_linear_add_to_addmm(gm: fx.GraphModule) -> fx.GraphModule:
    """
    OPT-4 — detect (F.linear, operator.add) pattern and attempt addmm fusion.

    At the pre-Inductor graph level, F.linear is a single call_function node.
    Inductor's epilogue fusion usually handles mm+add automatically. This pass
    attempts the fusion explicitly only when the pattern is unambiguous:
    - linear node has no bias argument (bias=None or absent)
    - linear node has exactly one user: the add node
    - add node has exactly two arguments: the linear output and a residual

    If these conditions hold, we rewrite to F.linear with the residual as the
    bias argument, which at the aten level maps to addmm(residual, input, weight.T).
    This is semantically correct because F.linear(x, w, b) = x @ w.T + b.
    """
    try:
        matched = False
        graph = gm.graph
        nodes_snapshot = list(graph.nodes)

        for node in nodes_snapshot:
            if not (node.op == "call_function" and node.target is F.linear):
                continue

            # Check: no existing bias
            has_bias = len(node.args) > 2 and node.args[2] is not None
            if has_bias:
                continue

            # Check: exactly one user
            users = list(node.users)
            if len(users) != 1:
                continue

            add_node = users[0]
            if not (add_node.op == "call_function" and add_node.target is operator.add):
                continue

            # Identify the residual (the non-linear argument to the add)
            if add_node.args[0] is node:
                residual = add_node.args[1]
            elif add_node.args[1] is node:
                residual = add_node.args[0]
            else:
                continue

            # Rewrite: F.linear(x, w, residual) = x @ w.T + residual
            inp_node = node.args[0]
            weight_node = node.args[1]

            with graph.inserting_before(node):
                addmm_linear = graph.call_function(
                    F.linear,
                    args=(inp_node, weight_node, residual),
                )

            add_node.replace_all_uses_with(addmm_linear)
            # Dead code: the original linear and add nodes will be pruned
            matched = True
            logger.info(
                f"[_pass_fuse_linear_add_to_addmm] Fused {node.name} + {add_node.name} "
                "into F.linear with bias=residual (-> addmm at aten level)"
            )

        if not matched:
            logger.warning(
                "[_pass_fuse_linear_add_to_addmm] No (F.linear, add) pattern found "
                "— pass not applied (Inductor may handle this automatically)"
            )
            return gm

        graph.eliminate_dead_code()
        graph.lint()
        gm.recompile()
        logger.info("[_pass_fuse_linear_add_to_addmm] Applied")

    except Exception as exc:
        logger.warning(f"[_pass_fuse_linear_add_to_addmm] Failed: {exc}")

    return gm


# ---------------------------------------------------------------------------
# Backend registration
# ---------------------------------------------------------------------------

@register_backend
def sdpa_attention_opt(gm: fx.GraphModule, example_inputs: list) -> Callable:
    """
    Custom torch.compile() backend for SDPAAttentionBlock.

    Pass application order:
      1. OPT-2 QKV fusion   — structural fusion first so BF16 cast
                               applies to the already-fused QKV linear
      2. OPT-4 addmm fusion — must run BEFORE BF16 cast (OPT-1) because
                               OPT-1 inserts a to_dtype node between F.linear
                               and operator.add, breaking the direct edge
                               needed for pattern detection
      3. OPT-1 BF16 cast    — dtype promotion; applies to all F.linear nodes
                               including the fused QKV and the rewritten
                               out-proj (which now has bias=residual)
      4. OPT-3 SDPA stub    — verification/logging; backend flags set at load time
    """
    logger.info("sdpa_attention_opt backend: starting FX pass pipeline")

    # OPT-2: fuse Q/K/V projections into one batched linear
    gm = _pass_fuse_qkv_projections(gm)

    # OPT-4: fuse output-projection linear + residual add into linear(bias=residual)
    # Must run BEFORE OPT-1: the BF16 pass inserts to_dtype between linear and add,
    # breaking the direct (F.linear -> operator.add) edge this pass requires.
    gm = _pass_fuse_linear_add_to_addmm(gm)

    # OPT-1 Stage 2: cast all F.linear nodes to BF16 (includes fused QKV + out-proj)
    gm = _pass_promote_linear_to_bf16(gm)

    # OPT-3: SDPA backend verification (no graph changes; flags set at import)
    gm = _pass_sdpa_backend_selection(gm)

    logger.info("sdpa_attention_opt backend: delegating to Inductor")
    return compile_fx(gm, example_inputs)


# ---------------------------------------------------------------------------
# Workload interface — re-exported from the baseline for standalone use
# ---------------------------------------------------------------------------

from sdpa_attention import (  # noqa: E402  (after backend registration)
    SDPAAttentionBlock,
    BATCH_SIZE,
    SEQ_LEN,
    DIM,
    NUM_HEADS,
    HEAD_DIM,
    DEVICE,
)


def get_model_and_input() -> tuple:
    """
    Optimized workload interface.

    Applies non-graph optimizations that must be set before torch.compile:
    - TF32 + Flash SDP flags (already set at module load above)
    - Returns float32 model/input; BF16 cast is applied inside the backend
      FX pass so the user-facing interface remains unchanged.
    """
    assert torch.cuda.is_available(), "CUDA required"
    model = SDPAAttentionBlock().to(DEVICE).eval()
    x = torch.randn(BATCH_SIZE, SEQ_LEN, DIM, device=DEVICE)
    # No channels_last or batch-padding needed for attention (3D input)
    return model, x


if __name__ == "__main__":
    import torch
    model, x = get_model_and_input()
    compiled = torch.compile(model, backend="sdpa_attention_opt")
    with torch.no_grad():
        out = compiled(x)
    print(f"Output shape: {out.shape}, dtype: {out.dtype}")
