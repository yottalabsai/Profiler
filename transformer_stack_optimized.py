"""
transformer_stack_optimized.py — Custom torch.compile backend for TransformerStack.

Optimizations implemented (from optimizations.json):
  OPT-1 [high]   BF16 dtype promotion — routes all matmuls through tensor core path
  OPT-2 [medium] QKV weight fusion — 3x mm → 1x mm + 3 slices (fewer kernel launches)
  OPT-3 [medium] SDPA replacement — softmax+bmm decomposition → F.scaled_dot_product_attention

Pipeline: dedup-aware (UniqueSubgraphRegistry detects 8 identical TransformerLayer partitions).
OPT-2 and OPT-3 apply only to unique representatives and are propagated to duplicates.
OPT-1 is non-graph (applied in get_model_and_input before torch.compile).
"""
from __future__ import annotations

import logging
import operator
from typing import Callable

import torch
import torch.fx as fx
import torch.nn.functional as F

from torch._dynamo import register_backend
from torch._inductor.compile_fx import compile_fx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Workload interface
# ---------------------------------------------------------------------------

DEVICE   = "cuda"
BATCH    = 4
SEQ_LEN  = 128
HIDDEN   = 512
N_HEADS  = 8
FFN_DIM  = 2048
N_LAYERS = 8
HEAD_DIM = HIDDEN // N_HEADS

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from examples.transformer_stack.transformer_stack import TransformerStack  # noqa: E402


def get_model_and_input() -> tuple:
    """Return model + input with OPT-1 (BF16) applied."""
    assert torch.cuda.is_available(), "CUDA required"
    model = TransformerStack().to(DEVICE).eval()

    # OPT-1: BF16 dtype promotion (non-graph — must be before torch.compile)
    if next(model.parameters()).dtype != torch.bfloat16:
        model = model.to(torch.bfloat16)
        logger.info("[transformer_stack_opt] OPT-1: model cast to BF16")

    x = torch.randn(BATCH, SEQ_LEN, HIDDEN, device=DEVICE, dtype=torch.bfloat16)
    return model, x


# ---------------------------------------------------------------------------
# FX passes — operate on the pre-Inductor Dynamo graph.
#
# The @register_backend function receives the graph BEFORE Inductor lowers it.
# At this level, ops are Python-level callables:
#   nn.Linear  →  call_function: torch.nn.functional.linear
#   @          →  call_function: operator.matmul  (the Python @ operator)
#   *          →  call_function: operator.mul
#   softmax    →  call_function: torch.softmax  (same object as F.softmax)
#   transpose  →  call_method: "transpose"
#
# Inductor later decomposes these into aten primitives (aten::mm, aten::_softmax,
# etc.).  Passes that target those aten forms find nothing at this stage.
# ---------------------------------------------------------------------------

def _pass_fuse_qkv(gm: fx.GraphModule, partition_inputs: list) -> fx.GraphModule:
    """
    Replace 3 consecutive F.linear(x, W_q/k/v) calls with one fused linear.

    Weight tensors are passed as placeholder inputs in the partitioned subgraph
    (Dynamo lifts all parameters to function arguments).  We resolve their
    actual values from `partition_inputs` — the tensors captured by a single
    interpreted forward pass through the partition.
    """
    try:
        # Map placeholder node → actual tensor value
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        ph_to_tensor = {ph: t for ph, t in zip(placeholders, partition_inputs)}

        # Group F.linear calls by their first argument (the shared x input)
        lin_groups: dict[str, list] = {}
        for n in gm.graph.nodes:
            if n.op == "call_function" and n.target is F.linear:
                lin_groups.setdefault(n.args[0].name, []).append(n)

        fused = False
        for x_name, lin_list in lin_groups.items():
            if len(lin_list) < 3:
                continue
            q_lin, k_lin, v_lin = lin_list[0], lin_list[1], lin_list[2]

            W_q = ph_to_tensor.get(q_lin.args[1])
            W_k = ph_to_tensor.get(k_lin.args[1])
            W_v = ph_to_tensor.get(v_lin.args[1])
            if W_q is None or W_k is None or W_v is None:
                logger.warning("[transformer_stack_opt] OPT-2: weight tensors not in partition inputs")
                break

            W_qkv = torch.cat([W_q, W_k, W_v], dim=0)
            gm.register_buffer("_fused_qkv", W_qkv)

            with gm.graph.inserting_before(q_lin):
                w_buf  = gm.graph.get_attr("_fused_qkv")
                fused_lin = gm.graph.call_function(F.linear, (q_lin.args[0], w_buf))
                chunks = gm.graph.call_function(torch.chunk, (fused_lin, 3), {"dim": -1})
                q_out  = gm.graph.call_function(operator.getitem, (chunks, 0))
                k_out  = gm.graph.call_function(operator.getitem, (chunks, 1))
                v_out  = gm.graph.call_function(operator.getitem, (chunks, 2))

            q_lin.replace_all_uses_with(q_out)
            k_lin.replace_all_uses_with(k_out)
            v_lin.replace_all_uses_with(v_out)
            for dead in (q_lin, k_lin, v_lin):
                gm.graph.erase_node(dead)

            gm.graph.lint()
            gm.recompile()
            logger.info("[transformer_stack_opt] OPT-2: QKV fused (input '%s')", x_name)
            fused = True
            break

        if not fused:
            logger.warning("[transformer_stack_opt] OPT-2: QKV pattern not found — pass not applied")
    except Exception as e:
        logger.warning("[transformer_stack_opt] OPT-2 failed: %s", e)
    return gm


def _pass_replace_sdpa(gm: fx.GraphModule) -> fx.GraphModule:
    """
    Replace  softmax(q @ k.T * scale) @ v  with F.scaled_dot_product_attention.

    Pattern in pre-Inductor Dynamo graph:
        k_t        = call_method  "transpose"   (k, -2, -1)
        qk         = call_function operator.matmul   (q, k_t)
        scaled     = call_function operator.mul      (qk, scale)
        attn       = call_function torch.softmax     (scaled, dim=-1)
        out        = call_function operator.matmul   (attn, v)

    SDPA expects (q, k, v) where k is NOT pre-transposed — it transposes
    internally.  We extract k from the transpose node's input.
    """
    try:
        # Anchor on the final matmul: out = attn @ v
        replaced = 0
        for n in list(gm.graph.nodes):
            if n.op != "call_function" or n.target is not operator.matmul:
                continue

            attn_node, v_node = n.args[0], n.args[1]

            # attn_node must be a softmax call
            if not (attn_node.op == "call_function" and
                    attn_node.target is torch.softmax):
                continue

            # softmax input: scaled = qk * scale
            scaled_node = attn_node.args[0]
            if not (scaled_node.op == "call_function" and
                    scaled_node.target is operator.mul):
                continue

            # scaled = operator.mul(qk, scale_float)
            qk_node = scaled_node.args[0]
            if not (qk_node.op == "call_function" and
                    qk_node.target is operator.matmul):
                continue

            q_node, k_t_node = qk_node.args[0], qk_node.args[1]

            # Unwrap the k.transpose(-2, -1) to get bare k for SDPA
            if k_t_node.op == "call_method" and k_t_node.target == "transpose":
                k_node = k_t_node.args[0]
            else:
                logger.warning("[transformer_stack_opt] OPT-3: k_t not a call_method transpose — skipping")
                continue

            with gm.graph.inserting_before(n):
                sdpa = gm.graph.call_function(
                    F.scaled_dot_product_attention,
                    (q_node, k_node, v_node),
                )

            n.replace_all_uses_with(sdpa)
            for dead in (n, attn_node, scaled_node, qk_node):
                try:
                    if not dead.users:
                        gm.graph.erase_node(dead)
                except Exception:
                    pass
            replaced += 1

        if replaced:
            gm.graph.lint()
            gm.recompile()
            logger.info("[transformer_stack_opt] OPT-3: replaced %d attention block(s) with SDPA", replaced)
        else:
            logger.warning("[transformer_stack_opt] OPT-3: attention pattern not found — pass not applied")
    except Exception as e:
        logger.warning("[transformer_stack_opt] OPT-3 failed: %s", e)
    return gm


# ---------------------------------------------------------------------------
# Partition input capture helper
# ---------------------------------------------------------------------------

def _capture_partition_inputs(split: fx.GraphModule, example_inputs: list) -> dict:
    captured: dict = {}
    saved: dict = {}
    for name, submod in split.named_children():
        if isinstance(submod, fx.GraphModule):
            saved[name] = submod.forward
            def _make_cap(n: str, orig: Callable) -> Callable:
                def _fwd(*args, **kwargs):
                    captured[n] = list(args)
                    return orig(*args, **kwargs)
                return _fwd
            submod.forward = _make_cap(name, submod.forward)
    with torch.no_grad():
        split(*example_inputs)
    for name, submod in split.named_children():
        if isinstance(submod, fx.GraphModule):
            submod.forward = saved[name]
    return captured


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

@register_backend
def transformer_stack_opt(gm: fx.GraphModule, example_inputs: list) -> Callable:
    """
    Dedup-aware backend: splits the graph by layer, applies OPT-2/3 to the unique
    representative, propagates compiled callables to all 7 structural duplicates.
    """
    logger.info("transformer_stack_opt backend: starting")

    from nvidia.operator_profiler.fx import UniqueSubgraphRegistry, FxPassRunner

    registry = UniqueSubgraphRegistry(gm)
    equiv_map = registry.build_partition_equivalence_map()

    if not equiv_map:
        # No repeated layers — flat compile (preserves cross-layer Inductor fusion)
        logger.info("transformer_stack_opt: no repeated layers, flat compile path")
        gm = _pass_fuse_qkv(gm, example_inputs)
        gm = _pass_replace_sdpa(gm)
        logger.info("transformer_stack_opt: delegating to Inductor")
        return compile_fx(gm, example_inputs)

    logger.info("transformer_stack_opt: %d duplicate partitions, dedup path", len(equiv_map))

    # Capture actual partition input tensors first — needed by _pass_fuse_qkv to
    # resolve weight values from placeholder nodes.
    partition_inputs = _capture_partition_inputs(registry.split, example_inputs)

    for rep_name, rep_mod in registry.unique_reps:
        inputs = partition_inputs.get(rep_name, example_inputs)

        # OPT-2: QKV fusion — weights resolved from partition_inputs
        _pass_fuse_qkv(rep_mod, inputs)

        # OPT-3: SDPA replacement — targets pre-Inductor operator.matmul / torch.softmax
        _pass_replace_sdpa(rep_mod)

        # Propagate modified graph structure to duplicates before compilation.
        # Duplicates share the same GraphModule structure as their representative,
        # so we copy any registered buffers (fused QKV weight) and replay SDPA
        # on their graphs so they carry the same node structure.
        for _, dup_mod in registry.duplicates_of(rep_name):
            for buf_name, buf in rep_mod.named_buffers():
                dup_mod.register_buffer(buf_name, buf)
            _pass_replace_sdpa(dup_mod)

        compiled = compile_fx(rep_mod, inputs)
        rep_mod.forward = compiled
        for _, dup_mod in registry.duplicates_of(rep_name):
            dup_mod.forward = compiled
        logger.info("transformer_stack_opt: compiled rep '%s', shared with %d duplicates",
                    rep_name, len(list(registry.duplicates_of(rep_name))))

    return lambda *args: registry.split(*args)
