"""
gpt2_optimized.py — GPT-2 small (117M) with custom torch.compile backend.

Implements three optimizations from optimizations.json:
  OPT-1 (high confidence)   : BF16 dtype promotion — routes GEMM kernels from
                              FP32 SIMT (ampere_sgemm_*) to BF16 Tensor Core
                              (sm80_xmma_gemm_bf16bf16). Expected ~50-60%
                              total latency reduction.
  OPT-2 (medium confidence) : Pre-transposed weight buffers — eliminates runtime
                              aten.t() overhead on every mm/addmm call. Expected
                              ~3-8% additional reduction on top of OPT-1.
  OPT-3 (medium confidence) : max-autotune mode — benchmarks CUTLASS/cuBLAS
                              tile candidates at compile time and selects the
                              fastest for each unique (M,N,K) shape. Expected
                              ~5-15% additional reduction on GEMM operators.

Backend name : gpt2_opt
Dedup path   : UniqueSubgraphRegistry detects GPT-2's 12 identical transformer
               blocks; OPT-2 FX pass is applied only to the unique representative
               and its compiled callable is shared with all 11 duplicates.
"""
from __future__ import annotations

import logging
from typing import Callable

import torch
import torch.fx as fx
import torch.nn as nn
from torch._dynamo import register_backend
from torch._inductor.compile_fx import compile_fx  # function, not module

from nvidia.operator_profiler.fx import UniqueSubgraphRegistry, FxPassRunner

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Non-graph configuration knobs (set at import time, before torch.compile)
# ---------------------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True  # belt-and-suspenders for OPT-1

DEVICE   = "cuda"
BATCH    = 4
SEQ_LEN  = 128
MODEL_ID = "gpt2"


# ---------------------------------------------------------------------------
# OPT-2: Pre-transposed weight FX pass
# ---------------------------------------------------------------------------
# Classification: Manual per-rep pass.
#   - Requires register_buffer (cannot use replace_pattern).
#   - Targets aten.t() nodes wrapping get_attr in the Inductor-traced graph.
#   - Only applied to weight matrices with K >= 512 to justify the memory cost.
#
# NOTE: This pass operates on the Inductor-lowered graph (post-Dynamo), where
# Inductor has already decomposed F.linear into aten.t(get_attr) + aten.mm or
# aten.addmm. The pattern to detect is therefore:
#   aten.addmm(bias, x, aten.t(get_attr('weight')))  — bias-bearing projections
#   aten.mm(x, aten.t(get_attr('weight')))            — bias-free projections
# ---------------------------------------------------------------------------

def _pass_pretranspose(gm: fx.GraphModule) -> fx.GraphModule:
    """
    Detect aten.t(get_attr) nodes feeding aten.mm or aten.addmm.
    Pre-transposes the weight, registers the contiguous buffer, inserts a new
    get_attr node, patches the mm/addmm args in-place, and erases the t() node.

    Only rewrites weights where K dimension >= 512.
    Wrapped in try/except; returns gm unchanged on any failure.
    """
    try:
        rewritten = 0
        nodes_to_process = list(gm.graph.nodes)  # snapshot — never iterate live

        for node in nodes_to_process:
            if node.op != "call_function":
                continue
            target = node.target

            # Match aten.mm.default(x, t_node)
            if target is torch.ops.aten.mm.default:
                if len(node.args) < 2:
                    continue
                x_node, t_node = node.args[0], node.args[1]
                bias_node = None
                is_addmm = False
            # Match aten.addmm.default(bias, x, t_node)
            elif target is torch.ops.aten.addmm.default:
                if len(node.args) < 3:
                    continue
                bias_node, x_node, t_node = node.args[0], node.args[1], node.args[2]
                is_addmm = True
            else:
                continue

            # t_node must be aten.t() wrapping a get_attr (weight parameter)
            if not (
                t_node.op == "call_function"
                and t_node.target is torch.ops.aten.t.default
            ):
                continue
            if len(t_node.args) < 1:
                continue
            weight_node = t_node.args[0]
            if weight_node.op != "get_attr":
                continue

            # Retrieve the actual weight tensor to check shape
            try:
                weight_tensor = gm.get_parameter(weight_node.target)
            except AttributeError:
                # May be a buffer, not a parameter — try get_buffer path
                try:
                    weight_tensor = gm.get_buffer(weight_node.target)
                except AttributeError:
                    logger.warning(
                        "[_pass_pretranspose] Cannot retrieve tensor for %s — skipping",
                        weight_node.target,
                    )
                    continue

            # Only apply to sufficiently large weight matrices
            if weight_tensor.ndim < 2 or weight_tensor.shape[-1] < 512:
                continue

            # Pre-transpose and register as a new buffer
            buf_name = weight_node.target.replace(".", "_") + "_pretransposed"
            # Avoid double-registration if the same weight appears in multiple mm nodes
            if not hasattr(gm, buf_name):
                W_T = weight_tensor.t().contiguous()
                gm.register_buffer(buf_name, nn.Parameter(W_T, requires_grad=False))

            # Insert get_attr node for the transposed buffer immediately before the mm
            with gm.graph.inserting_before(node):
                new_weight_node = gm.graph.get_attr(buf_name)
                new_weight_node.meta = {}

            # Patch the mm/addmm args — W_T is already transposed, no aten.t() needed
            if is_addmm:
                node.args = (bias_node, x_node, new_weight_node)
            else:
                node.args = (x_node, new_weight_node)

            # Erase the orphaned aten.t() node (only safe if it has no other users)
            if len(t_node.users) == 0:
                gm.graph.erase_node(t_node)

            rewritten += 1

        if rewritten:
            gm.graph.lint()
            gm.recompile()
            logger.info("[_pass_pretranspose] Pre-transposed %d weight(s)", rewritten)
        else:
            logger.warning(
                "[_pass_pretranspose] Pattern aten.t(get_attr) not found — "
                "pass not applied. This is expected if running on the pre-Inductor "
                "graph; the pass targets the post-Dynamo Inductor-lowered graph."
            )
    except Exception as e:
        logger.warning("[_pass_pretranspose] Failed: %s", e)
    return gm


# ---------------------------------------------------------------------------
# Utility: capture per-partition actual input tensors
# ---------------------------------------------------------------------------

def _capture_partition_inputs(
    split_gm: fx.GraphModule,
    example_inputs: list,
) -> dict[str, list]:
    """Run split_gm once to capture actual input tensors for each partition."""
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
# Backend registration
# ---------------------------------------------------------------------------

@register_backend
def gpt2_opt(gm: fx.GraphModule, example_inputs) -> Callable:
    """
    Custom torch.compile backend for GPT-2.

    Pass order (Rule 6):
      1. OPT-2 pre-transpose (manual per-rep) — applied first to the final
         Inductor-lowered graph so it sees aten.t() nodes.
      2. OPT-3 max-autotune — passed via options to compile_fx.

    Dedup awareness (Rule 10):
      - UniqueSubgraphRegistry detects the 12 structurally identical transformer
        blocks. OPT-2 runs on the unique representative only; its compiled
        callable is shared with all 11 duplicates.
      - If no duplicates are found (flat graph), passes are applied to gm
        directly and the result is compiled with compile_fx.
    """
    logger.info("gpt2_opt backend: starting")

    # OPT-3: max-autotune options forwarded to Inductor
    inductor_options = {"max_autotune": True}

    registry = UniqueSubgraphRegistry(gm)
    equiv_map = registry.build_partition_equivalence_map()

    if not equiv_map:
        # ---------------------------------------------------------------
        # Flat compile path — no repeated layers detected.
        # Apply OPT-2 directly to the full flat graph, then hand off to
        # Inductor. The flat path preserves cross-layer Inductor fusion.
        # ---------------------------------------------------------------
        logger.info("gpt2_opt: no repeated layers detected — flat compile path")
        gm = _pass_pretranspose(gm)
        logger.info("gpt2_opt: delegating to Inductor (max_autotune=True)")
        return compile_fx(gm, example_inputs, config_patches=inductor_options)

    # -------------------------------------------------------------------
    # Dedup compile path — repeated transformer blocks detected.
    # Apply OPT-2 to each unique representative, compile once per unique
    # signature, then share the compiled callable with all duplicates.
    # -------------------------------------------------------------------
    logger.info(
        "gpt2_opt: %d duplicate partition(s) detected — dedup compile path",
        len(equiv_map),
    )

    # Capture actual partition inputs (needed for shape-aware decisions;
    # also ensures the split graph is warmed up before we mutate submodules)
    partition_inputs = _capture_partition_inputs(registry.split, example_inputs)

    for rep_name, rep_mod in registry.unique_reps:
        # OPT-2: pre-transpose — manual pass, applied per unique rep
        _pass_pretranspose(rep_mod)

        # OPT-3: compile unique rep with max-autotune
        rep_inputs = partition_inputs.get(rep_name, example_inputs)
        try:
            compiled = compile_fx(rep_mod, rep_inputs, config_patches=inductor_options)
        except Exception as e:
            logger.warning(
                "gpt2_opt: compile_fx failed for rep %s (%s) — falling back to eager",
                rep_name, e,
            )
            compiled = rep_mod.forward

        # Patch representative's forward
        rep_mod.forward = compiled

        # Share compiled callable with all structural duplicates
        for dup_name, dup_mod in registry.duplicates_of(rep_name):
            dup_mod.forward = compiled
            logger.info(
                "gpt2_opt: shared compiled callable from %s → %s", rep_name, dup_name
            )

    # Return callable: registry.split routes each forward call through the
    # assembled graph with compiled partition .forward methods.
    logger.info("gpt2_opt: backend assembly complete")
    return lambda *args: registry.split(*args)


# ---------------------------------------------------------------------------
# get_model_and_input — applies OPT-1 (BF16 dtype promotion) before compile
# ---------------------------------------------------------------------------

def get_model_and_input() -> tuple:
    """
    Return (uncompiled GPT-2 model on CUDA, input_ids tensor on CUDA).

    OPT-1 applied here (non-graph, must precede torch.compile):
      - model.to(torch.bfloat16) casts all parameters including wte/wpe embeddings.
      - input_ids remains int64 (required for embedding lookup).
      - torch.backends.cuda.matmul.allow_tf32 = True set at module level.

    Downloads GPT-2 weights from HuggingFace on first call (~500 MB).
    Subsequent calls use the local cache.
    """
    assert torch.cuda.is_available(), "CUDA required"

    from transformers import GPT2Model  # imported here to keep top-level import-free

    hf_model = GPT2Model.from_pretrained(MODEL_ID)
    model = GPT2Wrapper(hf_model).to(DEVICE).eval()

    # OPT-1: BF16 dtype promotion
    # Cast model parameters to BF16 so all GEMM calls route to Tensor Core path.
    # input_ids stays int64 — embedding lookup requires integer indices.
    if next(model.parameters()).dtype != torch.bfloat16:
        model = model.to(torch.bfloat16)

    # Random token ids in [0, vocab_size)
    vocab_size = hf_model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (BATCH, SEQ_LEN), device=DEVICE)
    # input_ids intentionally kept as int64 (torch.long) — embedding lookup
    # does not accept BF16 indices.

    return model, input_ids


# ---------------------------------------------------------------------------
# GPT-2 wrapper (identical to baseline)
# ---------------------------------------------------------------------------

class GPT2Wrapper(nn.Module):
    """Thin wrapper so model(input_ids) returns the last hidden state tensor."""

    def __init__(self, hf_model: nn.Module):
        super().__init__()
        self.model = hf_model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids).last_hidden_state


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model, input_ids = get_model_and_input()

    compiled_model = torch.compile(model, backend="gpt2_opt")

    with torch.no_grad():
        y = compiled_model(input_ids)

    print(f"Output shape : {y.shape}")   # expect (4, 128, 768)
    print(f"Output dtype : {y.dtype}")   # expect torch.bfloat16
