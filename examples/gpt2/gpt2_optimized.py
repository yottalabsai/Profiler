"""
gpt2_optimized.py — Custom torch.compile() backend for GPT-2 small (117M).

Implements profiling-guided optimizations from optimizations.json:

  OPT-1 (HIGH, priority 1) — Tensor Core engagement via TF32/BF16 dtype promotion
                              Stage 1: torch.backends.cuda.matmul.allow_tf32 = True
                              and torch.set_float32_matmul_precision('high') set at
                              module load time (non-graph, immediate effect).
                              Stage 2: FX pass that inserts bfloat16 cast nodes before
                              every aten.mm.default and aten.addmm.default call, and
                              restores float32 on the output.
                              Evidence: smsp__pipe_tensor_cycles_active = 0.0% across
                              all 792 GEMM launches — cuBLAS was routing to the SIMT
                              Kernel2 (legacy SGEMM) path exclusively.

  OPT-2 (MEDIUM, priority 2) — Replace efficient_attention with SDPA + is_causal=True
                               Replace aten._scaled_dot_product_efficient_attention.default
                               with aten.scaled_dot_product_attention.default using
                               is_causal=True and attn_mask=None. Forces SDPA dispatcher
                               to select a Blackwell-native kernel (Flash Attention 3 or
                               cuDNN), eliminating the sm80-targeting
                               fmha_cutlassF_f32_aligned_64x64_rf_sm80 kernel that runs
                               in compatibility mode on sm100.
                               Also eliminates 4 Triton mask-prep kernels per block
                               (144 kernel launches total) by removing the materialized
                               [4,12,128,128] causal mask.
                               Evidence: achieved_occupancy = 8.5% (target: 6-8 warps),
                               registers_per_thread = 168, 49152 bytes local memory spills.

  OPT-3 (LOW, stub only)    — Batch same-shape sequential GEMMs into strided-batched
                               GEMM calls. Not implemented; see implementation_notes.md.
                               Requires complex topology analysis + model restructuring
                               that goes beyond pure FX surgery.

Backend registration name: gpt2_opt
Prerequisite order: OPT-1 → OPT-2 (OPT-2 in BF16 regime benefits from FA3 BF16 kernels).
"""
from __future__ import annotations

import logging
import math
import operator
from typing import Callable

import torch
import torch.fx as fx
import torch.nn.functional as F
from torch._dynamo import register_backend
from torch._inductor.compile_fx import compile_fx  # function, not module

from nvidia.operator_profiler.fx import UniqueSubgraphRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend name constant
# ---------------------------------------------------------------------------
BACKEND_NAME = "gpt2_opt"

# ---------------------------------------------------------------------------
# Module-load-time side effects for OPT-1 Stage 1 (Level A)
# Must be set BEFORE torch.compile traces the model.
# ---------------------------------------------------------------------------

# OPT-1 Stage 1: route all FP32 GEMMs through TF32 Tensor Core path.
# 'high' precision maps to TF32 in PyTorch 2.x; eliminates Kernel2 (legacy SGEMM).
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
logger.info(
    "gpt2_opt: set_float32_matmul_precision('high'), allow_tf32=True "
    "at module load [OPT-1 Stage 1 / Level A]"
)


# ---------------------------------------------------------------------------
# FX Pass: OPT-1 Stage 2 / Level B — BF16 casts around aten.mm and aten.addmm
# ---------------------------------------------------------------------------

def _pass_gemm_bf16_casts(gm: fx.GraphModule) -> fx.GraphModule:
    """
    OPT-1 Stage 2 (Level B) — insert bfloat16 cast nodes around every
    aten.mm.default and aten.addmm.default node in the graph.

    At the pre-Inductor FX IR level, nn.Linear in HuggingFace GPT-2 lowers to:
      - aten.addmm.default(bias, input, weight_T)  — linear with bias
      - aten.mm.default(input, weight_T)            — linear without bias

    Strategy:
      - Cast every positional argument (all are tensor inputs) to bfloat16
        before the GEMM node.
      - Cast the output back to float32 so downstream ops (LayerNorm, residual
        add, softmax) remain dtype-consistent.

    This forces cuBLAS to select a Tensor Core GEMM algorithm on Blackwell
    (sm100), replacing the SIMT Kernel2 path that had 0% Tensor Core activity
    in the baseline profile.

    All shapes in GPT-2 small (M=512, K=768/3072, N=768/2304/3072) are
    divisible by 16, so BF16 Tensor Core alignment requirements are satisfied.
    """
    try:
        matched = False
        graph = gm.graph
        # Partition subgraphs are at the torch-functional level (pre-ATen lowering),
        # so the GEMM targets are torch.addmm / torch.mm, not aten.addmm.default.
        gemm_targets = {torch.addmm, torch.mm}

        for node in list(graph.nodes):
            if not (node.op == "call_function" and node.target in gemm_targets):
                continue
            matched = True

            original_args = list(node.args)
            cast_args: list[fx.Node] = []

            # Cast each positional tensor argument to bfloat16.
            # For addmm(bias, mat1, mat2): cast all three — cuBLAS handles BF16 bias.
            with graph.inserting_before(node):
                for arg in original_args:
                    cast_node = graph.call_function(
                        torch.ops.aten.to.dtype,
                        args=(arg, torch.bfloat16),
                    )
                    cast_args.append(cast_node)

            node.args = tuple(cast_args)

            # Restore float32 output for downstream dtype consistency
            with graph.inserting_after(node):
                cast_out = graph.call_function(
                    torch.ops.aten.to.dtype,
                    args=(node, torch.float32),
                )
            # Replace all downstream uses with the float32 cast output,
            # then fix the self-reference in cast_out.args
            node.replace_all_uses_with(cast_out)
            cast_out.args = (node, torch.float32)

        if not matched:
            logger.warning(
                "[_pass_gemm_bf16_casts] No aten.mm / aten.addmm nodes found "
                "— pass not applied"
            )
            return gm

        graph.lint()
        gm.recompile()
        logger.info(
            "[_pass_gemm_bf16_casts] Applied BF16 casts to all aten.mm / "
            "aten.addmm nodes [OPT-1 Stage 2]"
        )

    except Exception as exc:
        logger.warning("[_pass_gemm_bf16_casts] Failed: %s", exc)

    return gm


# ---------------------------------------------------------------------------
# FX Pass: OPT-2 — Replace efficient_attention with SDPA (is_causal=True)
# ---------------------------------------------------------------------------

def _pass_replace_efficient_attn_with_sdpa(gm: fx.GraphModule) -> fx.GraphModule:
    """
    OPT-2 (medium confidence) — replace aten._scaled_dot_product_efficient_attention
    nodes with aten.scaled_dot_product_attention(is_causal=True, attn_mask=None).

    The efficient_attention kernel (fmha_cutlassF_f32_aligned_64x64_rf_sm80) was
    compiled for sm80 (A100) and runs in compatibility mode on Blackwell sm100.
    It also requires a materialized [4,12,128,128] causal mask tensor, constructed
    by 4 upstream Triton kernels per transformer block (48 kernels total for 12
    blocks × 3 iters = 144 eliminated kernel launches).

    Replacement: aten.scaled_dot_product_attention with is_causal=True dispatches
    via the SDPA dispatcher, which selects Flash Attention 3 or cuDNN kernels
    compiled natively for sm100. The causal mask is handled inside the fused kernel
    without materialization.

    Signature of aten._scaled_dot_product_efficient_attention.default:
      (query, key, value, attn_bias, compute_log_sumexp, dropout_p, is_causal,
       scale=None)
    Returns a tuple: (output, log_sumexp, seed, offset).
    Downstream users of the tuple index element [0] for the attention output.
    Indices [1-3] (log_sumexp, seed, offset) are dead in inference mode.

    Signature of aten.scaled_dot_product_attention.default:
      (query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
       scale=None)
    Returns the output tensor directly (not a tuple).

    Head dimension for GPT-2 small: 64 → scale = 1/sqrt(64) = 0.125.
    """
    try:
        matched = False
        graph = gm.graph
        # At the torch-functional level (pre-ATen lowering), attention appears as
        # F.scaled_dot_product_attention with an explicit attn_mask kwarg (HuggingFace
        # GPT-2 builds a causal float mask and passes it here). We modify the kwargs
        # in-place: set attn_mask=None and is_causal=True, then eliminate dead code to
        # remove the upstream mask-construction nodes (torch.tensor / torch.cat).
        sdpa_target = F.scaled_dot_product_attention

        for node in list(graph.nodes):
            if not (node.op == "call_function" and node.target is sdpa_target):
                continue

            # Skip if already using causal mask without explicit mask tensor
            attn_mask_val = node.kwargs.get("attn_mask", None)
            is_causal_val = node.kwargs.get("is_causal", False)
            if is_causal_val is True and not isinstance(attn_mask_val, fx.Node):
                continue  # already optimal, skip

            if not isinstance(attn_mask_val, fx.Node):
                continue  # no explicit mask node to remove, nothing to do

            matched = True

            # Replace attn_mask with None and set is_causal=True in-place.
            # This drops the reference to the mask tensor node, making the
            # mask-construction subgraph dead so eliminate_dead_code() removes it.
            new_kwargs = dict(node.kwargs)
            new_kwargs["attn_mask"] = None
            new_kwargs["is_causal"] = True
            node.kwargs = new_kwargs

        if not matched:
            logger.warning(
                "[_pass_replace_efficient_attn_with_sdpa] "
                "No F.scaled_dot_product_attention nodes with explicit attn_mask "
                "found — pass not applied"
            )
            return gm

        # Eliminate the now-dead mask-construction subgraph (torch.tensor, torch.cat,
        # comparisons that built the [4,12,128,128] causal mask float tensor).
        graph.eliminate_dead_code()
        graph.lint()
        gm.recompile()
        logger.info(
            "[_pass_replace_efficient_attn_with_sdpa] "
            "Replaced explicit attn_mask with is_causal=True on %d SDPA node(s); "
            "dead mask-construction nodes eliminated [OPT-2]",
            sum(1 for n in graph.nodes
                if n.op == "call_function" and n.target is sdpa_target),
        )

    except Exception as exc:
        logger.warning("[_pass_replace_efficient_attn_with_sdpa] Failed: %s", exc)

    return gm


# ---------------------------------------------------------------------------
# Utility: capture partition inputs for dedup compile path
# ---------------------------------------------------------------------------

def _capture_partition_inputs(
    split_gm: fx.GraphModule,
    example_inputs: list,
) -> dict[str, list]:
    """
    Run split_gm once under no_grad to capture actual input tensors for each
    partition submodule via forward pre-hooks.
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


# ---------------------------------------------------------------------------
# Backend: gpt2_opt
# ---------------------------------------------------------------------------

@register_backend
def gpt2_opt(gm: fx.GraphModule, example_inputs) -> Callable:
    """
    Custom torch.compile() backend for GPT-2.

    Pass application order (per prerequisite_for constraints in optimizations.json
    and the recommendation: OPT-1 → OPT-2 to benefit from FA3 BF16 path):

      1. OPT-2 — replace efficient_attention with SDPA (is_causal=True)
                 Applied before BF16 casts because the SDPA replacement operates
                 on the attention nodes; inserting BF16 casts first would change
                 the dtype of q/k/v tensors flowing into the attention op and
                 could confuse the pattern match.
      2. OPT-1 Stage 2 — BF16 casts around aten.mm / aten.addmm
                 Applied after SDPA replacement. If the SDPA replacement introduced
                 any new mm-class nodes they will be covered here automatically.
      3. Delegate to Inductor compile_fx

    OPT-1 Stage 1 (TF32 global flags) is applied at module-load time above.
    OPT-3 is a stub; no FX transformation is applied.

    GPT-2 has 12 structurally identical transformer blocks, so the dedup path
    (equiv_map non-empty) is the expected primary path. The flat-compile path
    is retained for robustness (e.g. if the splitter fails or batch=1).
    """
    logger.info("gpt2_opt backend: starting FX pass pipeline")

    # Build dedup registry to detect structurally identical transformer blocks.
    # For GPT-2, expect 12 identical subgraphs (modules_0 … modules_11).
    registry = UniqueSubgraphRegistry(gm)
    equiv_map = registry.build_partition_equivalence_map()

    if not equiv_map:
        # Flat path — no repeated partitions detected.
        # Apply passes to the full flat graph and delegate to Inductor.
        logger.info(
            "gpt2_opt: no repeated partitions detected — flat compile path"
        )

        # OPT-2: SDPA replacement (before BF16 casts to keep attn pattern clean)
        gm = _pass_replace_efficient_attn_with_sdpa(gm)

        # OPT-1 Stage 2: BF16 casts around all aten.mm / aten.addmm nodes
        gm = _pass_gemm_bf16_casts(gm)

        logger.info("gpt2_opt: delegating flat graph to Inductor compile_fx")
        return compile_fx(gm, example_inputs)

    # Dedup path — expected for GPT-2 (12 identical transformer blocks).
    logger.info(
        "gpt2_opt: %d duplicate partition(s) detected — dedup compile path",
        len(equiv_map),
    )

    # Apply passes to each unique representative; propagate compiled callable
    # to all structural duplicates.
    for rep_name, rep_mod in registry.unique_reps:
        # OPT-2 first, then OPT-1 Stage 2
        _pass_replace_efficient_attn_with_sdpa(rep_mod)
        _pass_gemm_bf16_casts(rep_mod)
        for _, dup_mod in registry.duplicates_of(rep_name):
            _pass_replace_efficient_attn_with_sdpa(dup_mod)
            _pass_gemm_bf16_casts(dup_mod)

    # Capture actual partition inputs by running split_gm once
    partition_inputs = _capture_partition_inputs(registry.split, example_inputs)

    # Compile each unique representative with its actual partition inputs;
    # share the resulting callable with all duplicates.
    for rep_name, rep_mod in registry.unique_reps:
        inputs = partition_inputs.get(rep_name, example_inputs)
        compiled = compile_fx(rep_mod, inputs)
        rep_mod.forward = compiled
        for _, dup_mod in registry.duplicates_of(rep_name):
            dup_mod.forward = compiled

    # Return a callable that routes through the assembled split graph.
    # registry.split is a GraphModule whose child partitions have compiled .forward.
    return lambda *args: registry.split(*args)


# ---------------------------------------------------------------------------
# Workload interface
# ---------------------------------------------------------------------------

DEVICE   = "cuda"
BATCH    = 4
SEQ_LEN  = 128
MODEL_ID = "gpt2"


class GPT2Wrapper(torch.nn.Module):
    """Thin wrapper so model(input_ids) returns the last hidden state tensor."""

    def __init__(self, hf_model: torch.nn.Module):
        super().__init__()
        self.model = hf_model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids).last_hidden_state


def get_model_and_input() -> tuple:
    """
    Return (uncompiled model on CUDA, input_ids tensor on CUDA).

    Non-graph optimizations applied here:
    - OPT-1 Stage 1: TF32 global flags are set at module-load time above.
      No model-level dtype conversion is applied here; BF16 casts are injected
      by the backend FX pass (OPT-1 Stage 2) so the public interface stays float32.

    Note: channels_last memory format does not apply here (GPT-2 has no 4-D
    NCHW convolution tensors — all GEMMs operate on 2-D/3-D tensors).
    """
    assert torch.cuda.is_available(), "CUDA required"

    from transformers import GPT2Model  # defer import to avoid top-level dep

    hf_model = GPT2Model.from_pretrained(MODEL_ID)
    model = GPT2Wrapper(hf_model).to(DEVICE).eval()

    vocab_size = hf_model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (BATCH, SEQ_LEN), device=DEVICE)

    return model, input_ids


if __name__ == "__main__":
    model, input_ids = get_model_and_input()
    compiled = torch.compile(model, backend=BACKEND_NAME)
    with torch.no_grad():
        out = compiled(input_ids)
    print(f"Output shape: {out.shape}, dtype: {out.dtype}")
    # expect (4, 128, 768), dtype: torch.float32
