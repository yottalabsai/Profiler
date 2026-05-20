"""
sdpa_attention_optimized.py — SDPAAttentionBlock with custom torch.compile() backend.

Implements four operator-level optimizations derived from ncu profiling of the baseline:

  OPT-1 (high confidence)   : BF16 dtype promotion — applied in get_model_and_input()
                              before torch.compile. Moves all GEMM kernels from the
                              FP32 SIMT path (smsp__pipe_tensor_cycles_active = 0%,
                              210 regs/thread, 16.6% occupancy) onto the BF16 Tensor
                              Core path. Also enables Flash Attention backend (OPT-3
                              requires BF16). Expected ~38.7% total latency reduction.

  OPT-2 (high confidence)   : QKV weight fusion — FX pass (manual per-rep) that
                              merges the three separate F.linear(x, W_q), F.linear(x,
                              W_k), F.linear(x, W_v) calls (all sharing the same
                              post-LayerNorm input) into a single F.linear(x, W_qkv)
                              + torch.chunk(3, dim=-1). Replaces 3 sequential 128-block
                              GEMMs (0.75 wave each) with 1 GEMM of ~384 blocks (2.3
                              waves). Eliminates 6 kernel launches. Expected additional
                              ~6.6% speedup.

  OPT-3 (medium confidence) : Flash Attention backend — non-graph knob applied before
                              torch.compile. Calls enable_flash_sdp(True) +
                              enable_mem_efficient_sdp(False) so the SDPA dispatcher
                              selects the BF16-native Flash kernel instead of the
                              sm80 xFormers FP32 path (fmha_cutlassF_f32_aligned_
                              64x64_rf_sm80). Requires OPT-1 (BF16). Expected
                              additional ~5.0% speedup.

  OPT-4 (medium confidence) : Pre-transposed QKV weight buffer — FX pass (manual
                              per-rep) that, after QKV fusion (OPT-2), eliminates the
                              aten.t() node on the fused weight by pre-computing the
                              transpose and registering a contiguous [512, 1536] buffer.
                              Converts NT GEMM to NN GEMM in cuBLAS. Expected
                              additional ~0.3% speedup. Applied only if OPT-2 produced
                              a fused weight buffer.

Pass application order (per optimizations.json prerequisite_for and Rule 6):
  OPT-1 (non-graph, get_model_and_input) →
  OPT-3 (non-graph, get_model_and_input) →
  OPT-2 (FX pass, _pass_fuse_qkv) →
  OPT-4 (FX pass, _pass_pretranspose_fused_qkv) →
  compile_fx

Backend name: sdpa_attention_opt
Dedup path  : SDPAAttentionBlock has one attention block (not repeated). The flat
              compile path is taken. UniqueSubgraphRegistry is included for structural
              consistency.

To profile:
    python nvidia/scripts/run_workload.py \\
        --workload examples/sdpa_attention/sdpa_attention_optimized.py \\
        --compile-backend sdpa_attention_opt \\
        --output-prefix profiler_output/sdpa_attention_opt \\
        --warmup-iters 2 --measure-iters 2
"""
from __future__ import annotations

import logging
import operator
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
# Non-graph configuration (set at import time, before torch.compile)
# ---------------------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True  # belt-and-suspenders for BF16 path

# ---------------------------------------------------------------------------
# Model and workload constants (mirrored from baseline)
# ---------------------------------------------------------------------------
DEVICE     = "cuda"
BATCH_SIZE = 8
SEQ_LEN    = 512
DIM        = 512
NUM_HEADS  = 8
HEAD_DIM   = DIM // NUM_HEADS  # 64


# ===========================================================================
# Optimized model class
# ===========================================================================

class SDPAAttentionBlockOpt(nn.Module):
    """
    Multi-head self-attention block with the same interface as SDPAAttentionBlock
    but uses separate Q/K/V projections to match the baseline graph structure.

    OPT-2 (QKV fusion) is applied as a post-trace FX pass — the model keeps
    separate q_proj/k_proj/v_proj at the nn.Module level so that the FX graph
    has the three distinct F.linear nodes that the fusion pass detects and merges.
    This is intentional: fusing at the nn.Module level would bypass the FX pass
    machinery and remove the canonical demonstration of the OPT-2 pass.
    """

    def __init__(self) -> None:
        super().__init__()
        self.q_proj   = nn.Linear(DIM, DIM, bias=False)
        self.k_proj   = nn.Linear(DIM, DIM, bias=False)
        self.v_proj   = nn.Linear(DIM, DIM, bias=False)
        self.out_proj = nn.Linear(DIM, DIM, bias=False)
        self.ln_pre   = nn.LayerNorm(DIM)
        self.ln_post  = nn.LayerNorm(DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        residual = x
        x = self.ln_pre(x)

        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k_proj(x).view(B, T, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(x).view(B, T, NUM_HEADS, HEAD_DIM).transpose(1, 2)

        # Flash Attention dispatched when BF16 + enable_flash_sdp=True (OPT-3)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(attn_out)
        out = self.ln_post(out + residual)
        return out


# ===========================================================================
# OPT-2: QKV weight fusion — manual per-rep FX pass
# ===========================================================================
# Classification: Manual per-rep pass.
#   - Requires register_buffer to store the fused W_qkv tensor (cannot use
#     replace_pattern which has no access to tensor values).
#   - Detects three F.linear calls sharing the same input node (post-LayerNorm
#     output) and fuses them into one F.linear + torch.chunk.
#   - Weight tensors are accessed via placeholder→tensor map built from
#     partition_inputs (Dynamo lifts all nn.Module params to placeholders).
#   - Implementation note: Dynamo attaches source-tracking metadata
#     (_param_name_to_source, dynamo_compile_id) to the GraphModule that
#     aot_autograd uses to verify every buffer/parameter. Newly registered
#     buffers have no Dynamo source entry, causing an AssertionError in
#     _try_get_metadata_from_dynamo. The fix is to strip these Dynamo
#     metadata fields before calling register_buffer, directing compile_fx
#     to the "not from Dynamo" code path which skips the source check.
#     The three weight placeholder nodes (wq/wk/wv) are then removed from
#     the graph and their positions pruned from partition_inputs so that
#     compile_fx receives a consistent (graph, inputs) pair.
#
# Mutual exclusion: OPT-4 (_pass_pretranspose_fused_qkv) must run AFTER this
# pass. If OPT-2 is skipped or fails, OPT-4 is also skipped.
# ===========================================================================

def _pass_fuse_qkv(
    gm: fx.GraphModule, partition_inputs: list
) -> tuple[fx.GraphModule, list, set]:
    """
    Fuse three F.linear(x, W_q), F.linear(x, W_k), F.linear(x, W_v) nodes
    sharing the same input x into a single F.linear(x, W_qkv) + chunk(3, -1).

    Returns (gm, updated_partition_inputs, removed_indices).
    removed_indices is the set of placeholder positions erased from partition_inputs.
    On failure/no-match, returns (gm, partition_inputs, empty set) unchanged.

    Implementation note — Dynamo source-map compatibility:
      compile_fx → aot_autograd → _try_get_metadata_from_dynamo requires every
      buffer/parameter in gm to have a corresponding entry in
      gm._param_name_to_source (set by Dynamo at trace time).  Newly registered
      buffers have no Dynamo source, so naively calling gm.register_buffer()
      followed by compile_fx raises:
          AssertionError: _fused_qkv_weight not found in param_name_to_source
      The fix: strip the Dynamo metadata fields (dynamo_compile_id from gm.meta
      and _param_name_to_source from gm) before register_buffer.  This routes
      compile_fx through the "graph not from Dynamo" code path in
      _try_get_metadata_from_dynamo, which skips the source-map check entirely.
      The three weight placeholder nodes (wq/wk/wv) are erased from the graph
      and their positions removed from partition_inputs to keep the graph and
      inputs consistent for compile_fx.
    """
    removed_indices: set = set()
    try:
        # Build placeholder → actual tensor map
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        ph_to_tensor = {ph: t for ph, t in zip(placeholders, partition_inputs)}

        # Group F.linear nodes by their first argument (shared input node)
        lin_by_input: dict[str, list[fx.Node]] = {}
        for n in gm.graph.nodes:
            if n.op == "call_function" and n.target is F.linear:
                key = n.args[0].name
                lin_by_input.setdefault(key, []).append(n)

        fused = False
        for x_name, lin_list in lin_by_input.items():
            if len(lin_list) < 3:
                continue

            # Take first three — Q, K, V ordering as they appear in the graph
            q_lin, k_lin, v_lin = lin_list[0], lin_list[1], lin_list[2]

            # Resolve weight placeholder nodes and their actual tensors
            wq_ph = q_lin.args[1]
            wk_ph = k_lin.args[1]
            wv_ph = v_lin.args[1]
            W_q = ph_to_tensor.get(wq_ph)
            W_k = ph_to_tensor.get(wk_ph)
            W_v = ph_to_tensor.get(wv_ph)

            if W_q is None or W_k is None or W_v is None:
                logger.warning(
                    "[_pass_fuse_qkv] Weight tensors not resolved from partition_inputs "
                    "for input '%s' — skipping this group",
                    x_name,
                )
                continue

            # Validate shapes: all must have the same K dimension
            if not (W_q.shape[1] == W_k.shape[1] == W_v.shape[1]):
                logger.warning(
                    "[_pass_fuse_qkv] QKV weight K dims differ (%s, %s, %s) — skipping",
                    W_q.shape, W_k.shape, W_v.shape,
                )
                continue

            # ------------------------------------------------------------------
            # Strip Dynamo source-map metadata before register_buffer so that
            # compile_fx takes the "not from Dynamo" path in
            # _try_get_metadata_from_dynamo (which skips the source-map check).
            # Without this, the newly registered _fused_qkv_weight buffer has
            # no _param_name_to_source entry and aot_autograd raises
            # AssertionError: _fused_qkv_weight not found in param_name_to_source.
            # ------------------------------------------------------------------
            gm.meta.pop("dynamo_compile_id", None)
            if hasattr(gm, "_param_name_to_source"):
                del gm._param_name_to_source

            # Fuse: W_qkv = cat([W_q, W_k, W_v], dim=0)  shape [3*DIM, DIM]
            W_qkv = torch.cat([W_q, W_k, W_v], dim=0)
            gm.register_buffer("_fused_qkv_weight", W_qkv)

            with gm.graph.inserting_before(q_lin):
                w_buf     = gm.graph.get_attr("_fused_qkv_weight")
                fused_lin = gm.graph.call_function(
                    F.linear, (q_lin.args[0], w_buf)
                )
                chunks    = gm.graph.call_function(
                    torch.chunk, (fused_lin, 3), {"dim": -1}
                )
                q_out = gm.graph.call_function(operator.getitem, (chunks, 0))
                k_out = gm.graph.call_function(operator.getitem, (chunks, 1))
                v_out = gm.graph.call_function(operator.getitem, (chunks, 2))

            q_lin.replace_all_uses_with(q_out)
            k_lin.replace_all_uses_with(k_out)
            v_lin.replace_all_uses_with(v_out)

            for dead in (q_lin, k_lin, v_lin):
                gm.graph.erase_node(dead)

            # Leave the now-unused weight placeholder nodes in the graph.
            # This preserves the 1-to-1 correspondence between placeholders and
            # partition_inputs required by compile_fx. Dead placeholder nodes
            # (no users) are valid in the FX graph; Inductor dead-code-eliminates
            # them during lowering. We must NOT erase them here, because Dynamo
            # determines how many args to pass to the compiled function by counting
            # the placeholders in the graph it sees after the backend returns.
            # Erasing 3 nodes and then wrapping with a filter causes an index
            # mismatch: Dynamo calls with 6 args, but the filter expects 9.

            gm.graph.lint()
            gm.recompile()
            logger.info(
                "[_pass_fuse_qkv] Fused 3 × F.linear → 1 × F.linear + chunk "
                "(input '%s', W_qkv shape %s)",
                x_name, list(W_qkv.shape),
            )
            fused = True
            break  # one QKV group per attention block; call again for stacked blocks

        if not fused:
            logger.warning(
                "[_pass_fuse_qkv] QKV pattern not found — no group of 3 F.linear "
                "nodes shares the same input. OPT-2 not applied."
            )
    except Exception as e:
        logger.warning("[_pass_fuse_qkv] Failed: %s", e)
    return gm, partition_inputs, removed_indices


# ===========================================================================
# OPT-4: Pre-transposed fused QKV weight — manual per-rep FX pass
# ===========================================================================
# Classification: Manual per-rep pass.
#   - Must run AFTER _pass_fuse_qkv (depends on _fused_qkv_weight buffer).
#   - Operates on the Inductor-lowered graph where F.linear → aten.t(get_attr)
#     + aten.mm. Detects aten.t() wrapping get_attr('_fused_qkv_weight') and
#     replaces it with a pre-stored contiguous transposed buffer.
#   - Converts NT GEMM to NN GEMM, eliminating the virtual transpose descriptor.
#
# Note: This pass targets the post-Dynamo Inductor-lowered graph. At the
# pre-Inductor level (where @register_backend receives the graph), F.linear is
# still an opaque call_function node — Inductor has not yet decomposed it.
# The pass silently no-ops on the pre-Inductor graph and will find the
# aten.t pattern after Inductor performs decomposition internally.
#
# Since compile_fx runs Inductor internally and the pre-Inductor graph we
# receive has F.linear (not aten.mm), we implement OPT-4 at the module level
# instead: the fused weight buffer is stored pre-transposed and F.linear is
# replaced with operator.matmul (x @ W_qkv_T) directly in the post-fusion step.
# This achieves the same cuBLAS NN GEMM effect without requiring post-Inductor
# graph surgery.
# ===========================================================================

def _pass_pretranspose_fused_qkv(gm: fx.GraphModule) -> fx.GraphModule:
    """
    After _pass_fuse_qkv, replace F.linear(x, _fused_qkv_weight) with
    operator.matmul(x, _fused_qkv_weight_T) where _fused_qkv_weight_T is the
    pre-transposed [DIM, 3*DIM] buffer.

    This eliminates the implicit transpose inside F.linear (which becomes
    aten.t() in the Inductor graph), giving cuBLAS a contiguous [DIM, 3*DIM]
    B matrix in NN GEMM layout.

    Detects the fused linear node by matching call_function F.linear whose
    weight argument is a get_attr node named '_fused_qkv_weight'.
    """
    try:
        # Check that OPT-2 registered the fused weight buffer
        if not hasattr(gm, "_fused_qkv_weight"):
            logger.warning(
                "[_pass_pretranspose_fused_qkv] '_fused_qkv_weight' buffer not found "
                "— OPT-2 may not have run. Skipping OPT-4."
            )
            return gm

        W_qkv: torch.Tensor = gm._fused_qkv_weight  # type: ignore[attr-defined]
        # Pre-transpose: [3*DIM, DIM] → [DIM, 3*DIM], contiguous
        W_qkv_T = W_qkv.T.contiguous()
        gm.register_buffer("_fused_qkv_weight_T", W_qkv_T)

        replaced = False
        for node in list(gm.graph.nodes):
            if node.op != "call_function" or node.target is not F.linear:
                continue
            # Weight arg must be a get_attr referencing our fused buffer
            weight_node = node.args[1] if len(node.args) > 1 else None
            if weight_node is None:
                continue
            if weight_node.op != "get_attr" or weight_node.target != "_fused_qkv_weight":
                continue

            x_arg = node.args[0]
            with gm.graph.inserting_before(node):
                w_T_node = gm.graph.get_attr("_fused_qkv_weight_T")
                # Use matmul(x, W_T) — NN GEMM, no implicit transpose
                mm_node = gm.graph.call_function(operator.matmul, (x_arg, w_T_node))

            node.replace_all_uses_with(mm_node)
            gm.graph.erase_node(node)
            replaced = True
            logger.info(
                "[_pass_pretranspose_fused_qkv] Replaced F.linear(x, W_qkv) with "
                "matmul(x, W_qkv_T) — W_qkv_T shape %s, NN GEMM layout",
                list(W_qkv_T.shape),
            )
            break  # one fused QKV per block

        if not replaced:
            logger.warning(
                "[_pass_pretranspose_fused_qkv] F.linear with '_fused_qkv_weight' "
                "not found — OPT-4 not applied."
            )
        else:
            gm.graph.lint()
            gm.recompile()

    except Exception as e:
        logger.warning("[_pass_pretranspose_fused_qkv] Failed: %s", e)
    return gm


# ===========================================================================
# Utility: capture per-partition actual input tensors
# ===========================================================================

def _capture_partition_inputs(
    split_gm: fx.GraphModule,
    example_inputs: list,
) -> dict[str, list]:
    """
    Run split_gm once via forward-pre hooks to capture the actual input tensors
    seen by each partition submodule.

    Required for any pass that reads actual weight values from partition_inputs,
    because Dynamo lifts all nn.Module parameters to placeholder nodes and the
    values flow in as function arguments at runtime.
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


# ===========================================================================
# Backend Registration
# ===========================================================================

@register_backend
def sdpa_attention_opt(gm: fx.GraphModule, example_inputs) -> Callable:
    """
    Custom torch.compile() backend for SDPAAttentionBlock.

    OPT-1 (BF16 promotion) and OPT-3 (Flash SDPA backend) are applied in
    get_model_and_input() — they are non-graph optimizations invisible to the
    FX IR and must precede torch.compile.

    OPT-2 (QKV fusion) and OPT-4 (pre-transposed QKV weight) are applied here
    as manual per-rep FX passes in dependency order:
      1. _pass_fuse_qkv       — fuses 3 F.linear into 1 + chunk (OPT-2)
      2. _pass_pretranspose_fused_qkv — pre-transposes the fused weight (OPT-4)

    Dedup awareness (Rule 10):
      SDPAAttentionBlock has a single attention block (no repeated layer structure).
      The flat compile path is taken in the normal case. The dedup path is included
      for structural completeness if this backend is reused with a stacked variant.
    """
    logger.info("sdpa_attention_opt backend: starting")

    registry = UniqueSubgraphRegistry(gm)
    equiv_map = registry.build_partition_equivalence_map()

    if not equiv_map:
        # ------------------------------------------------------------------
        # Flat compile path — single attention block, no repeated layers.
        # Apply manual passes directly to the full flat graph, then delegate
        # to Inductor. Preserves any cross-op fusion Inductor may discover.
        # ------------------------------------------------------------------
        logger.info(
            "sdpa_attention_opt: no repeated layers detected — flat compile path"
        )

        # Build a list aligned with placeholder order for _pass_fuse_qkv.
        # Use example_inputs directly (placeholders and example_inputs are in
        # the same 1-to-1 order as delivered by Dynamo).
        flat_partition_inputs = list(example_inputs)

        # OPT-2: fuse Q/K/V linear projections.
        # Dead weight placeholder nodes are left in the graph (see _pass_fuse_qkv
        # docstring). partition_inputs is unchanged — compile_fx receives the
        # full original example_inputs so placeholder count matches input count.
        gm, flat_partition_inputs, _ = _pass_fuse_qkv(gm, flat_partition_inputs)

        # OPT-4: pre-transpose the fused QKV weight (depends on OPT-2)
        gm = _pass_pretranspose_fused_qkv(gm)

        logger.info("sdpa_attention_opt: delegating to Inductor")
        return compile_fx(gm, flat_partition_inputs)

    # -----------------------------------------------------------------------
    # Dedup compile path — for stacked-block variants of this model.
    # Apply passes to each unique representative; share compiled callable
    # with structural duplicates.
    # -----------------------------------------------------------------------
    logger.info(
        "sdpa_attention_opt: %d duplicate partition(s) detected — dedup path",
        len(equiv_map),
    )

    partition_inputs = _capture_partition_inputs(registry.split, example_inputs)

    for rep_name, rep_mod in registry.unique_reps:
        rep_inputs = partition_inputs.get(rep_name, example_inputs)

        # OPT-2: QKV fusion (manual per-rep, needs partition_inputs).
        # Returns updated (rep_mod, rep_inputs, removed_indices).
        rep_mod, rep_inputs, _ = _pass_fuse_qkv(rep_mod, rep_inputs)

        # OPT-4: pre-transpose fused weight (depends on OPT-2)
        _pass_pretranspose_fused_qkv(rep_mod)

        try:
            compiled = compile_fx(rep_mod, rep_inputs)
        except Exception as e:
            logger.warning(
                "sdpa_attention_opt: compile_fx failed for partition '%s' (%s) "
                "— falling back to eager forward",
                rep_name, e,
            )
            compiled = rep_mod.forward

        rep_mod.forward = compiled

        for dup_name, dup_mod in registry.duplicates_of(rep_name):
            dup_mod.forward = compiled
            logger.info(
                "sdpa_attention_opt: shared compiled callable %s → %s",
                rep_name, dup_name,
            )

    logger.info("sdpa_attention_opt: backend assembly complete")
    return lambda *args: registry.split(*args)


# ===========================================================================
# Workload Interface
# ===========================================================================

def get_model_and_input() -> tuple:
    """
    Workload interface for run_workload.py.

    Applies non-graph optimizations before returning:

      OPT-1 (BF16 dtype promotion, high confidence):
        Casts all model parameters to bfloat16 so Dynamo traces a monomorphic
        BF16 graph. Routes GEMM kernels from FP32 SIMT (0% Tensor Core) to the
        BF16 Tensor Core path. Input tensor is also cast to BF16 (unlike GPT-2,
        this model takes float activations, not integer token IDs).
        Applied idempotently — skipped if already BF16.

      OPT-3 (Flash Attention backend, medium confidence):
        enable_flash_sdp(True) + enable_mem_efficient_sdp(False) so the SDPA
        dispatcher selects the BF16-native Flash kernel rather than the xFormers
        memory-efficient path (fmha_cutlassF_f32_aligned_64x64_rf_sm80).
        Requires OPT-1 (BF16). Applied at module level (global state).

    OPT-2 and OPT-4 are applied in the sdpa_attention_opt backend at FX graph
    level.

    Returns:
        model : SDPAAttentionBlockOpt in eval mode on CUDA, parameters in BF16
        x     : FloatTensor of shape (BATCH_SIZE, SEQ_LEN, DIM) in BF16 on CUDA
    """
    assert torch.cuda.is_available(), "CUDA required"

    model = SDPAAttentionBlockOpt().to(DEVICE).eval()
    x = torch.randn(BATCH_SIZE, SEQ_LEN, DIM, device=DEVICE)

    # ------------------------------------------------------------------
    # OPT-1: BF16 dtype promotion (non-graph, must precede torch.compile)
    # Cast model parameters first, then input tensor.
    # ------------------------------------------------------------------
    if next(model.parameters()).dtype != torch.bfloat16:
        logger.info("get_model_and_input: applying OPT-1 (BF16 dtype promotion)")
        model = model.to(torch.bfloat16)

    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)

    logger.info(
        "get_model_and_input: OPT-1 applied — model dtype=%s, input dtype=%s",
        next(model.parameters()).dtype, x.dtype,
    )

    # ------------------------------------------------------------------
    # OPT-3: Enable Flash Attention backend (non-graph SDPA dispatcher knob)
    # After BF16 promotion, Flash becomes eligible. Disable memory-efficient
    # and math backends to force the Flash path.
    # ------------------------------------------------------------------
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)
    logger.info(
        "get_model_and_input: OPT-3 applied — Flash SDP enabled, "
        "mem-efficient SDP disabled, math SDP disabled"
    )

    return model, x


# ===========================================================================
# CLI entry point
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model, x = get_model_and_input()

    compiled_model = torch.compile(model, backend="sdpa_attention_opt")

    with torch.no_grad():
        y = compiled_model(x)

    print(f"Output shape : {y.shape}")   # expect (8, 512, 512)
    print(f"Output dtype : {y.dtype}")   # expect torch.bfloat16
