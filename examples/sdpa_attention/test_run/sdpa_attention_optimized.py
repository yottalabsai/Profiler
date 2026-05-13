"""
sdpa_attention_optimized.py — SDPAAttentionBlock with custom torch.compile() backend.

Implements four operator-level optimizations:
  1. BF16 dtype promotion  — activates Tensor Core path (non-graph, in get_model_and_input)
  2. QKV weight fusion     — 3x [512x512] mm -> 1x [1536x512] mm + chunk (FX pass)
  3. SDPA replacement      — replace manual q@k.T/softmax/@v with F.scaled_dot_product_attention (FX pass)
  4. max-autotune compile  — exhaustive GEMM tile search for non-square shapes (torch.compile mode)

Backend name: sdpa_attention_opt
Registered via @register_backend from torch._dynamo.

To profile with optimizations:
    python nvidia/scripts/run_workload.py \\
        --workload examples/sdpa_attention/test_run/sdpa_attention_optimized.py \\
        --compile-backend sdpa_attention_opt \\
        --output-prefix examples/sdpa_attention/test_run/profile_optimized

To run as a standalone smoke test:
    python examples/sdpa_attention/test_run/sdpa_attention_optimized.py
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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_handler)

# ---------------------------------------------------------------------------
# Model constants (mirror the baseline)
# ---------------------------------------------------------------------------
DEVICE     = "cuda"
BATCH_SIZE = 8
SEQ_LEN    = 512
DIM        = 512
NUM_HEADS  = 8
HEAD_DIM   = DIM // NUM_HEADS  # 64


# ---------------------------------------------------------------------------
# Model definition (verbatim copy from baseline)
# ---------------------------------------------------------------------------

class SDPAAttentionBlock(nn.Module):
    """
    Multi-head self-attention using F.scaled_dot_product_attention.

    Uses separate Q, K, V projections (no bias) so the three linear layers are
    visible as distinct NVTX ranges.  With inductor they may be fused.
    """
    def __init__(self):
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

        # Dispatches to FlashAttention on Ampere / Hopper when available.
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(attn_out)
        out = self.ln_post(out + residual)
        return out


# ---------------------------------------------------------------------------
# FX Pass: OPT-2 — QKV Weight Fusion
# ---------------------------------------------------------------------------

def _pass_fuse_qkv(gm: fx.GraphModule, partition_inputs: list) -> fx.GraphModule:
    """
    Fuse 3 independent F.linear(x, W_q/k/v) calls that share the same input x
    into a single F.linear(x, W_qkv) followed by torch.chunk(3, dim=-1).

    Confidence: MEDIUM — pattern detection required; degrades gracefully if
    the three projections do not share an identical input node in the FX graph.

    Weight access strategy: Dynamo lifts all nn.Module parameters as placeholder
    nodes.  Actual tensor values flow in via partition_inputs; we match
    placeholder-node position to tensor value by zip(placeholders, partition_inputs).

    Classification: Manual per-rep (requires register_buffer + actual tensor values).
    """
    try:
        # Map placeholder node -> actual tensor from partition inputs
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        ph_to_tensor = {ph: t for ph, t in zip(placeholders, partition_inputs)}

        # Group F.linear calls by their first argument (the shared x input node)
        lin_groups: dict[str, list] = {}
        for n in gm.graph.nodes:
            if n.op == "call_function" and n.target is F.linear:
                # args[0] is the activation input; args[1] is the weight
                lin_groups.setdefault(n.args[0].name, []).append(n)

        fused = False
        for x_name, lin_list in lin_groups.items():
            if len(lin_list) < 3:
                continue

            q_lin, k_lin, v_lin = lin_list[0], lin_list[1], lin_list[2]

            # Resolve weight tensors from partition inputs via placeholder nodes
            W_q = ph_to_tensor.get(q_lin.args[1])
            W_k = ph_to_tensor.get(k_lin.args[1])
            W_v = ph_to_tensor.get(v_lin.args[1])

            if W_q is None or W_k is None or W_v is None:
                logger.warning(
                    "[pass_fuse_qkv] Weight tensors not found in partition inputs "
                    "(input '%s') — pass not applied", x_name
                )
                continue

            # Guard: K (input) dimensions must match for concatenation to be valid
            if not (W_q.shape[1] == W_k.shape[1] == W_v.shape[1]):
                logger.warning(
                    "[pass_fuse_qkv] Weight K-dims differ (%d/%d/%d) for input '%s' "
                    "— skipping fusion",
                    W_q.shape[1], W_k.shape[1], W_v.shape[1], x_name,
                )
                continue

            # Materialize fused weight and register as a buffer on the GraphModule
            # Buffer is registered at the current dtype (BF16 after OPT-1 is applied
            # in get_model_and_input before torch.compile sees the graph).
            W_qkv = torch.cat([W_q, W_k, W_v], dim=0)  # [3*N_out, K]
            gm.register_buffer("_fused_qkv_weight", W_qkv)

            # Insert graph nodes: get_attr -> fused linear -> chunk -> 3x getitem
            with gm.graph.inserting_before(q_lin):
                w_buf     = gm.graph.get_attr("_fused_qkv_weight")
                fused_lin = gm.graph.call_function(F.linear, (q_lin.args[0], w_buf))
                chunks    = gm.graph.call_function(torch.chunk, (fused_lin, 3), {"dim": -1})
                q_out     = gm.graph.call_function(operator.getitem, (chunks, 0))
                k_out     = gm.graph.call_function(operator.getitem, (chunks, 1))
                v_out     = gm.graph.call_function(operator.getitem, (chunks, 2))

            # Replace all downstream uses before erasing old nodes (Rule 3)
            q_lin.replace_all_uses_with(q_out)
            k_lin.replace_all_uses_with(k_out)
            v_lin.replace_all_uses_with(v_out)
            for dead in (q_lin, k_lin, v_lin):
                gm.graph.erase_node(dead)

            gm.graph.lint()
            gm.recompile()
            logger.info(
                "[pass_fuse_qkv] Fused 3 F.linear into 1 for input '%s' "
                "(W_qkv shape: %s)", x_name, list(W_qkv.shape)
            )
            fused = True
            break  # one group per unique rep; re-call for deeper stacks

        if not fused:
            logger.warning(
                "[pass_fuse_qkv] QKV pattern (3x F.linear sharing same input) "
                "not found — pass not applied. Possible causes: (a) Inductor already "
                "fused them, (b) weight nodes are not placeholders in this graph."
            )
    except Exception as e:
        logger.warning("[pass_fuse_qkv] Failed with exception: %s", e)
    return gm


# ---------------------------------------------------------------------------
# FX Pass: OPT-3 — SDPA Replacement
# ---------------------------------------------------------------------------

def _pass_replace_sdpa(gm: fx.GraphModule) -> fx.GraphModule:
    """
    Replace the manual attention pattern:
        scores  = operator.matmul(q, k_t)
        scaled  = operator.mul(scores, scale)
        attn    = torch.softmax(scaled, dim)
        out     = operator.matmul(attn, v)
    with a single F.scaled_dot_product_attention(q, k, v) call.

    Confidence: MEDIUM — degrades gracefully if the pattern is not present
    (e.g. model already uses F.sdpa at the Python level, in which case Dynamo
    traces it directly to aten::scaled_dot_product_attention and this pass is
    a no-op).

    NOTE: The model in sdpa_attention.py already uses F.scaled_dot_product_attention
    directly.  Dynamo will trace it to aten::_scaled_dot_product_efficient_attention
    before this pass sees the graph.  Expect the warning log; the pass is included
    so that if an alternate decomposition path is present it will be caught.

    Classification: Manual per-rep (involves call_method "transpose" nodes that
    replace_pattern cannot match).
    """
    try:
        replaced = 0
        for n in list(gm.graph.nodes):
            # Anchor: final matmul — out = attn_weights @ v
            if n.op != "call_function" or n.target is not operator.matmul:
                continue

            attn_node, v_node = n.args[0], n.args[1]

            # attn_weights must be output of torch.softmax
            if not (attn_node.op == "call_function"
                    and attn_node.target is torch.softmax):
                continue

            # softmax input: scaled scores = qk * scale (operator.mul)
            scaled_node = attn_node.args[0]
            if not (scaled_node.op == "call_function"
                    and scaled_node.target is operator.mul):
                continue

            # scores: q @ k_t (operator.matmul)
            qk_node = scaled_node.args[0]
            if not (qk_node.op == "call_function"
                    and qk_node.target is operator.matmul):
                continue

            q_node, k_t_node = qk_node.args[0], qk_node.args[1]

            # Unwrap k.transpose(-2, -1) to recover bare k.
            # F.scaled_dot_product_attention transposes k internally;
            # passing an already-transposed k would compute attention incorrectly.
            if k_t_node.op == "call_method" and k_t_node.target == "transpose":
                k_node = k_t_node.args[0]
            else:
                logger.warning(
                    "[pass_replace_sdpa] k_t node is not call_method transpose "
                    "(op=%s, target=%s) — skipping this attention block",
                    k_t_node.op, k_t_node.target,
                )
                continue

            # Insert F.scaled_dot_product_attention before the old final matmul
            with gm.graph.inserting_before(n):
                sdpa = gm.graph.call_function(
                    F.scaled_dot_product_attention,
                    (q_node, k_node, v_node),
                    {"is_causal": False},
                )

            n.replace_all_uses_with(sdpa)

            # Erase dead nodes in reverse dependency order
            for dead in (n, attn_node, scaled_node, qk_node):
                try:
                    if not dead.users:
                        gm.graph.erase_node(dead)
                except Exception:
                    pass  # node may have already been removed

            replaced += 1

        if replaced > 0:
            gm.graph.lint()
            gm.recompile()
            logger.info(
                "[pass_replace_sdpa] Replaced %d manual attention block(s) with "
                "F.scaled_dot_product_attention", replaced
            )
        else:
            logger.warning(
                "[pass_replace_sdpa] Manual attention pattern (matmul -> mul -> softmax "
                "-> matmul) not found — pass not applied. This is expected if the model "
                "already uses F.scaled_dot_product_attention at the Python level "
                "(Dynamo traces it directly to an aten SDPA op, bypassing this pattern)."
            )
    except Exception as e:
        logger.warning("[pass_replace_sdpa] Failed with exception: %s", e)
    return gm


# ---------------------------------------------------------------------------
# Utility: capture per-partition inputs
# ---------------------------------------------------------------------------

def _capture_partition_inputs(
    split_gm: fx.GraphModule,
    example_inputs: list,
) -> dict[str, list]:
    """
    Run one interpreted forward pass through split_gm, recording the actual
    input tensors received by each partition submodule.

    Dynamo lifts all nn.Module parameters as placeholder nodes; the parameter
    tensors flow in as function arguments.  This captured list is the
    partition_inputs required by _pass_fuse_qkv to resolve weight values.
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
# Backend: OPT-4 max-autotune + dedup-aware structure
# ---------------------------------------------------------------------------

@register_backend
def sdpa_attention_opt(gm: fx.GraphModule, example_inputs) -> Callable:
    """
    Custom torch.compile() backend for SDPAAttentionBlock.

    Pass order (see optimizations.json application_order):
      1. OPT-2: _pass_fuse_qkv    — fuse Q/K/V linear projections
      2. OPT-3: _pass_replace_sdpa — replace manual attention with SDPA (graceful no-op)
      3. OPT-4: compile_fx with max-autotune mode for optimal GEMM tile selection

    Non-graph optimizations (OPT-1 BF16 cast) are applied in get_model_and_input()
    before torch.compile() sees the graph — dtype is not visible in the FX IR.

    Dedup-aware structure: builds a UniqueSubgraphRegistry, applies passes only to
    unique structural representatives, propagates compiled callables to duplicates.
    Falls back to a flat compile when no repeated layer structure is detected.
    """
    from nvidia.operator_profiler.fx import UniqueSubgraphRegistry

    logger.info("sdpa_attention_opt backend: starting")

    # ------------------------------------------------------------------ #
    # Build unique subgraph registry                                       #
    # ------------------------------------------------------------------ #
    registry = UniqueSubgraphRegistry(gm)
    equiv_map = registry.build_partition_equivalence_map()

    if not equiv_map:
        # No repeated layers detected — flat compile path.
        # Applying passes to the full flat graph preserves cross-layer
        # Inductor fusion opportunities.
        logger.info("sdpa_attention_opt: no repeated layers detected, flat compile path")

        # _pass_fuse_qkv requires register_buffer on the GraphModule, which is not
        # compatible with the Dynamo-traced flat graph: Dynamo records each nn.Module
        # parameter as a placeholder node annotated with _dynamo_source, and AOT
        # autograd later validates all buffers against that source map.  A buffer
        # registered *after* tracing has no _dynamo_source entry and causes:
        #   AssertionError: _fused_qkv_weight not found in param_name_to_source
        # This is a fundamental constraint of the flat Dynamo-graph compile path.
        # The dedup path does NOT have this limitation because partition submodules
        # are isolated GraphModules created by the splitter (not the Dynamo trace
        # root), so register_buffer works there.
        #
        # Graceful degradation: skip _pass_fuse_qkv in the flat path.
        # Inductor (max-autotune) will still fuse the three GEMM kernels
        # through its own horizontal-fusion and loop-fusion passes.
        phs = [n for n in gm.graph.nodes if n.op == "placeholder"]
        is_dynamo_graph = any(hasattr(n, "_dynamo_source") for n in phs)
        if is_dynamo_graph:
            logger.warning(
                "[pass_fuse_qkv] Flat Dynamo graph detected — skipping QKV weight "
                "fusion (register_buffer not compatible with param_name_to_source "
                "tracking). Inductor max-autotune will fuse the three GEMMs natively."
            )
        else:
            flat_inputs = list(example_inputs)
            gm = _pass_fuse_qkv(gm, flat_inputs)

        gm = _pass_replace_sdpa(gm)

        logger.info("sdpa_attention_opt: delegating to Inductor (max-autotune)")
        # OPT-4: max-autotune — exhaustive GEMM tile search
        import torch._inductor.config as _ind_cfg
        original_max_autotune = getattr(_ind_cfg, "max_autotune", False)
        _ind_cfg.max_autotune = True
        try:
            compiled = compile_fx(gm, example_inputs)
        finally:
            _ind_cfg.max_autotune = original_max_autotune
        return compiled

    # ------------------------------------------------------------------ #
    # Dedup path: repeated layer structure found                           #
    # ------------------------------------------------------------------ #
    logger.info(
        "sdpa_attention_opt: %d duplicate partition(s) detected, dedup compile path",
        len(equiv_map),
    )

    # Capture per-partition inputs BEFORE applying passes so the placeholder
    # positions still correspond to the unmodified graph's parameter ordering.
    partition_inputs = _capture_partition_inputs(registry.split, list(example_inputs))

    # Apply manual passes to each unique representative, then propagate to duplicates
    for rep_name, rep_mod in registry.unique_reps:
        inputs = partition_inputs.get(rep_name, list(example_inputs))

        # OPT-2: QKV fusion (manual per-rep — needs register_buffer + tensor values)
        _pass_fuse_qkv(rep_mod, inputs)

        # OPT-3: SDPA replacement (manual per-rep — involves call_method transpose)
        _pass_replace_sdpa(rep_mod)

        # Propagate graph edits to structural duplicates
        for _, dup_mod in registry.duplicates_of(rep_name):
            dup_inputs = partition_inputs.get(_, list(example_inputs))
            _pass_fuse_qkv(dup_mod, dup_inputs)
            _pass_replace_sdpa(dup_mod)

    # OPT-4: Compile each unique representative with max-autotune;
    # share the compiled callable with all structural duplicates.
    import torch._inductor.config as _ind_cfg
    original_max_autotune = getattr(_ind_cfg, "max_autotune", False)
    _ind_cfg.max_autotune = True
    try:
        for rep_name, rep_mod in registry.unique_reps:
            rep_inputs = partition_inputs.get(rep_name, list(example_inputs))
            compiled_fn = compile_fx(rep_mod, rep_inputs)
            rep_mod.forward = compiled_fn
            for dup_name, dup_mod in registry.duplicates_of(rep_name):
                dup_mod.forward = compiled_fn
                logger.info(
                    "sdpa_attention_opt: partition '%s' shares compiled callable "
                    "with representative '%s'", dup_name, rep_name
                )
    finally:
        _ind_cfg.max_autotune = original_max_autotune

    logger.info("sdpa_attention_opt: all partitions compiled")
    return lambda *args: registry.split(*args)


# ---------------------------------------------------------------------------
# Workload Interface
# ---------------------------------------------------------------------------

def get_model_and_input() -> tuple:
    """
    Workload interface for run_workload.py.

    Returns:
        model  — SDPAAttentionBlock on CUDA, weights cast to torch.bfloat16
        x      — input tensor [8, 512, 512] on CUDA, dtype torch.bfloat16

    Optimizations applied here (OPT-1: BF16 dtype promotion):
        model = model.to(torch.bfloat16)
        x     = x.to(torch.bfloat16)

    BF16 must be applied BEFORE torch.compile(). Dynamo traces at the dtype
    present at compile time. Applying dtype inside the graph or after compile
    has no effect on cuBLAS kernel selection.

    Effect: routes cuBLAS from ampere_sgemm_128x64_tn (FP32 SIMT,
    smsp__pipe_tensor_cycles_active=0.0) to sm80_xmma_gemm_bf16bf16_tn
    (HMMA Tensor Core path, ~16x higher theoretical throughput on A100).
    SDPA transitions from fmha_cutlassF_f32 (168 regs/thread, 4.7 MB local
    spills, 15.9% occupancy) to the BF16 FlashAttention variant.
    """
    assert torch.cuda.is_available(), "CUDA required"

    model = SDPAAttentionBlock().to(DEVICE).eval()
    x     = torch.randn(BATCH_SIZE, SEQ_LEN, DIM, device=DEVICE)

    # OPT-1: BF16 dtype promotion — non-graph, applied before torch.compile()
    # Check current state so the function is idempotent
    if next(model.parameters()).dtype != torch.bfloat16:
        model = model.to(torch.bfloat16)
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)

    return model, x


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model, x = get_model_and_input()
    print(f"Model dtype : {next(model.parameters()).dtype}")
    print(f"Input shape : {x.shape}, dtype: {x.dtype}")
    with torch.no_grad():
        out = model(x)
    print(f"Output shape: {out.shape}, dtype: {out.dtype}")
    assert not torch.isnan(out).any(), "Output contains NaN"
    assert not torch.isinf(out).any(), "Output contains Inf"
    print("Smoke test passed")
