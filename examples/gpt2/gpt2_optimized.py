"""
gpt2_optimized.py — GPT-2 small (117M) with custom torch.compile() backend.

Implements five operator-level optimizations derived from profiling feedback:
  1. TF32 global flag (OPT-5)   — defense-in-depth for any residual FP32 GEMMs
  2. BF16 cast (OPT-1)          — activates A100 Tensor Cores on all 792 GEMMs
  3. BF16 SDPA dispatch (OPT-3) — cascades from OPT-1; Flash-Attention for attention kernels
  4. QKV fusion check (OPT-4)   — FX pass; safe no-op if HF c_attn already fuses Q/K/V
  5. max-autotune compile (OPT-2) — autotuned tile selection after BF16 cast

Optimizations 1 (TF32), 2 (BF16 cast), and the SDPA backend flags are applied
in get_model_and_input(). Optimization 4 (QKV check) runs inside the custom
FX backend before delegating to Inductor. Optimization 5 is expressed through
the torch.compile() mode argument in get_model_and_input().

Hardware target: NVIDIA A100-SXM4-80GB (Ampere, sm_major=8)
Expected combined speedup: 3x-6x end-to-end wall-clock vs FP32 baseline.

To profile with optimizations:
    python scripts/run_workload.py examples/gpt2/gpt2_optimized.py \\
        --warmup-iters 3 --measure-iters 10

Note: First compile with max-autotune is slow (~60-180 s for autotuning).
      Subsequent runs use the FX graph cache (inductor.config.fx_graph_cache=True).

Requires:
    pip install transformers
"""
from __future__ import annotations

import logging
import operator
from collections import defaultdict
from typing import Callable

import torch
import torch.fx as fx
import torch.nn as nn
from torch._dynamo import register_backend
from torch._inductor.compile_fx import compile_fx  # function, not module — see Rule 1

from nvidia.operator_profiler.fx import UniqueSubgraphRegistry, FxPassRunner

# ---------------------------------------------------------------------------
# OPT-5: TF32 global flags — set at module load time before any model creation.
# This is a no-op on hardware that defaults to allow_tf32=True (PyTorch >= 1.12),
# but explicit setting ensures correctness if the flag was toggled elsewhere.
# ---------------------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---------------------------------------------------------------------------
# OPT-3 (partial): enforce SDPA backend selection explicitly so the
# FlashAttention kernel is always preferred over the cutlass efficient-attention
# kernel when inputs are BF16.
# ---------------------------------------------------------------------------
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(False)

# ---------------------------------------------------------------------------
# Inductor cache: avoid re-autotuning on subsequent runs with identical shapes.
# ---------------------------------------------------------------------------
import torch._inductor.config as _inductor_config
_inductor_config.fx_graph_cache = True

DEVICE   = "cuda"
BATCH    = 4
SEQ_LEN  = 128
MODEL_ID = "gpt2"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ============================================================================
# Pass utilities
# ============================================================================

def _capture_partition_inputs(
    split_gm: fx.GraphModule,
    example_inputs: list,
) -> dict[str, list]:
    """
    Capture the real input tensors seen by each partition submodule by running
    a single forward pass with forward-pre hooks.

    Returns a dict mapping submodule name → list of input tensors so that each
    partition can be compiled with its actual input shapes rather than the
    top-level example_inputs.
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
# OPT-4: QKV fusion FX pass
#
# Confidence: MEDIUM — HuggingFace GPT2Attention uses a single Conv1D c_attn
# that projects to 3*n_embd in one call, so the three Q/K/V mm nodes that
# this pass targets may never appear in the Inductor-traced graph.  The pass
# exits cleanly as a no-op in that case; it is included for correctness and
# documentation purposes, and for models that split Q/K/V projections.
#
# Pattern: three aten.mm nodes that share the same input activation tensor,
#   each multiplied by a different weight matrix of identical shape.
# Transformation: concatenate the three weights along the output (N) dimension
#   to produce a single fused weight [K, 3N], issue one mm, then chunk(3, -1).
# Prerequisite: model must already be BF16 (OPT-1) so the fused buffer dtype
#   matches the runtime mm input dtype.
# ============================================================================

def _pass_fuse_qkv(gm: fx.GraphModule) -> fx.GraphModule:
    """
    Fuse 3x mm(x, W_q/k/v) sharing the same input into mm(x, W_qkv) + chunk.

    Weight detection handles two common Inductor graph shapes:
      - bare get_attr node:        mm(x, get_attr('weight'))
      - transposed get_attr node:  mm(x, t(get_attr('weight')))

    The pass is a safe no-op when fewer than 3 mm nodes share an input
    (e.g. when HuggingFace c_attn already fuses the projections).
    """
    try:
        mm_by_input: dict = defaultdict(list)
        for node in list(gm.graph.nodes):
            if (node.op == "call_function"
                    and node.target == torch.ops.aten.mm.default):
                input_node = node.args[0]
                mm_by_input[input_node].append(node)

        fused_count = 0
        for input_node, mm_nodes in mm_by_input.items():
            if len(mm_nodes) != 3:
                continue

            # Extract weight tensor for each mm, looking through aten.t() wrappers.
            weights = []
            attr_nodes = []
            for mm_node in mm_nodes:
                w_node = mm_node.args[1]
                # Case 1: direct get_attr (Inductor may not emit t() for some weights)
                if w_node.op == "get_attr":
                    attr_node = w_node
                # Case 2: t(get_attr(...)) — the most common Inductor pattern
                elif (w_node.op == "call_function"
                        and w_node.target == torch.ops.aten.t.default
                        and w_node.args[0].op == "get_attr"):
                    attr_node = w_node.args[0]
                else:
                    attr_node = None

                if attr_node is None:
                    break
                try:
                    W = gm.get_parameter(attr_node.target)
                except AttributeError:
                    # May be a buffer rather than a parameter — try get_buffer path
                    try:
                        W = gm.get_buffer(attr_node.target)
                    except AttributeError:
                        break
                weights.append(W)
                attr_nodes.append(attr_node)

            if len(weights) != 3:
                # Could not resolve all three weights — skip this group
                logger.warning(
                    "[_pass_fuse_qkv] Could not resolve weights for input %s — skipping",
                    input_node,
                )
                continue

            # Require identical weight shapes so the fused output can be evenly chunked.
            shapes = [tuple(w.shape) for w in weights]
            if len(set(shapes)) != 1:
                logger.warning(
                    "[_pass_fuse_qkv] Weight shapes differ %s — skipping", shapes
                )
                continue

            # Concatenate weights along the N (output) dimension: [K, N] -> [K, 3N].
            # If weights were transposed in the graph they are stored as [N, K] in the
            # parameter dict; in that case cat along dim=0 (which becomes the output
            # dimension after the implicit transpose in the mm call).
            # Detect orientation by comparing stored shape against mm input shape:
            #   - [seq*batch, K] @ [K, N]  => weight stored as [K, N], cat dim=1
            #   - [seq*batch, K] @ [N, K]^T => weight stored as [N, K], cat dim=0
            K_input = input_node.meta.get("tensor_meta", None)
            W0 = weights[0]
            # Heuristic: if the weight's second dim matches the input's last dim,
            # it is already in [K, N] form and we cat along dim=1 (output dim).
            # Otherwise assume [N, K] stored form (t() was applied in the graph)
            # and cat along dim=0.
            cat_dim = 1 if (K_input is None or W0.shape[0] != W0.shape[1]) else 0
            # Safer default: since Inductor typically stores [N, K] and applies t(),
            # use dim=0 when the weight node was accessed through aten.t().
            weight_through_t = [
                (mm_node.args[1].op == "call_function"
                 and mm_node.args[1].target == torch.ops.aten.t.default)
                for mm_node in mm_nodes
            ]
            if all(weight_through_t):
                cat_dim = 0  # stored as [N, K]; cat along N => [3N, K]; then t() => [K, 3N]
            else:
                cat_dim = 1  # stored as [K, N]; cat along output dim

            W_fused = torch.cat(weights, dim=cat_dim).contiguous()
            buf_name = f"_fused_qkv_weight_{id(input_node)}"
            gm.register_buffer(buf_name, W_fused)

            # Insert the fused weight get_attr and the single mm before the first mm.
            first_mm = mm_nodes[0]
            with gm.graph.inserting_before(first_mm):
                get_fused = gm.graph.get_attr(buf_name)

                if all(weight_through_t):
                    # W_fused is [3N, K]; we still need t() so mm(x, t(W_fused)) works.
                    t_fused = gm.graph.call_function(
                        torch.ops.aten.t.default, args=(get_fused,)
                    )
                    fused_mm = gm.graph.call_function(
                        torch.ops.aten.mm.default,
                        args=(input_node, t_fused),
                    )
                else:
                    # W_fused is [K, 3N]; direct mm.
                    fused_mm = gm.graph.call_function(
                        torch.ops.aten.mm.default,
                        args=(input_node, get_fused),
                    )

                # chunk(fused_mm, 3, dim=-1)
                chunks = gm.graph.call_function(
                    torch.ops.aten.chunk.default,
                    args=(fused_mm, 3, -1),
                )
                q_out = gm.graph.call_function(operator.getitem, args=(chunks, 0))
                k_out = gm.graph.call_function(operator.getitem, args=(chunks, 1))
                v_out = gm.graph.call_function(operator.getitem, args=(chunks, 2))

            # Replace uses then erase original mm nodes (snapshot already taken above).
            mm_nodes[0].replace_all_uses_with(q_out)
            mm_nodes[1].replace_all_uses_with(k_out)
            mm_nodes[2].replace_all_uses_with(v_out)
            for mm_node in reversed(mm_nodes):
                gm.graph.erase_node(mm_node)

            fused_count += 1
            logger.info("[_pass_fuse_qkv] Fused QKV group for input node %s", input_node)

        if fused_count == 0:
            logger.info(
                "[_pass_fuse_qkv] No 3-mm groups found — HF c_attn already fuses "
                "Q/K/V projections (expected no-op for standard GPT-2)"
            )
        else:
            gm.graph.lint()
            gm.recompile()
            logger.info("[_pass_fuse_qkv] Applied %d QKV fusion(s)", fused_count)

    except Exception as exc:
        logger.warning("[_pass_fuse_qkv] Failed: %s — graph left unmodified", exc)

    return gm


# ============================================================================
# Backend Registration
# ============================================================================

@register_backend
def gpt2_backend(gm: fx.GraphModule, example_inputs, **_kwargs) -> Callable:
    """
    Custom torch.compile() backend for the GPT-2 optimized workload.

    Dedup path (12-layer transformer): applies OPT-4 (QKV fusion check) only
    to structurally unique layer representatives, then propagates to all 11
    structural duplicates before compiling each representative with Inductor.

    Flat path (fallback): applies the same pass to the full flat graph and
    delegates directly to Inductor — preserves Inductor's cross-layer fusion
    opportunities when no duplicate partitions are detected.

    Pass order (inside backend):
      1. _pass_fuse_qkv  — MEDIUM confidence, safe no-op for HF GPT-2

    Non-graph optimizations (BF16 cast, SDPA flags, TF32) are applied in
    get_model_and_input() before torch.compile() is called, so Inductor traces
    the already-optimized dtype graph.
    """
    logger.info("gpt2_backend: starting")
    registry = UniqueSubgraphRegistry(gm)
    equiv_map = registry.build_partition_equivalence_map()

    if not equiv_map:
        # No repeated layers detected — flat compile path.
        # Applying passes to the full flat graph lets Inductor see cross-layer
        # fusion opportunities that the split path would break.
        logger.info("gpt2_backend: no repeated layers, flat compile path")
        gm = _pass_fuse_qkv(gm)
        logger.info("gpt2_backend: delegating to Inductor (flat)")
        return compile_fx(gm, example_inputs)

    logger.info(
        "gpt2_backend: %d duplicate partition(s) detected, using dedup path",
        len(equiv_map),
    )
    runner = FxPassRunner(registry)

    # QKV fusion is manual (uses register_buffer) — apply per-rep loop rather
    # than runner.apply_pass() (which is for pure replace_pattern passes).
    for rep_name, rep_mod in registry.unique_reps:
        _pass_fuse_qkv(rep_mod)
        for _, dup_mod in registry.duplicates_of(rep_name):
            _pass_fuse_qkv(dup_mod)

    # Compile each unique representative with its actual partition inputs;
    # share the resulting callable with all structural duplicates.
    partition_inputs = _capture_partition_inputs(registry.split, example_inputs)
    for rep_name, rep_mod in registry.unique_reps:
        compiled = compile_fx(
            rep_mod,
            partition_inputs.get(rep_name, example_inputs),
        )
        rep_mod.forward = compiled
        for _, dup_mod in registry.duplicates_of(rep_name):
            dup_mod.forward = compiled

    logger.info("gpt2_backend: all passes applied, returning dedup-compiled graph")
    return lambda *args: registry.split(*args)


# ============================================================================
# GPT-2 model wrapper (identical to baseline — preserved for self-containment)
# ============================================================================

class GPT2Wrapper(nn.Module):
    """Thin wrapper so model(input_ids) returns the last hidden state tensor."""

    def __init__(self, hf_model: nn.Module):
        super().__init__()
        self.model = hf_model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids).last_hidden_state


# ============================================================================
# Workload Interface
# ============================================================================

def get_model_and_input() -> tuple:
    """
    Workload interface for run_workload.py.

    Returns:
      model   — GPT2Wrapper compiled with gpt2_backend, weights in BF16, on CUDA
      input_ids — int64 token ids [BATCH, SEQ_LEN] on CUDA

    Non-graph optimizations applied here (before torch.compile so Inductor
    traces the already-optimized dtype graph):
      OPT-5: TF32 flags set at module load time (see top of file)
      OPT-1: model.to(torch.bfloat16) — unlocks A100 Tensor Cores for all GEMMs
      OPT-3: enable_flash_sdp(True) at module load time; BF16 input auto-dispatches
             to FlashAttention kernel (fmha_cutlassF_bf16 vs f32 variant)
      OPT-2: torch.compile(mode='max-autotune', fullgraph=True, backend='gpt2_backend')

    Note on input dtype:
      input_ids stays int64 — embedding lookup (aten::embedding) is dtype-agnostic
      on the weight matrix; the lookup is an integer gather, not a GEMM.
      The first float operation after the embedding is a LayerNorm whose input
      is the BF16 embedding output (embedding output dtype follows weight dtype).
    """
    assert torch.cuda.is_available(), "CUDA required"
    assert torch.cuda.get_device_capability()[0] >= 8, (
        "BF16 Tensor Core support requires sm_major >= 8 (Ampere or later). "
        f"Detected: sm{torch.cuda.get_device_capability()}"
    )

    from transformers import GPT2Model  # deferred import; 500 MB download on first call

    hf_model = GPT2Model.from_pretrained(MODEL_ID)
    model = GPT2Wrapper(hf_model).to(DEVICE).eval()

    # OPT-1: BF16 cast — must precede torch.compile so Inductor traces BF16 weight nodes.
    # Check before applying in case a future caller pre-casts the model.
    if next(model.parameters()).dtype != torch.bfloat16:
        model = model.to(torch.bfloat16)

    # Random token ids in [0, vocab_size) — dtype stays int64.
    vocab_size = hf_model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (BATCH, SEQ_LEN), device=DEVICE)

    # OPT-2: max-autotune with fullgraph — applied after BF16 cast.
    # First run will autotune ~60-180 s; subsequent runs hit the FX graph cache.
    model = torch.compile(
        model,
        backend="gpt2_backend",
        mode="max-autotune",
        fullgraph=True,
    )

    return model, input_ids


if __name__ == "__main__":
    model, input_ids = get_model_and_input()
    with torch.no_grad():
        y = model(input_ids)
    print(f"Output shape: {y.shape}")   # expect (4, 128, 768)
    print(f"Output dtype: {y.dtype}")   # expect torch.bfloat16
