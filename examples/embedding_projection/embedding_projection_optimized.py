"""
embedding_projection_optimized.py — Custom torch.compile() backend for the
EmbeddingProjection workload (embedding lookup + 2-layer MLP head + 32k logit
projection).

ALL graph passes run at the Aten IR level, inside ``_aten_inner_compile``, which
``compile_fx`` invokes (as ``inner_compile``) with the fully decomposed Aten
forward graph after AOTAutograd has run. At that level every nn.Linear is an
aten.addmm.default (proj1/proj2, biased) or aten.mm.default (logits, bias-free),
the LayerNorm is aten.native_layer_norm.default, and the gather is
aten.embedding.default.

Profiling-guided optimizations (from optimizations.json):

  OPT-1 (HIGH, priority 1) — dtype_promotion  [primary]
      Every matmul (mm + addmm) runs on the FP32 SIMT FFMA path with the Tensor
      Cores idle (ncu: tensor_core_active_pct=0.0, thread/inst ratio=32,
      211-212 regs/thread, occupancy pinned at 16.66%). Cast the FP32 operands
      of each aten.mm.default / aten.addmm.default to bfloat16 and cast the GEMM
      result back to float32. This switches cuBLAS dispatch onto the Blackwell
      HMMA/wgmma Tensor-Core GEMM. The four 512x32000 logit-head GEMMs are the
      single largest cost (~54% of attributed time), so this is the dominant win.
      Prerequisite for OPT-3. Target: every aten.mm / aten.addmm node.

  OPT-2 (MEDIUM, priority 2) — TF32  [mutually-exclusive alternative, NOT applied]
      Same root cause as OPT-1, the lower-risk numerics-preserving alternative
      (route FP32 matmuls to the TF32 Tensor-Core path via the matmul-precision
      toggle). OPT-1 (BF16) and OPT-2 (TF32) are mutually exclusive — we apply
      OPT-1 as the primary and leave OPT-2 as a documented stub that only logs.

  OPT-3 (LOW, priority 3) — memory_layout (logit-head bf16 downcast)  [depends OPT-1]
      The 512x32000 logit-head aten.mm.default writes ~991 MB fp32 per launch.
      After OPT-1 moves the multiply to Tensor Cores, keep the logit output in
      bf16 (skip OPT-1's fp32 back-cast on the head) to halve that write to
      ~495 MB. Identified by output dim == VOCAB_SIZE (32000) from
      node.meta['val'].shape. Runs AFTER OPT-1 — it deletes the fp32 back-cast
      OPT-1 inserted on the logit mm. Confidence LOW: realized win depends on how
      much of the write is exposed after Tensor-Core engagement.

Backend registration name: embedding_projection_opt
Prerequisite DAG: OPT-1 -> OPT-3.  (OPT-2 is an alternative to OPT-1, not additive.)
"""
from __future__ import annotations

import logging
from typing import Callable

import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
from torch._dynamo import register_backend

# compile_fx is the callable (NEVER `from torch._inductor import compile_fx`,
# which imports the module). compile_fx_inner is the post-AOTAutograd hook that
# receives the fully decomposed Aten IR graph; passing it as inner_compile lets
# our Aten-IR passes run on that graph and then delegate to the real inner
# compiler (Aten -> Triton).
from torch._inductor.compile_fx import compile_fx, compile_fx_inner

from nvidia.operator_profiler.fx import UniqueSubgraphRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend name constant (used for torch.compile(backend=...) and re-capture)
# ---------------------------------------------------------------------------
BACKEND_NAME = "embedding_projection_opt"

# Vocabulary dim that identifies the logit-head GEMM output (see OPT-3).
VOCAB_SIZE = 32_000

# Aten op targets used across passes
_MM = torch.ops.aten.mm.default
_ADDMM = torch.ops.aten.addmm.default
_GEMM_TARGETS = frozenset({_MM, _ADDMM})

# prims.convert_element_type is the canonical Inductor dtype-cast primitive. We
# use it instead of aten._to_copy.default (which the optimizations.json fx_steps
# name as aten.to.dtype) because on torch 2.11 _to_copy carries BOTH a fallback
# and a decomp registration; inserting it post-AOTAutograd makes Inductor raise
# "both a fallback and a decomp for same op". convert_element_type lowers cleanly
# to a Triton cast and is the form Inductor itself emits for dtype conversions,
# so OPT-1's casts fuse into the producing/consuming Triton epilogues.
_CONVERT = torch.ops.prims.convert_element_type.default


# ---------------------------------------------------------------------------
# OPT-1 (HIGH) — dtype_promotion: BF16 casts around aten.mm / aten.addmm
# ---------------------------------------------------------------------------

def _pass_gemm_bf16_casts(gm: fx.GraphModule) -> fx.GraphModule:
    """
    OPT-1 — force every tensor operand of every aten.mm.default and
    aten.addmm.default node to bfloat16, then cast the result back to float32.

    Confidence HIGH: assume the pattern exists; an exception is a real error.

    Robustness: aten rejects a GEMM whose operands disagree in dtype
    ("expected mat1 and mat2 to have the same dtype"). We therefore do NOT guard
    on float32 — instead we guarantee that *both* matmul operands (and, for
    addmm, the bias) end up bf16, regardless of each operand's current dtype.
    For every tensor operand we insert a convert_element_type(bf16) unless that
    operand is *already known* to be bf16 (meta dtype == torch.bfloat16), in
    which case the cast is skipped as a no-op. When meta is missing/unknown we
    cast defensively — convert_element_type(bf16) on a bf16 tensor is harmless.
    Non-tensor args (none expected on mm/addmm, but e.g. ints) are left as-is.

    The output is restored to float32 so downstream ops (the gelu between proj1
    and proj2, and the final logits) stay dtype-consistent; Inductor fuses the
    convert_element_type casts into neighbouring Triton kernels, hiding the cost.
    OPT-3 later removes the back-cast specifically on the 32000-wide logit mm.
    """
    try:
        matched = False
        graph = gm.graph
        for node in list(graph.nodes):
            if not (node.op == "call_function" and node.target in _GEMM_TARGETS):
                continue
            matched = True

            # Force every tensor operand to bfloat16 before the GEMM so both
            # matmul inputs (and the addmm bias) share one dtype. Skip the cast
            # only when the operand is already provably bf16.
            with graph.inserting_before(node):
                cast_args = []
                for a in node.args:
                    if isinstance(a, fx.Node):
                        val = a.meta.get("val")
                        cur_dtype = getattr(val, "dtype", None)
                        if cur_dtype == torch.bfloat16:
                            # Already bf16 — no cast needed.
                            cast_args.append(a)
                        else:
                            c = graph.call_function(_CONVERT, (a, torch.bfloat16))
                            cast_args.append(c)
                    else:
                        cast_args.append(a)
            node.args = tuple(cast_args)

            # Restore float32 on the output for downstream dtype consistency.
            with graph.inserting_after(node):
                back = graph.call_function(_CONVERT, (node, torch.float32))
            # Re-point all existing users to the float32 cast (but not the cast
            # itself, which consumes `node`).
            node.replace_all_uses_with(back, delete_user_cb=lambda u: u is not back)

        if not matched:
            logger.warning(
                "[_pass_gemm_bf16_casts] No aten.mm / aten.addmm nodes found "
                "— pass not applied"
            )
            return gm

        graph.lint()
        gm.recompile()
        logger.info(
            "[_pass_gemm_bf16_casts] Applied BF16 casts to aten.mm / aten.addmm "
            "operands [OPT-1, Aten IR]"
        )
    except Exception as exc:
        logger.warning("[_pass_gemm_bf16_casts] Failed: %s", exc)

    return gm


# ---------------------------------------------------------------------------
# OPT-2 (MEDIUM) — TF32: mutually-exclusive alternative to OPT-1 (NOT applied)
# ---------------------------------------------------------------------------

def _pass_tf32_alternative_stub(gm: fx.GraphModule) -> fx.GraphModule:
    """
    OPT-2 — TF32 Tensor-Core dispatch for FP32 matmuls.

    Stub (detect/log only, no transformation). OPT-2 is a backend-precision
    toggle (torch.set_float32_matmul_precision('high') /
    torch.backends.cuda.matmul.allow_tf32 = True), NOT a graph transform — it
    changes which cuBLAS kernel Inductor's GEMM lowering selects, with no Aten
    node change. It is mutually exclusive with OPT-1 (BF16), which we apply as
    the primary, higher-throughput optimization. Enabling TF32 here would be
    redundant once operands are bf16 (and would mask OPT-1's numerics), so this
    pass only reports the GEMM count it would have toggled and returns gm
    unchanged. To deploy OPT-2 INSTEAD of OPT-1: remove the OPT-1 pass from
    _aten_inner_compile and uncomment the two torch.backends lines below in
    get_model_and_input().
    """
    try:
        gemm_count = sum(
            1 for n in gm.graph.nodes
            if n.op == "call_function" and n.target in _GEMM_TARGETS
        )
        logger.info(
            "[_pass_tf32_alternative_stub] OPT-2 (TF32) is a mutually-exclusive "
            "alternative to OPT-1 and is NOT applied; %d GEMM node(s) would be "
            "affected by the matmul-precision toggle [OPT-2, stub]",
            gemm_count,
        )
    except Exception as exc:
        logger.warning("[_pass_tf32_alternative_stub] Failed: %s", exc)
    return gm


# ---------------------------------------------------------------------------
# OPT-3 (LOW) — memory_layout: keep the 32000-wide logit-head output in bf16
# ---------------------------------------------------------------------------

def _pass_logit_bf16_downcast(gm: fx.GraphModule) -> fx.GraphModule:
    """
    OPT-3 — for the logit-head aten.mm.default (output dim == VOCAB_SIZE), remove
    the fp32 back-cast OPT-1 inserted so the ~991 MB fp32 logit write collapses
    to a ~495 MB bf16 write. Consumers (the model output / downstream loss /
    argmax) read bf16 logits.

    Depends on OPT-1 (must run AFTER it — it deletes the convert_element_type ->
    float32 node OPT-1 placed on the logit mm's output edge).

    Confidence LOW: detect the logit mm by output dim from node.meta['val'].shape;
    if no 32000-wide mm or no fp32 back-cast is found, degrade to a no-op. The
    realized time win depends on how much of the DRAM write is exposed once the
    multiply is on the Tensor Cores, which the fp32 profile cannot measure.
    """
    try:
        matched = False
        graph = gm.graph
        for node in list(graph.nodes):
            if not (node.op == "call_function" and node.target is _MM):
                continue
            val = node.meta.get("val")
            if val is None or not hasattr(val, "shape"):
                continue
            if int(val.shape[-1]) != VOCAB_SIZE:
                continue

            # OPT-1 wraps the mm output in convert_element_type(node, float32).
            # Find that back-cast among the mm's users and splice it out so the
            # bf16 mm result flows straight to the consumer.
            for user in list(node.users):
                if (user.op == "call_function"
                        and user.target is _CONVERT
                        and len(user.args) >= 2
                        and user.args[1] == torch.float32):
                    user.replace_all_uses_with(node)
                    graph.erase_node(user)
                    matched = True

        if not matched:
            logger.warning(
                "[_pass_logit_bf16_downcast] No %d-wide logit mm with an OPT-1 "
                "fp32 back-cast found — pass not applied (did OPT-1 run?)",
                VOCAB_SIZE,
            )
            return gm

        graph.eliminate_dead_code()
        graph.lint()
        gm.recompile()
        logger.info(
            "[_pass_logit_bf16_downcast] Kept logit-head (dim=%d) output in bf16, "
            "removed fp32 back-cast [OPT-3, Aten IR]",
            VOCAB_SIZE,
        )
    except Exception as exc:
        logger.warning("[_pass_logit_bf16_downcast] Failed: %s", exc)

    return gm


# ---------------------------------------------------------------------------
# Aten IR inner compiler — all graph passes run here, in prerequisite order
# ---------------------------------------------------------------------------

def _aten_inner_compile(gm: fx.GraphModule, example_inputs, **kwargs) -> Callable:
    """
    Inductor ``inner_compile`` hook. ``compile_fx`` calls this with the fully
    decomposed **Aten IR** forward graph (after AOTAutograd has run), where every
    nn.Linear is an aten.addmm/aten.mm and the LayerNorm is native_layer_norm. We
    run the passes here, in prerequisite-DAG order, then delegate to the real
    ``compile_fx_inner`` (Aten -> Triton).

    Order (from optimizations.json prerequisite_for[]):
      OPT-1 (BF16 dtype_promotion)   — first; sets the bf16 dtype OPT-3 depends on.
      OPT-2 (TF32)                   — stub/log only; mutually exclusive with OPT-1.
      OPT-3 (logit bf16 downcast)    — last; removes the OPT-1 back-cast on the mm.

    Using inner_compile (rather than re-wrapping the functional graph with a
    second aot_autograd) avoids a double-AOTAutograd input-flattening bug on
    torch 2.11 where a non-tensor input reaches Inductor's copy_misaligned_inputs.
    """
    gm = _pass_gemm_bf16_casts(gm)          # OPT-1 (primary, HIGH)
    gm = _pass_tf32_alternative_stub(gm)    # OPT-2 (stub; alternative to OPT-1)
    gm = _pass_logit_bf16_downcast(gm)      # OPT-3 (depends on OPT-1, LOW)
    return compile_fx_inner(gm, example_inputs, **kwargs)


def _compile_with_aten_passes(gm: fx.GraphModule, example_inputs) -> Callable:
    """Compile a (sub)graph through Inductor with the Aten-IR passes installed."""
    return compile_fx(gm, example_inputs, inner_compile=_aten_inner_compile)


# ---------------------------------------------------------------------------
# Utility: capture per-partition input tensors for the dedup compile path
# ---------------------------------------------------------------------------

def _capture_partition_inputs(
    split_gm: fx.GraphModule,
    example_inputs: list,
) -> dict[str, list]:
    """Run split_gm once under no_grad to capture each partition's input tensors."""
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
# Backend: embedding_projection_opt
# ---------------------------------------------------------------------------

@register_backend
def embedding_projection_opt(gm: fx.GraphModule, example_inputs) -> Callable:
    """
    Custom torch.compile() backend for EmbeddingProjection.

    Structure (dedup-aware per the standard backend pattern):
      - UniqueSubgraphRegistry splits the functional FX graph into per-layer
        partitions and groups them by structural signature. EmbeddingProjection
        has NO repeated structure (embedding + ln + 2 MLP layers + logit head are
        all distinct), so build_partition_equivalence_map() returns empty and we
        take the flat compile path — preserving cross-op Inductor fusion of the
        BF16 casts into neighbouring kernels.
      - On the flat path the whole graph is compiled through
        compile_fx(..., inner_compile=_aten_inner_compile); OPT-1/OPT-2/OPT-3 run
        inside _aten_inner_compile at the Aten IR level (post-AOTAutograd).
      - The dedup branch is retained for protocol compliance and degrades safely
        to the flat path on any per-partition compile failure.
    """
    logger.info(
        "embedding_projection_opt backend: starting "
        "(Aten IR passes via compile_fx inner_compile)"
    )

    registry = UniqueSubgraphRegistry(gm)
    equiv_map = registry.build_partition_equivalence_map()

    if not equiv_map:
        logger.info(
            "embedding_projection_opt: no repeated layers — flat compile path"
        )
        return _compile_with_aten_passes(gm, example_inputs)

    logger.info(
        "embedding_projection_opt: %d duplicate partition(s) — dedup compile path",
        len(equiv_map),
    )

    try:
        partition_inputs = _capture_partition_inputs(registry.split, example_inputs)
        for rep_name, rep_mod in registry.unique_reps:
            inputs = partition_inputs.get(rep_name, example_inputs)
            compiled = _compile_with_aten_passes(rep_mod, inputs)
            rep_mod.forward = compiled
            for _, dup_mod in registry.duplicates_of(rep_name):
                dup_mod.forward = compiled
        return lambda *args: registry.split(*args)
    except Exception as exc:
        logger.warning(
            "embedding_projection_opt: dedup compile path failed (%s) — falling "
            "back to flat compile path", exc
        )
        return _compile_with_aten_passes(gm, example_inputs)


# ---------------------------------------------------------------------------
# Workload interface (mirrors embedding_projection.py)
# ---------------------------------------------------------------------------

DEVICE     = "cuda"
BATCH_SIZE = 64
SEQ_LEN    = 128
DIM        = 512
DIM_FF     = 2048


class EmbeddingProjection(nn.Module):
    """Token embedding lookup + two-layer projection + logit head."""

    def __init__(self):
        super().__init__()
        self.embed  = nn.Embedding(VOCAB_SIZE, DIM)
        self.ln     = nn.LayerNorm(DIM)
        self.proj1  = nn.Linear(DIM,    DIM_FF, bias=True)
        self.proj2  = nn.Linear(DIM_FF, DIM,    bias=True)
        self.logits = nn.Linear(DIM, VOCAB_SIZE, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(token_ids)      # (B, T, DIM)
        x = self.ln(x)
        x = F.gelu(self.proj1(x))      # (B, T, DIM_FF)
        x = self.proj2(x)              # (B, T, DIM)
        return self.logits(x)          # (B, T, VOCAB_SIZE)


def get_model_and_input() -> tuple:
    """
    Return (uncompiled model on CUDA, integer token-id tensor on CUDA).

    Non-graph optimizations: none applied here. OPT-1 (BF16) and OPT-3 (logit
    bf16 downcast) are graph passes applied inside the backend at the Aten IR
    level, so the public model/input stay float32 / int64. channels_last and
    batch-padding do not apply (no NCHW conv tensors; B*T = 8192 and all GEMM
    M/N/K dims are multiples of 16, already BF16 Tensor-Core aligned).

    OPT-2 (TF32) is the mutually-exclusive alternative to OPT-1. To deploy TF32
    INSTEAD of BF16, remove the OPT-1 pass from _aten_inner_compile and uncomment:
        # torch.backends.cuda.matmul.allow_tf32 = True
        # torch.set_float32_matmul_precision("high")
    """
    assert torch.cuda.is_available(), "CUDA required"
    model     = EmbeddingProjection().to(DEVICE).eval()
    token_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    return model, token_ids


if __name__ == "__main__":
    model, token_ids = get_model_and_input()
    compiled = torch.compile(model, backend=BACKEND_NAME)
    with torch.no_grad():
        out = compiled(token_ids)
    print(f"Output shape: {out.shape}, dtype: {out.dtype}")  # expect (64, 128, 32000)
