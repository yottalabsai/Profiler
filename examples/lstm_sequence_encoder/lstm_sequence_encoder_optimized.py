"""
lstm_sequence_encoder_optimized.py — Custom torch.compile() backend for the
stacked 2-layer LSTM sequence encoder.

When torch.compile + Inductor lowers ``nn.LSTM`` it does NOT keep the cuDNN fused
RNN kernel: it UNROLLS the recurrence into a per-timestep, per-layer chain of
small gate GEMMs + sigmoid/tanh/elementwise + view/split kernels. At the Aten IR
level (after AOTAutograd) the decomposed forward graph for this model is:

    258  aten.addmm.default        — recurrent gate GEMM, addmm(bias[2048], h[32,512], W[512,2048])
      1  aten.mm.default           — input-projection batched GEMM [4096,256] x [256,2048]
    768  aten.sigmoid.default      — i/f/o gate activations (3 gates x 256 steps)
    512  aten.tanh.default         — cell-gate + hidden-state tanh
    768  aten.mul.Tensor / 513 add — cell update  c_t = f*c + i*g ; h_t = o*tanh(c_t)
    516  aten.view.default         — gate/state reshapes  (NONE are identity-shape)
    261  aten.permute.default      — weight transpose feeding addmm
    256  aten.split.Tensor         — split the [32,2048] gate tensor into 4 x [32,512]

The dominant cost (~60% of attributed time, profile.json) is the serial chain of
M=32 gate addmm's. On the RTX PRO 6000 Blackwell these run the FP32 SIMT path
("Kernel2") with the tensor cores 100% idle (tensor_core_active_pct = 0.0) and a
paired splitKreduce_kernel per call.

Backend registration name: lstm_sequence_encoder_opt

Passes (prerequisite-DAG order OPT-1 -> OPT-2 -> OPT-3), all at Aten IR level
inside ``_aten_inner_compile`` (the Inductor inner_compile hook):

  OPT-1 (MEDIUM, priority 1) — dtype_promotion  [APPLIED]
      Cast the FP32 operands of every recurrent gate addmm (and the input mm) to
      bfloat16 and cast the GEMM result back to float32 (accumulation stays FP32
      inside the MMA). This steers cuBLAS off the FP32 SIMT "Kernel2" onto the
      Blackwell HMMA tensor-core path and typically drops the paired
      splitKreduce_kernel. Highest-leverage change; prerequisite for OPT-2/OPT-3.
      NOTE: optimizations.json named only aten.mm.default — but the recurrent gate
      GEMM is actually realized as aten.addmm.default (bias pre-folded). This pass
      covers BOTH aten.mm and aten.addmm so the 258 gate GEMMs are promoted.

  OPT-2 (LOW, priority 2) — memory_layout (weight pre-transpose / bias fold)  [STUB]
      optimizations.json proposed hoisting the recurrent weight transpose into a
      register_buffer and folding the gate bias into aten.addmm. In this graph the
      decomposition ALREADY emits aten.addmm.default with the bias folded and a
      pre-permuted weight, so the bias-fold half is a no-op. The register_buffer
      half is downgraded to a stub: registering a packed weight buffer per timestep
      across 258 nodes conflicts with the BF16 cast OPT-1 inserts on the same
      operand (register_buffer fixes the dtype at registration time, see OPT-2
      notes) and the optimizations.json itself rates this LOW with uncertain bias
      placement. The pass detects + reports; it does not mutate the graph.

  OPT-3 (MEDIUM, priority 3) — fusion (view/cat elimination)  [STUB]
      optimizations.json proposed erasing layout-noop aten._unsafe_view nodes so
      Inductor emits one fused pointwise kernel per timestep. This graph contains
      ZERO aten._unsafe_view nodes and ZERO identity-shape aten.view nodes (all 516
      views genuinely reshape between [32,2048] gate tensors and [32,512] gate
      slices), so there is no safe layout-noop to erase. Inductor already fuses the
      gate activation arithmetic (kernel name triton_poi_fused_add_addmm_cat).
      Erasing real reshapes would corrupt the graph, so this pass is detection-only:
      it reports the view/split structure and confirms no eliminable noop exists.

Hardware: NVIDIA RTX PRO 6000 Blackwell Server Edition. compile_mode: inductor.

torch._dynamo.config.allow_rnn = True is REQUIRED to trace nn.LSTM/GRU/RNN under
Dynamo; it is set at import time below so importing this module (and the test
suite) is sufficient.
"""
from __future__ import annotations

import logging
from typing import Callable

import torch
import torch.fx as fx
from torch._dynamo import register_backend

# compile_fx is the callable entry point; compile_fx_inner is the post-AOTAutograd
# inner hook that receives the fully decomposed Aten IR graph. Passing
# inner_compile=_aten_inner_compile lets our passes run on that Aten graph and then
# delegate to the real inner compiler (Aten -> Triton). This avoids the torch 2.11
# double-AOTAutograd input-flattening path entirely.
from torch._inductor.compile_fx import compile_fx, compile_fx_inner

from nvidia.operator_profiler.fx import UniqueSubgraphRegistry

logger = logging.getLogger(__name__)

# RNN tracing under Dynamo requires this flag. Set at import so the test suite and
# any direct torch.compile(model, backend="lstm_sequence_encoder_opt") works.
torch._dynamo.config.allow_rnn = True

BACKEND_NAME = "lstm_sequence_encoder_opt"

# --- Aten op targets used across passes ---------------------------------------
_MM = torch.ops.aten.mm.default
_ADDMM = torch.ops.aten.addmm.default
_GEMM_TARGETS = frozenset({_MM, _ADDMM})
_VIEW = torch.ops.aten.view.default
_SPLIT = torch.ops.aten.split.Tensor
_UNSAFE_VIEW = torch.ops.aten._unsafe_view.default

# prims.convert_element_type is the canonical Inductor dtype-cast primitive. We use
# it instead of aten._to_copy.default (which optimizations.json fx_steps name)
# because on torch 2.11 _to_copy carries BOTH a fallback and a decomp registration;
# inserting it post-AOTAutograd makes Inductor raise "both a fallback and a decomp
# for same op". convert_element_type lowers cleanly to a Triton cast and is the form
# Inductor itself emits for dtype conversions, so OPT-1's casts fuse into neighbours.
_CONVERT = torch.ops.prims.convert_element_type.default


# ---------------------------------------------------------------------------
# OPT-1 (MEDIUM) — dtype_promotion: BF16 casts around the recurrent gate GEMMs
# ---------------------------------------------------------------------------

def _pass_gemm_bf16_casts(gm: fx.GraphModule) -> fx.GraphModule:
    """
    OPT-1 — insert bfloat16 casts on the FP32 operands of every recurrent gate
    aten.addmm.default and the input-projection aten.mm.default, casting the GEMM
    result back to float32.

    Why addmm AND mm: the cuDNN-RNN decomposition realizes the recurrent gate GEMM
    as ``aten.addmm.default(bias[2048], h[32,512], W[512,2048])`` (258 of them, the
    ~60%-of-time bottleneck) and the one-shot input projection as a single
    ``aten.mm.default`` ([4096,256] x [256,2048]). Both must be promoted to flip
    cuBLAS dispatch off the FP32 SIMT "Kernel2" onto the Blackwell HMMA tensor-core
    GEMM, which also typically eliminates the paired splitKreduce_kernel.

    Only FP32 tensor operands are cast (guarded on node.meta['val'].dtype) so the
    integer/scalar args and the addmm bias (when already half) are left untouched.
    The output is cast back to float32 so the elementwise gate chain
    (sigmoid/tanh/mul/add) and the downstream mean-pool + classifier stay FP32 and
    numerically stable across the 128-step recurrence. Inductor fuses the
    convert_element_type casts into neighbouring Triton kernels, hiding most cost.

    Confidence MEDIUM: detect-first; if no GEMM node is present this is a graceful
    no-op (WARNING). An exception is logged (WARNING) and the original graph is
    returned unchanged so the compile never crashes.
    """
    try:
        matched = False
        graph = gm.graph
        n_cast = 0
        for node in list(graph.nodes):
            if not (node.op == "call_function" and node.target in _GEMM_TARGETS):
                continue
            matched = True

            with graph.inserting_before(node):
                cast_args = []
                for a in node.args:
                    if (isinstance(a, fx.Node)
                            and a.meta.get("val") is not None
                            and getattr(a.meta["val"], "dtype", None) == torch.float32):
                        cast_args.append(graph.call_function(_CONVERT, (a, torch.bfloat16)))
                    else:
                        cast_args.append(a)
            node.args = tuple(cast_args)

            with graph.inserting_after(node):
                back = graph.call_function(_CONVERT, (node, torch.float32))
            # Re-point every existing user to the float32 cast (but not the cast itself).
            node.replace_all_uses_with(back, delete_user_cb=lambda u: u is not back)
            n_cast += 1

        if not matched:
            logger.warning(
                "[_pass_gemm_bf16_casts] No aten.mm / aten.addmm nodes found "
                "— OPT-1 not applied"
            )
            return gm

        graph.lint()
        gm.recompile()
        logger.info(
            "[_pass_gemm_bf16_casts] Applied BF16 casts to %d gate GEMM node(s) "
            "(aten.addmm + aten.mm) [OPT-1, Aten IR]",
            n_cast,
        )
    except Exception as exc:
        logger.warning("[_pass_gemm_bf16_casts] Failed: %s", exc)

    return gm


# ---------------------------------------------------------------------------
# OPT-2 (LOW, stub) — memory_layout: recurrent weight pre-transpose / bias fold
# ---------------------------------------------------------------------------

def _pass_weight_pretranspose_stub(gm: fx.GraphModule) -> fx.GraphModule:
    """
    OPT-2 (STUB) — detection-only.

    optimizations.json proposed hoisting the recurrent gate weight transpose into a
    register_buffer and folding the gate bias into aten.addmm. Two facts about the
    actual decomposed graph make the transformative version inadvisable here:

      1. The decomposition ALREADY emits aten.addmm.default (bias pre-folded into
         the GEMM epilogue) with a pre-permuted weight feeding it, so the bias-fold
         half is a no-op — there is no standalone (mm -> add(bias)) chain to re-fuse.

      2. Hoisting the weight into a register_buffer fixes its dtype at registration
         time. OPT-1 casts that very weight operand to bf16 inside the graph; a
         pre-registered FP32 buffer would either defeat OPT-1 or require re-casting
         a registered buffer (the DAG rule the optimizations.json OPT-2 notes flag
         as illegal). optimizations.json itself rates OPT-2 LOW and notes the bias
         placement is uncertain.

    This pass therefore only counts/reports the addmm gate GEMMs and their permuted
    weight feeds, and never mutates the graph. The L2-residency win OPT-2 targets is
    already partly delivered by OPT-1 (bf16 halves the weight read traffic).
    """
    try:
        n_addmm = 0
        n_permuted_weight = 0
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target is _ADDMM:
                n_addmm += 1
                w = node.args[2] if len(node.args) > 2 else None
                if (isinstance(w, fx.Node)
                        and w.op == "call_function"
                        and w.target in (torch.ops.aten.permute.default,
                                         torch.ops.aten.t.default)):
                    n_permuted_weight += 1

        logger.info(
            "[_pass_weight_pretranspose_stub] OPT-2 STUB (not applied): %d recurrent "
            "gate addmm node(s), %d with pre-permuted weight feed — bias already "
            "folded into addmm epilogue; register_buffer hoist skipped (conflicts "
            "with OPT-1 bf16 cast on the same weight operand). bf16 from OPT-1 "
            "already halves the weight read traffic.",
            n_addmm, n_permuted_weight,
        )
    except Exception as exc:
        logger.warning("[_pass_weight_pretranspose_stub] Failed: %s", exc)

    return gm


# ---------------------------------------------------------------------------
# OPT-3 (MEDIUM, stub) — fusion: layout-noop view/cat elimination
# ---------------------------------------------------------------------------

def _is_identity_view(node: fx.Node) -> bool:
    """A view whose output shape equals its input shape is a pure layout no-op."""
    if not (node.op == "call_function" and node.target in (_VIEW, _UNSAFE_VIEW)):
        return False
    inp = node.args[0]
    if not isinstance(inp, fx.Node):
        return False
    iv = inp.meta.get("val")
    ov = node.meta.get("val")
    if not (hasattr(iv, "shape") and hasattr(ov, "shape")):
        return False
    return tuple(iv.shape) == tuple(ov.shape)


def _pass_eliminate_noop_views(gm: fx.GraphModule) -> fx.GraphModule:
    """
    OPT-3 — erase only the layout-noop reshapes (aten._unsafe_view / aten.view whose
    output shape == input shape) so Inductor schedules one fused pointwise kernel per
    timestep over the contiguous gate buffer instead of paying launch latency on a
    redundant reshape.

    Reality of this graph: it contains ZERO aten._unsafe_view nodes and ZERO
    identity-shape aten.view nodes — all 516 views genuinely reshape between the
    [32,2048] gate tensor and the four [32,512] gate slices (paired with 256
    aten.split.Tensor). There is therefore no SAFE layout-noop to erase, and
    Inductor already fuses the gate activation arithmetic (kernel name
    triton_poi_fused_add_addmm_cat). Erasing a genuine reshape would silently
    corrupt downstream shapes, so this pass guards every candidate with
    _is_identity_view and degrades to a detection-only no-op (WARNING) when none
    match — which is the expected outcome for this model.

    Confidence MEDIUM: applies the transform if (and only if) a true noop view is
    present in some future/variant graph; otherwise reports and returns unchanged.
    """
    try:
        n_view = n_split = 0
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target in (_VIEW, _UNSAFE_VIEW):
                n_view += 1
            elif node.op == "call_function" and node.target is _SPLIT:
                n_split += 1

        erased = 0
        for node in list(gm.graph.nodes):
            if _is_identity_view(node) and len(node.users) > 0:
                node.replace_all_uses_with(node.args[0])
                gm.graph.erase_node(node)
                erased += 1

        if erased:
            gm.graph.eliminate_dead_code()
            gm.graph.lint()
            gm.recompile()
            logger.info(
                "[_pass_eliminate_noop_views] Erased %d layout-noop view(s) of %d "
                "total view nodes (%d split nodes) [OPT-3, Aten IR]",
                erased, n_view, n_split,
            )
        else:
            logger.warning(
                "[_pass_eliminate_noop_views] OPT-3 no-op: %d view node(s) and %d "
                "split node(s) present, but NONE are identity-shape layout noops "
                "(graph has no aten._unsafe_view; gate activations already fused by "
                "Inductor). No safe reshape to erase — pass not applied.",
                n_view, n_split,
            )
    except Exception as exc:
        logger.warning("[_pass_eliminate_noop_views] Failed: %s", exc)

    return gm


# ---------------------------------------------------------------------------
# Aten IR inner compiler — all graph passes run here, in prerequisite-DAG order
# ---------------------------------------------------------------------------

def _aten_inner_compile(gm: fx.GraphModule, example_inputs, **kwargs) -> Callable:
    """
    Inductor ``inner_compile`` hook. ``compile_fx`` calls this with the fully
    decomposed **Aten IR** forward graph (post-AOTAutograd), where the cuDNN RNN has
    already been unrolled into per-timestep addmm + sigmoid/tanh/elementwise nodes.
    The three passes run here in prerequisite order, then delegate to the real
    ``compile_fx_inner`` (Aten -> Triton).

    Order (from optimizations.json prerequisite_for[]):
      OPT-1 (dtype_promotion)        — first; sets the bf16 dtype OPT-2/OPT-3 assume.
      OPT-2 (weight pretranspose)    — after OPT-1 (register_buffer-after-dtype rule);
                                        stub / detection-only here.
      OPT-3 (view/cat fusion)        — last; operates on the (now bf16) gate buffer;
                                        detection-only here (no eliminable noop view).
    """
    gm = _pass_gemm_bf16_casts(gm)            # OPT-1  (applied)
    gm = _pass_weight_pretranspose_stub(gm)   # OPT-2  (stub)
    gm = _pass_eliminate_noop_views(gm)       # OPT-3  (stub / guarded)
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
) -> dict:
    """Run split_gm once under no_grad to capture each partition's input tensors."""
    partition_inputs: dict = {}
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
# Backend: lstm_sequence_encoder_opt
# ---------------------------------------------------------------------------

@register_backend
def lstm_sequence_encoder_opt(gm: fx.GraphModule, example_inputs) -> Callable:
    """
    Custom torch.compile() backend for the LSTM sequence encoder.

    Dedup-aware structure (the canonical backend shape): UniqueSubgraphRegistry
    splits the functional FX graph into per-layer partitions and groups them by
    structural signature. The unrolled LSTM produces a single flat partition (no
    repeated block structure survives the unroll), so equiv_map is empty and the
    backend takes the flat compile path — which preserves cross-timestep Inductor
    fusion. The dedup branch is retained for robustness / parity with the other
    workloads and is exercised only if a future graph exposes repeated partitions.

    All three FX passes run inside ``_aten_inner_compile`` at the Aten IR level via
    ``compile_fx(..., inner_compile=...)``; every pass degrades gracefully (INFO on
    apply, WARNING on no-match/failure) and never crashes the compile.
    """
    logger.info(
        "lstm_sequence_encoder_opt backend: starting "
        "(Aten IR passes via compile_fx inner_compile)"
    )

    registry = UniqueSubgraphRegistry(gm)
    equiv_map = registry.build_partition_equivalence_map()

    if not equiv_map:
        logger.info(
            "lstm_sequence_encoder_opt: no repeated layers (unrolled flat graph) "
            "— flat compile path"
        )
        return _compile_with_aten_passes(gm, example_inputs)

    logger.info(
        "lstm_sequence_encoder_opt: %d duplicate partition(s) — dedup compile path",
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
            "lstm_sequence_encoder_opt: dedup compile path failed (%s) — falling "
            "back to flat compile path", exc
        )
        return _compile_with_aten_passes(gm, example_inputs)


# ---------------------------------------------------------------------------
# Workload interface
# ---------------------------------------------------------------------------

DEVICE      = "cuda"
BATCH_SIZE  = 32
SEQ_LEN     = 128
INPUT_SIZE  = 256
HIDDEN_SIZE = 512
NUM_LAYERS  = 2
NUM_CLASSES = 10


class LSTMSequenceEncoder(torch.nn.Module):
    """Stacked 2-layer LSTM + mean-pool temporal reduction + linear classifier."""

    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=0.0,
        )
        self.classifier = torch.nn.Linear(HIDDEN_SIZE, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)           # (B, T, HIDDEN_SIZE)
        pooled = out.mean(dim=1)        # (B, HIDDEN_SIZE)
        return self.classifier(pooled)  # (B, NUM_CLASSES)


def get_model_and_input() -> tuple:
    """
    Return (uncompiled model on CUDA, input tensor on CUDA).

    Non-graph optimizations: NONE. OPT-1 (bf16), OPT-2 (layout) and OPT-3 (view
    fusion) are all graph passes applied inside the backend at the Aten IR level, so
    the public model/input stay float32. dtype/memory_format/batch-shape rewrites do
    not apply: there is no NCHW conv, batch=32 is already a tensor-core-friendly
    multiple of 16, and forcing bf16 weights here would skip the controlled in-graph
    cast-back-to-fp32 that keeps the 128-step recurrence numerically stable.

    allow_rnn is enabled at module import so torch.compile can trace nn.LSTM.
    """
    assert torch.cuda.is_available(), "CUDA required"
    model = LSTMSequenceEncoder().to(DEVICE).eval()
    x = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE, device=DEVICE)
    return model, x


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model, x = get_model_and_input()
    compiled = torch.compile(model, backend=BACKEND_NAME)
    with torch.no_grad():
        out = compiled(x)
    print(f"Output shape: {out.shape}, dtype: {out.dtype}")  # expect (32, 10)
