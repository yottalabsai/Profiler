"""
lstm_sequence_encoder_optimized.py — Custom torch.compile() backend for the
stacked-LSTM sequence encoder.

Backend name (registered with torch._dynamo): ``lstm_sequence_encoder_opt``

Source proposal: examples/lstm_sequence_encoder/optimizations.json
Cross-validated against: examples/lstm_sequence_encoder/profile.json
    device   = NVIDIA RTX PRO 6000 Blackwell Server Edition
    dtype    = float32 (baseline)
    shapes   = input (B=32, T=128, INPUT_SIZE=256); classifier addmm [32,512]x[512,10]

================================================================================
ROUTE DECISION (the load-bearing analysis)
================================================================================
The proposal lists four optimizations grouped into two MUTUALLY EXCLUSIVE routes
over the recurrent matmuls (optimizations.json global_notes + OPT-1/OPT-2/OPT-3
notes):

  * Route 1 (OPT-1): keep ``nn.LSTM`` EAGER so cuDNN's fused RNN path
    (elemWiseRNNcell + cutlass_80_tensorop_s1688gemm, Tensor Cores engaged)
    runs, instead of letting Inductor unroll the loop into ~1020 tiny
    cutlass_80_simt_sgemm_128x32 + splitKreduce launches. Ranked priority 1 and
    described as "the single largest structural win".

  * Route 2 (OPT-2 + OPT-3 + OPT-4): retain the unrolled custom-cell loop and
    transform it in place (hoist input projection, bf16/tf32 promotion, freezing).

This backend implements Route 1, because empirically ``nn.LSTM`` is a HARD
Dynamo graph break on torch 2.11: Dynamo never traces into it (``dynamo.explain``
reports graph_count 0 for this model; the custom backend is never even invoked
for the recurrent region). The cuDNN fused RNN therefore runs eagerly by default
— exactly the OPT-1 end state — and the only subgraph that can reach a backend
is the ``mean(dim=1) + classifier`` tail. There is no shared-``F.linear`` gate
triplet at the functional level to apply OPT-2's input-projection hoist to, so
Route 2 is unsatisfiable as an FX pass on the natural graph. OPT-2 is therefore
implemented as a detection-only stub (mutually exclusive with OPT-1, not applied).

OPT-3 (aten dtype promotion) and OPT-4 (inductor_config freezing) still apply
to whatever GEMMs DO reach the compiled graph — the classifier addmm, and any
gate mm should a future caller force the loop into the graph (e.g. fullgraph or
an unrolled custom cell). They are implemented as real passes routed through the
funnel so they fire wherever an in-graph GEMM exists.
================================================================================
"""
from __future__ import annotations

import functools
import logging
from typing import Callable

import torch
import torch.fx as fx
import torch.nn as nn
from torch._dynamo import register_backend
from torch._inductor.compile_fx import compile_fx, compile_fx_inner

from nvidia.operator_profiler.fx import UniqueSubgraphRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Workload constants (mirror the baseline; cross-validated against profile.json)
# ---------------------------------------------------------------------------
DEVICE      = "cuda"
BATCH_SIZE  = 32
SEQ_LEN     = 128
INPUT_SIZE  = 256
HIDDEN_SIZE = 512
NUM_LAYERS  = 2
NUM_CLASSES = 10


# ===========================================================================
# LEVEL 1 — functional passes (run on the Dynamo graph, before compile_fx)
# ===========================================================================

def _fpass_opt1_keep_lstm_eager(gm: fx.GraphModule) -> fx.GraphModule:
    """
    OPT-1 (functional, op_substitution) — confidence medium.

    The substitution "unrolled per-timestep gate-GEMM loop -> single fused
    cuDNN RNN op" is realised STRUCTURALLY: ``nn.LSTM`` is left eager so the
    cuDNN fused RNN (elemWiseRNNcell + tensorop s1688 GEMM) runs and Inductor
    never sees the recurrent region. This pass therefore only AUDITS the graph
    that reached the backend: it confirms the unrolled gate-GEMM cohort
    ([B,H] x [H,4H] mm triplets sharing one activation) is NOT present, i.e.
    the recurrent loop was correctly kept out of the compiled region. If it
    ever IS present (a caller forced the loop into the graph), we log a warning
    and leave it for OPT-3 to promote — we do not unroll/rebuild the cell here.
    """
    try:
        gate_mm = 0
        for n in gm.graph.nodes:
            if n.op != "call_function":
                continue
            tname = getattr(n.target, "__name__", "")
            if tname in ("linear", "mm", "matmul", "addmm"):
                gate_mm += 1
        if gate_mm <= 1:
            logger.info(
                "[OPT-1] LSTM kept eager (cuDNN fused RNN path); compiled region "
                "is the mean+classifier head only — no unrolled gate loop in graph "
                "[functional]"
            )
        else:
            logger.warning(
                "[OPT-1] %d projection ops found in the compiled graph — the "
                "recurrent loop may have been forced into the graph. OPT-1 "
                "(eager cuDNN) is the preferred route; leaving GEMMs for OPT-3 "
                "bf16/tf32 promotion.",
                gate_mm,
            )
    except Exception as e:  # confidence medium — graceful
        logger.warning("[OPT-1] functional audit no-op: %s", e)
    return gm


_LINEAR_FNS = {torch.nn.functional.linear}
try:  # builtin alias on some builds
    _LINEAR_FNS.add(torch._C._nn.linear)  # type: ignore[attr-defined]
except Exception:
    pass


def _is_linear(n: fx.Node) -> bool:
    return (
        n.op == "call_function"
        and (n.target in _LINEAR_FNS or getattr(n.target, "__name__", "") == "linear")
    )


def _fpass_opt2_hoist_input_projection_stub(gm: fx.GraphModule) -> fx.GraphModule:
    """
    OPT-2 (functional, fusion) — confidence high BUT MUTUALLY EXCLUSIVE with OPT-1.

    Detection-only stub. OPT-2 hoists the input-to-hidden projection out of the
    unrolled recurrence by grouping per-timestep ``F.linear`` calls that share a
    single ``W_ih`` weight node and concatenating their independent input slices
    into one [seq*batch, in] x [in, 4*hidden] GEMM. It is only meaningful when
    the custom-cell loop is RETAINED inside the graph (Route 2). Under Route 1
    the LSTM is eager and no such triplet exists, so this never transforms — it
    only reports whether a hoistable shared-W_ih triplet is present.
    """
    try:
        groups: dict[fx.Node, list[fx.Node]] = {}
        for n in gm.graph.nodes:
            if _is_linear(n) and len(n.args) > 1 and isinstance(n.args[1], fx.Node):
                groups.setdefault(n.args[1], []).append(n)  # group by shared weight
        hoistable = [w for w, lins in groups.items() if len(lins) >= 3]
        if hoistable:
            logger.warning(
                "[OPT-2] %d shared-weight F.linear group(s) look hoistable, but "
                "OPT-2 is MUTUALLY EXCLUSIVE with OPT-1 (eager cuDNN LSTM) which "
                "is the adopted route — input-projection hoist NOT applied.",
                len(hoistable),
            )
        else:
            logger.info(
                "[OPT-2] no unrolled per-timestep shared-W_ih F.linear triplet in "
                "graph (LSTM is eager under OPT-1) — stub, not applied [functional]"
            )
    except Exception as e:
        logger.warning("[OPT-2] functional stub no-op: %s", e)
    return gm


# ===========================================================================
# LEVEL 2 — aten passes (run inside _aten_inner_compile, post-decomposition)
# ===========================================================================

_MM_TARGETS = (torch.ops.aten.mm.default, torch.ops.aten.addmm.default)


def _apass_opt3_bf16_promotion(gm: fx.GraphModule) -> fx.GraphModule:
    """
    OPT-3 (aten, dtype_promotion) — confidence high.

    Promote float32 GEMM operands to bfloat16 so cuBLAS/cutlass selects a
    tensorop GEMM (cutlass_80_tensorop_s1688gemm) instead of the scalar FP32
    SIMT path (cutlass_80_simt_sgemm, tensor_core_active_pct == 0). For each
    ``aten.mm.default`` / ``aten.addmm.default`` whose matmul operands are
    float32, insert ``aten._to_copy`` casts to bf16 on the operands and cast the
    result back to float32 to preserve the downstream dtype contract (the
    classifier's logits, and — if the loop were retained — the recurrent-state
    sigmoid/tanh/mul ops). In the adopted Route 1 the only in-graph GEMM is the
    classifier addmm [32,512]x[512,10]; this still routes it to Tensor Cores.

    TF32 (the lower-risk variant, FP32 accumulation preserved) is additionally
    enabled globally in get_model_and_input() so even GEMMs that keep float32
    operands select the tensorop kernel.
    """
    try:
        matched = 0
        for node in list(gm.graph.nodes):
            if node.op != "call_function" or node.target not in _MM_TARGETS:
                continue
            is_mm = node.target is torch.ops.aten.mm.default
            mat_idx = (0, 1) if is_mm else (1, 2)
            # For addmm, the bias (idx 0) must be promoted alongside the matmul
            # operands: Inductor fuses bias+addmm into bias_addmm, which requires
            # bias.dtype == mat.dtype. Casting only mat1/mat2 leaves bias f32 and
            # crashes at runtime ("self and mat2 must have the same dtype").
            cast_idx = mat_idx if is_mm else (0, 1, 2)

            def _is_f32(arg: object) -> bool:
                if not isinstance(arg, fx.Node):
                    return False
                v = arg.meta.get("val", arg.meta.get("example_value"))
                return getattr(v, "dtype", None) == torch.float32

            if not all(_is_f32(node.args[i]) for i in mat_idx):
                continue

            with gm.graph.inserting_before(node):
                for i in cast_idx:
                    src = node.args[i]
                    if not _is_f32(src):
                        continue
                    cast = gm.graph.call_function(
                        torch.ops.prims.convert_element_type.default,
                        args=(src, torch.bfloat16),
                    )
                    node.update_arg(i, cast)
            with gm.graph.inserting_after(node):
                out_f32 = gm.graph.call_function(
                    torch.ops.prims.convert_element_type.default,
                    args=(node, torch.float32),
                )
            node.replace_all_uses_with(out_f32)
            out_f32.update_arg(0, node)  # restore the self-edge severed above
            matched += 1

        if not matched:
            logger.warning(
                "[OPT-3] no float32 aten.mm/addmm operands found — pass not applied "
                "[aten]"
            )
            return gm
        gm.graph.lint()
        gm.recompile()
        logger.info(
            "[OPT-3] Promoted %d GEMM(s) to bf16 (operands->bf16, result->f32) "
            "to engage Tensor Cores [aten]",
            matched,
        )
    except Exception as e:  # confidence high — exception is a real error, log + continue
        logger.warning("[OPT-3] Failed: %s", e)
    return gm


# ===========================================================================
# LEVEL 3 — inductor_config "passes" (scoped config_patches on compile_fx)
# ===========================================================================

def _cfg_opt4_freezing() -> dict:
    """
    OPT-4 (inductor_config, freezing) — confidence medium.

    Freeze constant weights (folds the _tn_ transpose, drops per-call weight
    guards) and autotune a GEMM config that avoids the split-K reduction
    epilogue (cublasLt::splitKreduce_kernel) for the small-M shapes. With OPT-3's
    bf16 weights in place (funnel order functional -> aten -> inductor_config
    guarantees OPT-3 already ran), freezing additionally emits a pre-packed
    tensorop weight layout. Scoped to THIS compile_fx call only — no global
    config mutation. Requires eval()/inference (no autograd).
    """
    return {
        "freezing": True,
        "max_autotune": True,
        "max_autotune_gemm_backends": "ATEN,TRITON",
    }


# ===========================================================================
# Pass registry + router
# ===========================================================================

PASS_REGISTRY = [
    {"id": "OPT-1", "level": "functional",      "fn": _fpass_opt1_keep_lstm_eager},
    {"id": "OPT-2", "level": "functional",      "fn": _fpass_opt2_hoist_input_projection_stub},
    {"id": "OPT-3", "level": "aten",            "fn": _apass_opt3_bf16_promotion},
    {"id": "OPT-4", "level": "inductor_config", "fn": _cfg_opt4_freezing},
]


def _passes(level: str) -> list:
    return [p for p in PASS_REGISTRY if p["level"] == level]


def _run_functional_passes(gm: fx.GraphModule) -> fx.GraphModule:
    """LEVEL 1 — rewrite the Dynamo (functional) graph before compile_fx owns it."""
    for p in _passes("functional"):
        try:
            gm = p["fn"](gm)
        except Exception as e:
            logger.warning("[%s] functional pass no-op: %s", p["id"], e)
    return gm


def _repropagate_meta(gm: fx.GraphModule, example_inputs) -> None:
    """Repopulate meta['val'] on inserted nodes after a structural aten rewrite."""
    try:
        from torch.fx.passes.fake_tensor_prop import FakeTensorProp

        fake_mode = None
        for ex in example_inputs:
            fm = getattr(ex, "fake_mode", None)
            if fm is not None:
                fake_mode = fm
                break
        if fake_mode is not None:
            FakeTensorProp(gm, mode=fake_mode).propagate(*example_inputs)
        else:
            FakeTensorProp(gm).propagate(*example_inputs)
    except Exception as e:
        logger.warning("[_repropagate_meta] skipped: %s", e)


def _aten_inner_compile(gm: fx.GraphModule, example_inputs, *, real_inputs=None, **kwargs) -> Callable:
    """LEVEL 2 — Inductor inner_compile hook (fully decomposed Aten IR graph)."""
    for p in _passes("aten"):
        try:
            gm = p["fn"](gm)  # OPT-3 reads dtypes from meta, not weight values
            _repropagate_meta(gm, example_inputs)
        except Exception as e:
            logger.warning("[%s] aten pass no-op: %s", p["id"], e)
    return compile_fx_inner(gm, example_inputs, **kwargs)


def _config_patches() -> dict:
    """LEVEL 3 — scoped Inductor config_patches merged into THIS compile_fx call."""
    patches: dict = {}
    for p in _passes("inductor_config"):
        try:
            patches.update(p["fn"]() or {})
        except Exception as e:
            logger.warning("[%s] config pass skipped: %s", p["id"], e)
    return patches


def _compile_unit(gm: fx.GraphModule, example_inputs) -> Callable:
    """The fixed three-stage funnel: functional -> compile_fx(aten + config)."""
    gm = _run_functional_passes(gm)
    inner = functools.partial(_aten_inner_compile, real_inputs=list(example_inputs))
    try:
        return compile_fx(gm, example_inputs, inner_compile=inner,
                          config_patches=_config_patches())
    except Exception as e:
        # OPT-4 freezing/max_autotune can be unavailable in some Inductor builds;
        # fall back to a plain compile so the head still lowers.
        logger.warning("[funnel] compile_fx with config_patches failed (%s); "
                       "retrying without inductor_config patches", e)
        return compile_fx(gm, example_inputs, inner_compile=inner)


def _capture_partition_inputs(split_gm: fx.GraphModule, example_inputs: list) -> dict:
    """Capture actual input tensors for each partition by running split_gm once."""
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
# Registered backend
# ===========================================================================

@register_backend
def lstm_sequence_encoder_opt(gm: fx.GraphModule, example_inputs) -> Callable:
    logger.info(
        "lstm_sequence_encoder_opt backend: starting "
        "(functional -> aten -> inductor_config)"
    )
    registry = UniqueSubgraphRegistry(gm)
    equiv_map = registry.build_partition_equivalence_map()

    if not equiv_map:
        logger.info(
            "lstm_sequence_encoder_opt: no repeated layers, flat compile path"
        )
        return _compile_unit(gm, example_inputs)

    logger.info(
        "lstm_sequence_encoder_opt: %d duplicate partition(s), dedup path",
        len(equiv_map),
    )
    partition_inputs = _capture_partition_inputs(registry.split, example_inputs)
    for rep_name, rep_mod in registry.unique_reps:
        inputs = partition_inputs.get(rep_name, example_inputs)
        compiled = _compile_unit(rep_mod, inputs)
        rep_mod.forward = compiled
        for _, dup_mod in registry.duplicates_of(rep_name):
            dup_mod.forward = compiled

    return lambda *args: registry.split(*args)


# ===========================================================================
# Workload interface — same contract as the baseline lstm_sequence_encoder.py
# ===========================================================================

class LSTMSequenceEncoder(nn.Module):
    """Stacked 2-layer LSTM + mean-pool + linear classifier (identical to baseline)."""

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=0.0,
        )
        self.classifier = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)           # (B, T, HIDDEN_SIZE) — eager cuDNN fused RNN (OPT-1)
        pooled = out.mean(dim=1)        # (B, HIDDEN_SIZE)
        return self.classifier(pooled)  # (B, NUM_CLASSES)


def get_model_and_input() -> tuple:
    """
    Workload interface — return (raw_model, input_tensor) on CUDA.

    Non-graph optimizations applied here (per Rule 7 / OPT-3 conservative variant):
      * TF32: route FP32 GEMMs / cuDNN RNN to the tensorop (s1688) pipeline while
        preserving FP32 accumulation. This is the low-risk half of OPT-3 and also
        benefits the eager cuDNN LSTM (OPT-1), which Inductor never sees.

    Compilation and warmup are handled externally by run_workload.py / the test.
    """
    assert torch.cuda.is_available(), "CUDA required"

    # OPT-3 (conservative, non-graph): enable TF32 tensorop GEMM selection.
    if not torch.backends.cuda.matmul.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    if not torch.backends.cudnn.allow_tf32:
        torch.backends.cudnn.allow_tf32 = True

    model = LSTMSequenceEncoder().to(DEVICE).eval()
    x = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE, device=DEVICE)
    return model, x


if __name__ == "__main__":
    m, x = get_model_and_input()
    compiled = torch.compile(m, backend="lstm_sequence_encoder_opt")
    with torch.no_grad():
        print(compiled(x).shape)
