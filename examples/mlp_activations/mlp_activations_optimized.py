"""
mlp_activations_optimized.py — Custom torch.compile() backend for the
MLPActivations workload (four nn.Linear layers, each with a different activation:
ReLU, GELU, SiLU, Tanh).

ALL graph passes run at the Aten IR level, inside ``_aten_inner_compile``, which
``compile_fx`` invokes (as its ``inner_compile`` hook) with the fully decomposed
Aten forward graph after AOTAutograd has run. At that level every biased
nn.Linear is ``aten.addmm.default(bias, x, aten.t.default(weight))``, and the
activations are ``aten.relu.default`` / ``aten.gelu.default`` / ``aten.silu.default``
/ ``aten.tanh.default``.

Profiling-guided optimizations (from optimizations.json):

  OPT-1 (HIGH, priority 1) — dtype_promotion -> bf16 / Tensor-Core path  [primary]
      Every GEMM (the 8 distinct aten::mm + the addmm prologue) executes on the
      FP32 SIMT FFMA datapath with the Tensor Cores completely idle (ncu:
      smsp__pipe_tensor_cycles_active = 0, tensor_core_active_pct = 0.0 on 100%
      of GEMM kernels, 210 regs/thread, 16.5% occupancy on the 2048x2048 GEMM).
      Cast each matmul's FP32 operands (and the addmm bias) to bfloat16 and cast
      the GEMM result back to fp32 for the activation epilogue. This switches
      cuBLAS/cutlass kernel selection from cutlass_80_simt_sgemm onto a
      *_tensorop_* HMMA path. Root of the dependency DAG and the dominant win
      (conservative ~18.8% of total). Prerequisite for OPT-2.
      Target: every aten.mm.default / aten.addmm.default node (Aten IR graph pass).

  OPT-2 (MEDIUM, priority 2) — max-autotune GEMM templates  [config; depends OPT-1]
      The skinny/small GEMMs (2048->512, 512->2048) are wave- and launch-starved
      (sm_throughput 6.3%, occupancy 8.3%, 176 blocks << one wave) because the
      single fixed cutlass_80_simt tile is the wrong shape. Enable Inductor GEMM
      autotuning so it benchmarks Triton matmul templates / cutlass / ATEN
      candidates and selects a per-shape-optimal tile. With OPT-1 applied the
      candidate set includes Tensor-Core templates. This is a configuration-level
      optimization (no node surgery) applied via torch._inductor.config; it
      directly changes the kernel chosen for each aten.mm / aten.addmm lowering.
      Must run after OPT-1 (eligible template set depends on operand dtype).

  OPT-3 (MEDIUM, priority 3) — bias+activation epilogue fusion  [config; depends OPT-2]
      Each layer currently runs as two kernels: a cutlass GEMM that writes the
      full [256,2048] (or [256,512]) result to DRAM, immediately followed by a
      separate triton_poi_fused_addmm_<act> kernel that reloads it, adds bias,
      applies the activation, and writes it back — a redundant DRAM round-trip and
      an extra launch per layer. Once OPT-2 selects a Triton GEMM template,
      Inductor's pointwise scheduler folds the bias add and the pointwise
      activation (relu/gelu/silu/tanh) into the GEMM epilogue, collapsing each
      layer to a single triton_mm_* kernel. Configuration-level (no node surgery):
      enable epilogue_fusion and constrain max_autotune_gemm_backends to TRITON
      (extern cutlass kernels cannot host a fused Triton epilogue). Depends on
      OPT-2 selecting the Triton template.

Backend registration name: mlp_activations_opt
Prerequisite DAG (strict order): OPT-1 -> OPT-2 -> OPT-3.
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
# which imports the module -> "'module' object is not callable"). compile_fx_inner
# is the post-AOTAutograd leaf compiler that receives the fully decomposed Aten IR
# graph; passing _aten_inner_compile as inner_compile lets our Aten-IR passes run
# on that graph and then delegate to the real inner compiler (Aten -> Triton).
from torch._inductor.compile_fx import compile_fx, compile_fx_inner
import torch._inductor.config as inductor_config

from nvidia.operator_profiler.fx import UniqueSubgraphRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend name constant (used for torch.compile(backend=...) and --compile-backend)
# ---------------------------------------------------------------------------
BACKEND_NAME = "mlp_activations_opt"

# Aten op targets used across passes
_MM = torch.ops.aten.mm.default
_ADDMM = torch.ops.aten.addmm.default
_GEMM_TARGETS = frozenset({_MM, _ADDMM})

# prims.convert_element_type is the canonical Inductor dtype-cast primitive. We
# use it instead of aten._to_copy.default (which optimizations.json names as
# aten.to.dtype) because on torch 2.11 _to_copy carries BOTH a fallback and a
# decomp registration; inserting it post-AOTAutograd makes Inductor raise "both a
# fallback and a decomp for same op". convert_element_type lowers cleanly to a
# Triton cast and is the form Inductor itself emits, so OPT-1's casts fuse into
# the producing/consuming Triton epilogues.
_CONVERT = torch.ops.prims.convert_element_type.default

# OPT-1 cast target dtype. bf16 preferred over fp16 for numerical range on
# inference; activations (gelu/silu/tanh) are computed in fp32 after the
# cast-back so activation accuracy is unaffected.
TARGET_DTYPE = torch.bfloat16


# ---------------------------------------------------------------------------
# OPT-2 / OPT-3 — Inductor configuration (max-autotune + epilogue fusion)
# ---------------------------------------------------------------------------

def _apply_inductor_config() -> None:
    """
    OPT-2 + OPT-3 — configuration-level optimizations applied to the Inductor
    compile config. Neither is a graph-node transformation:

      OPT-2 (max-autotune GEMM): max_autotune / max_autotune_gemm let Inductor
        benchmark per-shape GEMM templates instead of the single fixed
        cutlass_80_simt tile, fixing the wave-starved skinny/small GEMMs.
      OPT-3 (epilogue fusion): epilogue_fusion + a TRITON-only GEMM backend let
        the bias add and the pointwise activation fold into the Triton matmul
        template epilogue, collapsing each layer to one kernel.

    OPT-3 constrains max_autotune_gemm_backends to "TRITON" because extern
    cutlass GEMM kernels cannot host a fused Triton epilogue. This is set *after*
    (and overrides) OPT-2's broader backend list, honouring the OPT-1 -> OPT-2 ->
    OPT-3 prerequisite DAG. Each setattr is individually guarded so a missing
    attribute on this torch build degrades to a no-op instead of breaking compile.
    """
    # OPT-2 — max-autotune GEMM templates.
    opt2_settings = {
        "max_autotune": True,
        "max_autotune_gemm": True,
        # Broad candidate set for OPT-2; OPT-3 narrows this to TRITON below.
        "max_autotune_gemm_backends": "TRITON,CUTLASS,ATEN",
        "coordinate_descent_tuning": True,
    }
    applied_opt2 = []
    for key, value in opt2_settings.items():
        if hasattr(inductor_config, key):
            try:
                setattr(inductor_config, key, value)
                applied_opt2.append(key)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[OPT-2] could not set inductor_config.%s: %s", key, exc)
        else:
            logger.warning("[OPT-2] inductor_config has no attribute '%s' — skipped", key)
    if applied_opt2:
        logger.info(
            "[OPT-2] max-autotune GEMM enabled: %s [config, depends OPT-1]",
            ", ".join(applied_opt2),
        )
    else:
        logger.warning("[OPT-2] no max-autotune settings applied — pass not effective")

    # OPT-3 — epilogue fusion; constrain GEMM backends to TRITON so the activation
    # folds into the matmul template epilogue (extern cutlass cannot fuse it).
    opt3_settings = {
        "epilogue_fusion": True,
        "max_autotune_gemm_backends": "TRITON",
    }
    applied_opt3 = []
    for key, value in opt3_settings.items():
        if hasattr(inductor_config, key):
            try:
                setattr(inductor_config, key, value)
                applied_opt3.append(key)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[OPT-3] could not set inductor_config.%s: %s", key, exc)
        else:
            logger.warning("[OPT-3] inductor_config has no attribute '%s' — skipped", key)
    if applied_opt3:
        logger.info(
            "[OPT-3] bias+activation epilogue fusion enabled (TRITON GEMM backend): "
            "%s [config, depends OPT-2]",
            ", ".join(applied_opt3),
        )
    else:
        logger.warning("[OPT-3] no epilogue-fusion settings applied — pass not effective")


# ---------------------------------------------------------------------------
# OPT-1 (HIGH) — dtype_promotion: BF16 casts around aten.mm / aten.addmm
# ---------------------------------------------------------------------------

def _pass_gemm_bf16_casts(gm: fx.GraphModule) -> fx.GraphModule:
    """
    OPT-1 — force every tensor operand of every aten.mm.default and
    aten.addmm.default node to bfloat16, then cast the result back to float32.

    Confidence HIGH (smsp__pipe_tensor_cycles_active = 0 on 100% of GEMM kernels):
    assume the pattern exists; an exception is a real error -> log WARNING + return
    gm unchanged. A `matched` guard still degrades to a no-op if no GEMM is found.

    Robustness: aten rejects a GEMM whose operands disagree in dtype ("expected
    mat1 and mat2 to have the same dtype"). We therefore do NOT guard on float32 —
    instead we guarantee that *both* matmul operands (and, for addmm, the bias)
    end up bf16 regardless of their current dtype. For every tensor operand we
    insert a convert_element_type(bf16) unless the operand is *already provably*
    bf16 (meta dtype == torch.bfloat16), in which case the cast is skipped. When
    meta is missing we cast defensively (convert to bf16 of a bf16 tensor is a
    harmless no-op). Non-tensor args (ints, None bias) are left as-is.

    The output is restored to float32 so the downstream activation (relu/gelu/
    silu/tanh) runs in fp32 — activation accuracy is unaffected by the bf16 GEMM.
    Inductor fuses the convert_element_type casts into neighbouring Triton kernels,
    hiding their cost; with OPT-2's Triton template + OPT-3's epilogue fusion the
    back-cast collapses into the layer's single fused kernel.
    """
    try:
        matched = False
        graph = gm.graph
        for node in list(graph.nodes):
            if not (node.op == "call_function" and node.target in _GEMM_TARGETS):
                continue
            matched = True

            # Cast every tensor operand to bf16 before the GEMM so both matmul
            # inputs (and the addmm bias) share one dtype. Skip the cast only when
            # the operand is already provably bf16.
            with graph.inserting_before(node):
                cast_args = []
                for a in node.args:
                    if isinstance(a, fx.Node):
                        val = a.meta.get("val")
                        cur_dtype = getattr(val, "dtype", None)
                        if cur_dtype == TARGET_DTYPE:
                            cast_args.append(a)
                        else:
                            c = graph.call_function(_CONVERT, (a, TARGET_DTYPE))
                            cast_args.append(c)
                    else:
                        cast_args.append(a)
            node.args = tuple(cast_args)

            # Restore float32 on the output for the fp32 activation epilogue.
            with graph.inserting_after(node):
                back = graph.call_function(_CONVERT, (node, torch.float32))
            # Re-point all existing users to the float32 cast (but NOT the cast
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
            "operands, fp32 back-cast on output [OPT-1, Aten IR]"
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[_pass_gemm_bf16_casts] Failed: %s", exc)

    return gm


# ---------------------------------------------------------------------------
# Aten IR inner compiler — graph passes run here, in prerequisite order
# ---------------------------------------------------------------------------

def _aten_inner_compile(gm: fx.GraphModule, example_inputs, **kwargs) -> Callable:
    """
    Inductor ``inner_compile`` hook. ``compile_fx`` calls this with the fully
    decomposed **Aten IR** forward graph (after AOTAutograd has run), where every
    biased nn.Linear is an aten.addmm.default and each activation is a pointwise
    aten op. We run OPT-1 here (the only graph-node transformation), then delegate
    to the real ``compile_fx_inner`` (Aten -> Triton).

    OPT-2 (max-autotune) and OPT-3 (epilogue fusion) are configuration-level and
    are applied to torch._inductor.config in the backend before this compile runs;
    they steer compile_fx_inner's GEMM lowering and pointwise scheduling rather
    than editing nodes.

    OPT-1 is the dtype root of the DAG, so it must precede the OPT-2/OPT-3 config
    effects: the Tensor-Core template candidate set OPT-2 autotunes over depends
    on the bf16 operands OPT-1 inserts here.

    Using inner_compile (rather than re-wrapping the functional graph with a
    second aot_autograd) avoids a torch 2.11 double-AOTAutograd input-flattening
    bug ("Expected tensors only, but got list" in copy_misaligned_inputs).
    """
    gm = _pass_gemm_bf16_casts(gm)  # OPT-1 (HIGH) — Aten IR graph pass
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
# Backend: mlp_activations_opt
# ---------------------------------------------------------------------------

@register_backend
def mlp_activations_opt(gm: fx.GraphModule, example_inputs) -> Callable:
    """
    Custom torch.compile() backend for MLPActivations.

    Structure (dedup-aware per the standard backend pattern):
      - OPT-2/OPT-3 are configuration-level: set the Inductor config (max-autotune
        + epilogue fusion, TRITON GEMM backend) BEFORE any compile so they steer
        every GEMM lowering and the pointwise scheduler.
      - UniqueSubgraphRegistry splits the functional FX graph into per-layer
        partitions and groups them by structural signature. MLPActivations' four
        layers have DISTINCT shapes/activations (relu/gelu/silu/tanh), so there is
        no repeated structure; build_partition_equivalence_map() returns empty and
        we take the flat compile path — preserving cross-op Inductor fusion of the
        BF16 casts and the activation epilogues into neighbouring kernels.
      - On the flat path the whole graph is compiled through
        compile_fx(..., inner_compile=_aten_inner_compile); OPT-1 runs inside
        _aten_inner_compile at the Aten IR level (post-AOTAutograd).
      - The dedup branch is retained for protocol compliance and degrades safely
        to the flat path on any per-partition compile failure.
    """
    logger.info(
        "mlp_activations_opt backend: starting "
        "(Aten IR passes via compile_fx inner_compile)"
    )

    # OPT-2 + OPT-3 — apply Inductor config before compiling (depends on OPT-1
    # supplying bf16 operands inside _aten_inner_compile; config does not edit nodes).
    _apply_inductor_config()

    registry = UniqueSubgraphRegistry(gm)
    equiv_map = registry.build_partition_equivalence_map()

    if not equiv_map:
        logger.info("mlp_activations_opt: no repeated layers — flat compile path")
        return _compile_with_aten_passes(gm, example_inputs)

    logger.info(
        "mlp_activations_opt: %d duplicate partition(s) — dedup compile path",
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
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "mlp_activations_opt: dedup compile path failed (%s) — falling back "
            "to flat compile path", exc
        )
        return _compile_with_aten_passes(gm, example_inputs)


# ---------------------------------------------------------------------------
# Workload interface (mirrors mlp_activations.py)
# ---------------------------------------------------------------------------

DEVICE     = "cuda"
BATCH_SIZE = 256
DIM_IN     = 512
DIM_HIDDEN = 2048
DIM_OUT    = 512


class MLPActivations(nn.Module):
    """
    Four-layer MLP with heterogeneous activations.

      Linear(512  -> 2048) -> ReLU
      Linear(2048 -> 2048) -> GELU
      Linear(2048 -> 2048) -> SiLU
      Linear(2048 -> 512)  -> Tanh
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(DIM_IN,     DIM_HIDDEN, bias=True)
        self.fc2 = nn.Linear(DIM_HIDDEN, DIM_HIDDEN, bias=True)
        self.fc3 = nn.Linear(DIM_HIDDEN, DIM_HIDDEN, bias=True)
        self.fc4 = nn.Linear(DIM_HIDDEN, DIM_OUT,    bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = F.silu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x


def get_model_and_input() -> tuple:
    """
    Workload interface — return (uncompiled model on CUDA, input tensor on CUDA).

    Non-graph optimizations: none applied here. OPT-1 (BF16 dtype promotion) is a
    graph pass applied inside the backend at the Aten IR level, so the public
    model/input stay float32. OPT-2 (max-autotune) and OPT-3 (epilogue fusion) are
    Inductor configuration applied inside the backend, not model state. No
    channels_last (no NCHW conv tensors) and no batch padding (B=256, all GEMM
    M/N/K dims are multiples of 16, already BF16 Tensor-Core aligned).

    Compilation and warmup are handled externally by run_workload.py via
    --compile-backend mlp_activations_opt.
    """
    assert torch.cuda.is_available(), "CUDA required"
    model = MLPActivations().to(DEVICE).eval()
    x     = torch.randn(BATCH_SIZE, DIM_IN, device=DEVICE)
    return model, x


if __name__ == "__main__":
    model, x = get_model_and_input()
    compiled = torch.compile(model, backend=BACKEND_NAME)
    with torch.no_grad():
        out = compiled(x)
    print(f"Output shape: {out.shape}, dtype: {out.dtype}")  # expect (256, 512)
