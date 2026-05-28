"""
mlp_activations_optimized.py — custom torch.compile() backend for MLPActivations.

Implements the two transformations proposed in optimizations.json at the Aten IR
level (post-AOTAutograd, pre-Inductor):

  OPT-1  dtype_promotion  (priority 1, high confidence)
         Wrap every GEMM node with bf16 casts on the matmul operands and an
         fp32 cast on the result, so Inductor selects the Blackwell bf16
         tensor-core GEMM template instead of the FP32 SIMT path
         (tensor_core_active_pct = 0 on every GEMM in the baseline profile).

  OPT-2  epilogue fusion  (priority 2, medium confidence; MUST run after OPT-1)
         Enable Inductor epilogue_fusion + max_autotune_gemm so the per-layer
         bias-add and activation (relu/gelu/silu/tanh) fuse into the bf16 matmul
         template, removing the separate triton_poi_fused_addmm_* pointwise
         kernels and their GEMM-output round trip.

IR-level note
-------------
With bias=True, nn.Linear lowers to `aten.addmm.default(bias, x, w_t)` at the
Aten IR level (NOT bare `aten.mm.default`). The profile shows `aten::mm` because
Inductor decomposes addmm into mm+add *during* lowering, which is downstream of
this pass. OPT-1 therefore targets `aten.addmm.default` (the real node we see)
and also handles bare `aten.mm.default` defensively. The bf16 cast of the
addmm's matmul operands is what reaches Inductor's GEMM template selection.

The Aten-IR passes run via Inductor's `post_grad_custom_pre_pass` hook (the
decomposed Aten graph just before lowering), and the backend delegates to
`compile_fx` directly. On torch 2.11 a manual `aot_autograd(fw_compiler=
compile_fx)` composition mis-boxes runtime inputs (AssertionError in
copy_misaligned_inputs); routing through `compile_fx` as the dynamo backend lets
it drive AOTAutograd internally and apply the pass at the correct stage.

Backend registered name (for Stage 4 capture):  mlp_activations_opt
"""
from __future__ import annotations

import logging
from typing import Callable

import torch
import torch.fx as fx
import torch._inductor.config as ind_cfg
from torch._dynamo import register_backend
from torch._inductor.compile_fx import compile_fx

from nvidia.operator_profiler.fx import UniqueSubgraphRegistry

logger = logging.getLogger("mlp_activations_opt")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# Aten op handles
_ATEN_MM = torch.ops.aten.mm.default
_ATEN_ADDMM = torch.ops.aten.addmm.default
_ATEN_TO_DTYPE = torch.ops.aten.to.dtype
_ATEN_ADD_TENSOR = torch.ops.aten.add.Tensor

_ACTIVATIONS = {
    torch.ops.aten.relu.default,
    torch.ops.aten.gelu.default,
    torch.ops.aten.silu.default,
    torch.ops.aten.tanh.default,
}


def _val_to_dtype(node: fx.Node, dtype: torch.dtype):
    """Return a fake-tensor `meta['val']` for a to.dtype node, if the source has one."""
    src_val = node.meta.get("val")
    if src_val is not None and hasattr(src_val, "to"):
        try:
            return src_val.to(dtype)
        except Exception:  # pragma: no cover - defensive
            return None
    return None


# --------------------------------------------------------------------------- #
# OPT-1 — dtype promotion: route every GEMM onto the bf16 tensor-core path.    #
# --------------------------------------------------------------------------- #
def _promote_gemm_to_bf16(graph: fx.Graph) -> None:
    """
    Aten-IR graph pass (runs as Inductor's post_grad_custom_pre_pass, i.e. on the
    decomposed Aten graph just before Inductor lowering).

    For each GEMM node, cast its matmul operands to bf16, run the GEMM in bf16,
    and cast the result back to fp32 so downstream consumers (bias-add /
    activation epilogues, and the final model output) still see fp32.

      aten.addmm(bias, m1, m2)  ->  to_fp32(aten.addmm(to_bf16(bias),
                                                       to_bf16(m1),
                                                       to_bf16(m2)))
      aten.mm(m1, m2)           ->  to_fp32(aten.mm(to_bf16(m1), to_bf16(m2)))

    With bias=True, nn.Linear is `aten.addmm.default` at this IR level (the GEMM
    lives inside addmm); bare `aten.mm.default` is handled defensively. Inserted
    `to.dtype` nodes get a populated `meta['val']` so Inductor's post-grad fake
    propagation does not KeyError on a missing 'val'.

    High confidence: the GEMM pattern is guaranteed present in this MLP, so we do
    not gate on a `matched` flag; an exception is downgraded to a warning and the
    graph is left unchanged.
    """
    try:
        gemm_nodes = [
            n
            for n in list(graph.nodes)
            if n.op == "call_function" and n.target in (_ATEN_ADDMM, _ATEN_MM)
        ]
        if not gemm_nodes:
            logger.warning(
                "[OPT-1 promote_gemm_to_bf16] No addmm/mm nodes found — pass not applied"
            )
            return

        promoted = 0
        for node in gemm_nodes:
            # addmm(bias, mat1, mat2): cast all three (addmm needs uniform dtype).
            # mm(mat1, mat2): cast both operands.
            cast_arg_idx = (0, 1, 2) if node.target is _ATEN_ADDMM else (0, 1)

            new_args = list(node.args)
            with graph.inserting_before(node):
                for i in cast_arg_idx:
                    src = node.args[i]
                    cast = graph.call_function(_ATEN_TO_DTYPE, (src, torch.bfloat16))
                    v = _val_to_dtype(src, torch.bfloat16)
                    if v is not None:
                        cast.meta["val"] = v
                    new_args[i] = cast
            node.args = tuple(new_args)

            # The GEMM node now produces bf16; reflect that in its own meta['val'].
            self_val = _val_to_dtype(node, torch.bfloat16)
            if self_val is not None:
                node.meta["val"] = self_val

            # Cast the bf16 GEMM result back to fp32 for the epilogue/output.
            with graph.inserting_after(node):
                back = graph.call_function(_ATEN_TO_DTYPE, (node, torch.float32))
            back_val = _val_to_dtype(node, torch.float32)
            if back_val is not None:
                back.meta["val"] = back_val
            node.replace_all_uses_with(back)
            # replace_all_uses_with rewrote back's own arg to itself; restore it.
            back.args = (node, torch.float32)
            promoted += 1

        graph.lint()
        logger.info(
            "[OPT-1 promote_gemm_to_bf16] Applied: %d GEMM node(s) promoted to bf16",
            promoted,
        )
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("[OPT-1 promote_gemm_to_bf16] Failed: %s — graph unchanged", e)


# --------------------------------------------------------------------------- #
# OPT-2 — epilogue fusion: bias-add + activation fold into the GEMM template.  #
# --------------------------------------------------------------------------- #
def _tag_epilogue_fusions(graph: fx.Graph) -> None:
    """
    Tag each `(GEMM) -> (fp32 cast) -> (activation)` chain so the relationship is
    discoverable for logging / inspection. The actual fusion is enabled by the
    Inductor config flags in `_configure_inductor()`; this pass annotates
    node.meta and counts fusible chains, degrading to a clean no-op when none
    are present.

    Note: after OPT-1 the GEMM's direct user is the fp32 `to.dtype` cast, and the
    activation consumes that cast — so we look one hop past the cast to find the
    activation. Medium confidence: gated on a `matched` count.
    """
    try:
        matched = 0
        for node in list(graph.nodes):
            if node.op != "call_function" or node.target not in (_ATEN_ADDMM, _ATEN_MM):
                continue
            # Walk GEMM -> {direct users, and users-of-fp32-cast}.
            candidates = set(node.users)
            for u in list(node.users):
                if u.target is _ATEN_TO_DTYPE:
                    candidates.update(u.users)
            for user in candidates:
                if user.target is _ATEN_ADD_TENSOR or user.target in _ACTIVATIONS:
                    user.meta["inductor_epilogue_of"] = node
                    matched += 1
        if matched == 0:
            logger.warning(
                "[OPT-2 tag_epilogue_fusions] No GEMM->pointwise epilogue chains "
                "found — relying on Inductor config flags only"
            )
            return
        logger.info(
            "[OPT-2 tag_epilogue_fusions] Applied: tagged %d epilogue node(s) "
            "for matmul-template fusion",
            matched,
        )
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("[OPT-2 tag_epilogue_fusions] Failed: %s — graph unchanged", e)


def _aten_pre_lowering_pass(graph: fx.Graph) -> None:
    """
    Inductor `post_grad_custom_pre_pass` entry point. Runs on the decomposed Aten
    graph immediately before Inductor lowering — the correct IR level for both
    OPT-1 and OPT-2 per optimizations.json. Order: OPT-1 (high conf) then OPT-2
    (prereq: OPT-1).
    """
    _promote_gemm_to_bf16(graph)
    _tag_epilogue_fusions(graph)


def _configure_inductor() -> None:
    """
    OPT-2 config half + OPT-1 pass registration. Enables epilogue fusion and the
    max-autotune GEMM path so the bf16 matmul template (selected after OPT-1)
    absorbs the bias + activation epilogue, and wires the Aten-IR pass into the
    Inductor post-grad pipeline. Idempotent; safe to call before every compile.
    """
    ind_cfg.post_grad_custom_pre_pass = _aten_pre_lowering_pass
    ind_cfg.epilogue_fusion = True
    ind_cfg.max_autotune_gemm = True
    logger.info(
        "[OPT-2 configure_inductor] epilogue_fusion=True, max_autotune_gemm=True; "
        "OPT-1/OPT-2 Aten pass registered as post_grad_custom_pre_pass"
    )


# --------------------------------------------------------------------------- #
# Registered backend.                                                          #
# --------------------------------------------------------------------------- #
@register_backend
def mlp_activations_opt(gm: fx.GraphModule, example_inputs) -> Callable:
    """
    Dedup-aware backend.

    The four MLP layers have distinct shapes / activations, so the registry finds
    no structural duplicates and the flat compile path is taken (which also
    preserves cross-layer Inductor fusion). The dedup branch is retained for
    parity with the standard backend template.

    IR note: the OPT-1/OPT-2 graph passes run at the Aten IR level via Inductor's
    `post_grad_custom_pre_pass` hook (wired in `_configure_inductor`), then we
    delegate to `compile_fx` as the dynamo backend directly. On torch 2.11 a
    manual `aot_autograd(fw_compiler=compile_fx)` wrapper mis-boxes runtime inputs
    (AssertionError in copy_misaligned_inputs); calling `compile_fx` as the
    backend lets it drive AOTAutograd internally and apply the pass at the right
    stage without that boxing bug.
    """
    logger.info("mlp_activations_opt backend: starting")
    _configure_inductor()

    registry = UniqueSubgraphRegistry(gm)
    equiv_map = registry.build_partition_equivalence_map()

    if not equiv_map:
        logger.info("mlp_activations_opt: no repeated layers, flat compile path")
        return compile_fx(gm, example_inputs)

    logger.info(
        "mlp_activations_opt: %d duplicate partition(s), dedup path", len(equiv_map)
    )
    for rep_name, rep_mod in registry.unique_reps:
        compiled = compile_fx(rep_mod, example_inputs)
        rep_mod.forward = compiled
        for _dup_name, dup_mod in registry.duplicates_of(rep_name):
            dup_mod.forward = compiled

    return lambda *args: registry.split(*args)


# --------------------------------------------------------------------------- #
# Workload interface.                                                          #
# --------------------------------------------------------------------------- #
DEVICE = "cuda"
BATCH_SIZE = 256
DIM_IN = 512
DIM_HIDDEN = 2048
DIM_OUT = 512

import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402


class MLPActivations(nn.Module):
    """Four-layer MLP with heterogeneous activations (matches baseline)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(DIM_IN, DIM_HIDDEN, bias=True)
        self.fc2 = nn.Linear(DIM_HIDDEN, DIM_HIDDEN, bias=True)
        self.fc3 = nn.Linear(DIM_HIDDEN, DIM_HIDDEN, bias=True)
        self.fc4 = nn.Linear(DIM_HIDDEN, DIM_OUT, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = F.silu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x


def get_model_and_input() -> tuple:
    """
    Return (raw_model, input_tensor) on CUDA, uncompiled and unwarmed.

    OPT-1 (dtype promotion) and OPT-2 (epilogue fusion) are graph-level passes
    that run inside the backend, so no non-graph (dtype/layout/batch) changes are
    applied here — the input stays fp32, [256, 512], exactly as the baseline.
    """
    assert torch.cuda.is_available(), "CUDA required"
    model = MLPActivations().to(DEVICE).eval()
    x = torch.randn(BATCH_SIZE, DIM_IN, device=DEVICE)
    return model, x


if __name__ == "__main__":
    model, x = get_model_and_input()
    compiled = torch.compile(model, backend="mlp_activations_opt")
    with torch.no_grad():
        out = compiled(x)
    print("output shape:", tuple(out.shape), "dtype:", out.dtype)
