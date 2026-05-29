"""
mlp_activations_optimized.py — Custom torch.compile() backend for MLPActivations.

Registered backend: ``mlp_activations_opt``

Implements the three optimizations from optimizations.json. Dependency DAG:

    OPT-1  ->  {OPT-2, OPT-3}

  OPT-1  dtype_promotion  — BF16 cast of every aten.mm / aten.addmm matmul operand
                            at the post-grad Aten IR level, routing the FP32-SIMT
                            cuBLAS GEMMs (tensor_core_active_pct == 0.0 on all 8)
                            onto the Blackwell WGMMA Tensor-Core path. The matmul
                            result is cast back to FP32 so the downstream bias add
                            and ReLU/GELU/SiLU/Tanh epilogues and the final output
                            are numerically unchanged (cuBLAS/Triton accumulate in
                            FP32). A global TF32 fallback is also set at import.
                            Confidence: high. Prerequisite for OPT-2 and OPT-3.

                            NOTE on placement: optimizations.json proposes OPT-1 as a
                            non-graph ``model.to(torch.bfloat16)`` in
                            get_model_and_input(). We instead promote only the matmul
                            operands inside the graph and cast the result back to FP32.
                            Same Tensor-Core engagement, but the activation epilogues,
                            bias adds, and the final output stay FP32 — tighter
                            numerical agreement with the FP32 baseline than a
                            whole-model BF16 cast, and the change is localized to the
                            8 GEMMs the profile flagged. The equivalent non-graph cast
                            is documented in get_model_and_input() as an alternative.

  OPT-2  fusion           — Inductor lowering policy (NOT a graph node insertion):
                            enable Triton GEMM templates with epilogue fusion so the
                            bias add + activation fuse onto the matmul output tile and
                            the separate triton_poi_fused_addmm_<act> kernels (and
                            their full-tensor DRAM round-trip) are eliminated.
                            Configured via torch._inductor.config:
                              max_autotune_gemm         = True
                              epilogue_fusion           = True
                              max_autotune_gemm_backends = "TRITON"
                            Confidence: medium (autotuner-dependent template choice).
                            Most effective after OPT-1 sets operands to BF16, because
                            the Tensor-Core MMA template keeps the output tile on-chip
                            for the epilogue.

  OPT-3  dtype_promotion  — FP8 (e4m3) row/tensor-wise scaled_mm on the four square
                            [256,2048]x[2048,2048] hidden GEMMs (op_id=5/7/15/17).
                            Confidence: LOW. Per the pass-confidence policy a low
                            confidence pass is DETECTION-ONLY: it walks the graph,
                            identifies the qualifying square K=2048 matmuls, logs an
                            FP8 candidacy report, and returns the graph UNCHANGED.
                            It does not insert aten._scaled_mm. Reasons it is gated as
                            a no-op (from optimizations.json notes): accuracy for the
                            GELU/SiLU dynamic range under per-tensor scaling is
                            unvalidated, the M=256 batch may not amortize quantize
                            overhead, and the transform was not corroborated by search
                            tooling. The detection log gives /report the candidate set
                            and the _scaled_mm recipe so the transform can be promoted
                            once validated against the FP32 baseline.

IR-level mechanics (torch 2.11, RTX PRO 6000 Blackwell):
  The graph torch.compile hands a @register_backend function is the *functional*
  Dynamo graph (`linear`, `relu`, `gelu`, `silu`, `tanh`) — NOT Aten IR. Aten IR
  (`aten.mm`, `aten.addmm`) only appears after AOTAutograd decomposition, inside
  Inductor. The supported torch 2.11 injection point for Aten-IR passes is
  Inductor's ``post_grad_custom_pre_pass`` hook, which runs on the fully
  decomposed, functionalized Aten graph immediately before lowering. We install
  the OPT-1/OPT-3 pass chain there and delegate AOTAutograd + lowering to
  ``compile_fx``. (The aot_autograd fw_compiler injection point is broken on
  torch 2.11 — boxed-args AssertionError in copy_misaligned_inputs.)

  At this post-grad level dtype casts must use ``prims.convert_element_type.default``
  — ``aten._to_copy.default`` collides with an Inductor "both fallback and decomp"
  assertion. Graph inputs are FakeTensors, so OPT-1 is a purely structural rewrite
  that never reads weight values.

  A bias-carrying nn.Linear can lower either to ``aten.addmm.default(bias, x, wT)``
  or, post-grad, to a separate ``aten.mm.default`` + ``aten.add``; OPT-1 handles
  both aten.mm and aten.addmm targets so all 8 GEMMs are promoted regardless.

compile_mode = "inductor" (from optimizations.json): standard FX pass approach.
"""
from __future__ import annotations

import logging
from typing import Callable

import torch
import torch.fx as fx
import torch._inductor.config as inductor_config
from torch._dynamo import register_backend
from torch._inductor.compile_fx import compile_fx  # function, not module
from torch._subclasses.fake_tensor import FakeTensor

from nvidia.operator_profiler.fx import UniqueSubgraphRegistry

logger = logging.getLogger(__name__)

# OPT-1 TF32 fallback (optimizations.json code_hint): engages TF32 Tensor Cores
# for any residual FP32 matmul even if the BF16 rewrite does not apply. Cheap,
# global, accuracy-safe relative to BF16. Set once at module import.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

_PROMOTE_DTYPE = torch.bfloat16

_MM = torch.ops.aten.mm.default
_ADDMM = torch.ops.aten.addmm.default
_CONVERT = torch.ops.prims.convert_element_type.default

# addmm.default(bias, mat1, mat2); mm.default(mat1, mat2). `base` is the index of
# the first matmul operand (skipping the bias for addmm — the bias add stays FP32).
_MM_TARGET_BASE = {_MM: 0, _ADDMM: 1}

# OPT-3 candidacy: the K dimension of the four square hidden GEMMs that satisfy the
# FP8 64/64/32 WGMMA tile minimum (per optimizations.json op_id=5/7/15/17).
_FP8_K_DIM = 2048
_FP8_N_DIM = 2048


# ---------------------------------------------------------------------------
# OPT-2 — Inductor epilogue-fusion lowering policy. Not a graph pass: enabling
# these config flags lets Inductor emit a Triton matmul template that fuses the
# bias add + activation onto the GEMM output tile, removing the standalone
# triton_poi_fused_addmm_<act> kernels and their DRAM round-trip. Set once at
# import so they are in effect before any compile_fx call. Confidence: medium —
# whether max-autotune selects the Triton template over cuBLAS is autotuner- and
# shape-dependent; verify in the Inductor debug dir.
# ---------------------------------------------------------------------------
def _install_inductor_fusion_config() -> None:
    inductor_config.max_autotune_gemm = True
    inductor_config.epilogue_fusion = True
    # Force the Triton template path — the cuBLAS extern call cannot fuse an
    # epilogue. Guard the assignment so import never fails if the attribute name
    # shifts across point releases.
    try:
        inductor_config.max_autotune_gemm_backends = "TRITON"
    except Exception as e:  # pragma: no cover
        logger.warning("[OPT-2 fusion] could not set max_autotune_gemm_backends: %s", e)
    logger.info(
        "[OPT-2 fusion] Inductor epilogue fusion enabled "
        "(max_autotune_gemm=True, epilogue_fusion=True, backends=TRITON); "
        "bias+activation fuse onto the GEMM template, removing standalone "
        "triton_poi_fused_addmm_<act> kernels"
    )


# ---------------------------------------------------------------------------
# Meta propagation for newly inserted nodes.
#
# Nodes created via g.call_function() inside a post_grad_custom_pre_pass carry
# no FakeTensor metadata. g.lint() validates structure only, so the pass returns
# cleanly, but Inductor's built-in post-grad pattern matcher
# (unfuse_bias_add_to_pointwise -> match.replace_by_example) later reads
# arg.meta["val"] (preferred) / arg.meta["example_value"] (fallback) on the
# operands feeding addmm/mm. A missing key raises KeyError deep inside Inductor.
#
# We therefore compute each created convert_element_type's FakeTensor result the
# same way the rest of the post-grad graph carries it — under the source
# operand's own FakeTensorMode, by applying the prims op to the operand's fake
# val — and set BOTH meta["val"] and meta["example_value"].
# ---------------------------------------------------------------------------
def _set_convert_meta(new_node: fx.Node, src_node: fx.Node, dtype: torch.dtype) -> None:
    """Populate meta['val'] and meta['example_value'] on a convert_element_type
    node created from ``src_node`` cast to ``dtype``."""
    src_val = src_node.meta.get("val", src_node.meta.get("example_value"))
    if not isinstance(src_val, FakeTensor):
        logger.warning(
            "[OPT-1 promote_dtype] source node %s carries no FakeTensor val; "
            "skipping meta propagation for its cast",
            src_node,
        )
        return
    fake_mode = src_val.fake_mode
    with fake_mode:
        out_val = torch.ops.prims.convert_element_type.default(src_val, dtype)
    new_node.meta["val"] = out_val
    new_node.meta["example_value"] = out_val
    if "tensor_meta" in src_node.meta:
        from torch.fx.passes.shape_prop import _extract_tensor_metadata

        new_node.meta["tensor_meta"] = _extract_tensor_metadata(out_val)


def _mm_operand_shapes(n: fx.Node):
    """Return (M, K, N) for an aten.mm/aten.addmm node from operand FakeTensor meta,
    or None if meta is unavailable. For addmm(bias, mat1, mat2): mat1=[M,K], mat2=[K,N]."""
    base = _MM_TARGET_BASE[n.target]
    try:
        mat1 = n.args[base]
        mat2 = n.args[base + 1]
        s1 = mat1.meta.get("val", mat1.meta.get("example_value")).shape
        s2 = mat2.meta.get("val", mat2.meta.get("example_value")).shape
        return int(s1[0]), int(s1[1]), int(s2[1])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# OPT-1 — dtype promotion to BF16 on aten.mm / aten.addmm matmul operands.
# Confidence: high (no `matched` gate; exception => WARNING + return graph).
# Prerequisite for OPT-2 and OPT-3.
# ---------------------------------------------------------------------------
def _pass_promote_dtype(g: fx.Graph) -> int:
    """Insert prims.convert_element_type casts so every aten.mm / aten.addmm runs
    its matmul in BF16. Each matmul operand is cast to BF16 (the addmm bias at
    args[0] is left in FP32 so the bias add itself is unaffected), and the matmul
    result is cast back to FP32 so the downstream activation epilogue and final
    output are numerically unchanged. No weight values are read."""
    mm_count = 0
    for n in list(g.nodes):
        if n.op != "call_function" or n.target not in _MM_TARGET_BASE:
            continue
        base = _MM_TARGET_BASE[n.target]

        # Cast each matmul operand (args[base:]) to BF16 just before the node.
        with g.inserting_before(n):
            for i in range(base, len(n.args)):
                operand = n.args[i]
                cast = g.call_function(_CONVERT, (operand, _PROMOTE_DTYPE))
                _set_convert_meta(cast, operand, _PROMOTE_DTYPE)
                n.update_arg(i, cast)

        # The matmul node n now produces a BF16 result; refresh its own fake val so
        # the result cast below derives correct meta from it.
        n_src_val = n.meta.get("val", n.meta.get("example_value"))
        if isinstance(n_src_val, FakeTensor):
            with n_src_val.fake_mode:
                n.meta["val"] = n_src_val.to(_PROMOTE_DTYPE)
            n.meta["example_value"] = n.meta["val"]

        # Cast the matmul result back to FP32 for the downstream FP32 consumers.
        with g.inserting_after(n):
            back = g.call_function(_CONVERT, (n, torch.float32))
        _set_convert_meta(back, n, torch.float32)
        n.replace_all_uses_with(back, delete_user_cb=lambda u: u is not back)
        mm_count += 1

    g.lint()
    logger.info(
        "[OPT-1 promote_dtype] Promoted %d aten.mm/aten.addmm matmul(s) to %s "
        "[Aten IR]; accumulation/bias kept FP32; result cast back to FP32; "
        "TF32 fallback on",
        mm_count,
        _PROMOTE_DTYPE,
    )
    return mm_count


# ---------------------------------------------------------------------------
# OPT-3 — FP8 (e4m3) scaled_mm on the four square K=2048 hidden GEMMs.
# Confidence: LOW => DETECTION-ONLY. Identifies the qualifying matmuls and logs an
# FP8 candidacy report + the _scaled_mm recipe; returns the graph UNCHANGED. The
# transform is gated on accuracy validation against the FP32 baseline (see module
# docstring and optimizations.json OPT-3 notes). Runs after OPT-1 so the reported
# operands are already on the BF16 path.
# ---------------------------------------------------------------------------
def _pass_detect_fp8_candidates(g: fx.Graph) -> int:
    """Detect square K=2048 GEMMs that satisfy the FP8 64/64/32 WGMMA tile minimum.

    DOES NOT TRANSFORM. For each candidate it logs M/K/N and the scaled_mm recipe:
        scale_w = W.abs().amax() / 448.0          # e4m3 max
        W_fp8   = (W / scale_w).to(torch.float8_e4m3fn)
        out     = aten._scaled_mm(x_fp8, W_fp8, scale_a=sx, scale_b=scale_w,
                                  out_dtype=torch.bfloat16)
    """
    candidates = 0
    cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
    fp8_supported = cap >= (9, 0)
    for n in list(g.nodes):
        if n.op != "call_function" or n.target not in _MM_TARGET_BASE:
            continue
        shapes = _mm_operand_shapes(n)
        if shapes is None:
            continue
        m, k, n_dim = shapes
        if k == _FP8_K_DIM and n_dim == _FP8_N_DIM:
            candidates += 1
            logger.info(
                "[OPT-3 detect_fp8] candidate square GEMM %s (M=%d, K=%d, N=%d) — "
                "FP8 e4m3 _scaled_mm eligible (K satisfies 64/64/32 tile min). "
                "DETECTION ONLY: not transformed (low confidence, accuracy "
                "unvalidated). Recipe: scale_w=W.abs().amax()/448; "
                "W_fp8=(W/scale_w).to(float8_e4m3fn); "
                "aten._scaled_mm(x_fp8, W_fp8, scale_a=sx, scale_b=scale_w, "
                "out_dtype=bfloat16)",
                n.name, m, k, n_dim,
            )
    if candidates:
        logger.warning(
            "[OPT-3 detect_fp8] %d FP8 candidate GEMM(s) found; device capability "
            "%s, FP8 Tensor-Core path %s. Pass is detection-only — promote to a "
            "scaled_mm rewrite only after validating BF16/FP8 output vs the FP32 "
            "baseline.",
            candidates, cap, "available" if fp8_supported else "UNAVAILABLE",
        )
    else:
        logger.warning(
            "[OPT-3 detect_fp8] No square K=2048 GEMM candidates found — pass not "
            "applied"
        )
    return candidates


# ---------------------------------------------------------------------------
# Aten-IR pass chain, installed as Inductor's post_grad_custom_pre_pass. Runs on
# the decomposed, functionalized Aten graph just before lowering. Order respects
# the DAG OPT-1 -> {OPT-2, OPT-3}: OPT-1 (BF16) first so OPT-3 detection observes
# the promoted operands; OPT-2 is Inductor config applied at backend entry.
# ---------------------------------------------------------------------------
def _aten_pass_chain(g: fx.Graph) -> fx.Graph:
    try:
        _pass_promote_dtype(g)        # OPT-1 (high) — must run first; unblocks OPT-2/3
        _pass_detect_fp8_candidates(g)  # OPT-3 (low) — detection only, no transform
    except Exception as e:  # never crash the compile
        logger.warning("[mlp_activations_opt] Aten pass chain failed: %s", e)
    return g


def _install_aten_passes() -> None:
    """Register the Aten-IR pass chain as Inductor's post-grad pre-pass."""
    inductor_config.post_grad_custom_pre_pass = _aten_pass_chain


def _capture_partition_inputs(split_gm: fx.GraphModule, example_inputs: list) -> dict:
    """Capture actual input tensors for each partition by running split_gm once."""
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


@register_backend
def mlp_activations_opt(gm: fx.GraphModule, example_inputs) -> Callable:
    """Custom torch.compile backend for MLPActivations.

    Installs the OPT-1 BF16 promotion + OPT-3 FP8 detection passes via Inductor's
    post_grad_custom_pre_pass and the OPT-2 epilogue-fusion lowering policy via
    torch._inductor.config, then delegates AOTAutograd + lowering to compile_fx.
    Dedup-aware per Rule 9: the four MLP layers are structurally distinct (different
    activations and shapes), so build_partition_equivalence_map() finds no repeats
    and the flat compile path is taken — preserving cross-layer Inductor fusion. The
    dedup branch is retained for structural reuse if the model grows repeated blocks.
    """
    logger.info("mlp_activations_opt backend: starting")
    _install_inductor_fusion_config()  # OPT-2 (config, must precede compile_fx)
    _install_aten_passes()             # OPT-1 + OPT-3 (post-grad Aten-IR passes)

    registry = UniqueSubgraphRegistry(gm)
    equiv_map = registry.build_partition_equivalence_map()

    if not equiv_map:
        logger.info("mlp_activations_opt: no repeated layers, flat compile path")
        return compile_fx(gm, example_inputs)

    logger.info(
        "mlp_activations_opt: %d duplicate partition(s), dedup path", len(equiv_map)
    )
    partition_inputs = _capture_partition_inputs(registry.split, example_inputs)
    for rep_name, rep_mod in registry.unique_reps:
        inputs = partition_inputs.get(rep_name, example_inputs)
        compiled = compile_fx(rep_mod, inputs)
        rep_mod.forward = compiled
        for _, dup_mod in registry.duplicates_of(rep_name):
            dup_mod.forward = compiled

    return lambda *args: registry.split(*args)


# ---------------------------------------------------------------------------
# Workload interface. Per optimizations.json, OPT-1 (dtype) is implemented as a
# graph-level matmul-operand promotion (keeping activations/bias/output FP32) and
# OPT-2 (fusion) is an Inductor config policy — neither is a non-graph transform.
# There is no Conv2d (no channels_last) and the batch is already a multiple of
# common tiles (256), so get_model_and_input() applies no non-graph optimization.
# The model is returned in FP32; the backend promotes the matmuls inside the graph.
#
# Equivalent non-graph alternative for OPT-1 (proposal's literal code_hint), kept
# here for reference — uncomment to cast the whole model + input to BF16 instead of
# the per-matmul graph promotion (looser numerics, simpler):
#     model = MLPActivations().to(DEVICE).eval().to(torch.bfloat16)
#     x = torch.randn(BATCH_SIZE, DIM_IN, device=DEVICE, dtype=torch.bfloat16)
# ---------------------------------------------------------------------------
DEVICE = "cuda"
BATCH_SIZE = 256
DIM_IN = 512


def get_model_and_input() -> tuple:
    """Return (raw_model, input_tensor) on CUDA — uncompiled, unwarmed."""
    assert torch.cuda.is_available(), "CUDA required"
    from mlp_activations import MLPActivations

    model = MLPActivations().to(DEVICE).eval()
    x = torch.randn(BATCH_SIZE, DIM_IN, device=DEVICE)
    return model, x


if __name__ == "__main__":
    model, x = get_model_and_input()
    compiled = torch.compile(model, backend="mlp_activations_opt")
    with torch.no_grad():
        out = compiled(x)
    print("output shape:", out.shape, "dtype:", out.dtype)
