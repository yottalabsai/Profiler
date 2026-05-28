"""
sdpa_attention_optimized.py — Custom torch.compile() backend for SDPAAttentionBlock.

Registered backend: ``sdpa_attention_opt``

Implements three FX graph passes at the Aten IR level. Each pass corresponds to
one optimization from optimizations.json (linear chain OPT-1 -> OPT-2 -> OPT-3):

  OPT-1  dtype_promotion  — BF16 cast of aten.mm operands and SDPA Q/K/V inputs,
                            routing FP32-SIMT GEMMs and the FP32 fmha kernel onto
                            the Tensor-Core (HMMA) path. LayerNorm kept in FP32.
                            Confidence: high. Prerequisite for OPT-2 and OPT-3.
                            A global TF32 fallback is also enabled at import.
  OPT-2  fusion           — fuse the 3 sibling Q/K/V aten.mm projections (shared
                            post-LayerNorm activation) into one [4096,512]@[512,1536]
                            GEMM + 3x aten.slice. Confidence: high. After OPT-1.
  OPT-3  memory_layout    — speculative weight pre-transpose, gated behind OPT-2.
                            Confidence: low -> detection + WARNING, no transform
                            (cuBLAS on Blackwell typically absorbs the transpose;
                            constant pre-transpose is not materializable from the
                            FakeTensor graph inputs at this IR level).

IR-level mechanics (torch 2.11):
  The graph torch.compile hands a @register_backend function is the *functional*
  Dynamo graph (`linear`, `scaled_dot_product_attention`, `layer_norm`) — NOT
  Aten IR. Aten IR (`aten.mm`, `aten._scaled_dot_product_efficient_attention`,
  `aten.permute`) only appears after AOTAutograd decomposition, inside Inductor.
  The supported torch 2.11 injection point for Aten-IR passes is Inductor's
  ``post_grad_custom_pre_pass`` hook, which runs on the fully decomposed,
  functionalized Aten graph immediately before lowering. We install our pass
  chain there and delegate the AOTAutograd + lowering pipeline to ``compile_fx``.

  At this post-grad level a bias-free ``nn.Linear`` weight transpose appears as
  ``aten.permute.default(weight_placeholder, [1, 0])`` (not ``aten.t.default``),
  and dtype casts must use ``prims.convert_element_type.default`` (``aten._to_copy``
  collides with an Inductor decomp/fallback). Graph inputs are FakeTensors, so all
  passes are *structural* rewrites — they never read weight values; OPT-2 fuses by
  concatenating the weight placeholder nodes with an ``aten.cat`` graph node.

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

from nvidia.operator_profiler.fx import UniqueSubgraphRegistry

logger = logging.getLogger(__name__)

# OPT-1 TF32 fallback (optimizations.json code_hint): engages TF32 Tensor Cores
# for any residual FP32 matmul even if a BF16 rewrite does not apply. Cheap,
# global, accuracy-safe relative to BF16. Set once at module import.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

_PROMOTE_DTYPE = torch.bfloat16

_MM = torch.ops.aten.mm.default
_PERMUTE = torch.ops.aten.permute.default
_CAT = torch.ops.aten.cat.default
_SLICE = torch.ops.aten.slice.Tensor
_CONVERT = torch.ops.prims.convert_element_type.default


def _is_sdpa(node: fx.Node) -> bool:
    return node.op == "call_function" and "scaled_dot_product" in str(node.target)


# ---------------------------------------------------------------------------
# OPT-1 — dtype promotion to BF16 on aten.mm operands and SDPA Q/K/V inputs.
# Confidence: high (no `matched` gate; exception => WARNING + return graph).
# Prerequisite for OPT-2 and OPT-3.
# ---------------------------------------------------------------------------
def _pass_promote_dtype(g: fx.Graph) -> int:
    """Insert prims.convert_element_type casts so every aten.mm and the SDPA
    inputs run in BF16. Each mm's operands are cast to BF16 and the mm result is
    cast back to FP32 so downstream FP32 consumers (LayerNorm, residual add) are
    unaffected. SDPA Q/K/V are cast to BF16 in place so the efficient/flash kernel
    selects its BF16 variant, removing the FP32 register spill. LayerNorm is never
    touched, so its reductions stay in FP32 (accuracy guard)."""
    mm_count = 0
    for n in list(g.nodes):
        if n.op == "call_function" and n.target is _MM:
            with g.inserting_before(n):
                lhs_b = g.call_function(_CONVERT, (n.args[0], _PROMOTE_DTYPE))
                rhs_b = g.call_function(_CONVERT, (n.args[1], _PROMOTE_DTYPE))
            n.update_arg(0, lhs_b)
            n.update_arg(1, rhs_b)
            with g.inserting_after(n):
                back = g.call_function(_CONVERT, (n, torch.float32))
            n.replace_all_uses_with(back, delete_user_cb=lambda u: u is not back)
            mm_count += 1

    sdpa_count = 0
    for n in list(g.nodes):
        if _is_sdpa(n):
            with g.inserting_before(n):
                q_b = g.call_function(_CONVERT, (n.args[0], _PROMOTE_DTYPE))
                k_b = g.call_function(_CONVERT, (n.args[1], _PROMOTE_DTYPE))
                v_b = g.call_function(_CONVERT, (n.args[2], _PROMOTE_DTYPE))
            n.update_arg(0, q_b)
            n.update_arg(1, k_b)
            n.update_arg(2, v_b)
            sdpa_count += 1

    g.lint()
    logger.info(
        "[OPT-1 promote_dtype] Promoted %d aten.mm operand-pair(s) and %d SDPA "
        "input-triple(s) to %s [Aten IR]; LayerNorm kept FP32; TF32 fallback on",
        mm_count,
        sdpa_count,
        _PROMOTE_DTYPE,
    )
    return mm_count


def _mm_activation_source(mm: fx.Node) -> fx.Node:
    """Walk back through the (post-OPT-1) convert/reshape chain on mm.args[0] to
    the underlying activation producer, used to group siblings sharing one input."""
    cur = mm.args[0]
    for _ in range(3):
        if getattr(cur, "op", None) == "call_function" and cur.args:
            cur = cur.args[0]
        else:
            break
    return cur


def _mm_weight_permute(mm: fx.Node):
    """For a post-OPT-1 mm whose args[1] is convert(permute(placeholder)), return
    (placeholder_node, permute_dims). Returns (None, None) if it doesn't match."""
    conv = mm.args[1]
    if not (getattr(conv, "op", None) == "call_function" and conv.target is _CONVERT):
        return None, None
    perm = conv.args[0]
    if not (getattr(perm, "op", None) == "call_function" and perm.target is _PERMUTE):
        return None, None
    ph = perm.args[0]
    if getattr(ph, "op", None) != "placeholder":
        return None, None
    return ph, perm.args[1]


# ---------------------------------------------------------------------------
# OPT-2 — QKV fusion: merge 3 sibling aten.mm projections sharing one activation
# into one wide GEMM + 3 aten.slice. Confidence: high (graceful no-op if absent).
# Runs after OPT-1, so operands are already wrapped in BF16 convert nodes.
# ---------------------------------------------------------------------------
def _pass_fuse_qkv(g: fx.Graph) -> bool:
    """Detect three aten.mm nodes that share the same post-LayerNorm activation
    source and whose weights are distinct permuted placeholders. Concatenate the
    three weight placeholders along the output (N) axis with an aten.cat graph
    node (structural — no host-side tensor read, FakeTensor-safe), cast to BF16,
    emit one aten.mm, cast back to FP32, and slice the [M, 3N] result into three
    [M, N] tensors feeding the existing reshape/SDPA chain.

    Fresh permute nodes for the weights are inserted at the earliest sibling mm so
    the fused subgraph stays topologically valid (placeholders are always defined
    at graph top, so the permutes are available there)."""
    order = {n: i for i, n in enumerate(g.nodes)}
    groups: dict[int, list] = {}
    for n in g.nodes:
        if n.op == "call_function" and n.target is _MM:
            groups.setdefault(id(_mm_activation_source(n)), []).append(n)

    for _, mm_list in groups.items():
        if len(mm_list) < 3:
            continue
        q_n, k_n, v_n = mm_list[0], mm_list[1], mm_list[2]
        act_b = q_n.args[0]  # BF16-converted activation node (shared)

        ph_q, dims_q = _mm_weight_permute(q_n)
        ph_k, dims_k = _mm_weight_permute(k_n)
        ph_v, dims_v = _mm_weight_permute(v_n)
        if ph_q is None or ph_k is None or ph_v is None:
            logger.warning("[OPT-2 fuse_qkv] Weight placeholders not resolvable — skipping group")
            continue

        # Output (N) widths from the FakeTensor meta on each placeholder.
        try:
            n_q = ph_q.meta["val"].shape[0]
            n_k = ph_k.meta["val"].shape[0]
            n_v = ph_v.meta["val"].shape[0]
        except Exception:
            logger.warning("[OPT-2 fuse_qkv] Missing shape meta on weight placeholders — skipping")
            continue

        first_mm = min(mm_list[:3], key=lambda n: order[n])
        with g.inserting_before(first_mm):
            p_q = g.call_function(_PERMUTE, (ph_q, dims_q))
            p_k = g.call_function(_PERMUTE, (ph_k, dims_k))
            p_v = g.call_function(_PERMUTE, (ph_v, dims_v))
            # cat permuted weights ([in, out]) along output axis -> [in, n_q+n_k+n_v]
            w_cat = g.call_function(_CAT, ([p_q, p_k, p_v], 1))
            w_cat_b = g.call_function(_CONVERT, (w_cat, _PROMOTE_DTYPE))
            fused_mm = g.call_function(_MM, (act_b, w_cat_b))
            fused_fp32 = g.call_function(_CONVERT, (fused_mm, torch.float32))
            q_s = g.call_function(_SLICE, (fused_fp32, 1, 0, n_q))
            k_s = g.call_function(_SLICE, (fused_fp32, 1, n_q, n_q + n_k))
            v_s = g.call_function(_SLICE, (fused_fp32, 1, n_q + n_k, n_q + n_k + n_v))

        # After OPT-1, each original mm's users point at its FP32 convert-back node;
        # redirect those convert-back nodes to the corresponding slice.
        for orig, sl in ((q_n, q_s), (k_n, k_s), (v_n, v_s)):
            users = list(orig.users)
            if not users:
                continue
            users[0].replace_all_uses_with(sl)

        g.eliminate_dead_code()
        g.lint()
        logger.info(
            "[OPT-2 fuse_qkv] Fused 3 Q/K/V aten.mm into 1 GEMM (N=%d) + 3 slices "
            "[Aten IR]; fused weight cat cast to %s",
            n_q + n_k + n_v,
            _PROMOTE_DTYPE,
        )
        return True

    logger.warning("[OPT-2 fuse_qkv] 3-sibling QKV pattern not found — pass not applied")
    return False


# ---------------------------------------------------------------------------
# OPT-3 — weight pre-transpose. Confidence: LOW => detection + WARNING only.
# Gated behind OPT-2: the fused QKV weight is already concatenated/permuted by
# OPT-2, so the only remaining permute(placeholder)->mm is the output projection.
# A genuine constant pre-transpose is not materializable here (graph inputs are
# FakeTensors with no readable storage), and modern cuBLAS on Blackwell usually
# absorbs the transpose internally, so this pass reports candidates and no-ops.
# ---------------------------------------------------------------------------
def _pass_pretranspose_weights(g: fx.Graph) -> None:
    candidates = 0
    for n in g.nodes:
        if not (n.op == "call_function" and n.target is _MM):
            continue
        w = n.args[1]
        inner = w.args[0] if getattr(w, "op", None) == "call_function" and w.args else None
        if (
            inner is not None
            and getattr(inner, "op", None) == "call_function"
            and inner.target is _PERMUTE
            and getattr(inner.args[0], "op", None) == "placeholder"
        ):
            candidates += 1

    if candidates:
        logger.warning(
            "[OPT-3 pretranspose] Detected %d permute(weight)->mm candidate(s) "
            "(e.g. output projection). Speculative/low-confidence: constant "
            "pre-transpose is not materializable from FakeTensor graph inputs and "
            "cuBLAS on Blackwell typically absorbs the transpose — no transform applied.",
            candidates,
        )
    else:
        logger.info(
            "[OPT-3 pretranspose] No remaining permute(weight)->mm candidates "
            "(QKV weights already fused by OPT-2) — pass not applied"
        )


# ---------------------------------------------------------------------------
# Aten-IR pass chain, installed as Inductor's post_grad_custom_pre_pass. Runs on
# the decomposed, functionalized Aten graph just before lowering. Order respects
# the prerequisite chain OPT-1 -> OPT-2 -> OPT-3 from optimizations.json.
# ---------------------------------------------------------------------------
def _aten_pass_chain(g: fx.Graph) -> fx.Graph:
    try:
        promoted = _pass_promote_dtype(g)  # OPT-1 (high) — must run first
        if promoted:
            _pass_fuse_qkv(g)  # OPT-2 (high) — fusion before layout pass
        _pass_pretranspose_weights(g)  # OPT-3 (low) — detection only
    except Exception as e:  # never crash the compile
        logger.warning("[sdpa_attention_opt] Aten pass chain failed: %s", e)
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
def sdpa_attention_opt(gm: fx.GraphModule, example_inputs) -> Callable:
    """Custom torch.compile backend for SDPAAttentionBlock.

    Installs the Aten-IR pass chain (OPT-1/2/3) via Inductor's
    post_grad_custom_pre_pass, then delegates AOTAutograd + lowering to
    compile_fx. Dedup-aware per Rule 9: a single attention block has no repeated
    structure, so the flat compile path is taken (preserving cross-layer Inductor
    fusion); the dedup branch is retained for structural reuse if the model grows.
    """
    logger.info("sdpa_attention_opt backend: starting")
    _install_aten_passes()

    registry = UniqueSubgraphRegistry(gm)
    equiv_map = registry.build_partition_equivalence_map()

    if not equiv_map:
        logger.info("sdpa_attention_opt: no repeated layers, flat compile path")
        return compile_fx(gm, example_inputs)

    logger.info(
        "sdpa_attention_opt: %d duplicate partition(s), dedup path", len(equiv_map)
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
# Workload interface. Non-graph optimizations would live here; for this workload
# no layout/batch-shape change is required (no Conv2d; batch already tiled), and
# the dtype work is a graph-level pass (OPT-1) per optimizations.json. The model
# is returned in FP32; the backend promotes to BF16 inside the graph.
# ---------------------------------------------------------------------------
DEVICE = "cuda"
BATCH_SIZE = 8
SEQ_LEN = 512
DIM = 512


def get_model_and_input() -> tuple:
    """Return (raw_model, input_tensor) on CUDA — uncompiled, unwarmed."""
    assert torch.cuda.is_available(), "CUDA required"
    from sdpa_attention import SDPAAttentionBlock

    model = SDPAAttentionBlock().to(DEVICE).eval()
    x = torch.randn(BATCH_SIZE, SEQ_LEN, DIM, device=DEVICE)
    return model, x


if __name__ == "__main__":
    model, x = get_model_and_input()
    compiled = torch.compile(model, backend="sdpa_attention_opt")
    with torch.no_grad():
        out = compiled(x)
    print("output shape:", out.shape, "dtype:", out.dtype)
