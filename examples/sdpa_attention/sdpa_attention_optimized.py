"""
sdpa_attention_optimized.py — optimized workload + custom torch.compile() backend.

Implements the three proposals from optimizations.json against the
SDPAAttentionBlock multi-head self-attention workload:

  OPT-1  bf16 dtype promotion        (ir_level: aten)
         Casts the operands of every aten.mm / aten.addmm / aten.bmm to
         bfloat16 so the GEMMs route to the Blackwell tensor-core HGEMM path
         instead of the CUTLASS SIMT FP32 (cutlass_80_simt_sgemm) kernel, and
         casts the SDPA q/k/v to bf16 so attention routes to the bf16
         FlashAttention path instead of the f32 sm80 mem-efficient kernel.
         Each GEMM/SDPA output is cast back to fp32 immediately so the bf16
         region stays local to the compute op and the surrounding LayerNorm /
         residual-add graph remains fp32 (numerically safe, baseline-comparable).

  OPT-2  QKV fusion                   (ir_level: functional)
         Fuses the three bias-free [512,512] q/k/v projections that all consume
         the single shared ln_pre activation into one wide [512,1536] F.linear +
         three slices. One wide GEMM (grid ~384 blocks) replaces three narrow
         serial launches (grid ~128 blocks each). Runs at the functional level
         because the shared-activation identity is destroyed by AOTAutograd's
         per-consumer views at the aten level.

  OPT-3  Inductor freezing            (ir_level: inductor_config)
         Scoped config_patches={"freezing": True} so eval-mode constant weights
         are folded and pre-laid-out for the (post-OPT-1) tensor-core HGEMM,
         removing per-call weight transpose/pack work.

The backend funnel is: functional -> compile_fx(inner=aten, config_patches).
This guarantees OPT-2 (functional) precedes OPT-1 (aten) precedes OPT-3
(inductor_config) without any explicit cross-level prerequisite edge.

get_model_and_input() exposes the same interface as the baseline so the capture
pipeline can profile this script unchanged. Importing this module registers the
backend "sdpa_attention_opt" via @register_backend.
"""
from __future__ import annotations

import functools
import logging
from typing import Callable

import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
from torch._dynamo import register_backend

# Rule 1: import the callable functions, NOT the module.
from torch._inductor.compile_fx import compile_fx, compile_fx_inner

from nvidia.operator_profiler.fx import UniqueSubgraphRegistry

logger = logging.getLogger(__name__)

DEVICE     = "cuda"
BATCH_SIZE = 8
SEQ_LEN    = 512
DIM        = 512
NUM_HEADS  = 8
HEAD_DIM   = DIM // NUM_HEADS   # 64

TARGET_DTYPE = torch.bfloat16


# ---------------------------------------------------------------------------
# Model (identical structure to the baseline workload)
# ---------------------------------------------------------------------------
class SDPAAttentionBlock(nn.Module):
    """Multi-head self-attention using F.scaled_dot_product_attention.

    Separate bias-free Q/K/V projections so the three linears are visible as
    distinct nodes feeding the SAME ln_pre activation — the signal OPT-2's
    functional QKV-fusion pass keys on.
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
        residual = x
        x = self.ln_pre(x)

        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k_proj(x).view(B, T, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(x).view(B, T, NUM_HEADS, HEAD_DIM).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(attn_out)
        out = self.ln_post(out + residual)
        return out


# ===========================================================================
# OPT-2 — QKV fusion (ir_level: functional)
# ===========================================================================
# Match F.linear by identity against the functional builtin, with a __name__
# fallback (it binds to a builtin on some torch builds).
_LINEAR_FNS = {torch.nn.functional.linear}
try:  # pragma: no cover - build dependent
    _LINEAR_FNS.add(torch._C._nn.linear)
except Exception:
    pass


def _is_linear(n: fx.Node) -> bool:
    return (
        n.op == "call_function"
        and (n.target in _LINEAR_FNS or getattr(n.target, "__name__", "") == "linear")
    )


def _fpass_fuse_qkv(gm: fx.GraphModule) -> fx.GraphModule:
    """Fuse 3 bias-free projections sharing one activation into one wide F.linear.

    Functional level: the three projections are F.linear(x, W) nodes that share
    the IDENTICAL activation node and carry clean weight params. After
    AOTAutograd decomposition the activation is shattered into per-consumer
    views and (with OPT-1) per-mm casts, so an aten-level matcher would no-op.
    """
    g = gm.graph
    groups: dict[fx.Node, list[fx.Node]] = {}
    for n in g.nodes:
        if _is_linear(n) and isinstance(n.args[0], fx.Node):
            groups.setdefault(n.args[0], []).append(n)

    fused = 0
    for act, lins in groups.items():
        if len(lins) < 3:
            continue
        q, k, v = lins[:3]
        # bias-free only: bias as 3rd positional arg or as kwarg must be absent.
        if any(
            (len(n.args) > 2 and n.args[2] is not None) or n.kwargs.get("bias") is not None
            for n in (q, k, v)
        ):
            continue
        wq, wk, wv = q.args[1], k.args[1], v.args[1]

        def _out(w: fx.Node):
            mv = w.meta.get("example_value", w.meta.get("val"))
            return int(mv.shape[0]) if mv is not None else None

        nq, nk, nv = _out(wq), _out(wk), _out(wv)
        if None in (nq, nk, nv):
            logger.warning("[OPT-2 fuse_qkv] missing weight meta — pass not applied")
            continue

        with g.inserting_before(q):
            w_cat = g.call_function(torch.ops.aten.cat.default, ([wq, wk, wv], 0))
            fused_lin = g.call_function(q.target, (act, w_cat))

            def _chunk(lo: int, hi: int) -> fx.Node:
                s = g.call_function(torch.ops.aten.slice.Tensor, (fused_lin, -1, lo, hi))
                # Last-dim slice is a strided view; clone to contiguous so the
                # downstream .view(B,T,H,Hd) stays valid.
                return g.call_function(
                    torch.ops.aten.clone.default,
                    (s,),
                    {"memory_format": torch.contiguous_format},
                )

            q_out = _chunk(0, nq)
            k_out = _chunk(nq, nq + nk)
            v_out = _chunk(nq + nk, nq + nk + nv)

        q.replace_all_uses_with(q_out)
        k.replace_all_uses_with(k_out)
        v.replace_all_uses_with(v_out)
        for d in (q, k, v):
            if not d.users:
                g.erase_node(d)
        fused += 1
        logger.info(
            "[OPT-2 fuse_qkv] Fused 3 projections into 1 linear (N=%d+%d+%d) [functional]",
            nq, nk, nv,
        )
        break

    if not fused:
        logger.warning(
            "[OPT-2 fuse_qkv] No 3-way shared-activation linear triplet found — pass not applied"
        )
        return gm

    g.eliminate_dead_code()
    g.lint()
    gm.recompile()
    return gm


# ===========================================================================
# OPT-1 — bf16 dtype promotion (ir_level: aten)
# ===========================================================================
_MM_OVERLOADS = {
    torch.ops.aten.mm.default,
    torch.ops.aten.addmm.default,
    torch.ops.aten.bmm.default,
}
def _sdpa_targets() -> set:
    """SDPA aten overloads present in this torch build (names vary across builds).

    The dispatcher hardware-selects one of these for F.scaled_dot_product_attention;
    we cast operands for whichever the decomposed graph actually contains.
    """
    targets = set()
    for name in (
        "_scaled_dot_product_efficient_attention",
        "_scaled_dot_product_flash_attention",
        "_scaled_dot_product_cudnn_attention",
        "scaled_dot_product_flash_attention",  # bare-name variant on some builds
    ):
        op = getattr(torch.ops.aten, name, None)
        if op is not None:
            try:
                targets.add(op.default)
            except Exception:
                pass
    return targets


_SDPA_TARGETS = _sdpa_targets()


def _apass_promote_bf16(gm: fx.GraphModule) -> fx.GraphModule:
    """Cast GEMM operands and SDPA q/k/v to bf16; cast each output back to fp32.

    Op-target pass (does not read weight VALUES): matches purely by node target,
    so it is robust to FakeTensors. Casting the GEMM operands routes them to the
    Blackwell tensor-core HGEMM path; casting the SDPA q/k/v routes attention to
    the bf16 FlashAttention path. Each compute op's result is immediately cast
    back to fp32, keeping the bf16 region local to the GEMM/SDPA so the
    surrounding LayerNorm / residual-add graph stays fp32 (numerically safe and
    directly comparable to the fp32 baseline output).
    """
    try:
        g = gm.graph
        matched = False

        def _cast(arg: fx.Node, before: fx.Node, dtype) -> fx.Node:
            with g.inserting_before(before):
                return g.call_function(
                    torch.ops.prims.convert_element_type.default,
                    (arg, dtype),
                )

        def _cast_outputs_back(node: fx.Node, getitem_indices=None) -> None:
            """Insert fp32 cast on each downstream tensor consumer of `node`.

            For single-output ops (mm/addmm/bmm) we wrap the op result directly.
            For tuple-output ops (SDPA) we wrap the getitem(node, 0) consumers.
            """
            with g.inserting_after(node):
                back = g.call_function(
                    torch.ops.prims.convert_element_type.default,
                    (node, torch.float32),
                )
            # Re-point every existing user (except the cast itself) at the fp32 cast.
            for user in list(node.users):
                if user is back:
                    continue
                user.replace_input_with(node, back)

        # --- GEMMs: cast operands to bf16, result back to fp32 ---
        for node in list(g.nodes):
            if node.op == "call_function" and node.target in _MM_OVERLOADS:
                new_args = [
                    _cast(a, node, TARGET_DTYPE) if isinstance(a, fx.Node) else a
                    for a in node.args
                ]
                node.args = tuple(new_args)
                _cast_outputs_back(node)
                matched = True

        # --- SDPA: cast q/k/v to bf16; cast getitem(0) result back to fp32 ---
        for node in list(g.nodes):
            if node.op == "call_function" and node.target in _SDPA_TARGETS:
                new_args = list(node.args)
                for i in range(min(3, len(new_args))):
                    if isinstance(new_args[i], fx.Node):
                        new_args[i] = _cast(new_args[i], node, TARGET_DTYPE)
                node.args = tuple(new_args)
                # Output is a tuple; cast getitem(0) consumers back to fp32.
                import operator as _operator
                for user in list(node.users):
                    if (
                        user.op == "call_function"
                        and user.target is _operator.getitem
                        and user.args[1] == 0
                    ):
                        with g.inserting_after(user):
                            back = g.call_function(
                                torch.ops.prims.convert_element_type.default,
                                (user, torch.float32),
                            )
                        for cons in list(user.users):
                            if cons is back:
                                continue
                            cons.replace_input_with(user, back)
                matched = True

        if not matched:
            logger.warning("[OPT-1 promote_bf16] No GEMM/SDPA nodes found — pass not applied")
            return gm

        g.lint()
        gm.recompile()
        logger.info("[OPT-1 promote_bf16] Promoted GEMM + SDPA operands to bf16 [aten]")
    except Exception as exc:  # high confidence: log + return gm
        logger.warning("[OPT-1 promote_bf16] Failed: %s", exc)
    return gm


# ===========================================================================
# OPT-3 — Inductor freezing (ir_level: inductor_config)
# ===========================================================================
def _cfg_freeze_constants() -> dict:
    """Return scoped Inductor config_patches enabling weight freezing.

    Valid only for inference (eval / no_grad), which this workload uses. Folds
    and pre-lays-out the (post-OPT-1) bf16 constant weights for the tensor-core
    HGEMM, eliminating per-call weight transpose/pack.
    """
    return {"freezing": True}


# ===========================================================================
# Pass registry + IR-level router
# ===========================================================================
PASS_REGISTRY = [
    {"id": "OPT-2", "level": "functional",      "fn": _fpass_fuse_qkv},
    {"id": "OPT-1", "level": "aten",            "fn": _apass_promote_bf16},
    {"id": "OPT-3", "level": "inductor_config", "fn": _cfg_freeze_constants},
]


def _passes(level: str):
    return [p for p in PASS_REGISTRY if p["level"] == level]


def _run_functional_passes(gm: fx.GraphModule) -> fx.GraphModule:
    """LEVEL 1 — rewrite the Dynamo graph BEFORE compile_fx owns it."""
    for p in _passes("functional"):
        try:
            gm = p["fn"](gm)
        except Exception as exc:
            logger.warning("[%s] functional pass no-op: %s", p["id"], exc)
    return gm


def _repropagate_meta(gm: fx.GraphModule, example_inputs) -> None:
    """Repopulate meta['val'] on inserted nodes after a structural rewrite.

    OPT-1 inserts convert_element_type nodes; give them fake meta so Inductor's
    lowering sees correct dtypes. Best-effort: a failure here is non-fatal
    because compile_fx_inner re-derives meta during lowering.
    """
    try:
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.passes.fake_tensor_prop import FakeTensorProp

        fake_mode = None
        for n in gm.graph.nodes:
            v = n.meta.get("val")
            if v is not None and hasattr(v, "fake_mode"):
                fake_mode = v.fake_mode
                break
        if fake_mode is None:
            fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
        with fake_mode:
            fake_args = [
                fake_mode.from_tensor(a) if isinstance(a, torch.Tensor) else a
                for a in example_inputs
            ]
        FakeTensorProp(gm, mode=fake_mode).propagate(*fake_args)
    except Exception as exc:
        logger.warning("[_repropagate_meta] best-effort meta prop skipped: %s", exc)


def _aten_inner_compile(
    gm: fx.GraphModule, example_inputs, *, real_inputs=None, **kwargs
) -> Callable:
    """LEVEL 2 — Inductor inner_compile hook (post-AOTAutograd, decomposed Aten).

    Runs aten-level passes, then delegates to the real compile_fx_inner
    (Aten -> Triton). OPT-1 is an op-target pass and does not need real_inputs,
    but we build the lookup for forward-compatibility / weight-value passes.
    """
    weight_source = real_inputs if real_inputs is not None else example_inputs
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    _ = {ph: t for ph, t in zip(placeholders, weight_source)}  # ph_to_tensor (unused by OPT-1)

    for p in _passes("aten"):
        try:
            gm = p["fn"](gm)
            _repropagate_meta(gm, example_inputs)
        except Exception as exc:
            logger.warning("[%s] aten pass no-op: %s", p["id"], exc)

    return compile_fx_inner(gm, example_inputs, **kwargs)


def _config_patches() -> dict:
    """LEVEL 3 — scoped Inductor config_patches for THIS compile_fx call only."""
    patches: dict = {}
    for p in _passes("inductor_config"):
        try:
            patches.update(p["fn"]() or {})
        except Exception as exc:
            logger.warning("[%s] config pass skipped: %s", p["id"], exc)
    return patches


def _compile_unit(gm: fx.GraphModule, example_inputs) -> Callable:
    """The fixed three-stage funnel: functional -> aten (inner) -> inductor_config."""
    gm = _run_functional_passes(gm)
    inner = functools.partial(_aten_inner_compile, real_inputs=list(example_inputs))
    return compile_fx(gm, example_inputs, inner_compile=inner, config_patches=_config_patches())


def _capture_partition_inputs(split_gm: fx.GraphModule, example_inputs: list) -> dict:
    """Capture actual input tensors per partition by running split_gm once."""
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


# ===========================================================================
# Backend entry point
# ===========================================================================
@register_backend
def sdpa_attention_opt(gm: fx.GraphModule, example_inputs) -> Callable:
    logger.info("sdpa_attention_opt backend: starting (functional -> aten -> inductor_config)")
    registry = UniqueSubgraphRegistry(gm)
    equiv_map = registry.build_partition_equivalence_map()

    if not equiv_map:
        logger.info("sdpa_attention_opt: no repeated layers, flat compile path")
        return _compile_unit(gm, example_inputs)

    logger.info("sdpa_attention_opt: %d duplicate partition(s), dedup path", len(equiv_map))
    partition_inputs = _capture_partition_inputs(registry.split, example_inputs)
    for rep_name, rep_mod in registry.unique_reps:
        inputs = partition_inputs.get(rep_name, example_inputs)
        compiled = _compile_unit(rep_mod, inputs)
        rep_mod.forward = compiled
        for _, dup_mod in registry.duplicates_of(rep_name):
            dup_mod.forward = compiled

    return lambda *args: registry.split(*args)


# ---------------------------------------------------------------------------
# Workload interface
# ---------------------------------------------------------------------------
def get_model_and_input() -> tuple:
    """Return (raw_model, input_tensor) on CUDA — same interface as the baseline.

    No non-graph optimizations apply here: bf16 is handled inside the backend
    (OPT-1) so the baseline-comparable fp32 model + fp32 input are returned, and
    there is no Conv/channels_last or batch-padding opportunity in this workload.
    """
    assert torch.cuda.is_available(), "CUDA required"
    model = SDPAAttentionBlock().to(DEVICE).eval()
    x = torch.randn(BATCH_SIZE, SEQ_LEN, DIM, device=DEVICE)
    return model, x


if __name__ == "__main__":
    model, x = get_model_and_input()
    compiled = torch.compile(model, backend="sdpa_attention_opt")
    with torch.no_grad():
        out = compiled(x)
    print("output shape:", out.shape, "dtype:", out.dtype)
