"""
sdpa_attention_optimized.py — Custom torch.compile() backend for SDPAAttentionBlock.

Registered backend: ``sdpa_attention_opt``

Implements the three optimizations from optimizations.json as named FX graph passes
operating on the *decomposed Aten IR*. The passes are installed via Inductor's
``inner_compile`` hook (Strategy D): ``compile_fx`` owns AOTAutograd, the decomp
table, the boxed calling convention and the fwd/bwd partitioner; we only swap the
leaf compiler (Aten -> Triton). We do NOT use ``aot_autograd(fw_compiler=compile_fx)``
— on torch 2.11 that raises ``AssertionError: Expected tensors only, but got list``
inside ``copy_misaligned_inputs``.

Dependency DAG / apply order (from the proposal global_notes):

    OPT-1 (dtype) -> OPT-2 (QKV fusion) -> OPT-3 (layout)   ;   apply order: 1, 2, 3

  OPT-1  dtype_promotion (high)  — bf16 promotion. Casts the projection-GEMM
                          activation + weight operands and the SDPA q/k/v operands
                          to bfloat16, re-routing every aten.mm from the FP32 cutlass
                          SIMT path (tensor_core_active_pct = 0.0) onto a bf16
                          tensor-core HMMA GEMM and the FP32 mem-efficient SDPA kernel
                          onto the bf16 flash/efficient path. A single down-cast
                          restores fp32 at the block output so the external contract
                          is unchanged. MUST run first: OPT-2/OPT-3 build derived
                          buffers that must inherit the bf16 runtime dtype.
  OPT-2  fusion (high)    — QKV weight fusion. The three bias-free projections share
                          the same ln_pre activation; decomposed they are three
                          aten.mm(act, t(W_q/k/v)) nodes. Concatenate the three
                          weights into one wide operand (aten.cat), issue one
                          aten.mm, and recover Q/K/V with three aten.slice views (no
                          copy). Three sub-1-wave launches -> one wider GEMM. Runs
                          after OPT-1 so the concatenated weight is bf16.
  OPT-3  memory_layout (low) — weight pre-transpose / alignment. After OPT-2 the
                          fused QKV weight and the out-proj weight feed aten.mm via an
                          aten.t.default. Fold the transpose into a contiguous
                          pre-transposed buffer so CUTLASS can pick an aligned
                          (align8) tile. Confidence low: with bf16 in place CUTLASS may
                          already select an aligned tile, so this is a graceful no-op
                          when the transpose chain is absent or already folded.

IR-level mechanics (torch 2.11):
  All passes run inside ``_aten_inner_compile``, which ``compile_fx`` calls with the
  fully decomposed Aten IR graph (after AOTAutograd). aten.mm /
  aten._scaled_dot_product_efficient_attention / the native_layer_norm decomposition
  only appear at this level. ``example_inputs`` here may be FakeTensors (Inductor
  traces under FakeTensorMode), so weight-VALUE-reading passes (OPT-2/OPT-3) use the
  ``real_inputs`` threaded from the backend for the placeholder->tensor lookup.

compile_mode = "inductor" (from optimizations.json): standard FX pass approach.
"""
from __future__ import annotations

import functools
import logging
import operator
from typing import Callable, Optional

import torch
import torch.fx as fx
from torch._dynamo import register_backend
from torch._inductor.compile_fx import compile_fx, compile_fx_inner  # functions, not module

from nvidia.operator_profiler.fx import UniqueSubgraphRegistry

logger = logging.getLogger(__name__)

_BF16 = torch.bfloat16
_FP32 = torch.float32

_MM = torch.ops.aten.mm.default
_T = torch.ops.aten.t.default
_CAT = torch.ops.aten.cat.default
_SLICE = torch.ops.aten.slice.Tensor
_INT_MAX = 9223372036854775807

# prims.convert_element_type.default is lowered directly by Inductor; aten._to_copy
# is in Inductor's decomp table and can trip a fallback assertion when inserted into
# an already-decomposed Aten graph.
_CONVERT_DTYPE = torch.ops.prims.convert_element_type.default

# SDPA decomposes (hardware-selected) to one of these tuple-returning ops.
_SDPA_TARGETS = frozenset(
    {
        torch.ops.aten._scaled_dot_product_efficient_attention.default,
        torch.ops.aten._scaled_dot_product_flash_attention.default,
        torch.ops.aten.scaled_dot_product_attention.default,
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _is_mm(n) -> bool:
    return isinstance(n, fx.Node) and n.op == "call_function" and n.target is _MM


def _is_sdpa(n) -> bool:
    return isinstance(n, fx.Node) and n.op == "call_function" and n.target in _SDPA_TARGETS


def _node_dtype(n) -> Optional[torch.dtype]:
    if not isinstance(n, fx.Node):
        return None
    val = n.meta.get("val", None)
    if val is None or not hasattr(val, "dtype"):
        return None
    return val.dtype


def _insert_bf16_cast(g: fx.Graph, src: fx.Node, before: fx.Node) -> fx.Node:
    """Insert a prims.convert_element_type to bf16 of ``src`` directly before ``before``.
    No-op (returns src) if src is already bf16."""
    if _node_dtype(src) is _BF16:
        return src
    with g.inserting_before(before):
        return g.call_function(_CONVERT_DTYPE, (src, _BF16))


def _build_ph_to_tensor(gm: fx.GraphModule, weight_source) -> dict:
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    return {ph: t for ph, t in zip(placeholders, weight_source)}


# ---------------------------------------------------------------------------
# OPT-1 — bf16 dtype promotion. Confidence: high. MUST run first.
# Casts every projection-GEMM operand (activation + weight) and the SDPA q/k/v
# operands to bf16, then down-casts the final block output back to fp32.
# Graceful no-op if no mm / SDPA node is present.
# ---------------------------------------------------------------------------
def _pass_bf16_promotion(gm: fx.GraphModule) -> fx.GraphModule:
    try:
        g = gm.graph
        promoted = 0

        # 1) Promote the matmul operands. Each projection is aten.mm(act, W_or_t(W)).
        for mm in list(g.nodes):
            if not _is_mm(mm):
                continue
            act, w = mm.args[0], mm.args[1]
            if not (isinstance(act, fx.Node) and isinstance(w, fx.Node)):
                continue
            cast_act = _insert_bf16_cast(g, act, mm)
            cast_w = _insert_bf16_cast(g, w, mm)
            if cast_act is act and cast_w is w:
                continue  # already bf16
            mm.args = (cast_act, cast_w) + tuple(mm.args[2:])
            promoted += 1

        # 2) Promote SDPA q/k/v operands (args[0:3]).
        for sdpa in list(g.nodes):
            if not _is_sdpa(sdpa):
                continue
            new_args = list(sdpa.args)
            changed = False
            for i in range(min(3, len(new_args))):
                a = new_args[i]
                if isinstance(a, fx.Node):
                    cast = _insert_bf16_cast(g, a, sdpa)
                    if cast is not a:
                        new_args[i] = cast
                        changed = True
            if changed:
                sdpa.args = tuple(new_args)
                promoted += 1

        if promoted == 0:
            logger.warning("[OPT-1 bf16_promotion] No mm/SDPA nodes found — pass not applied")
            return gm

        # 3) Down-cast the graph outputs back to fp32 so the block contract is unchanged.
        output_node = next((n for n in g.nodes if n.op == "output"), None)
        downcast = 0
        if output_node is not None:

            def _maybe_downcast(val):
                nonlocal downcast
                if isinstance(val, fx.Node) and _node_dtype(val) is _BF16:
                    with g.inserting_before(output_node):
                        dc = g.call_function(_CONVERT_DTYPE, (val, _FP32))
                    downcast += 1
                    return dc
                return val

            output_node.args = torch.fx.map_arg(output_node.args, _maybe_downcast)

        g.lint()
        gm.recompile()
        logger.info(
            "[OPT-1 bf16_promotion] Promoted %d mm/SDPA operand group(s) to bf16, "
            "%d output down-cast(s) to fp32 [Aten IR]",
            promoted,
            downcast,
        )
    except Exception as e:
        logger.warning("[OPT-1 bf16_promotion] Failed: %s", e)
    return gm


# ---------------------------------------------------------------------------
# OPT-2 — QKV weight fusion. Confidence: high. Runs AFTER OPT-1.
# Three bias-free projections sharing the same activation become one wider mm
# plus three slice views. Operates on the (bf16-cast) operands so the fused
# weight inherits bf16. Graceful no-op if no qualifying triplet found.
# ---------------------------------------------------------------------------
def _unwrap_weight_operand(w_node: fx.Node):
    """Return (weight_carrier_node, is_transposed). The mm weight arg is either a
    raw placeholder/cast or an aten.t.default wrapping one."""
    if isinstance(w_node, fx.Node) and w_node.op == "call_function" and w_node.target is _T:
        return w_node.args[0], True
    return w_node, False


def _pass_fuse_qkv(gm: fx.GraphModule) -> fx.GraphModule:
    try:
        g = gm.graph

        # Group mm nodes by their shared activation operand (args[0]).
        groups: dict[fx.Node, list[fx.Node]] = {}
        for mm in list(g.nodes):
            if not _is_mm(mm):
                continue
            act = mm.args[0]
            if isinstance(act, fx.Node):
                groups.setdefault(act, []).append(mm)

        fused = 0
        for act, mm_list in groups.items():
            if len(mm_list) < 3:
                continue
            q_mm, k_mm, v_mm = mm_list[0], mm_list[1], mm_list[2]
            w_q, w_k, w_v = q_mm.args[1], k_mm.args[1], v_mm.args[1]
            if not all(isinstance(w, fx.Node) for w in (w_q, w_k, w_v)):
                continue

            # Resolve per-projection output width N from weight meta (cat on dim 1).
            try:
                n_q = int(w_q.meta["val"].shape[1])
                n_k = int(w_k.meta["val"].shape[1])
            except Exception:
                logger.warning("[OPT-2 fuse_qkv] Missing weight shape meta — skipping fusion")
                continue

            with g.inserting_before(q_mm):
                w_cat = g.call_function(_CAT, ([w_q, w_k, w_v], 1))  # [K, N_q+N_k+N_v]
                fused_mm = g.call_function(_MM, (act, w_cat))        # [M, N_q+N_k+N_v]
                q_slice = g.call_function(_SLICE, (fused_mm, 1, 0, n_q))
                k_slice = g.call_function(_SLICE, (fused_mm, 1, n_q, n_q + n_k))
                v_slice = g.call_function(_SLICE, (fused_mm, 1, n_q + n_k, _INT_MAX))

            q_mm.replace_all_uses_with(q_slice)
            k_mm.replace_all_uses_with(k_slice)
            v_mm.replace_all_uses_with(v_slice)
            for dead in (q_mm, k_mm, v_mm):
                if not dead.users:
                    g.erase_node(dead)

            fused += 1
            logger.info(
                "[OPT-2 fuse_qkv] Fused 3 projections into 1 mm (N=%d+%d+...) [Aten IR]",
                n_q,
                n_k,
            )
            break  # one QKV block per (sub)graph

        if fused == 0:
            logger.warning(
                "[OPT-2 fuse_qkv] No 3-way shared-activation mm triplet found — pass not applied"
            )
            return gm

        g.eliminate_dead_code()
        g.lint()
        gm.recompile()
    except Exception as e:
        logger.warning("[OPT-2 fuse_qkv] Failed: %s", e)
    return gm


# ---------------------------------------------------------------------------
# OPT-3 — weight pre-transpose / alignment. Confidence: low. Runs AFTER OPT-1/2.
# Detection-first: locate aten.t.default(weight) consumed as the weight arg of an
# aten.mm.default, materialize a contiguous pre-transposed buffer from the real
# weight tensor, and re-point the mm to consume it directly (eliminating the 'tn'
# transpose and raising load alignment). Graceful no-op when no such transpose
# chain exists (e.g. Inductor already folded it, or the weights were fused in OPT-2
# without a surviving aten.t node).
# ---------------------------------------------------------------------------
def _pass_pretranspose_weights(gm: fx.GraphModule, ph_to_tensor: dict) -> fx.GraphModule:
    try:
        g = gm.graph
        replaced = 0
        for t_node in list(g.nodes):
            if not (
                t_node.op == "call_function" and t_node.target is _T
            ):
                continue
            ph_node = t_node.args[0]
            if not (isinstance(ph_node, fx.Node) and ph_node.op == "placeholder"):
                continue
            weight = ph_to_tensor.get(ph_node)
            if weight is None or not isinstance(weight, torch.Tensor):
                continue

            for user in list(t_node.users):
                if not (_is_mm(user) and user.args[1] is t_node):
                    continue
                buf_name = f"_pretransposed_weight_{replaced}"
                weight_T = weight.t().contiguous().to(_BF16)
                gm.register_buffer(buf_name, weight_T)
                with g.inserting_before(user):
                    buf_node = g.get_attr(buf_name)
                    new_mm = g.call_function(_MM, (user.args[0], buf_node))
                user.replace_all_uses_with(new_mm)
                g.erase_node(user)
                replaced += 1

        if replaced == 0:
            logger.info(
                "[OPT-3 pretranspose] No aten.t->mm weight chain found — left intact "
                "(transpose already folded or weights fused in OPT-2). No-op."
            )
            return gm

        for t_node in list(g.nodes):
            if t_node.op == "call_function" and t_node.target is _T and not t_node.users:
                g.erase_node(t_node)

        g.eliminate_dead_code()
        g.lint()
        gm.recompile()
        logger.info(
            "[OPT-3 pretranspose] Pre-transposed %d weight(s) into contiguous bf16 buffer(s) [Aten IR]",
            replaced,
        )
    except Exception as e:
        logger.warning("[OPT-3 pretranspose] Failed: %s", e)
    return gm


# ---------------------------------------------------------------------------
# Re-propagate FakeTensor meta after structural rewrites so downstream passes and
# Inductor can read meta['val'] on inserted mm/cat/convert nodes.
# ---------------------------------------------------------------------------
def _repropagate_meta(gm: fx.GraphModule, example_inputs) -> None:
    try:
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        fake_inputs = []
        fake_mode = None
        for ph, ex in zip(placeholders, example_inputs):
            val = ph.meta.get("val", None)
            if val is None:
                val = ex
            fake_inputs.append(val)
            fm = getattr(val, "fake_mode", None)
            if fm is not None:
                fake_mode = fm
        from torch.fx.passes.fake_tensor_prop import FakeTensorProp

        if fake_mode is not None:
            with fake_mode:
                FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(*fake_inputs)
        else:
            FakeTensorProp(gm).propagate_dont_convert_inputs(*fake_inputs)
    except Exception as e:
        logger.warning("[sdpa_attention_opt] meta re-propagation skipped: %s", e)


# ---------------------------------------------------------------------------
# Inductor inner_compile hook (Strategy D). compile_fx hands us the fully
# decomposed Aten IR graph. Apply OPT-1 -> OPT-2 -> OPT-3 then delegate to the
# real compile_fx_inner (Aten -> Triton). Each pass is independently try/guarded.
# ---------------------------------------------------------------------------
def _aten_inner_compile(
    gm: fx.GraphModule, example_inputs, *, real_inputs=None, **kwargs
) -> Callable:
    weight_source = real_inputs if real_inputs is not None else example_inputs
    ph_to_tensor = _build_ph_to_tensor(gm, weight_source)

    # OPT-1 (high) — MUST run first. Op-target pass (no weight values needed).
    gm = _pass_bf16_promotion(gm)
    _repropagate_meta(gm, example_inputs)

    # OPT-2 (high) — needs weight shape meta from OPT-1's operands.
    gm = _pass_fuse_qkv(gm)
    _repropagate_meta(gm, example_inputs)

    # OPT-3 (low) — weight-VALUE-reading; uses real_inputs ph_to_tensor lookup.
    gm = _pass_pretranspose_weights(gm, ph_to_tensor)
    _repropagate_meta(gm, example_inputs)

    return compile_fx_inner(gm, example_inputs, **kwargs)


def _compile_with_aten_passes(gm: fx.GraphModule, example_inputs) -> Callable:
    """Compile a (sub)graph through Inductor with the Aten-IR passes installed via
    inner_compile. compile_fx owns AOTAutograd / decomp / boxing / partitioner."""
    inner = functools.partial(_aten_inner_compile, real_inputs=list(example_inputs))
    return compile_fx(gm, example_inputs, inner_compile=inner)


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

    Installs the Aten-IR pass chain (OPT-1 -> OPT-2 -> OPT-3) via compile_fx's
    inner_compile hook (Strategy D). Dedup-aware per Rule 9: a single attention
    block has no repeated structural partitions, so the flat compile path is taken
    (preserving cross-op Inductor fusion). The dedup branch is retained for models
    with repeated identical blocks.
    """
    logger.info("sdpa_attention_opt backend: starting")

    registry = UniqueSubgraphRegistry(gm)
    equiv_map = registry.build_partition_equivalence_map()

    if not equiv_map:
        logger.info("sdpa_attention_opt: no repeated layers, flat compile path")
        return _compile_with_aten_passes(gm, example_inputs)

    logger.info("sdpa_attention_opt: %d duplicate partition(s), dedup path", len(equiv_map))
    partition_inputs = _capture_partition_inputs(registry.split, example_inputs)
    for rep_name, rep_mod in registry.unique_reps:
        inputs = partition_inputs.get(rep_name, example_inputs)
        compiled = _compile_with_aten_passes(rep_mod, inputs)
        rep_mod.forward = compiled
        for _, dup_mod in registry.duplicates_of(rep_name):
            dup_mod.forward = compiled

    return lambda *args: registry.split(*args)


# ---------------------------------------------------------------------------
# Workload interface. OPT-1/2/3 are all graph-level passes installed by the
# backend; there is no non-graph (eager-side) lever for this workload. The model
# and input are returned in FP32 (matches the profile dtype) — bf16 promotion is
# applied selectively inside the graph, not by casting the whole module.
# ---------------------------------------------------------------------------
DEVICE = "cuda"
BATCH_SIZE = 8
SEQ_LEN = 512
DIM = 512


def get_model_and_input() -> tuple:
    """Return (raw_model, input_tensor) on CUDA — uncompiled, unwarmed.

    No non-graph optimization applies here (dtype promotion is selective and done
    in-graph; there is no conv/layout lever). Returned in FP32 to match the profile
    and to let OPT-1 own the bf16 boundary.
    """
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
