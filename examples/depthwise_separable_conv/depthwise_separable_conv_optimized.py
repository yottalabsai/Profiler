"""
depthwise_separable_conv_optimized.py — custom torch.compile() backend for the
MobileNet-style depthwise-separable conv stack in depthwise_separable_conv.py.

Backend name:  depthwise_separable_conv_opt   (registered via @register_backend)

Optimizations implemented (from optimizations.json, routed by ir_level):

  OPT-1  Conv-BN fold via Inductor freezing      level: inductor_config  (high)
         -> config_patches={"freezing": True}. With eval()/no_grad the model
            weights + BN running stats become constants; Inductor folds each
            _native_batch_norm_legit_no_training into the preceding conv's
            weight/bias, killing the per-channel BN read-modify-write traffic
            (34% of attributed time in the baseline).

  OPT-2  bf16 dtype promotion                     level: aten             (medium)
         -> aten-level pass inserts a prims.convert_element_type(bf16) cast on the
            input of every aten.convolution.default (the dtype-conversion primitive
            Inductor lowers natively; aten._to_copy would abort lowering). Routes
            the 1x1 pointwise GEMMs onto
            a native bf16 tensor-core path (relieving the 8% occupancy / 224-reg
            wall on the TF32 cutlass_80 kernels) and halves byte traffic for the
            memory-bound depthwise + BN kernels. Runs BEFORE OPT-1 by funnel
            level order, so the folded conv weight inherits bf16.

  OPT-3  channels_last / NHWC                      level: non-graph        (medium)
         -> model + input converted to torch.channels_last in
            get_model_and_input(). The conv library kernels are already
            NHWC-native, so this removes the standalone NCHW->NHWC layout-copy
            kernel (triton_poi_fused_convolution_0) at the graph boundary and
            improves coalescing for the depthwise / BN epilogues.

The backend is the canonical three-stage funnel
    _run_functional_passes(gm) -> compile_fx(inner_compile=_aten_inner_compile,
                                              config_patches=_config_patches())
invoked identically on the flat graph and on every dedup representative. The
three DWSepBlocks have different channel counts (32->64, 64->128, 128->256) so
they are NOT structurally identical; in practice the dedup registry yields an
empty equivalence map and the flat-compile path is taken. The dedup path is kept
for interface uniformity.
"""
from __future__ import annotations

import functools
import logging
from typing import Callable

import torch
import torch.fx as fx
import torch.nn as nn

from torch._dynamo import register_backend
# Import the callable functions — NOT the module (module is not callable).
from torch._inductor.compile_fx import compile_fx, compile_fx_inner

from nvidia.operator_profiler.fx import UniqueSubgraphRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Workload constants (cross-validated against profile.json)
#   batch=16, in_channels=32, 56x56, fp32, CUDA. Output of the 3 blocks is
#   [16, 256, 56, 56].
# ---------------------------------------------------------------------------
DEVICE      = "cuda"
BATCH_SIZE  = 16
IN_CHANNELS = 32
HEIGHT      = 56
WIDTH       = 56


# ===========================================================================
# LEVEL 2 (aten) pass — OPT-2: bf16 dtype promotion on convolution inputs
# ===========================================================================
_CONV_TARGETS = frozenset({
    torch.ops.aten.convolution.default,
    torch.ops.aten.cudnn_convolution.default,
})


def _apass_bf16_conv_promotion(gm: fx.GraphModule) -> fx.GraphModule:
    """OPT-2 (aten, medium): insert a bf16 cast on the activation input of every
    aten.convolution.default node. This is an op-target pass (it keys on the conv
    target, not on weight values), so it does not need the ph_to_tensor lookup.

    Cast op: `torch.ops.prims.convert_element_type.default(a, dtype)`. This is the
    primitive Inductor itself emits for dtype conversion and the one it lowers
    natively (it has a registered lowering and is NOT in the decomposition table).

    Do NOT use `aten._to_copy.default` here: on torch 2.11 that op has BOTH a
    registered Inductor fallback AND a decomposition, so lowering aborts with
    `AssertionError: both a fallback and a decomp for same op: aten._to_copy.default`.
    That crash fires inside compile_fx_inner (after this pass returns), so it would
    not be caught by this pass's try/except — it would kill the whole compile and
    OPT-1 freezing would never run. `aten.to.dtype` decomposes to `_to_copy` and
    hits the same wall. `prims.convert_element_type` avoids both.

    The weight argument is left as-is: Inductor's constant handling promotes the
    folded/constant conv weight to match the bf16 input dtype, and a per-channel
    bf16 weight keeps the GEMM on the native bf16 tensor-core path. Reductions
    inside BN stay in fp32 the way autocast would handle them, since the BN op
    is folded away by OPT-1 freezing.

    Note `prims.convert_element_type` takes the target dtype as a POSITIONAL arg
    (schema: `convert_element_type(Tensor a, ScalarType dtype)`), not a kwarg.
    """
    try:
        matched = False
        for node in list(gm.graph.nodes):
            if not (node.op == "call_function" and node.target in _CONV_TARGETS):
                continue
            act = node.args[0]
            if not isinstance(act, fx.Node):
                continue
            with gm.graph.inserting_before(node):
                cast = gm.graph.call_function(
                    torch.ops.prims.convert_element_type.default,
                    (act, torch.bfloat16),
                )
            node.update_arg(0, cast)
            matched = True

        if not matched:
            logger.warning(
                "[OPT-2 bf16_conv_promotion] No aten.convolution nodes found "
                "— pass not applied"
            )
            return gm

        gm.graph.lint()
        gm.recompile()
        logger.info(
            "[OPT-2 bf16_conv_promotion] Applied bf16 cast on convolution "
            "inputs [Aten IR]"
        )
    except Exception as e:  # noqa: BLE001 — medium confidence: degrade gracefully
        logger.warning("[OPT-2 bf16_conv_promotion] Failed: %s", e)
    return gm


# ===========================================================================
# LEVEL 3 (inductor_config) pass — OPT-1: Conv-BN fold via Inductor freezing
# ===========================================================================
def _cfg_freeze_conv_bn() -> dict:
    """OPT-1 (inductor_config, high): enable Inductor freezing. For an eval /
    no_grad model this treats weights + BN running stats as constants and runs
    Inductor's constant-folding + conv-BN fold, eliminating the standalone
    _native_batch_norm_legit_no_training + hardtanh epilogue kernels.

    Returned as scoped config_patches merged into THIS compile_fx call only — no
    global torch._inductor.config mutation. Because aten passes (OPT-2) run
    first in the funnel, the folded conv weight inherits bf16.
    """
    logger.info("[OPT-1 freeze_conv_bn] Enabling Inductor freezing (conv-BN fold)")
    return {"freezing": True}


# ===========================================================================
# Pass registry — routed by ir_level
# ===========================================================================
# Each entry: id, level, fn. "reads_weights" flags aten passes that need the
# ph_to_tensor map (real parameter tensors); op-target passes do not.
PASS_REGISTRY = [
    {"id": "OPT-2", "level": "aten",            "fn": _apass_bf16_conv_promotion, "reads_weights": False},
    {"id": "OPT-1", "level": "inductor_config", "fn": _cfg_freeze_conv_bn},
    # OPT-3 (channels_last) is a non-graph optimization applied in
    # get_model_and_input(); it is intentionally NOT in this graph-pass registry.
]


def _passes(level: str) -> list:
    return [p for p in PASS_REGISTRY if p["level"] == level]


# ===========================================================================
# Funnel stage 1 — functional passes (none for this workload)
# ===========================================================================
def _run_functional_passes(gm: fx.GraphModule) -> fx.GraphModule:
    """LEVEL 1 — rewrite the Dynamo graph BEFORE compile_fx owns it. This
    workload has no fusion / op-substitution opportunities (no shared-activation
    linear triplets, no SDPA), so this is a no-op pass-through that keeps the
    funnel structure uniform."""
    for p in _passes("functional"):
        try:
            gm = p["fn"](gm)
        except Exception as e:  # noqa: BLE001 — degrade gracefully
            logger.warning("[%s] functional pass no-op: %s", p["id"], e)
    return gm


# ===========================================================================
# Funnel stage 2 — aten inner-compile hook
# ===========================================================================
def _repropagate_meta(gm: fx.GraphModule, example_inputs) -> None:
    """Re-run FakeTensorProp so nodes inserted by an aten pass get meta['val'].
    Best-effort: a failure here must never break compilation."""
    try:
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.passes.fake_tensor_prop import FakeTensorProp

        fake_mode = None
        for inp in example_inputs:
            fm = getattr(inp, "fake_mode", None)
            if fm is not None:
                fake_mode = fm
                break
        if fake_mode is None:
            fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
        FakeTensorProp(gm, mode=fake_mode).propagate(*example_inputs)
    except Exception as e:  # noqa: BLE001
        logger.debug("[_repropagate_meta] skipped: %s", e)


def _aten_inner_compile(
    gm: fx.GraphModule,
    example_inputs,
    *,
    real_inputs=None,
    apply_aten_passes: bool = True,
    **kwargs,
) -> Callable:
    """LEVEL 2 — Inductor's inner_compile hook. compile_fx calls this with the
    fully decomposed Aten IR graph (after AOTAutograd). Run aten-level passes,
    then delegate to compile_fx_inner (Aten -> Triton).

    example_inputs may be FakeTensors (Inductor traces under FakeTensorMode), so
    weight-VALUE-reading passes would use the threaded real_inputs. The only aten
    pass here (OPT-2 bf16) is an op-target pass and does not read weight values.
    Forward **kwargs verbatim to stay forward-compatible with compile_fx_inner.

    `apply_aten_passes` lets _compile_unit retry with the aten passes disabled if a
    bf16-promoted graph fails to LOWER (the failure surfaces inside
    compile_fx_inner, below, not inside the per-pass guard above), so the backend
    falls back to a working compile rather than hard-crashing.
    """
    if apply_aten_passes:
        weight_source = real_inputs if real_inputs is not None else example_inputs
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        ph_to_tensor = {ph: t for ph, t in zip(placeholders, weight_source)}

        for p in _passes("aten"):
            try:
                if p.get("reads_weights"):
                    gm = p["fn"](gm, ph_to_tensor)
                else:
                    gm = p["fn"](gm)
                _repropagate_meta(gm, example_inputs)
            except Exception as e:  # noqa: BLE001
                logger.warning("[%s] aten pass no-op: %s", p["id"], e)

    return compile_fx_inner(gm, example_inputs, **kwargs)


# ===========================================================================
# Funnel stage 3 — inductor config patches
# ===========================================================================
def _config_patches() -> dict:
    """LEVEL 3 — scoped Inductor config_patches merged into THIS compile_fx call
    only. Each inductor_config pass returns a dict to merge; no global mutation.
    """
    patches: dict = {}
    for p in _passes("inductor_config"):
        try:
            patches.update(p["fn"]() or {})
        except Exception as e:  # noqa: BLE001
            logger.warning("[%s] config pass skipped: %s", p["id"], e)
    return patches


def _compile_unit(gm: fx.GraphModule, example_inputs) -> Callable:
    """The fixed three-stage funnel. compile_fx owns AOTAutograd, the decomp
    table, the boxed calling convention and the partitioner — we only run
    functional passes ahead of it, swap the leaf compiler, and pass scoped
    config_patches. No second AOTAutograd (which would raise on torch 2.11)."""
    gm = _run_functional_passes(gm)
    real_inputs = list(example_inputs)
    patches = _config_patches()

    try:
        inner = functools.partial(
            _aten_inner_compile, real_inputs=real_inputs, apply_aten_passes=True
        )
        return compile_fx(
            gm, example_inputs, inner_compile=inner, config_patches=patches
        )
    except Exception as e:  # noqa: BLE001
        # A lowering failure (e.g. a cast op Inductor cannot lower) surfaces here,
        # inside compile_fx_inner — outside the per-pass guard. Fall back to a
        # compile with the aten passes disabled so the backend still produces a
        # working callable (OPT-1 freezing + OPT-3 channels_last remain in effect)
        # rather than hard-crashing the whole compile.
        logger.warning(
            "[_compile_unit] aten-pass compile failed (%s) — retrying with aten "
            "passes disabled (OPT-1/OPT-3 still applied)",
            e,
        )
        inner = functools.partial(
            _aten_inner_compile, real_inputs=real_inputs, apply_aten_passes=False
        )
        return compile_fx(
            gm, example_inputs, inner_compile=inner, config_patches=patches
        )


# ===========================================================================
# Dedup-input capture utility
# ===========================================================================
def _capture_partition_inputs(split_gm: fx.GraphModule, example_inputs: list) -> dict:
    """Capture the actual input tensors for each partition by running split_gm
    once, so compile_fx can lower each rep with correct example inputs."""
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
# The backend
# ===========================================================================
@register_backend
def depthwise_separable_conv_opt(gm: fx.GraphModule, example_inputs) -> Callable:
    logger.info(
        "depthwise_separable_conv_opt backend: starting "
        "(functional -> aten -> inductor_config)"
    )
    registry = UniqueSubgraphRegistry(gm)
    equiv_map = registry.build_partition_equivalence_map()

    if not equiv_map:
        # The 3 DWSepBlocks differ by channel count, so no repeated structure is
        # detected — flat compile preserves cross-layer Inductor fusion.
        logger.info(
            "depthwise_separable_conv_opt: no repeated layers, flat compile path"
        )
        return _compile_unit(gm, example_inputs)

    logger.info(
        "depthwise_separable_conv_opt: %d duplicate partition(s), dedup path",
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
# Workload definition (same architecture as the baseline) + interface
# ===========================================================================
class DWSepBlock(nn.Module):
    """Depthwise-separable convolution block (unchanged from baseline)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False
        )
        self.bn_dw = nn.BatchNorm2d(in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn_pw = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn_dw(self.depthwise(x)))
        x = self.act(self.bn_pw(self.pointwise(x)))
        return x


class DepthwiseSepConv(nn.Module):
    """Three stacked DWSepBlocks (32->64->128->256), unchanged from baseline."""

    def __init__(self):
        super().__init__()
        self.block1 = DWSepBlock(32, 64)
        self.block2 = DWSepBlock(64, 128)
        self.block3 = DWSepBlock(128, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


def get_model_and_input() -> tuple:
    """Workload interface — return (raw_model, input_tensor) on CUDA, eval mode.

    Applies OPT-3 (channels_last / NHWC) as a non-graph optimization: the conv
    library kernels are NHWC-native, so converting model + input to
    torch.channels_last removes the boundary NCHW->NHWC layout-copy kernel and
    improves coalescing. Guarded by an is_contiguous check so it is a no-op if
    the baseline already supplies channels_last tensors.

    OPT-1 (freezing) and OPT-2 (bf16) are applied inside the backend, not here.
    """
    assert torch.cuda.is_available(), "CUDA required"
    model = DepthwiseSepConv().to(DEVICE).eval()
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH, device=DEVICE)

    # OPT-3: channels_last (NHWC) — check current state before applying.
    if not next(model.parameters()).is_contiguous(memory_format=torch.channels_last):
        model = model.to(memory_format=torch.channels_last)
    if not x.is_contiguous(memory_format=torch.channels_last):
        x = x.to(memory_format=torch.channels_last)

    return model, x


if __name__ == "__main__":
    model, x = get_model_and_input()
    compiled = torch.compile(model, backend="depthwise_separable_conv_opt")
    with torch.no_grad():
        out = compiled(x)
    print("output shape:", tuple(out.shape), "dtype:", out.dtype)
