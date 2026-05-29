"""
conv_block_optimized.py — Custom torch.compile() backend for ConvBlock.

Registered backend: ``conv_block_opt``

Implements the three optimizations from optimizations.json. Dependency DAG
(from the proposal): OPT-1 -> OPT-2 -> OPT-3 (OPT-3 is subsumed by OPT-2).

  OPT-1  fusion         — Conv-BatchNorm folding (inference). In eval() with frozen
                          running stats, BatchNorm2d is a pure per-channel affine.
                          After AOTAutograd decomposition the eval BN does NOT survive
                          as aten._native_batch_norm_legit_no_training; Inductor lowers
                          it to a decomposed elementwise affine epilogue on the conv
                          output:
                              sub(conv, unsqueeze(unsqueeze(running_mean)))
                              mul(., unsqueeze(unsqueeze(rstd)))      # rstd=1/sqrt(var+eps)
                              mul(., unsqueeze(unsqueeze(gamma)))     # affine (optional)
                              add(., unsqueeze(unsqueeze(beta)))      # affine (optional)
                          This pass detects that chain and folds gamma/sqrt(var+eps)
                          into the conv weight and (beta - mean*scale) into a synthesized
                          conv bias as *structural* graph nodes (FakeTensor-safe — never
                          reads weight values), then rewires the affine tail back to a new
                          conv. Inductor constant-folds the weight math, so the standalone
                          BN normalize/broadcast Triton kernels (triton_*_fused__native_
                          batch_3..9) are deleted. Confidence: high. Prereq for OPT-2/3.
  OPT-2  memory_layout  — channels_last (NHWC) propagation so cuDNN runs its native-NHWC
                          implicit-GEMM and drops the convertTensor_kernel NCHW<->NHWC
                          relayout shuffles. Primary lever is the eager-side model + input
                          .to(memory_format=channels_last) in get_model_and_input(); the
                          graph pass strips now-redundant aten.clone/_to_copy(channels_last)
                          layout copies. Confidence: medium. Applied after OPT-1.
  OPT-3  first_conv     — The 3->64 first conv is tensor-core-ineligible by construction
                          (C_in=3). Per the proposal this is SUBSUMED by OPT-2 (its
                          convertTensor feeders are removed by channels_last) and channel
                          padding is explicitly NOT recommended. Implemented as a
                          detection-only stub (low confidence): it locates the 3-channel
                          conv and logs that OPT-2 covers it. No transformation.

IR-level mechanics (torch 2.11, RTX PRO 6000 Blackwell, per repo memory notes):
  The graph torch.compile hands a @register_backend function is the *functional*
  Dynamo graph, NOT Aten IR. Aten IR (aten.convolution, the BN affine decomposition)
  only appears after AOTAutograd decomposition, inside Inductor. The supported torch
  2.11 injection point for Aten-IR passes is Inductor's ``post_grad_custom_pre_pass``
  hook (the aot_autograd fw_compiler path is broken on 2.11). We install the pass chain
  there and delegate AOTAutograd + lowering to ``compile_fx``. Graph inputs are
  FakeTensors with no readable storage, so the BN fold is a *structural* rewrite — it
  materializes folded weights as aten.mul / aten.reshape graph nodes on the parameter
  placeholders, which Inductor then constant-folds.

compile_mode = "inductor" (from optimizations.json): standard FX pass approach.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

import torch
import torch.fx as fx
import torch._inductor.config as inductor_config
from torch._dynamo import register_backend
from torch._inductor.compile_fx import compile_fx  # function, not module

from nvidia.operator_profiler.fx import UniqueSubgraphRegistry

logger = logging.getLogger(__name__)

# Engage TF32 Tensor Cores for any residual FP32 matmul/conv tile path (cheap, global).
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --------------------------------------------------------------------------- #
# Aten IR targets. Conv may appear as aten.convolution.default or
# aten.cudnn_convolution.default. The eval-mode BN affine reaches the post-grad
# graph as a decomposed sub/mul/mul/add chain with unsqueeze-broadcast params.
# --------------------------------------------------------------------------- #
_CONV = torch.ops.aten.convolution.default
_CONV_CUDNN = torch.ops.aten.cudnn_convolution.default
_CONV_TARGETS = frozenset({_CONV, _CONV_CUDNN})

_UNSQUEEZE = torch.ops.aten.unsqueeze.default
_SUB = torch.ops.aten.sub.Tensor
_ADD = torch.ops.aten.add.Tensor
_MUL = torch.ops.aten.mul.Tensor
_RESHAPE = torch.ops.aten.reshape.default
_CLONE = torch.ops.aten.clone.default
_TO_COPY = torch.ops.aten._to_copy.default


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _is_conv(n) -> bool:
    return getattr(n, "op", None) == "call_function" and n.target in _CONV_TARGETS


def _unwrap_unsqueeze(n) -> Optional[fx.Node]:
    """Walk back through a chain of aten.unsqueeze.default to the broadcast source.

    BatchNorm params reach the affine chain via two unsqueeze(-1) ops that turn a
    [C] vector into a [C,1,1] broadcastable tensor. Return the underlying source node.
    """
    cur = n
    while (
        getattr(cur, "op", None) == "call_function"
        and cur.target is _UNSQUEEZE
        and cur.args
    ):
        cur = cur.args[0]
    return cur if isinstance(cur, fx.Node) else None


# =========================================================================== #
# OPT-1 — Conv-BatchNorm folding (inference). Confidence: high.
# Prerequisite for OPT-2/OPT-3. Any exception => WARNING + return graph unchanged.
# =========================================================================== #
def _pass_fold_conv_bn(g: fx.Graph) -> int:
    """Fold the decomposed inference-BatchNorm affine epilogue into the preceding conv.

    Post-grad shape of ``Conv2d(bias=False) -> BatchNorm2d(eval)``:

        conv  = aten.convolution(x, W, None, ...)
        sub   = aten.sub(conv, unsqueeze(unsqueeze(running_mean)))
        mul_a = aten.mul(sub,  unsqueeze(unsqueeze(rstd)))        # rstd = 1/sqrt(var+eps)
        mul_b = aten.mul(mul_a, unsqueeze(unsqueeze(gamma)))      # optional (affine)
        out   = aten.add(mul_b, unsqueeze(unsqueeze(beta)))       # optional (affine)

    Fold:  scale = rstd * gamma ;  W_folded = W * scale[:, None, None, None]
           bias_folded = (bias - mean) * scale + beta   (conv bias defaults to 0 here)

    Emitted as *graph nodes* on the existing parameter placeholders, so it is
    FakeTensor-safe; Inductor constant-folds the weight arithmetic at lower time,
    deleting the per-conv BN normalize/broadcast passes from the trace.
    """
    folded = 0
    for conv in list(g.nodes):
        if not _is_conv(conv):
            continue
        # conv output must feed exactly the BN sub() (single consumer) to fold safely.
        if len(conv.users) != 1:
            continue
        sub = next(iter(conv.users))
        if not (sub.op == "call_function" and sub.target is _SUB and sub.args[0] is conv):
            continue

        mean_src = _unwrap_unsqueeze(sub.args[1])
        if mean_src is None:
            continue

        # sub -> mul(rstd)
        if len(sub.users) != 1:
            continue
        mul_a = next(iter(sub.users))
        if not (mul_a.op == "call_function" and mul_a.target is _MUL):
            continue
        rstd_arg = mul_a.args[1] if mul_a.args[0] is sub else mul_a.args[0]
        rstd_src = _unwrap_unsqueeze(rstd_arg)
        if rstd_src is None:
            continue

        # Optional gamma mul and beta add following mul_a.
        gamma_src = None
        beta_src = None
        affine_tail = mul_a  # last node of the affine chain to rewire
        if len(mul_a.users) == 1:
            nxt = next(iter(mul_a.users))
            if nxt.op == "call_function" and nxt.target is _MUL:
                gamma_arg = nxt.args[1] if nxt.args[0] is mul_a else nxt.args[0]
                gamma_src = _unwrap_unsqueeze(gamma_arg)
                affine_tail = nxt
                if len(nxt.users) == 1:
                    nxt2 = next(iter(nxt.users))
                    if nxt2.op == "call_function" and nxt2.target is _ADD:
                        beta_arg = nxt2.args[1] if nxt2.args[0] is nxt else nxt2.args[0]
                        beta_src = _unwrap_unsqueeze(beta_arg)
                        affine_tail = nxt2
            elif nxt.op == "call_function" and nxt.target is _ADD:
                beta_arg = nxt.args[1] if nxt.args[0] is mul_a else nxt.args[0]
                beta_src = _unwrap_unsqueeze(beta_arg)
                affine_tail = nxt

        conv_w = conv.args[1]
        conv_b = conv.args[2] if len(conv.args) > 2 else None
        c_out = None
        try:
            c_out = int(conv_w.meta["val"].shape[0])
        except Exception:
            pass

        # The rstd/mean/gamma/beta sources are computed *after* the conv in the
        # decomposed graph, so the folded weight/bias and the replacement conv must
        # be inserted just before the affine-chain tail (where every dependency is
        # already defined) to keep the graph topologically ordered.
        with g.inserting_before(affine_tail):
            # scale = rstd * gamma  (shape [C_out])
            if gamma_src is not None:
                scale = g.call_function(_MUL, (rstd_src, gamma_src))
            else:
                scale = rstd_src
            # W_folded = W * scale.reshape(C_out, 1, 1, 1)
            scale_w = g.call_function(_RESHAPE, (scale, [-1, 1, 1, 1]))
            w_folded = g.call_function(_MUL, (conv_w, scale_w))

            # bias_folded = (bias - mean) * scale + beta   (conv bias optional/None)
            if conv_b is not None:
                bias_centered = g.call_function(_SUB, (conv_b, mean_src))
            else:
                bias_centered = g.call_function(_MUL, (mean_src, -1.0))
            b_scaled = g.call_function(_MUL, (bias_centered, scale))
            if beta_src is not None:
                b_folded = g.call_function(_ADD, (b_scaled, beta_src))
            else:
                b_folded = b_scaled

            # Emit a replacement conv consuming the folded weight + synthesized bias.
            new_conv_args = list(conv.args)
            new_conv_args[1] = w_folded
            if len(new_conv_args) > 2:
                new_conv_args[2] = b_folded
            else:
                new_conv_args.append(b_folded)
            new_conv = g.call_function(_CONV, tuple(new_conv_args))

        # The affine-chain output now equals the new conv output; rewire consumers.
        affine_tail.replace_all_uses_with(new_conv)
        g.eliminate_dead_code()
        folded += 1
        if c_out is not None:
            logger.info(
                "[OPT-1 fold_conv_bn] Folded BatchNorm into conv (C_out=%d) [Aten IR]",
                c_out,
            )
        else:
            logger.info("[OPT-1 fold_conv_bn] Folded BatchNorm into conv [Aten IR]")

    if folded:
        g.lint()
    else:
        logger.warning(
            "[OPT-1 fold_conv_bn] No conv->BN affine chain found — pass not applied"
        )
    return folded


# =========================================================================== #
# OPT-2 — channels_last propagation: strip redundant layout-copy nodes.
# Confidence: medium. Primary lever is the eager-side conversion in
# get_model_and_input(); this pass removes graph-level NCHW<->NHWC copies that are
# already no-ops once producer/consumer agree on layout. Applied after OPT-1 so the
# folded conv weights are the nodes whose layout is re-evaluated.
# =========================================================================== #
def _pass_strip_layout_copies(g: fx.Graph) -> int:
    """Erase aten.clone / aten._to_copy(memory_format=channels_last) whose input is
    already channels_last contiguous, so Inductor does not emit a standalone
    convertTensor / layout-copy kernel around each conv."""
    stripped = 0
    for node in list(g.nodes):
        if not (node.op == "call_function" and node.target in (_CLONE, _TO_COPY)):
            continue
        if node.kwargs.get("memory_format", None) is not torch.channels_last:
            continue
        src = node.args[0]
        try:
            meta = src.meta["val"]
            if meta.is_contiguous(memory_format=torch.channels_last):
                node.replace_all_uses_with(src)
                stripped += 1
        except Exception:
            continue

    if stripped:
        g.eliminate_dead_code()
        g.lint()
        logger.info(
            "[OPT-2 strip_layout_copies] Removed %d redundant channels_last copy node(s) "
            "[Aten IR]",
            stripped,
        )
    else:
        logger.info(
            "[OPT-2 strip_layout_copies] No redundant channels_last copy nodes — "
            "layout handled eager-side in get_model_and_input()"
        )
    return stripped


# =========================================================================== #
# OPT-3 — first conv (3->64) handling. Confidence: low => detection-only stub.
# Per optimizations.json the 3-channel first conv is tensor-core-ineligible by
# construction and is SUBSUMED by OPT-2 (channels_last removes its convertTensor
# feeders). Channel padding (3->4) is explicitly NOT recommended as a first move.
# This pass locates the first conv and logs that OPT-2 covers it; never transforms.
# =========================================================================== #
def _pass_first_conv_stub(g: fx.Graph) -> None:
    for conv in g.nodes:
        if not _is_conv(conv):
            continue
        c_in = None
        try:
            c_in = int(conv.args[1].meta["val"].shape[1])
        except Exception:
            continue
        if c_in == 3:
            logger.info(
                "[OPT-3 first_conv] 3->C first conv detected (tensor-core-ineligible "
                "at C_in=3) — subsumed by OPT-2 channels_last (its convertTensor feeders "
                "are removed); channel padding NOT applied per proposal [Aten IR]"
            )
            return
    logger.warning(
        "[OPT-3 first_conv] No 3-channel first conv found — stub not applicable"
    )


# =========================================================================== #
# Aten-IR pass chain, installed as Inductor's post_grad_custom_pre_pass. Runs on
# the decomposed, functionalized Aten graph just before lowering. Order respects the
# DAG OPT-1 -> OPT-2 -> OPT-3 from optimizations.json.
# =========================================================================== #
def _repropagate_meta(g: fx.Graph) -> None:
    """Re-run FakeTensorProp so the nodes inserted by OPT-1 acquire ``meta['val']``.

    Downstream Inductor post-grad passes read ``node.meta['val']`` on the new
    fold/conv nodes; without re-propagation those reads raise KeyError. The fake mode
    and fake placeholder inputs are recovered from the existing placeholder meta, so
    this stays inside the active FakeTensorMode.
    """
    gm = g.owning_module
    if gm is None:
        return
    placeholders = [n for n in g.nodes if n.op == "placeholder"]
    fake_inputs = []
    fake_mode = None
    for ph in placeholders:
        val = ph.meta.get("val", None)
        if val is None:
            return  # cannot reconstruct inputs — skip (graph already had full meta)
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


def _aten_pass_chain(g: fx.Graph) -> fx.Graph:
    try:
        folded = _pass_fold_conv_bn(g)   # OPT-1 (high) — must run first
        if folded:
            # New fold/conv nodes need meta['val'] before OPT-2 reads strides and
            # before downstream Inductor post-grad passes run.
            _repropagate_meta(g)
        _pass_strip_layout_copies(g)     # OPT-2 (medium) — re-eval layout post-fold
        _pass_first_conv_stub(g)         # OPT-3 (low) — detection only
    except Exception as e:  # never crash the compile
        logger.warning("[conv_block_opt] Aten pass chain failed: %s", e)
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
def conv_block_opt(gm: fx.GraphModule, example_inputs) -> Callable:
    """Custom torch.compile backend for ConvBlock.

    Installs the Aten-IR pass chain (OPT-1 fold_conv_bn, OPT-2 strip_layout_copies,
    OPT-3 first_conv stub) via Inductor's post_grad_custom_pre_pass, then delegates
    AOTAutograd + lowering to compile_fx. OPT-2's primary (non-graph) lever is the
    channels_last conversion in get_model_and_input() (Rule 7).

    Dedup-aware per Rule 9: ConvBlock has three structurally distinct conv stages
    (3->64, 64->128, 128->256) plus a classifier, so there are no repeated layers
    and the flat compile path is taken (preserving cross-stage Inductor fusion).
    The dedup branch is retained for structural reuse if the model grows.
    """
    logger.info("conv_block_opt backend: starting")
    _install_aten_passes()

    registry = UniqueSubgraphRegistry(gm)
    equiv_map = registry.build_partition_equivalence_map()

    if not equiv_map:
        logger.info("conv_block_opt: no repeated layers, flat compile path")
        return compile_fx(gm, example_inputs)

    logger.info("conv_block_opt: %d duplicate partition(s), dedup path", len(equiv_map))
    partition_inputs = _capture_partition_inputs(registry.split, example_inputs)
    for rep_name, rep_mod in registry.unique_reps:
        inputs = partition_inputs.get(rep_name, example_inputs)
        compiled = compile_fx(rep_mod, inputs)
        rep_mod.forward = compiled
        for _, dup_mod in registry.duplicates_of(rep_name):
            dup_mod.forward = compiled

    return lambda *args: registry.split(*args)


# =========================================================================== #
# Workload interface — non-graph optimizations live here (Rule 7).
#   OPT-2 (channels_last): model + input cast to torch.channels_last so cuDNN runs
#          native-NHWC implicit-GEMM and drops the convertTensor_kernel relayouts.
#          This is the primary lever for OPT-2; the graph pass only cleans up
#          residual copies. Idempotent — checked before converting (Rule 7).
# OPT-1 (conv-BN fold) is realized as the OPT-1 graph pass at compile time, NOT here;
# leaving the BN in the module lets the post-grad fold pass demonstrably remove it.
# =========================================================================== #
DEVICE = "cuda"
BATCH_SIZE = 16
IN_CHANNELS = 3
HEIGHT = 64
WIDTH = 64


def get_model_and_input() -> tuple:
    """Return (model, input_tensor) on CUDA — uncompiled, unwarmed.

    Applies OPT-2's non-graph half: channels_last for both the model weights and the
    input (idempotent — checked before converting per Rule 7). OPT-1 (conv-BN fold)
    and the OPT-2 graph cleanup / OPT-3 detection run inside the conv_block_opt backend
    at compile time.
    """
    assert torch.cuda.is_available(), "CUDA required"
    from conv_block import ConvBlock

    model = ConvBlock().to(DEVICE).eval()
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH, device=DEVICE)

    # OPT-2 (channels_last propagation) — eager-side, only if not already NHWC.
    if not next(model.parameters()).is_contiguous(memory_format=torch.channels_last):
        model = model.to(memory_format=torch.channels_last)
        logger.info("[OPT-2 channels_last] Model cast to channels_last (NHWC)")
    if not x.is_contiguous(memory_format=torch.channels_last):
        x = x.to(memory_format=torch.channels_last)
        logger.info("[OPT-2 channels_last] Input cast to channels_last (NHWC)")

    return model, x


if __name__ == "__main__":
    model, x = get_model_and_input()
    compiled = torch.compile(model, backend="conv_block_opt")
    with torch.no_grad():
        out = compiled(x)
    print("output shape:", out.shape, "dtype:", out.dtype)
