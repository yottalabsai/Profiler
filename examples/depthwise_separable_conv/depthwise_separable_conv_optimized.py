"""
depthwise_separable_conv_optimized.py — custom torch.compile() backend for the
MobileNet-style depthwise-separable conv workload.

Implements (at the Aten IR level, inside ``_aten_fw_compiler``):

  OPT-1  Conv-BatchNorm folding (eval inference): absorb each
         aten._native_batch_norm_legit_no_training that consumes a conv output
         into the conv weight/bias. Removes ~12 standalone BN-affine Triton
         kernels. Foundational — runs first because it rewrites the conv
         weight/bias constants that OPT-2 reshapes.

  OPT-2  1x1 pointwise conv -> explicit GEMM (aten.mm/addmm). Reroutes the six
         occupancy-bound cuDNN 'Kernel2' pointwise convs to a well-occupied
         cuBLAS / Inductor GEMM template. Runs after OPT-1.

  OPT-4  channels_last (NHWC) end-to-end. NON-GRAPH optimization applied in
         get_model_and_input() via model.to(memory_format=channels_last) and
         x.to(...). Collapses the per-stage NCHW<->NHWC layout copies into
         metadata views.

  OPT-3  Depthwise Triton-codegen fusion. LOW confidence, autotune-gated, and
         mutually exclusive with OPT-4's copy elimination. Implemented as a
         DETECT-ONLY stub: it logs the depthwise convs it would force onto the
         Triton path but performs no transformation, so we never apply two
         redundant copy-eliminators. OPT-4 is the chosen copy eliminator.

Dependency order (from optimizations.json):
    OPT-1 (fold) -> OPT-2 (1x1->GEMM) -> OPT-4 (channels_last, non-graph)
    OPT-3 is the autotune-gated alternative to OPT-4 and is left as a stub.

Backend registered with @register_backend as: depthwise_separable_conv_opt
"""
from __future__ import annotations

import logging
import operator
from typing import Callable

import torch
import torch.fx as fx
import torch.nn as nn
from torch._dynamo import register_backend
from torch._dynamo.backends.common import aot_autograd
# compile_fx_inner is the POST-AOT Inductor compiler. It is the correct callable
# to return from an aot_autograd fw/inference compiler: returning the full
# pre-AOT compile_fx instead nests a second AOTAutograd pass, which in torch 2.11
# corrupts the boxed calling convention (the runtime hands the inner callable a
# single list of 31 tensors -> "Expected tensors only, but got <class 'list'>").
from torch._inductor.compile_fx import compile_fx_inner  # function, NOT the module
from torch._inductor.decomposition import select_decomp_table
from torch.fx.passes.fake_tensor_prop import FakeTensorProp

from nvidia.operator_profiler.fx import UniqueSubgraphRegistry

logger = logging.getLogger(__name__)

DEVICE = "cuda"
BATCH_SIZE = 16
IN_CHANNELS = 32
HEIGHT = 56
WIDTH = 56

_BN_TARGET = torch.ops.aten._native_batch_norm_legit_no_training.default
_CONV_TARGETS = frozenset(
    {
        torch.ops.aten.convolution.default,
        torch.ops.aten.cudnn_convolution.default,
    }
)


# ----------------------------------------------------------------------------
# OPT-1: Conv-BatchNorm fold (high confidence) — Aten IR weight-access pass
# ----------------------------------------------------------------------------
def _pass_fold_bn(gm: fx.GraphModule, ph_to_tensor: dict) -> fx.GraphModule:
    """
    Fold every aten._native_batch_norm_legit_no_training that consumes a conv
    output directly into the conv weight/bias. Loops over ALL conv->BN pairs
    (6 in this model) rather than stopping at the first match.

    W' = W * (gamma / sqrt(var + eps)).reshape(-1,1,1,1)
    b' = beta + (b - mean) * gamma / sqrt(var + eps)
    """
    try:
        folded = 0
        for bn_node in list(gm.graph.nodes):
            if not (bn_node.op == "call_function" and bn_node.target is _BN_TARGET):
                continue

            # aten._native_batch_norm_legit_no_training(
            #     input, weight, bias, running_mean, running_var, momentum, eps)
            conv_node = bn_node.args[0]
            if not (
                conv_node.op == "call_function" and conv_node.target in _CONV_TARGETS
            ):
                continue
            # Conv output must feed only this BN (single consumer) to fold safely.
            if len(conv_node.users) != 1:
                continue

            bn_weight = ph_to_tensor.get(bn_node.args[1])
            bn_bias = ph_to_tensor.get(bn_node.args[2])
            run_mean = ph_to_tensor.get(bn_node.args[3])
            run_var = ph_to_tensor.get(bn_node.args[4])
            eps = bn_node.args[6] if len(bn_node.args) > 6 else 1e-5

            if any(t is None for t in (bn_weight, bn_bias, run_mean, run_var)):
                continue

            # aten.convolution.default args:
            #   (input, weight, bias, stride, padding, dilation,
            #    transposed, output_padding, groups)
            conv_weight = ph_to_tensor.get(conv_node.args[1])
            conv_bias = (
                ph_to_tensor.get(conv_node.args[2])
                if conv_node.args[2] is not None
                else None
            )
            if conv_weight is None:
                continue

            # Folding must run on REAL data: a FakeTensor operand would yield a
            # fake folded buffer that leaks into the Inductor runtime and trips
            # copy_misaligned_inputs' data_ptr() call. Skip the match if any
            # operand is fake (the backend threads real tensors in, so this is
            # a guard, not the normal path).
            _fold_tensors = [bn_weight, bn_bias, run_mean, run_var, conv_weight]
            if conv_bias is not None:
                _fold_tensors.append(conv_bias)
            if any(isinstance(t, torch._subclasses.FakeTensor) for t in _fold_tensors):
                logger.warning(
                    "[OPT-1 fold_bn] Skipping a conv->BN match: operand is a "
                    "FakeTensor (no real data to fold)"
                )
                continue

            # AOTAutograd runs this compiler under an active FakeTensorMode, so
            # arithmetic on the (real) parameter tensors would be intercepted
            # ("convert all Tensors to FakeTensors first"). Disable fake mode
            # for the pure-data fold so we get genuine, storage-backed buffers.
            with torch.utils._python_dispatch._disable_current_modes():
                scale = bn_weight / torch.sqrt(run_var + eps)
                new_weight = conv_weight * scale.view(-1, 1, 1, 1)
                if conv_bias is not None:
                    new_bias = (conv_bias - run_mean) * scale + bn_bias
                else:
                    new_bias = bn_bias - run_mean * scale
                new_weight = new_weight.contiguous()
                new_bias = new_bias.contiguous()

            buf_w = f"_folded_conv_weight_{folded}"
            buf_b = f"_folded_conv_bias_{folded}"
            gm.register_buffer(buf_w, new_weight)
            gm.register_buffer(buf_b, new_bias)

            with gm.graph.inserting_before(conv_node):
                fw = gm.graph.get_attr(buf_w)
                fb = gm.graph.get_attr(buf_b)

            new_conv_args = (conv_node.args[0], fw, fb) + tuple(conv_node.args[3:])
            with gm.graph.inserting_before(bn_node):
                new_conv = gm.graph.call_function(
                    torch.ops.aten.convolution.default, new_conv_args
                )

            # bn_node returns (output, save_mean, save_rstd) — redirect getitem(0).
            for user in list(bn_node.users):
                if (
                    user.op == "call_function"
                    and user.target is operator.getitem
                    and user.args[1] == 0
                ):
                    user.replace_all_uses_with(new_conv)
                    gm.graph.erase_node(user)
            if not bn_node.users:
                gm.graph.erase_node(bn_node)
            if not conv_node.users:
                gm.graph.erase_node(conv_node)
            folded += 1

        if folded == 0:
            logger.warning(
                "[OPT-1 fold_bn] No conv->BN pair found — pass not applied"
            )
            return gm

        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()
        logger.info(
            "[OPT-1 fold_bn] Folded %d Conv-BatchNorm pair(s) into conv weights "
            "[Aten IR]",
            folded,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("[OPT-1 fold_bn] Failed: %s", e)
    return gm


# ----------------------------------------------------------------------------
# Fake-tensor metadata propagation
# ----------------------------------------------------------------------------
def _propagate_fake_meta(gm: fx.GraphModule, fake_inputs) -> None:
    """
    Recompute ``node.meta['val']`` for every node by replaying the graph under
    the active FakeTensorMode. Newly inserted nodes (the permute/reshape/mm
    chain from OPT-2) have NO meta['val'] after insertion; without it Inductor's
    input-collection mis-handles the constant-list reshape args and surfaces
    them as a stray ``list`` runtime input ("Expected tensors only, but got
    <class 'list'>"). Running FakeTensorProp restores correct metadata so the
    graph lowers cleanly.
    """
    import torch._guards
    import contextlib

    fake_mode = torch._guards.detect_fake_mode(fake_inputs)

    # The graph carries real get_attr buffers (OPT-1's folded weights and OPT-2's
    # pre-transposed weights). The active fake mode rejects real tensors unless
    # allow_non_fake_inputs is set; with it FakeTensorProp fakeifies them on the
    # fly. IMPORTANT: leave the flag ENABLED — compile_fx_inner (called right
    # after this) re-runs lowering under the SAME fake mode and likewise needs to
    # tolerate the real buffers. Restoring the flag here re-triggers the
    # "convert all Tensors to FakeTensors first" assert inside compile_fx_inner.
    if fake_mode is not None:
        fake_mode.allow_non_fake_inputs = True
    ctx = fake_mode if fake_mode is not None else contextlib.nullcontext()
    with ctx:
        FakeTensorProp(gm, mode=fake_mode).propagate(*fake_inputs)


# ----------------------------------------------------------------------------
# OPT-2: 1x1 pointwise conv -> explicit GEMM (medium confidence) — Aten IR
# ----------------------------------------------------------------------------
def _pass_conv1x1_to_gemm(
    gm: fx.GraphModule, ph_to_tensor: dict
) -> fx.GraphModule:
    """
    Rewrite each 1x1, stride-1, pad-0, dilation-1, groups-1 convolution as a GEMM:
        (N,C_in,H,W) -> permute NHWC -> reshape (N*H*W, C_in)
        @ weight_T (C_in, C_out) [+ bias]
        -> reshape (N,H,W,C_out) -> permute back NCHW

    Routes the occupancy-bound cuDNN pointwise convs to cuBLAS / Inductor GEMM.
    Runs after OPT-1 so the folded weight/bias are the ones used.

    The transposed weight (C_in, C_out) is materialized as a REAL contiguous
    buffer (``_gemm_weight_T_*``) so the GEMM is a plain ``aten.mm`` — we do NOT
    emit an ``aten.t.default`` node. That op is in Inductor's decomposition
    table, and inserting it *after* AOT decomposition makes compile_fx_inner
    assert ("both a fallback and a decomp for same op: aten.t.default"). Pre-
    transposing also removes a runtime transpose kernel.
    """
    try:
        rewritten = 0
        for node in list(gm.graph.nodes):
            if not (
                node.op == "call_function"
                and node.target is torch.ops.aten.convolution.default
            ):
                continue

            inp, w_node, bias_node = node.args[0], node.args[1], node.args[2]
            stride, padding, dilation = node.args[3], node.args[4], node.args[5]
            groups = node.args[8]

            # Weight tensor: either a folded buffer (get_attr) or a placeholder.
            if w_node.op == "get_attr":
                weight = getattr(gm, w_node.target, None)
            else:
                weight = ph_to_tensor.get(w_node)
            if weight is None or not isinstance(weight, torch.Tensor):
                continue

            c_out, c_in, kh, kw = (int(s) for s in weight.shape)
            if not (
                kh == 1
                and kw == 1
                and list(stride) == [1, 1]
                and list(padding) == [0, 0]
                and list(dilation) == [1, 1]
                and groups == 1
            ):
                continue

            # Pre-transposing needs REAL weight data. A FakeTensor here means the
            # backend could not thread real inputs through — skip rather than
            # materialize a fake buffer that would leak into the runtime.
            if isinstance(weight, torch._subclasses.FakeTensor):
                logger.warning(
                    "[OPT-2 conv1x1_to_gemm] Skipping a 1x1 conv: weight is a "
                    "FakeTensor (no real data to pre-transpose)"
                )
                continue

            # Output reshape must use STATIC python ints, not sym_size nodes.
            # A list of sym_size Node outputs leaked into the Inductor runtime as
            # a `list` graph input and tripped copy_misaligned_inputs' tensor
            # assert. The workload has fully static shapes; read N,H,W from the
            # input node's fake-val metadata. Skip the match if shape is unknown.
            inp_val = inp.meta.get("val", None)
            if inp_val is None or any(
                isinstance(s, torch.SymInt) for s in inp_val.shape
            ):
                logger.warning(
                    "[OPT-2 conv1x1_to_gemm] Skipping a 1x1 conv: input has no "
                    "static shape metadata (dynamic shapes unsupported here)"
                )
                continue
            n_i, _c, h_i, w_i = (int(s) for s in inp_val.shape)

            # Materialize weight_T (C_in, C_out) as a real contiguous buffer,
            # under a disabled fake mode so we get genuine storage (mirrors the
            # OPT-1 fold). No aten.t node is emitted.
            with torch.utils._python_dispatch._disable_current_modes():
                weight_T = (
                    weight.reshape(c_out, c_in).t().contiguous()
                )
            buf_wt = f"_gemm_weight_T_{rewritten}"
            gm.register_buffer(buf_wt, weight_T)

            with gm.graph.inserting_before(node):
                # x: (N,C_in,H,W) -> (N,H,W,C_in) -> (N*H*W, C_in)
                x_perm = gm.graph.call_function(
                    torch.ops.aten.permute.default, (inp, [0, 2, 3, 1])
                )
                x_2d = gm.graph.call_function(
                    torch.ops.aten.reshape.default, (x_perm, [n_i * h_i * w_i, c_in])
                )
                w_t = gm.graph.get_attr(buf_wt)  # (C_in, C_out), pre-transposed

                if bias_node is not None:
                    mm = gm.graph.call_function(
                        torch.ops.aten.addmm.default, (bias_node, x_2d, w_t)
                    )
                else:
                    mm = gm.graph.call_function(
                        torch.ops.aten.mm.default, (x_2d, w_t)
                    )

                # Static output shape (N,H,W,C_out) -> permute back to NCHW.
                out_p = gm.graph.call_function(
                    torch.ops.aten.reshape.default,
                    (mm, [n_i, h_i, w_i, c_out]),
                )
                out = gm.graph.call_function(
                    torch.ops.aten.permute.default, (out_p, [0, 3, 1, 2])
                )

            node.replace_all_uses_with(out)
            gm.graph.erase_node(node)
            rewritten += 1

        if rewritten == 0:
            logger.warning(
                "[OPT-2 conv1x1_to_gemm] No 1x1 pointwise conv found — "
                "pass not applied"
            )
            return gm

        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()
        # Node meta['val'] for the inserted permute/reshape/mm chain is restored
        # by the _propagate_fake_meta sweep the compiler runs after all passes.
        logger.info(
            "[OPT-2 conv1x1_to_gemm] Rewrote %d 1x1 conv(s) as GEMM [Aten IR]",
            rewritten,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("[OPT-2 conv1x1_to_gemm] Failed: %s", e)
    return gm


# ----------------------------------------------------------------------------
# OPT-3: Depthwise Triton-codegen fusion (low confidence) — DETECT-ONLY STUB
# ----------------------------------------------------------------------------
def _pass_depthwise_triton_stub(
    gm: fx.GraphModule, ph_to_tensor: dict
) -> fx.GraphModule:
    """
    STUB — not applied. OPT-3 forces grouped/depthwise convs onto Inductor's
    Triton codegen to drop the NCHW->NHWC layout copy. It is an AUTOTUNE-GATED
    alternative to OPT-4 (channels_last) and the two MUST NOT both run, since
    both eliminate the same copy kernel. OPT-4 is the chosen copy eliminator
    here, so this pass only detects and logs the depthwise convs it would target.
    """
    try:
        detected = 0
        for node in list(gm.graph.nodes):
            if not (
                node.op == "call_function"
                and node.target is torch.ops.aten.convolution.default
            ):
                continue
            w_node = node.args[1]
            if w_node.op == "get_attr":
                weight = getattr(gm, w_node.target, None)
            else:
                weight = ph_to_tensor.get(w_node)
            groups = node.args[8]
            # depthwise: groups == in_channels == out_channels-per-group
            if weight is not None and groups == weight.shape[0]:
                detected += 1
        if detected:
            logger.info(
                "[OPT-3 depthwise_triton] STUB — detected %d depthwise conv(s); "
                "not applied (autotune-gated alternative to OPT-4 channels_last)",
                detected,
            )
        else:
            logger.info(
                "[OPT-3 depthwise_triton] STUB — no depthwise conv detected; "
                "not applied"
            )
    except Exception as e:  # noqa: BLE001
        logger.warning("[OPT-3 depthwise_triton] Failed: %s", e)
    return gm


# ----------------------------------------------------------------------------
# Aten forward compiler — all graph passes run here in dependency order
# ----------------------------------------------------------------------------
# Decomposition table handed to aot_autograd. This is Inductor's standard
# decomposition table MINUS BatchNorm, so the graph the fw_compiler receives is
# decomposed exactly like the stock ``inductor`` backend EXCEPT that BatchNorm
# stays as aten._native_batch_norm_legit_no_training.default — the op OPT-1
# folds. Everything else is pre-decomposed, which is the contract
# ``compile_fx_inner`` enforces (it asserts if a graph op is both in the decomp
# table and still present un-decomposed). OPT-1 erases every BN node by folding,
# so compile_fx_inner never sees an un-decomposed BN.
_INDUCTOR_DECOMPS = dict(select_decomp_table())
_INDUCTOR_DECOMPS.pop(
    torch.ops.aten._native_batch_norm_legit_no_training.default, None
)


def _make_aten_fw_compiler(real_inputs) -> Callable:
    """
    Build the AOTAutograd forward / inference compiler, closing over the REAL
    parameter tensors captured at the backend entry.

    Two distinct bugs are fixed here:

    1. FakeTensor weight leak. AOTAutograd invokes the fw_compiler with
       **FakeTensor** example inputs (shape/dtype/device only, no storage).
       OPT-1's BN fold computes ``new_weight = W * scale`` then
       ``register_buffer`` — on fakes the result is a FakeTensor, so the folded
       buffers are fakes that later trip ``copy_misaligned_inputs``' data_ptr().
       Fix: Dynamo lifts all params to graph inputs and the AOT placeholder
       order is positionally identical to the backend's ``example_inputs``
       (verified) — which are REAL. We map placeholders to the captured real
       tensors, run the fold under a disabled fake mode, and get real buffers.

    2. Double-AOT calling-convention corruption. Returning the *full*
       ``compile_fx`` from inside an ``aot_autograd`` fw_compiler nests a second
       AOTAutograd pass; in torch 2.11 the inner runtime is then handed the
       outer boxed arg list as a single positional arg -> "Expected tensors
       only, but got <class 'list'>". Fix: call ``compile_fx_inner`` (the
       post-AOT Inductor compiler) for a single AOT pass with the correct boxed
       calling convention. It requires a pre-decomposed graph, hence
       ``_INDUCTOR_DECOMPS`` (minus BN) on aot_autograd, plus a FakeTensorProp
       sweep so the freshly inserted/folded nodes (and the real folded buffers)
       carry proper fake ``meta['val']`` for lowering.
    """

    def _aten_fw_compiler(gm: fx.GraphModule, fw_example_inputs) -> Callable:
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]

        # Prefer REAL captured tensors; fall back to the fakes only if the
        # placeholder count diverges (defensive — passes then self-skip on fakes).
        if real_inputs is not None and len(real_inputs) == len(placeholders):
            value_inputs = real_inputs
        else:
            logger.warning(
                "[backend] placeholder/real-input count mismatch (%d vs %d) — "
                "weight-folding passes will see fake tensors and self-skip",
                len(placeholders),
                len(real_inputs) if real_inputs is not None else -1,
            )
            value_inputs = fw_example_inputs
        ph_to_tensor = {ph: t for ph, t in zip(placeholders, value_inputs)}

        # Dependency order: OPT-1 (fold) -> OPT-2 (1x1->GEMM).
        # OPT-1 registers folded weights as REAL get_attr buffers and erases the
        # BN nodes; OPT-2 reads the folded weights back via getattr(gm, ...).
        gm = _pass_fold_bn(gm, ph_to_tensor)
        gm = _pass_conv1x1_to_gemm(gm, ph_to_tensor)
        # OPT-3 stub (detect-only; mutually exclusive with OPT-4).
        gm = _pass_depthwise_triton_stub(gm, ph_to_tensor)

        # Give every node — including the real folded buffers and the inserted
        # GEMM chain — a fake meta['val'] so compile_fx_inner can lower cleanly.
        _propagate_fake_meta(gm, fw_example_inputs)

        # POST-AOT Inductor compiler: single AOT pass, boxed-call-correct.
        return compile_fx_inner(gm, fw_example_inputs)

    return _aten_fw_compiler


def _capture_partition_inputs(
    split_gm: fx.GraphModule,
    example_inputs: list,
) -> dict:
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


# ----------------------------------------------------------------------------
# Registered backend
# ----------------------------------------------------------------------------
@register_backend
def depthwise_separable_conv_opt(
    gm: fx.GraphModule, example_inputs
) -> Callable:
    logger.info("depthwise_separable_conv_opt backend: starting")
    registry = UniqueSubgraphRegistry(gm)
    equiv_map = registry.build_partition_equivalence_map()

    if not equiv_map:
        # No structurally repeated layers — flat compile preserves cross-layer
        # Inductor fusion. The three DWSepBlocks have distinct channel counts
        # (32->64, 64->128, 128->256) so they typically do NOT dedup.
        logger.info(
            "depthwise_separable_conv_opt: no repeated layers, flat compile path"
        )
        # example_inputs here are REAL tensors; thread them through so the
        # weight-folding passes operate on real data (AOTAutograd hands the
        # fw_compiler fakes — see _make_aten_fw_compiler).
        fw_compiler = _make_aten_fw_compiler(list(example_inputs))
        return aot_autograd(
            fw_compiler=fw_compiler,
            inference_compiler=fw_compiler,
            decompositions=_INDUCTOR_DECOMPS,
        )(gm, example_inputs)

    logger.info(
        "depthwise_separable_conv_opt: %d duplicate partition(s), dedup path",
        len(equiv_map),
    )
    partition_inputs = _capture_partition_inputs(registry.split, example_inputs)
    for rep_name, rep_mod in registry.unique_reps:
        inputs = partition_inputs.get(rep_name, example_inputs)
        # Per-rep real inputs feed the fold math for that partition.
        fw_compiler = _make_aten_fw_compiler(list(inputs))
        compiled = aot_autograd(
            fw_compiler=fw_compiler,
            inference_compiler=fw_compiler,
            decompositions=_INDUCTOR_DECOMPS,
        )(rep_mod, inputs)
        rep_mod.forward = compiled
        for _, dup_mod in registry.duplicates_of(rep_name):
            dup_mod.forward = compiled

    return lambda *args: registry.split(*args)


# ----------------------------------------------------------------------------
# Workload interface — non-graph optimizations (OPT-4) live here
# ----------------------------------------------------------------------------
class DWSepBlock(nn.Module):
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
    """
    Workload interface — returns (model, input_tensor) on CUDA in eval mode.

    OPT-4 (channels_last) is a NON-GRAPH optimization and is applied here:
    convert the whole model and the input tensor to channels_last (NHWC) so the
    NHWC-native depthwise convs and the post-OPT-2 GEMM permutes become
    metadata-only views, collapsing the per-stage NCHW<->NHWC layout copies.
    """
    assert torch.cuda.is_available(), "CUDA required"
    model = DepthwiseSepConv().to(DEVICE).eval()
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH, device=DEVICE)

    # OPT-4: channels_last (NHWC). Check current state first (Rule 7).
    first_param = next(model.parameters())
    if not first_param.is_contiguous(memory_format=torch.channels_last):
        model = model.to(memory_format=torch.channels_last)
    if not x.is_contiguous(memory_format=torch.channels_last):
        x = x.to(memory_format=torch.channels_last)

    return model, x


if __name__ == "__main__":
    model, x = get_model_and_input()
    compiled = torch.compile(model, backend="depthwise_separable_conv_opt")
    with torch.no_grad():
        print(compiled(x).shape)
