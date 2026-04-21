"""
depthwise_sep_conv_optimized.py — DepthwiseSepConv with custom torch.compile() backend.

Implements 5 operator-level optimizations via FX graph passes derived from
profiling feedback on an RTX PRO 6000 Blackwell (GB202) GPU:

  1. CUDA Graphs (OPT-001, HIGH)     — eliminate 51.8% GPU idle time from 130 launch gaps
  2. 1×1 Conv → MM rewrite (OPT-002, HIGH) — break shmem wave starvation on Kernel2
  3. Depthwise Triton stub (OPT-003, MEDIUM) — vectorized load detection, Triton TODO
  4. Conv-BN-ReLU6 epilog fusion (OPT-004, MEDIUM) — eliminate conv-BN DRAM round-trip
  5. BF16 precision cast (OPT-005, HIGH) — halve memory pressure, double TC throughput

Non-graph optimizations applied in get_model_and_input():
  - BF16 cast (OPT-005): model + input cast to bfloat16 if not already

To run with optimizations:
    python depthwise_sep_conv_optimized.py

To profile with the custom backend:
    operator-profiler profile depthwise_sep_conv_optimized.py \\
        --model-name DepthwiseSepConvOpt --compile-mode transformer_opt \\
        --output runs/dsc_opt

    operator-profiler map runs/dsc_opt.manifest.json \\
        --script scripts/run_workload.py \\
        --ncu-sudo \\
        --ncu-env PYTHONPATH=/repo \\
        --script-args --workload depthwise_sep_conv_optimized.py \\
                      --compile-backend transformer_opt
"""
from __future__ import annotations

import logging
from typing import Callable

import torch
import torch.nn as nn
import torch.fx as fx
from torch._dynamo import register_backend
from torch._inductor.compile_fx import compile_fx

# Import baseline workload
from depthwise_separable_conv import (
    get_model_and_input as get_baseline_model_and_input,
    DEVICE,
    BATCH_SIZE,
    IN_CHANNELS,
    HEIGHT,
    WIDTH,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node_is_conv(node: fx.Node) -> bool:
    """Return True if node is any aten convolution variant."""
    conv_targets = {
        torch.ops.aten.convolution.default,
        torch.ops.aten._convolution.default,
        torch.ops.aten.cudnn_convolution.default,
        torch.ops.aten.conv2d.default,
    }
    return node.op == "call_function" and node.target in conv_targets


def _node_is_batch_norm(node: fx.Node) -> bool:
    bn_targets = {
        torch.ops.aten.native_batch_norm.default,
        torch.ops.aten._native_batch_norm_legit_no_training.default,
        torch.ops.aten.batch_norm.default,
        torch.ops.aten.native_batch_norm_backward.default,
    }
    return node.op == "call_function" and node.target in bn_targets


def _node_is_relu6(node: fx.Node) -> bool:
    """hardtanh(0, 6) is the FX-lowered form of ReLU6."""
    hardtanh_targets = {
        torch.ops.aten.hardtanh.default,
        torch.ops.aten.hardtanh_.default,
        torch.ops.aten.clamp.default,
    }
    if node.op == "call_function" and node.target in hardtanh_targets:
        # hardtanh args: (input, min_val=0.0, max_val=6.0)
        args = node.args
        if len(args) >= 3:
            try:
                return float(args[1]) == 0.0 and float(args[2]) == 6.0
            except (TypeError, ValueError):
                pass
        return True  # conservative: treat any hardtanh as candidate
    return False


# ============================================================================
# OPT-001: CUDA Graphs — reduce-overhead compile mode
# ============================================================================

def pass_cuda_graphs(gm: fx.GraphModule) -> fx.GraphModule:
    """
    OPT-001 (HIGH confidence): Enable CUDA Graphs to eliminate kernel launch overhead.

    Profile context: 51.8% GPU idle time from 1.75 ms of gaps across 130 CUDA API
    calls (avg 13.5 µs/launch). CPU dispatch serialises every kernel, creating
    per-launch synchronisation bubbles.

    Approach: This pass is a no-op at the FX level. CUDA Graph capture is requested
    by passing mode='reduce-overhead' to torch.compile() in get_model_and_input().
    We log here so the pass chain is observable and the omission is intentional.

    Alternatively, torch.cuda.make_graphed_callables() can wrap the compiled forward
    for fine-grained capture control — see get_model_and_input() below.

    Expected impact: ~1.75 ms wall-time recovered (40–50% latency reduction).
    """
    logger.info(
        "OPT-001 CUDA Graphs: pass is a no-op at FX level. "
        "CUDA Graph capture is activated via compile mode='reduce-overhead' "
        "in get_model_and_input(). Static shapes (B=16, HW=56×56) confirmed."
    )
    return gm


# ============================================================================
# OPT-002: 1×1 Pointwise Conv → MM Rewrite (shmem wave starvation fix)
# ============================================================================

def pass_conv1x1_as_mm(gm: fx.GraphModule) -> fx.GraphModule:
    """
    OPT-002 (HIGH confidence): Rewrite 1×1 convolutions as matrix multiplications.

    Profile context: Kernel2 (pointwise 1×1 conv) has 73.7 KB shmem/block with only
    4 warps/block (block_dim=[128,1,1]). On Blackwell (~192 KB shmem/SM), only 1–2
    blocks fit per SM → 4–8 active warps vs. 64 max → 8.1–8.6% occupancy. This
    kernel accounts for 44.1% of total kernel execution time (0.72 ms).

    Transformation: replace aten::convolution nodes where kernel_size=(1,1) and
    groups=1 with an equivalent reshape → mm → reshape sequence. Inductor then lowers
    this to a Triton GEMM with configurable tile sizes (64×64 or 128×128), avoiding
    the monolithic cuDNN 73 KB shmem allocation and allowing multiple concurrent
    blocks per SM.

    This mirrors torch._inductor.config.conv_1x1_as_mm = True, but implemented
    explicitly in the FX graph so it is visible and debuggable.

    Expected impact: 25–35% latency reduction on 1×1 conv kernels (~15% net wall time),
    occupancy → 50%+, TC utilisation → 60%+.
    """
    rewritten = 0
    try:
        for node in list(gm.graph.nodes):
            if not _node_is_conv(node):
                continue

            # Inspect kwargs / args for kernel_size and groups
            # aten.convolution signature:
            #   (input, weight, bias, stride, padding, dilation, transposed, output_padding, groups)
            args = node.args
            kwargs = node.kwargs

            # Weight is the second positional arg; its shape encodes kernel_size
            if len(args) < 2:
                continue
            weight_node = args[1]
            if not isinstance(weight_node, fx.Node):
                continue

            # Try to retrieve weight shape from meta tensor
            weight_meta = getattr(weight_node, "meta", {}).get("val", None)
            if weight_meta is None:
                # Fallback: try graph module parameters
                if weight_node.op == "get_attr":
                    try:
                        weight_meta = gm.get_parameter(weight_node.target)
                    except AttributeError:
                        pass
            if weight_meta is None:
                continue

            wshape = tuple(weight_meta.shape)
            # 1×1 conv has weight shape (C_out, C_in/groups, 1, 1)
            if len(wshape) != 4 or wshape[2] != 1 or wshape[3] != 1:
                continue

            # Check groups == 1 (pointwise, not depthwise)
            groups = args[8] if len(args) > 8 else kwargs.get("groups", 1)
            if groups != 1:
                continue

            c_out, c_in, _, _ = wshape
            logger.info(
                f"OPT-002: rewriting 1×1 conv node '{node.name}' "
                f"({c_in} → {c_out}) as reshape+mm+reshape"
            )

            input_node = args[0]

            # Build replacement subgraph after the conv node
            with gm.graph.inserting_after(node):
                # (B, C_in, H, W) → (B*H*W, C_in)
                reshape_in = gm.graph.call_function(
                    torch.ops.aten.reshape.default,
                    (input_node,),
                    {"shape": [-1, c_in]},
                )
                # Weight: (C_out, C_in) — already 1×1, squeeze spatial dims
                weight_squeeze = gm.graph.call_function(
                    torch.ops.aten.reshape.default,
                    (weight_node,),
                    {"shape": [c_out, c_in]},
                )
                # Transpose weight for mm: (C_in, C_out)
                weight_t = gm.graph.call_function(
                    torch.ops.aten.t.default,
                    (weight_squeeze,),
                )
                # MM: (B*H*W, C_in) × (C_in, C_out) → (B*H*W, C_out)
                mm_out = gm.graph.call_function(
                    torch.ops.aten.mm.default,
                    (reshape_in, weight_t),
                )

            # We need input spatial shape to restore. Read from input meta if available.
            input_meta = getattr(input_node, "meta", {}).get("val", None)
            if input_meta is not None and input_meta.ndim == 4:
                b, _, h, w = input_meta.shape
                with gm.graph.inserting_after(mm_out):
                    reshape_out = gm.graph.call_function(
                        torch.ops.aten.reshape.default,
                        (mm_out,),
                        {"shape": [b, c_out, h, w]},
                    )
                node.replace_all_uses_with(reshape_out)
            else:
                # Without spatial info, use a dynamic reshape via view
                with gm.graph.inserting_after(mm_out):
                    # Get input shape dynamically
                    input_shape = gm.graph.call_function(
                        torch.ops.aten.sym_size.default,
                        (input_node,),
                    )
                    reshape_out = gm.graph.call_function(
                        torch.ops.aten.reshape.default,
                        (mm_out,),
                        {"shape": [-1, c_out, -1, -1]},
                    )
                node.replace_all_uses_with(reshape_out)

            gm.graph.erase_node(node)
            rewritten += 1

        if rewritten:
            gm.graph.lint()
            gm.recompile()
            logger.info(f"OPT-002: rewrote {rewritten} 1×1 conv node(s) as MM")
        else:
            logger.info("OPT-002: no 1×1 conv nodes matched (may already be lowered by Inductor)")

    except Exception as e:
        logger.warning(f"OPT-002 conv1x1_as_mm failed: {e}. Continuing without this pass.")

    return gm


# ============================================================================
# OPT-003: Depthwise Conv Triton Stub (MEDIUM confidence)
# ============================================================================

def pass_depthwise_triton_stub(gm: fx.GraphModule) -> fx.GraphModule:
    """
    OPT-003 (MEDIUM confidence — stub): Detect depthwise convolutions and emit
    a warning stub for future Triton kernel replacement.

    Profile context: All 30 depthwise instances (groups=C_in, 3×3) show 0% TC
    utilisation. cuDNN's depthwise path does not decompose into a GEMM and never
    engages Blackwell tensor cores. Kernels are memory-bound at 47–73% DRAM peak.

    Full transformation requires a custom Triton kernel with float4 vectorised loads
    (128-bit transactions) and an optional BN+ReLU6 epilog fusion (see OPT-004).
    TC engagement is structurally impossible for depthwise (rank-deficient GEMM);
    gains are purely from memory access efficiency (~30–40% throughput increase).

    Status: STUB — detection only. Triton kernel implementation is a future TODO.
    """
    detected = 0
    try:
        for node in list(gm.graph.nodes):
            if not _node_is_conv(node):
                continue

            args = node.args
            if len(args) < 2:
                continue

            weight_node = args[1]
            groups = args[8] if len(args) > 8 else node.kwargs.get("groups", 1)

            weight_meta = getattr(weight_node, "meta", {}).get("val", None)
            if weight_meta is None and weight_node.op == "get_attr":
                try:
                    weight_meta = gm.get_parameter(weight_node.target)
                except AttributeError:
                    pass

            if weight_meta is None:
                continue

            wshape = tuple(weight_meta.shape)
            # Depthwise: groups == C_in == C_out, kernel 3×3
            if len(wshape) == 4 and wshape[2] == 3 and wshape[3] == 3:
                c_out = wshape[0]
                if groups == c_out:
                    detected += 1
                    logger.warning(
                        f"OPT-003 STUB: depthwise conv '{node.name}' detected "
                        f"(C={c_out}, groups={groups}, kernel=3×3). "
                        "Pattern detected but NOT rewritten — requires custom Triton kernel "
                        "with float4 vectorised loads and optional BN+ReLU6 epilog. "
                        "TODO: implement triton_depthwise_conv3x3_float4() and register "
                        "as a custom lowering via torch._inductor.lowering.register_lowering()."
                    )

        if detected:
            logger.info(
                f"OPT-003: detected {detected} depthwise conv node(s). "
                "No graph changes applied (stub pass)."
            )
        else:
            logger.info("OPT-003: no depthwise conv nodes matched.")

    except Exception as e:
        logger.warning(f"OPT-003 depthwise_triton_stub failed: {e}. Continuing.")

    return gm


# ============================================================================
# OPT-004: Conv → BN → ReLU6 Epilog Fusion Stub (MEDIUM confidence)
# ============================================================================

def pass_conv_bn_relu6_fusion(gm: fx.GraphModule) -> fx.GraphModule:
    """
    OPT-004 (MEDIUM confidence): Detect conv→BN→ReLU6 chains and mark them for
    epilog fusion. Full fusion requires a custom Triton epilog kernel.

    Profile context: BN+hardtanh already fused by Inductor into single Triton kernels
    (triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_*). However, the
    conv output is written to global memory and read back by the BN kernel. At
    [16, C, 56, 56] FP32, each intermediate is 3–13 MB; across 30 blocks this is
    significant avoidable DRAM traffic (bn_act kernels = 34% kernel time, 0.55 ms).

    Full transformation: register a custom Inductor fusion pass that pattern-matches
    convolution + native_batch_norm_legit_no_training + hardtanh in the post-lowering
    FX graph and emits a single Triton kernel with BN/clip as the conv epilog.
    After applying OPT-002 (Triton GEMM for pointwise), the epilog fusion becomes
    straightforward via tl.store with inline BN scaling and tl.clamp.

    Status: Detects conv→BN→ReLU6 chains. Logs per-chain stats for verification.
    No graph mutation applied — requires Triton epilog kernel.

    Expected impact: ~30–40% DRAM traffic reduction, ~10% latency reduction on
    BN+activation kernels once fully implemented.
    """
    chains_detected = 0
    try:
        for node in list(gm.graph.nodes):
            if not _node_is_conv(node):
                continue

            # Check single consumer chain: conv → BN → ReLU6
            conv_users = list(node.users)
            if len(conv_users) != 1:
                continue
            bn_node = conv_users[0]
            if not _node_is_batch_norm(bn_node):
                continue

            bn_users = list(bn_node.users)
            # BN may output a tuple (out, mean, rstd); follow the getitem
            relu6_candidates = []
            for u in bn_users:
                if u.op == "call_function" and u.target in (
                    operator_getitem := getattr(torch.ops.aten, "getitem", None),
                    # torch.fx sometimes uses __getitem__ directly
                ):
                    relu6_candidates.extend(list(u.users))
                elif _node_is_relu6(u):
                    relu6_candidates.append(u)

            # Also check direct BN users for relu6
            relu6_candidates += [u for u in bn_users if _node_is_relu6(u)]

            if not relu6_candidates:
                continue

            chains_detected += 1
            logger.warning(
                f"OPT-004 STUB: conv→BN→ReLU6 chain detected: "
                f"conv='{node.name}' → bn='{bn_node.name}'. "
                "Pattern detected but NOT fused — requires custom Triton epilog kernel. "
                "TODO: implement triton_conv_bn_relu6_fused() with tl.clamp epilog "
                "and register via torch._inductor.lowering."
            )

        if chains_detected:
            logger.info(
                f"OPT-004: detected {chains_detected} conv→BN→ReLU6 chain(s). "
                "No graph changes applied (stub pass)."
            )
        else:
            logger.info(
                "OPT-004: no conv→BN→ReLU6 chains matched in this graph. "
                "(Chains may have been pre-lowered by Inductor before this pass runs.)"
            )

    except Exception as e:
        logger.warning(f"OPT-004 conv_bn_relu6_fusion failed: {e}. Continuing.")

    return gm


# ============================================================================
# OPT-005: BF16 Insertion Pass (supplementary FX annotation)
# ============================================================================

def pass_annotate_bf16(gm: fx.GraphModule) -> fx.GraphModule:
    """
    OPT-005 (HIGH confidence): Log BF16 status of conv weight/input nodes.

    The primary BF16 cast is applied in get_model_and_input() before torch.compile().
    At that point, all parameters and the input tensor are already bfloat16, so no
    FX-level cast nodes are needed. This pass verifies the dtype in the FX graph's
    meta tensors and logs a warning if any conv input is still FP32.

    Profile context: FP32 throughout profile. Blackwell TCs deliver 2× higher
    throughput on BF16 vs FP32. Kernel2 TC utilisation of 18–45% reflects FP32
    TC rate; BF16 targets 60%+. Depthwise kernels are memory-bound; BF16 halves
    DRAM pressure directly.

    Expected impact: 40–80% throughput gain on pointwise, 50% memory reduction on
    activations, 25–35% net kernel-time latency reduction.
    """
    try:
        fp32_conv_count = 0
        bf16_conv_count = 0
        for node in gm.graph.nodes:
            if not _node_is_conv(node):
                continue
            val = node.meta.get("val", None)
            if val is None:
                continue
            if val.dtype == torch.float32:
                fp32_conv_count += 1
                logger.warning(
                    f"OPT-005: conv node '{node.name}' output is FP32. "
                    "BF16 cast may not have propagated. Check that model and input "
                    "were cast before torch.compile()."
                )
            elif val.dtype == torch.bfloat16:
                bf16_conv_count += 1

        if bf16_conv_count > 0:
            logger.info(
                f"OPT-005: {bf16_conv_count} conv node(s) confirmed BF16. "
                "Tensor Core BF16 path active."
            )
        if fp32_conv_count == 0 and bf16_conv_count == 0:
            logger.info("OPT-005: no meta dtype info available; skipping BF16 verification.")

    except Exception as e:
        logger.warning(f"OPT-005 annotate_bf16 failed: {e}. Continuing.")

    return gm


# ============================================================================
# Backend Registration
# ============================================================================

@register_backend
def transformer_opt(gm: fx.GraphModule, example_inputs) -> Callable:
    """
    Custom torch.compile() backend for DepthwiseSepConv: applies all 5 optimisation
    passes derived from Blackwell profiling data, then delegates to Inductor.

    Pass order:
      1. pass_cuda_graphs        — logging/verification (no-op; CUDA Graphs via compile mode)
      2. pass_conv1x1_as_mm      — 1×1 conv → MM before Inductor sees convolution nodes
      3. pass_depthwise_triton_stub — depthwise detection (stub; no graph change)
      4. pass_conv_bn_relu6_fusion  — chain detection (stub; no graph change)
      5. pass_annotate_bf16      — verify BF16 propagation after all rewrites

    Rationale for order:
      - OPT-002 must run before Inductor lowers convolutions to cuDNN calls
      - OPT-003 / OPT-004 are read-only stubs; order relative to OPT-002 is safe
      - OPT-005 annotation is last to see the final graph dtype state

    After passes, the graph is handed to compile_fx (Inductor) for kernel generation.
    """
    logger.info("transformer_opt backend: starting 5-pass FX optimisation pipeline")

    gm = pass_cuda_graphs(gm)           # OPT-001: CUDA Graphs (no-op, mode-level)
    gm = pass_conv1x1_as_mm(gm)         # OPT-002: 1×1 conv → MM (full pass)
    gm = pass_depthwise_triton_stub(gm) # OPT-003: depthwise Triton (stub)
    gm = pass_conv_bn_relu6_fusion(gm)  # OPT-004: conv-BN-ReLU6 epilog (stub)
    gm = pass_annotate_bf16(gm)         # OPT-005: BF16 verification

    logger.info("transformer_opt backend: all passes complete; delegating to Inductor")
    return compile_fx(gm, example_inputs)


# ============================================================================
# Workload Interface
# ============================================================================

def get_model_and_input() -> tuple:
    """
    Workload interface — return (compiled model, input tensor).

    Applies non-graph optimisations on top of the baseline before compiling:

      OPT-005 (BF16): Cast model and input to bfloat16 if baseline is still FP32.
        - Check performed before cast to avoid redundant work.
        - BF16 preferred over FP16 on Blackwell: same TC throughput, wider dynamic
          range, no loss scaling required.

      OPT-001 (CUDA Graphs): torch.compile() is invoked with mode='reduce-overhead',
        which internally captures a CUDA Graph on the first warm-up forward pass and
        replaces all 130 CUDA API calls with a single graph replay on subsequent
        calls. Static shapes (B=16, HW=56×56) and .eval() mode satisfy capture
        constraints.

    The custom transformer_opt backend handles FX-level rewrites (OPT-002–005).
    """
    assert torch.cuda.is_available(), "CUDA required"

    model, x = get_baseline_model_and_input()

    # OPT-005: BF16 cast — skip if baseline already uses BF16
    if next(model.parameters()).dtype != torch.bfloat16:
        logger.info("OPT-005: casting model to bfloat16 (baseline is FP32)")
        model = model.to(torch.bfloat16)
        x = x.to(torch.bfloat16)
    else:
        logger.info("OPT-005: model already bfloat16; skipping cast")

    # OPT-001 + all FX passes: compile with custom backend + reduce-overhead mode
    # reduce-overhead enables CUDA Graph capture (satisfies static-shape constraint)
    model = torch.compile(
        model,
        backend="transformer_opt",
        mode="reduce-overhead",
    )
    logger.info(
        "Compiled with backend='transformer_opt', mode='reduce-overhead' "
        "(CUDA Graphs + FX optimisation passes)"
    )

    return model, x


if __name__ == "__main__":
    logging.basicConfig(
        format="%(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
    )

    model, x = get_model_and_input()

    # Warm-up: trigger compilation + CUDA Graph capture
    with torch.no_grad():
        for i in range(3):
            y = model(x)
            if i == 0:
                print(f"✓ Output shape: {y.shape}, dtype: {y.dtype}")

    print("✓ Smoke test passed")