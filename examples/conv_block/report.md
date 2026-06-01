# Optimization Report — conv_block

This optimization achieved **1.63× total speedup** on conv_block (B=16, RTX PRO 6000 Blackwell), driven by moving the convolution GEMMs onto bf16 Tensor Cores and eliminating the cuDNN NCHW↔NHWC convert kernels and the standalone BatchNorm kernels; a standout per-operator win — the 64→128 conv dropping from ~104 µs to ~41 µs on the bf16 tensor-op path (Section 7) — is partly reabsorbed by six newly-introduced Triton dtype-convert/elementwise kernels (6.66 µs) that the bf16 promotion adds on the optimized side.

## 1. Hardware Context

| Field | Value |
|---|---|
| GPU model | NVIDIA RTX PRO 6000 Blackwell Server Edition (~188 SMs) |
| Architecture family | Blackwell (GB202) |
| PyTorch version | 2.11.0+cu128 |
| Compile mode (baseline) | inductor |
| Compile mode (optimized) | conv_block_opt (custom backend) |
| Batch size | 16 (3×64×64 input, eval mode) |
| Iteration count | warmup 2 / measure 2 (nsys capture — durations measured at locked GPU clocks; relative comparison) |
| Locked clocks | 1837 MHz graphics / 12481 MHz memory (both captures) |

**Timing source.** Per-operator **durations** are nsys-derived GPU kernel times from the capture phase, taken at locked GPU clocks (1837/12481 MHz) identical across both captures — so the baseline-vs-optimized comparison is fair and reproducible. The ncu replay phase contributes only the hardware **counters** (Tensor-core %, SM/DRAM throughput, occupancy), collected at its own base-clock lock. No clock-lock warning was present, so durations are treated as reproducible.

## 2. Operator Summary (baseline)

Total attributed time: 529,757 ns across 10 operators, 0 unattributed.

| Operator | Time (%) | Duration (ns) | Kernels | Bottleneck Class |
|---|---|---|---|---|
| aten::cudnn_convolution (op_id=12, 64→128) | 19.6 | 103,872 | 3 | Compute / Tensor-core bound (tc 66%) |
| aten::cudnn_convolution (op_id=33, 64→128, iter2) | 19.5 | 103,040 | 3 | Compute / Tensor-core bound (tc 66%) |
| aten::cudnn_convolution (op_id=18, 128→256) | 17.6 | 93,471 | 3 | Compute / Tensor-core bound (tc 73%) |
| aten::cudnn_convolution (op_id=39, 128→256, iter2) | 17.3 | 91,743 | 3 | Compute / Tensor-core bound (tc 73%) |
| aten::_native_batch_norm_legit_no_training | 16.1 | 85,088 | 14 | Memory bound (DRAM ~86–90%, tc 0%) |
| aten::cudnn_convolution (op_id=7, 3→64) | 3.4 | 18,272 | 3 | Mixed (convert kernels + tc 16% GEMM) |
| aten::cudnn_convolution (op_id=28, 3→64, iter2) | 3.4 | 17,888 | 3 | Mixed (convert kernels + tc 16% GEMM) |
| aten::convolution (NCHW→NHWC input prep) | 1.4 | 7,488 | 4 | Memory bound layout repack (tc 0%) |
| aten::addmm (op_id=21, linear head) | 1.0 | 5,087 | 1 | Small GEMM (cuBLAS) |
| aten::addmm (op_id=42, linear head, iter2) | 0.7 | 3,808 | 1 | Small GEMM (cuBLAS) |

## 3. Reading the Metrics

- **tensor_core_active_pct** — fraction of cycles the Tensor Cores were issuing. The four large baseline conv GEMMs sit at 66–73% (genuinely tensor-op bound), but they ran on the **TF32 s1688** single-precision path: full FP32 byte width at half-rate tensor ops. A value of **0.0 (not null)** on the BN/ReLU and input-prep kernels confirms those are pure SIMT memory passes with Tensor Cores idle — the signal that the math is elsewhere. A **null** value (the linear-head addmm and the bf16 convs whose dominant kernel ncu did not counter-sample) is expected for those small/cuBLAS kernels and is not a problem.
- **memory_throughput_pct (DRAM % of peak)** — above ~80% means the kernel is DRAM-bound. The baseline BN/ReLU triton kernels hit 86–90%, the clearest "memory-bound, fold-me" signal in the profile.
- **sm_throughput_pct** — compute-pipe utilization. High on the conv GEMMs (52–64%), low on the BN/convert kernels.
- **achieved_occupancy** — low (~8%) on the cutlass conv kernels is expected for 230+ register tensor-op kernels; they are tensor-core bound, not occupancy-limited, so the lever is precision, not launch geometry.

## 4. Optimizations Applied

| ID | Type | Target Operators | Hardware Evidence | Confidence | Status |
|---|---|---|---|---|---|
| OPT-1 | memory_layout (channels_last / NHWC) | all 6 cudnn_convolution ops + input-prep triton | 12 `convertTensor_kernel` NCHW↔NHWC permutes + 4-kernel input repack = 55,391 ns of zero-math overhead | high | APPLIED |
| OPT-2 | dtype_promotion (bf16) | the 4 large conv GEMMs (+ linear head) | GEMMs on TF32 s1688 path (tc 66–73%, sm 52–64%, DRAM 6–28%) — compute-bound, halve tensor-op cycles | medium | APPLIED |
| OPT-3 | fusion (eval-mode Conv-BN fold) | aten::_native_batch_norm_legit_no_training (14 nodes → 3 folded) | standalone BN/ReLU triton kernels DRAM 86–90%, sm 23%, tc 0% — memory-bound parameter re-reads | high | APPLIED |

All three passes report `status == APPLIED` in `validation_report.json` (syntax/import/registration/test_suite all pass; overall `READY_FOR_PROFILING`). The `OPT-3 fallback (aten)` entry is `NOT_APPLIED` by design — the functional-IR fold path already handled the fold, so the aten fallback gracefully no-ops.

## 5. Implementation Notes

# Implementation Notes — conv_block_opt

Custom `torch.compile()` backend for the ConvBlock workload (`examples/conv_block/conv_block.py`):
a VGG-style CNN of three `Conv2d-BN-ReLU` blocks (3->64, 64->128, 128->256) with
MaxPool/AdaptiveAvgPool and a Linear classifier head, eval mode, batch=16, 3x64x64 input,
fp32 inference on an RTX PRO 6000 Blackwell (torch 2.11.0+cu128, CUDA 12.8). Backend name
registered via `@register_backend`: **`conv_block_opt`**.

The backend is the canonical three-stage funnel
`_run_functional_passes(gm) -> compile_fx(inner_compile=_aten_inner_compile, config_patches=_build_config_patches())`,
invoked identically on the flat graph and on every dedup representative. ConvBlock's three
stages have different channel counts (3->64, 64->128, 128->256), so they are **not**
structurally identical — `UniqueSubgraphRegistry` returns an empty equivalence map and the
backend takes the **flat compile path**. The dedup branch is preserved for models with
repeated identical blocks.

## Backend Architecture

| Pass | Level | Method | Reason |
|---|---|---|---|
| OPT-1 — channels_last (NHWC) memory format | non-graph | `get_model_and_input()` (whole-module `.to(memory_format=torch.channels_last)`) | cuDNN's TF32/bf16 tensor-op fprop path is native NHWC; in default NCHW it inserts a `convertTensor` permute before AND after every conv (12 zero-math repack kernels, ~10.5% of attributed time) plus a 4-kernel triton input-prep repack. A whole-module memory_format conversion lets NHWC propagate through the conv stack without per-op transpose nodes; an in-graph `aten.contiguous` per conv risks reintroducing stray repacks. Checks current state before converting. |
| OPT-3 — eval-mode Conv-BN fold | functional | `_run_functional_passes` (`_fpass_fold_conv_bn`, reads weight values via placeholder->tensor lookup) | Folds each eval-mode BatchNorm into the preceding conv (`w'=w*gamma/sqrt(var+eps)`, `b'=beta-mu*gamma/sqrt(var+eps)`), exact and lossless. The JSON tags this `aten`, but at the aten seam it is a silent no-op: AOTAutograd decomposes eval BatchNorm BEFORE `inner_compile`, so no `aten._native_batch_norm_legit_no_training` node survives to match. On the Dynamo (functional) graph eval BN is still a single `F.batch_norm` node (training=False) fed directly by a conv — the only level where the fold matches. |
| OPT-2 — bf16 dtype promotion (conv + linear head) | aten | `_aten_inner_compile` (`_apass_bf16_promotion`) | Keys on the decomposed `aten.convolution.default` / `aten.addmm.default` / `aten.mm.default` targets; casts activation+weight (and bias / both matrix operands) to bf16 and the result back to fp32, fp32 accumulate. Routes the four compute-bound `cutlass_tensorop_s1688fprop_optimized_tf32` conv GEMMs (~74% of attributed time, tensor-core active 66-73%) off the half-rate TF32 path onto the bf16 (s16816-class) tensor-core path, halving tensor-op cycles and the bytes the DRAM-bound BN/ReLU triton kernels stage. Op-target pass — no weight-value lookup needed. |
| OPT-3-fallback — Conv-BN fold at the aten level | aten | `_aten_inner_compile` (`_apass_fold_conv_bn`) | Defensive fallback matching `aten._native_batch_norm_legit_no_training.default`. Expected no-op on torch 2.11 (eval BN is decomposed before the aten seam); logs a WARNING and returns the graph unchanged. Present so the fold still applies on a torch build that preserves the aten BN node. |

## Key Design Decisions

**Why OPT-3 (Conv-BN fold) runs at the functional level, not aten.** This is the critical
routing decision and it contradicts the `ir_level: "aten"` tag in `optimizations.json`.
`compile_fx` runs AOTAutograd (which decomposes eval-mode BatchNorm into primitive ops)
*before* the `inner_compile` seam, so by the time `_aten_inner_compile` sees the graph there
are zero `aten._native_batch_norm_legit_no_training` nodes left to match — the aten fold
would be a silent no-op. On the Dynamo functional graph the backend receives *before*
handing off to `compile_fx`, eval BN is still a single `torch.nn.functional.batch_norm` node
(training=False) fed directly by a conv2d node, so the matcher runs there. The aten-level
`OPT-3-fallback` stays registered defensively and gracefully no-ops on torch 2.11.

**How the OPT-2 -> OPT-3 prerequisite is honored across levels.** The proposal requires the
folded conv weight/bias to exist at the bf16 runtime dtype (register-buffer-after-dtype
rule), with OPT-2 listed as a prerequisite for OPT-3. Because OPT-3 is forced to the
functional level (runs *before* the aten bf16 pass), within-level sequencing cannot satisfy
this. Instead the fold computes `w'`/`b'` in fp32 for numerical accuracy, then registers the
folded buffers as **bf16** — matching the dtype OPT-2's cast would have produced. OPT-2's
cast in front of the conv then sees an already-bf16 weight and folds to a no-op. Folded 4D
weights are kept channels_last to preserve OPT-1.

**Why the functional fold wraps the conv with input/output casts.** The folded buffers are
bf16 but the conv input (graph input or a prior fp32 activation) is fp32; a bf16-weight conv
fed an fp32 input raises "Input type (float) and bias type (BFloat16) should be the same" at
AOTAutograd trace time. The fold therefore inserts an fp32->bf16 cast before the rewritten
conv and a bf16->fp32 cast after it, keeping the surrounding fp32 functional graph
dtype-consistent. The adjacent casts OPT-2 later adds at the aten level become redundant and
Inductor folds them away — the intended reconciled behavior.

**Why `prims.convert_element_type` instead of `aten._to_copy`.** All backend-inserted casts
(both the functional fold and the aten bf16 promotion) use
`torch.ops.prims.convert_element_type.default`. On torch 2.11 a hand-inserted
`aten._to_copy.default` trips Inductor's "both a fallback and a decomp for the same op"
assertion because that op has both registrations. `convert_element_type` lowers cleanly to a
Triton elementwise cast and its redundant adjacent pairs are folded by Inductor CSE/peephole.

**Why OPT-1 is non-graph and the flat compile path.** Whole-module memory_format is applied
in `get_model_and_input()` so NHWC propagates through the conv stack without per-op transpose
kernels (the funnel's non-graph rule for dtype/memory_format whole-module changes). ConvBlock
has no repeated structure, so `UniqueSubgraphRegistry` returns an empty equivalence map and
`_compile_unit` runs once on the flat graph — preserving cross-layer Inductor fusion. The
backend lets `compile_fx` own AOTAutograd exactly once (functional passes before it, aten
passes inside its `inner_compile` seam, config patches scoped to the call); it does not use
`aot_autograd(fw_compiler=compile_fx)`, which raises `AssertionError: Expected tensors only`
inside `copy_misaligned_inputs` on torch 2.11.

## 6. Before/After Results

Both profiles share batch size 16 and the same device (RTX PRO 6000 Blackwell). Captures are ~48 minutes apart (00:52 vs 01:41 UTC) — under the 6-hour cross-session threshold and on the same GPU, so no cross-session caveat applies; both were taken at identical locked clocks.

**Step A — operator matching.** Operators are matched by `operator_name`. In the optimized graph the standalone BatchNorm operator is gone (folded by OPT-3) and its remaining ReLU/pool/mean elementwise work is fused by Inductor into the `aten::convolution` / `aten::cudnn_convolution` epilogues, so the baseline BN operator (85,088 ns) and baseline NCHW→NHWC input-prep (`aten::convolution`, 7,488 ns) collapse into the optimized `aten::convolution` fused family (67,007 ns). The six new `triton_poi_fused_convert_element_type_*` / `triton_poi_fused_7/8` kernels (6,656 ns) are dtype-cast and elementwise kernels that bf16 promotion introduced — they land in `unattributed_kernels` and are charged to the optimized total as new overhead.

| Operator | Baseline (ns) | Optimized (ns) | Speedup |
|---|---|---|---|
| aten::cudnn_convolution 3→64 (op_id 7/6, iter1) | 18,272 | 31,039 | 0.59× |
| aten::cudnn_convolution 3→64 (op_id 28/27, iter2) | 17,888 | 30,656 | 0.58× |
| aten::cudnn_convolution 64→128 (op_id 12/10, iter1) | 103,872 | 40,704 | 2.55× |
| aten::cudnn_convolution 64→128 (op_id 33/31, iter2) | 103,040 | 40,735 | 2.53× |
| aten::cudnn_convolution 128→256 (op_id 18/15, iter1) | 93,471 | 50,623 | 1.85× |
| aten::cudnn_convolution 128→256 (op_id 39/36, iter2) | 91,743 | 49,567 | 1.85× |
| aten::_native_batch_norm + aten::convolution input-prep (fused into optimized conv family) | 92,576 | 67,007 | 1.38× |
| aten::addmm linear head (op_id 21/20, iter1) | 5,087 | 3,456 | 1.47× |
| aten::addmm linear head (op_id 42/41, iter2) | 3,808 | 3,136 | 1.21× |
| aten::t (transpose, optimized only) | — | 2,112 | new overhead |
| Triton convert/elementwise kernels (bf16-introduced, unattributed) | — | 6,656 | new overhead |
| **Total** | **529,757** | **325,691** | **1.63×** |

**Step B — speedup attribution.** All three passes are `APPLIED`, the expected counters moved in the right direction, and the targeted operators improved (see Section 7), so the speedup is attributed to OPT-1+OPT-2+OPT-3 jointly. The small 3→64 first-stage convs are the one regression (0.58–0.59×): at this tiny channel count (3 input channels) the bf16 cutlass/convolve path is slower than the baseline TF32 sm80_xmma kernel, and the new bf16 cast kernels add fixed overhead that the tiny GEMM cannot amortize. This is more than offset by the 64→128 and 128→256 stages.

**Step C — residual opportunity.** Re-ranking the optimized profile, the new top costs are the two 128→256 bf16 convs (~50 µs each, tc ~79%, now genuinely tensor-core bound — little headroom) and the fused `aten::convolution` Triton family (67 µs, DRAM ~64%, occupancy ~69% — memory-bound). The first-stage 3→64 convs are now a net regression and are the clearest residual target (force them back onto the TF32/fp32 path, or skip bf16 for ≤4-channel inputs). No unapplied FX proposals remain in `optimizations.json`.

## 7. What Drove Each Speedup

**channels_last / NHWC layout (OPT-1, applied across all convs):** marking conv activations and weights channels_last lets cuDNN consume NHWC natively, so the 12 `convertTensor_kernel<float,float,float,(cudnnKernelDataType_t)2>` permute kernels (55,391 ns of zero-math repacking in the baseline) and the 4-kernel NCHW→NHWC input-prep no longer appear in the optimized trace. Evidence: every `convertTensor_kernel` present in baseline conv ops is absent from the optimized profile; the input-prep `triton_poi_fused_convolution_0/1` is gone.

**bf16 dtype promotion (OPT-2, +2.55× on the 64→128 conv):** casting conv operands to bf16 routes the GEMMs off the TF32 `s1688` single-precision tensor-op path onto the Blackwell bf16 `cutlass_tensorop_bf16_s16816fprop` HGEMM path, roughly halving tensor-op cycles and staged bytes. Evidence: the dominant 64→128 kernel changed from a TF32 cutlass kernel at 103,872 ns to `cutlass_tensorop_bf16_s16816fprop_optimized_bf16_128x128` at 40,704 ns with tensor_core_active_pct rising to 70.67% (128→256 reaches 79.49%).

**eval-mode Conv-BN folding (OPT-3, folds 3 BN nodes; contributes to the 1.38× on the fused conv family):** folding each `_native_batch_norm_legit_no_training` into the preceding conv weight/bias deletes the standalone DRAM-bound BN/ReLU Triton kernels (baseline 85,088 ns at 86–90% DRAM, tc 0%) — the BatchNorm operator no longer exists in the optimized profile, and the surviving ReLU/pool/mean work is fused into the conv epilogue (optimized `aten::convolution` family, 67,007 ns). Evidence: the entire `aten::_native_batch_norm_legit_no_training` operator (14 kernels) is absent from the optimized trace; validation confirms "folded 3 eval-mode batch_norm node(s) into preceding conv at functional IR."

## 8. Remaining Opportunities

All proposed optimizations (OPT-1, OPT-2, OPT-3) were applied. No further FX-level gains were identified in `optimizations.json`.

The one actionable residual is not a proposed pass but a refinement of OPT-2: the first-stage 3→64 convolutions regressed (~0.58×, ~+13 µs each per iteration) because bf16 is counterproductive at 3 input channels. Reverting those two ops to the baseline TF32/fp32 path would recover roughly 25 µs across the two measured iterations (~8% of the optimized total), lifting the end-to-end speedup from ~1.63× toward ~1.77×.

## Reproduction Commands

```bash
# Baseline capture (inductor)
operator-profiler profile examples/conv_block/conv_block.py \
    --profile-name baseline --warmup-iters 2 --measure-iters 2
# -> examples/conv_block/profile.json

# Optimized capture (custom backend conv_block_opt)
operator-profiler profile examples/conv_block/conv_block_optimized.py \
    --profile-name optimized --warmup-iters 2 --measure-iters 2
# -> examples/conv_block/profile_optimized.json

# Validate the generated backend before profiling
#   -> profiler_output/validation_report.json

# Regenerate this report
operator-profiler report
```

Both captures used locked GPU clocks (1837 MHz graphics / 12481 MHz memory) and identical iteration counts (warmup 2 / measure 2); durations are nsys GPU kernel times compared relatively, and hardware counters come from the ncu replay phase.
