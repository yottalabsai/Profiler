# ConvBlock — GPU Optimization Report

**This optimization achieved a 1.49× total speedup on ConvBlock (B=16, RTX PRO 6000 Blackwell Server Edition)** by eliminating cuDNN NCHW↔NHWC layout transposes (channels_last) and routing the convolutions and the DRAM-bound BatchNorm/ReLU/pool epilogue from the TF32 path onto the BF16 tensor-core path.

---

## 1. Hardware Context

| Field | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Architecture | Blackwell (sm_100-class) |
| PyTorch | 2.11.0+cu128 |
| Compile mode (baseline) | `dedup-inductor` (built-in dedup backend) |
| Compile mode (optimized) | `conv_block_opt` (custom registered backend) |
| Batch size | 16 (input `[16, 3, 64, 64]`) |
| Iterations | 2 measured forward passes per capture (ncu replay — relative timing only) |

All durations below are from ncu application-mode replay and are inflated ~2–5× relative to real wall-clock execution. **Every figure is a relative comparison within this profile, never an absolute latency.**

---

## 2. Operator Summary (baseline)

Operator-class durations summed over the 2 measured iterations (the `layer::unique::prologue` dedup representative is excluded — it re-attributes the same physical kernels and would double-count). Total attributed kernel time = **453,535 ns**.

| Operator | Time (%) | Duration (ns) | Kernels | Bottleneck Class |
|---|---|---|---|---|
| `cudnn_convolution` 64→128 (`[16,64,64,64]`) | 38.1% | 172,864 | 2× (cutlass tf32 + 2 transposes ea.) | Compute / tensor-core (tc ~59%) |
| `cudnn_convolution` 128→256 (`[16,128,32,32]`) | 33.5% | 151,712 | 2× (cutlass tf32 + 2 transposes ea.) | Compute / tensor-core (tc ~72%) |
| `_native_batch_norm…relu` (BN+ReLU+MaxPool+mean, fused) | 15.5% | 70,496 | 14 | Memory-bound (DRAM 62–84%) |
| `cudnn_convolution` 3→64 stage-1 (+input pack) | 11.5% | 52,223 | sm80_xmma + triton pack | Small-channel (tc ~16%) |
| `addmm` linear head (256→10) | 1.4% | 6,240 | 2 (cuBLAS small-N GEMM) | Latency-bound (tiny GEMM) |

The two large convolutions consume ~72% of the budget. They are *already* on the TF32 NHWC tensor-op path (tensor cores 59–72% active) — so the recoverable waste is **not** idle tensor cores, but (a) the layout-transpose churn wrapping every conv and (b) TF32→BF16 throughput headroom plus the DRAM-bound BN epilogue.

---

## 3. Reading the Metrics

Only the counters that drive this workload's bottlenecks:

- **`smsp__pipe_tensor_cycles_active` / `tensor_core_active_pct`** — fraction of cycles the tensor-core MMA pipe is busy. The baseline big convs sit at 59–72% on the **TF32** path; that is not idle, so the lever is dtype throughput (BF16 ≈ 2× TF32 MMA rate on Blackwell), not "turn the tensor cores on." A value of `0.0` on the BN/ReLU/pool triton kernels is expected — they do no matrix math.
- **`dram__throughput.avg.pct_of_peak`** — for the BN+ReLU+MaxPool+mean group this runs 62–84%, the signature of a memory-bound elementwise/reduction pass. BF16 halves the bytes moved, giving near-linear speedup on this slice.
- **`achieved_occupancy`** — the big convs run at only ~8.3% occupancy (150 regs/thread + 73.7 KB smem/block in baseline). BF16's lower register footprint lifts this (to ~14.6% on the 64→128 conv), recovering latency-hiding headroom.
- **`convertTensor_kernel` launches (duration only, no counters)** — these are pure NCHW↔NHWC transposes cuDNN inserts because the framework tensors were NCHW while the conv kernels are native NHWC. 12 launches, ~49,600 ns of zero-arithmetic work. Their *disappearance* in the optimized profile is the direct signal that channels_last applied.

Note: `warp_cycles_per_instruction` is `null` for all operators — that counter is removed on Blackwell; `eligible_cycles_pct` is used as the latency-bound proxy instead.

---

## 4. Optimizations Applied

| ID | Type | Target Operators | Hardware Evidence | Confidence | Status |
|---|---|---|---|---|---|
| OPT-1 | memory_layout (channels_last / NHWC) | all 6 `cudnn_convolution` ops | 12× `convertTensor_kernel` transposes, ~49,600 ns (10.9%), zero useful math | high | **APPLIED** (eager-side boundary cast; in-graph pass no-op as designed) |
| OPT-2 | dtype_promotion (TF32 → BF16) | 4 large convs + BN/ReLU/pool/mean group | convs already TF32 tensor-op (tc 59–72%), occupancy 8.3%; BN group DRAM-bound 62–84% | medium | **APPLIED** (eager-side boundary cast; in-graph pass no-op as designed) |
| OPT-3 | fusion (eval-mode conv-BN fold) | `_native_batch_norm_legit_no_training` | DRAM-bound BN epilogue, 15.5% | low | **NOT_APPLIED** (no foldable conv→BN pair found — Inductor already bakes the affine; graceful no-op) |

OPT-1 and OPT-2 are implemented as a single combined boundary cast — `model.to(memory_format=channels_last).bfloat16()` and the matching input cast in `get_model_and_input()` — with the FX-level passes retained as documented fallbacks that detect the already-converted state and degrade to a logged no-op. This is the proposal's preferred application path (set layout/dtype once at the graph boundary rather than re-cloning/re-casting per conv).

---

## 5. Implementation Notes

# conv_block_opt — Implementation Notes

Custom `torch.compile()` backend for `ConvBlock` (VGG-style Conv2d → BatchNorm2d →
ReLU pipeline + Linear head). Registered backend name: **`conv_block_opt`**.

The backend uses the fixed three-stage funnel (Rule 9):
`_run_functional_passes(gm)` → `compile_fx(inner_compile=_aten_inner_compile,
config_patches=...)`. `compile_fx` owns AOTAutograd, the decomposition table, the
boxed calling convention, and the partitioner; the aten passes run at the
`inner_compile` seam on the fully decomposed Aten IR graph. The weight-VALUE-reading
conv-BN fold reads the genuine parameter tensors threaded as `real_inputs` (because
`inner_compile`'s `example_inputs` may be FakeTensors), and `_repropagate_meta`
repopulates `meta['val']` on inserted nodes after each structural rewrite.

`compile_mode` is `dedup-inductor`; the backend is dedup-aware. ConvBlock's three
ConvBnRelu stages have distinct channel counts (3→64, 64→128, 128→256), so
`UniqueSubgraphRegistry` finds no repeated layers and the flat compile path is taken
(preserving cross-stage Inductor fusion). The dedup branch is retained for structural
reuse if the model grows repeated blocks.

## Backend Architecture

| Pass | Level | Method | Reason |
|---|---|---|---|
| OPT-1 channels_last (NHWC) — eliminate cuDNN convertTensor NCHW↔NHWC transposes (12 launches, ~49,600 ns) around every conv | non-graph (primary) + `aten` (fallback) | `get_model_and_input()` (`model`/`x.to(memory_format=channels_last)`) primary; `_aten_inner_compile` (`_pass_channels_last_conv_inputs`) fallback | Layout is set once at the graph boundary (Rule 7) so cuDNN consumes NHWC framework tensors directly; the aten pass inserts `aten.clone(memory_format=channels_last)` only when the boundary layout was not set, and no-ops otherwise. Layout-only and numerically exact, so highest confidence. |
| OPT-2 BF16 dtype promotion — route the four large convs from the TF32 tensor-op path to the ~2x-faster BF16 path (and recover the ~8.3% occupancy), halve the DRAM-bound BN+ReLU+pool epilogue bytes | non-graph (primary) + `aten` (fallback) | `get_model_and_input()` (`model.bfloat16()` / `x.bfloat16()`) primary; `_aten_inner_compile` (`_pass_bf16_conv_operands`) fallback | dtype is cleanest as a whole-module boundary cast combined with OPT-1 (Rule 7, proposal application order); the aten pass casts conv operands to bf16 and restores fp32 on the output, no-opping when the operands are already bf16. Medium confidence — changes numerics. |
| OPT-3 conv-BatchNorm fold (eval mode) — bake the BN per-channel affine into the preceding conv weight/bias, removing the BN scale/shift arithmetic from the DRAM-bound epilogue | `aten` | `_aten_inner_compile` (`_pass_fold_conv_bn`, reads `real_inputs`) | The fold pattern (`aten.convolution` → `aten._native_batch_norm_legit_no_training`) is cleanly expressed only after AOTAutograd decomposition, and it needs the real running-stats/weight tensors. Low confidence — Inductor may already bake the eval-mode affine into the elementwise epilogue, in which case the explicit fold is a near no-op. |

No `functional`-level passes (no QKV/SDPA fusion in this CNN) and no
`inductor_config`-level passes; `_run_functional_passes` and `_config_patches` are
uniform structural pass-throughs that keep the funnel shape consistent per Rule 9.

## Key Design Decisions

**Why OPT-1 and OPT-2 are non-graph (boundary) primaries with aten fallbacks.**
The proposal explicitly prefers a single combined boundary cast
`model.to(memory_format=channels_last).bfloat16()` over per-conv FX surgery, because
it sets layout and dtype once at the graph boundary rather than re-cloning/re-casting
per convolution. Per Rule 7 these live in `get_model_and_input()` and are
idempotent (checked before applying). The aten-level FX passes are retained as the
documented fallbacks from the proposal's `fx_steps` so the optimization still applies
if a future call site cannot set the boundary layout/dtype; both detect the
already-converted state via `node.meta['val']` and degrade to a logged no-op, which
is the expected path here.

**Why OPT-3 reads `real_inputs` rather than the `inner_compile` example inputs.**
At the Aten IR level all BN/conv parameters are `placeholder` nodes; their actual
tensors are positionally matched in graph order. Inductor traces `inner_compile`
under FakeTensorMode, so the `example_inputs` there can be FakeTensors with no real
data. The conv-BN fold needs genuine `running_mean`/`running_var`/`gamma`/`beta` and
the conv weight to compute `scale = γ/√(var+ε)`, `W' = W·scale`,
`bias' = β − γ·mean/√(var+ε)`, so the backend threads `real_inputs=list(example_inputs)`
into `_aten_inner_compile` via `functools.partial` and builds the placeholder→tensor
map from those.

**Why OPT-3 must run after OPT-2 (within the aten level).** The fold allocates new
folded weight/bias buffers via `register_buffer`. A buffer registered before dtype
promotion cannot be recast, so OPT-3 must observe the conv's final runtime dtype. The
pass infers that dtype from the conv weight node's `meta['val']` (bf16 once OPT-2 or
the boundary cast applied) and registers the folded buffers accordingly, adding a
fp32 re-cast on the folded conv output when the conv runs bf16 so downstream
consumers keep their expected dtype.

**Bias introduction in the fold.** The model's convs are constructed with
`bias=False`, so the fold must introduce a `bias'` term and rewire a fresh *biased*
`aten.convolution` (appending the folded-bias arg when the original conv had a `None`
bias slot). The BN op returns a 3-tuple, so the rewrite redirects the live
`getitem(bn, 0)` consumer to the new conv output and then erases the getitem, the BN
node, any intermediate dtype cast, and the dead original conv, followed by
`eliminate_dead_code()`.

**Conv target set.** Both `aten.cudnn_convolution.default` (the form the profile
attributes the bottleneck to) and `aten.convolution.default` are matched, since the
exact decomposed op depends on the dtype/layout path Inductor selects after the
boundary casts.

---

## 6. Before/After Results

Both profiles were captured on the same GPU (RTX PRO 6000 Blackwell) ~16 minutes apart at batch size 16 — no cross-session caveat applies. Operators are matched by class and summed over the 2 measured iterations in each capture.

| Operator | Baseline (ns) | Optimized (ns) | Speedup |
|---|---|---|---|
| `cudnn_convolution` 64→128 | 172,864 | 80,480 | **2.15×** |
| `cudnn_convolution` 128→256 | 151,712 | 99,328 | **1.53×** |
| `_native_batch_norm…relu` (BN+ReLU+MaxPool+mean) | 70,496 | 58,400 | **1.21×** |
| `cudnn_convolution` 3→64 stage-1 (+input pack) | 52,223 | 59,008 | 0.89× (regression) |
| `addmm` linear head | 6,240 | 6,496 | 0.96× |
| **Total attributed kernel time** | **453,535** | **303,712** | **1.49×** |

Kernel-count evidence: the optimized BN group dropped from 14 → 10 kernels (7 → 5 per iteration) and **all 12 `convertTensor_kernel` transposes vanished** (baseline had 2 per conv; optimized has none).

---

## 7. What Drove Each Speedup

**channels_last / NHWC (OPT-1, contributes across all convs):** Casting the model and input to `torch.channels_last` makes the framework tensors NHWC-contiguous, matching the layout the cuDNN tensor-op conv kernels already consume, so cuDNN no longer wraps each convolution in a transpose-in/transpose-out pair. Evidence: the 12 `void cudnn::…convertTensor_kernel<float,…>` launches (~49,600 ns, 10.9% of baseline) present in `profile.json` are entirely absent from `profile_optimized.json`.

**TF32 → BF16 dtype promotion (OPT-2, +2.15× on 64→128, +1.53× on 128→256, +1.21× on the BN group):** Casting to bfloat16 switches the conv compute kernel from `cutlass_tensorop_s1688fprop_optimized_tf32_…_nhwc_align4` to `cutlass_tensorop_bf16_s16816fprop_optimized_bf16_…_nhwc_align8`. Evidence: tensor-core activity rose (64→128: 59% → 71%; 128→256: 72% → 79%), occupancy on the 64→128 conv lifted from 8.3% → 14.6% as register pressure fell, and on the DRAM-bound BN+ReLU+pool+mean group BF16 halved the bytes moved, collapsing 7 kernels/iteration to 5 and cutting its time 1.21×.

OPT-3 (conv-BN fold) is **not** credited with any speedup — it reported `NOT_APPLIED` (no foldable conv→BN pair survived AOTAutograd, because Inductor already bakes the eval-mode affine into the fused epilogue). The 1.21× on the BN group is attributed to OPT-2's BF16 byte reduction, not to OPT-3.

---

## 8. Remaining Opportunities

| ID | Type | Target | Reason Not Applied | Projected Gain |
|---|---|---|---|---|
| OPT-3 | fusion (conv-BN fold) | `_native_batch_norm_legit_no_training` | No foldable conv→BN pair found — Inductor already bakes the eval-mode affine into the elementwise epilogue | ~4.4% (already realized by Inductor; explicit fold would be a near no-op) |

**New second-order finding — stage-1 regression.** The 3→64 stage-1 convolution got *slower* under BF16 (52,223 → 59,008 ns, 0.89×). With only 3 input channels the tensor-core tiles are under-filled, and cuDNN selected `convolve_common_engine_float_NHWC<bf16>` (~29,500 ns/iter) instead of the baseline `sm80_xmma` implicit-GEMM path (~14,800 ns/iter). This is an intrinsic small-channel inefficiency, made worse by BF16 for this one layer. The clean remedy is to **exclude stage-1 from the BF16 cast** (keep the first conv in TF32/FP32 while the 64→128 and 128→256 convs stay BF16); recovering the ~6,800 ns regression would push the total speedup from 1.49× toward ~1.52×. The linear head (`addmm`, 0.96×) is negligible at 1.4% of the budget and not worth targeting.

Aside from re-tuning stage-1's dtype, no further FX-level gains are identified: the layout churn is eliminated, the dominant convs and the DRAM-bound epilogue are on the BF16 tensor-core path, and the BN affine fold is already handled by Inductor.

---

## Reproduction

```bash
# Baseline capture
python3 nvidia/scripts/run_workload.py --workload examples/conv_block/conv_block.py \
    --output-prefix profiler_output/conv_block --correlation-pass
nsys profile --trace=cuda,nvtx --output=profiler_output/conv_block \
    python3 nvidia/scripts/run_workload.py --workload examples/conv_block/conv_block.py \
    --output-prefix profiler_output/conv_block
# → profile.json   (12 operators, 0% unattributed)

# Optimized capture (registered backend conv_block_opt)
#   run_workload.py with --compile-backend conv_block_opt on conv_block_optimized.py
# → profile_optimized.json   (9 operators, 0% unattributed)

# Validate the backend before profiling
pytest examples/conv_block/test_conv_block_optimized.py
```

*Durations are ncu application-replay values (~2–5× inflated); treat every number as a relative comparison within this report, not a wall-clock latency. Batch size = 16 throughout.*
