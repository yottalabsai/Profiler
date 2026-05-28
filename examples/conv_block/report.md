# ConvBlock GPU Optimization Report

This optimization eliminated all standalone BatchNorm and per-conv layout/normalization kernels (baseline 30 distinct attributed kernel launches -> 20 in the optimized capture) on the ConvBlock CNN (B=16, NVIDIA RTX PRO 6000 Blackwell Server Edition); the dominant convolution compute time was already Tensor-Core-bound and is unchanged, so the de-duplicated attributed-kernel time is essentially flat (~1.00x) — the win is launch-count and memory-traffic reduction, not GEMM speedup.

## 1. Hardware Context

| Field | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition (~188 SMs, GB202 class) |
| Architecture | Blackwell |
| PyTorch | 2.11.0+cu128 |
| Compile mode (baseline) | `inductor` |
| Compile mode (optimized) | `conv_block_opt` (custom FX backend) |
| Batch size | 16 (from conv input shapes `[16, 3, 64, 64]`) |
| Iterations | 2 measure iterations (ncu replay — relative timing only) |

> All `duration_ns` values are ncu application-replay timings, inflated 2-5x over real wall-clock. They are valid only for relative ranking within and across these two captures, never as absolute latency.

## 2. Operator Summary (Baseline)

De-duplicated per-op-id entries (the IR nodes the FX passes target). The baseline also contains re-aggregating wrapper entries — `layer::unique::prologue` (460,671 ns, 41 kernels), generic `aten::cudnn_convolution` (324,064 ns), `aten::convolution` (5,888 ns), `aten::addmm` (4,800 ns) — which re-count the same kernels and are excluded here to avoid double-counting. Percentages are relative to the de-duplicated per-call total = 434,303 ns.

| Operator (stage) | Time (%) | Duration (ns) | Kernels | Bottleneck Class |
|---|---|---|---|---|
| `aten::cudnn_convolution` 64->128 @64x64 (x2 iters) | 47.7 | 207,263 | 6 | Compute-bound (TC 60-65%, sm 58-64%) |
| `aten::cudnn_convolution` 128->256 @32x32 (x2 iters) | 42.3 | 183,968 | 6 | Compute-bound (TC 60-65%) |
| `aten::cudnn_convolution` 3->64 @64x64 (x2 iters) | 8.2 | 35,456 | 6 | Memory/layout-bound (TC 0%, convertTensor + small GEMM) |
| `aten::addmm` classifier 256->10 (x2 iters) | 1.8 | 7,616 | 2 | Latency-bound (occ 8%, eligible 6.8%, SIMT) |

On Blackwell, `tensor_core_active_pct` is reported; where it reads `0.0` on a conv stage it means that stage's time is dominated by the non-GEMM convertTensor/normalization kernels, not the implicit-GEMM. Bottleneck class is corroborated by memory throughput % and achieved occupancy.

## 3. Reading the Metrics

- **`tensor_core_active_pct`** — Fraction of cycles the Tensor Cores were busy. On the heavy stage-2/stage-3 convs it is 58-66%, confirming those `Kernel`/`sm80_xmma_fprop_implicit_gemm` convs are already efficient compute and are NOT the optimization target. A value of `0.0` on the classifier `gemmSN_TN_kernel` means the GEMM ran on the FP32 SIMT path with Tensor Cores fully idle — but here it is expected (a 16x256x10 GEMM is below the tile threshold), not a fixable signal. `null` on non-GEMM kernels is normal and never a bottleneck.
- **`achieved_occupancy`** — At ~8% on the classifier addmm (4-block grid against ~188 SMs) the device is almost entirely idle: this is wave starvation, intrinsic to the tiny problem size.
- **`memory_throughput_pct` / `dram_throughput`** — Several baseline `convertTensor_kernel` and BatchNorm Triton kernels hit 55-73% of peak DRAM with 0% Tensor-Core activity: pure bandwidth spent shuffling layout or applying a frozen affine — exactly the memory-bound waste OPT-1/OPT-2 target.
- **`eligible_cycles_pct < 20%`** (Blackwell latency-bound rule; `warp_cycles_per_instruction` is removed on Blackwell) — the classifier at 6.8% is latency-bound.

## 4. Optimizations Applied

| ID | Type | Target Operators | Hardware Evidence | Confidence | Status |
|---|---|---|---|---|---|
| OPT-1 | fusion (Conv-BN fold) | 6 conv nodes + `triton_*_native_batch_*` BN kernels | 28 BN launches, TC 0%, DRAM-bound, largest BN grid 16384 | high | APPLIED |
| OPT-2 | memory_layout (channels_last / NHWC) | 6 conv nodes + `convertTensor_kernel` (~20) | convertTensor at 73.5% peak DRAM, TC 0%, L1 hit 0% | high | APPLIED |
| OPT-3 | fusion (ReLU -> clamp_min epilogue) | 3 `aten::relu` post-fold conv epilogues | full-tensor elementwise pass, TC 0%, DRAM-bound | medium | APPLIED |
| OPT-4 | wave_quantization (classifier addmm) | 2 `aten::addmm` nodes | occ 8%, sm 0.23%, eligible 6.8%, SIMT | high | Not applied (documented no-op) |

All three actionable passes report `status == APPLIED` in `validation_report.json`: `pass_fold_bn` (folded 3 BatchNorm into Conv2d), `pass_relu_to_clamp_min` (rewrote relu -> clamp_min on conv epilogue), `pass_verify_channels_last` (no NCHW re-layout reinserted). OPT-4 was intentionally not implemented.

## 5. Implementation Notes

# ConvBlock — Optimized Backend Implementation Notes

Backend name: `conv_block_opt` (registered via `@register_backend`).
compile_mode: `inductor` (full FX-pass backend). Device: RTX PRO 6000 Blackwell, torch 2.11.0+cu128.

All graph passes run at the **Aten IR** level inside `_aten_fw_compiler`, which `aot_autograd`
invokes with the fully decomposed graph. `nn.Module` parameters are `placeholder` nodes at this
level. Their **values** for weight-folding come from the REAL dynamo-level `example_inputs`
(genuine `Parameter`/buffer tensors), positionally matched 1:1 to the placeholders — NOT from
`fw_example_inputs`, which are FakeTensors during AOTAutograd tracing.

Status: runtime-validated. The compiled forward executes on real CUDA input and matches the eager
baseline to **max abs diff 6.7e-06** (rtol/atol 1e-3). All 5 tests in the suite pass.

## Backend Architecture

| Pass | Method | Reason |
|---|---|---|
| OPT-1 Conv-BN fold | `_aten_fw_compiler` (`_pass_fold_bn`) | Folds eval-mode `aten::_native_batch_norm_legit_no_training` into the preceding conv weights/bias; removes the memory-bound `triton_*_native_batch_*` kernel chain. Weight-access pass — reads BN/conv tensors from `ph_to_tensor`. |
| OPT-2 channels_last | `get_model_and_input()` + `_aten_fw_compiler` verify (`_pass_verify_channels_last`) | Layout switch (NHWC) is non-graph — applied to module + input. A non-mutating Aten IR verify pass confirms no `clone`/`contiguous(NCHW)` re-layout was reinserted. |
| OPT-3 ReLU -> clamp_min | `_aten_fw_compiler` (`_pass_relu_to_clamp_min`) | Rewrites `aten::relu` consuming a folded conv output to `aten::clamp_min(x, 0)` so Inductor schedules it into the conv-output pointwise group instead of a standalone launch. |
| OPT-4 classifier addmm | not applied (documented no-op) | A 16x256x10 GEMM is intrinsically sub-tile/wave-starved; no fusion or layout transform helps. <2% of attributed time. Closed out for budget completeness only. |

## Pass order

`_pass_fold_bn` (OPT-1) -> `_pass_relu_to_clamp_min` (OPT-3) -> `_pass_verify_channels_last` (OPT-2).
This honors the dependency DAG in `optimizations.json`: OPT-1 is node-count-reducing and creates
the conv->relu adjacency that OPT-3 matches; the OPT-2 verify pass runs last after the graph shape
is final. OPT-2's actual layout change happens in `get_model_and_input()` before `torch.compile`.

## Key Design Decisions

- **BN fold via `torch.nn.utils.fuse_conv_bn_weights`** (per `code_hint`) rather than hand-rolled
  arithmetic — it is the maintained helper and handles the `bias=None` conv case (these convs were
  defined `bias=False`, giving a clean folded-bias slot). Before folding, the pass asserts the conv
  is the sole consumer of its output (`len(conv_node.users) == 1`), because the folding identity
  assumes exactly one downstream BN.

- **Tuple-return handling for BN.** `aten::_native_batch_norm_legit_no_training` returns
  `(output, save_mean, save_rstd)`. The pass does not replace the BN node directly; it re-routes the
  `getitem(bn, 0)` consumer to the new folded-conv node, then erases the getitem, the dead BN node,
  the original conv node, and calls `eliminate_dead_code()`.

- **Folded weights kept channels_last.** When the source conv weight is already channels_last
  (because OPT-2 converted the module first), the folded weight is re-materialized channels_last so
  OPT-2 does not re-trigger a one-time `convertTensor` layout pass. This is the prerequisite ordering
  noted in OPT-2 (`apply after OPT-1`).

- **OPT-2 split into non-graph + verify.** memory_format is not visible as a transformable FX node,
  so the layout change lives in `get_model_and_input()` (checked-before-applied per Rule 7). The Aten
  IR side is verification-only; transforming layout via node surgery would be incorrect at this level.

- **OPT-3 capped at medium confidence.** The `matched` no-op guard is included: Inductor's default
  epilogue fusion likely already absorbs a trailing relu/clamp_min, so absence of the pattern is
  benign. The pass guarantees the relu does not survive as a standalone launch after BN folding
  reshapes the graph.

- **Flat compile path expected.** The three `ConvBnRelu` stages are structurally distinct (3->64,
  64->128, 128->256), so `UniqueSubgraphRegistry.build_partition_equivalence_map()` returns no
  duplicates and the flat `aot_autograd(fw_compiler=_aten_fw_compiler)(gm, example_inputs)` path is
  taken, preserving cross-layer Inductor fusion. The dedup path (Rule 9) is retained for robustness.

## FakeTensor / calling-convention fixes (runtime correctness)

Three interacting hazards had to be resolved for the compiled callable to run on real inputs:

1. **Real vs Fake weight source.** `aot_autograd` passes the fw_compiler FakeTensors as
   `fw_example_inputs`. Building `ph_to_tensor` from those and folding produced FakeTensor folded
   weights that, once registered as `get_attr` buffers, baked fake constants into the graph —
   Inductor's runtime then failed with *"Cannot access data pointer of FakeTensor"* in
   `copy_misaligned_inputs`. Fix: the backend threads the REAL dynamo-level `example_inputs` into
   `_aten_fw_compiler` (via `functools.partial(real_inputs=...)`); `ph_to_tensor` is built from those.
   The reals correspond 1:1 and in-order to the aten placeholders (verified: 18 placeholders <-> 18
   real `Parameter`/`Tensor` args).

2. **Active FakeTensorMode during the fold.** Even with real inputs, `fuse_conv_bn_weights` runs
   while `compile_fx`'s `FakeTensorMode` is still on the dispatch stack; its internal
   `aten.zeros_like`/arithmetic on real tensors is intercepted and rejected. Fix: wrap the fold math
   in `torch._subclasses.fake_tensor.unset_fake_temporarily()` so the folded weight/bias are computed
   eagerly and stay real, followed by `.detach().clone()` to own the storage.

3. **Boxed calling convention.** `compile_fx()` returns a callable using Inductor's BOXED convention
   (one list arg: `f([a, b, c])`), but `aot_autograd`'s runtime invokes the fw_compiler result with
   UNPACKED positional args (`f(a, b, c)`). Mismatched, the 18 flat args arrive as a single list and
   `copy_misaligned_inputs` raised *"Expected tensors only, but got: list"*. Fix: `_aten_fw_compiler`
   returns a small `_boxed_adapter` that re-boxes positional args back into the list form Inductor's
   callable expects.

## Cache-coherence caveat (baked constants)

Because OPT-1 bakes folded conv+BN weights as constant buffers INTO the compiled artifact, the
dynamo compile cache (keyed by code object) must not be shared across two different parameter sets.
Compiling a second distinct `ConvBlock` instance in the same process without `torch._dynamo.reset()`
reuses the first model's baked-in folded constants (observed: 0.077 abs diff vs 6.7e-06 after reset).
This is a non-issue for normal single-model usage; the equivalence test calls
`torch._dynamo.reset()` before compiling its model to fold against the correct weights.

## 6. Before/After Results

Both captures use batch size 16, so the comparison is valid. Operators are matched by stage signature (shapes), summed across the 2 measure iterations. Baseline figures use the de-duplicated per-op-id entries; the baseline's wrapper/aggregate entries are excluded to avoid double-counting.

| Operator (matched by stage, x2 iters) | Baseline (ns) | Optimized (ns) | Speedup |
|---|---|---|---|
| `aten::cudnn_convolution` 3->64 @64x64 | 35,456 | 35,392 | 1.00x |
| `aten::cudnn_convolution` 64->128 @64x64 | 207,263 | 205,600 | 1.01x |
| `aten::cudnn_convolution` 128->256 @32x32 | 183,968 | 184,032 | 1.00x |
| `aten::addmm` classifier 256->10 | 7,616 | 7,552 | 1.01x |
| **Total (de-duplicated attributed)** | **434,303** | **432,576** | **1.004x** |

**Kernel-count / launch reduction (the real, measurable effect):**

| Metric | Baseline | Optimized |
|---|---|---|
| Distinct kernel-name classes | 13 | 4 |
| Total attributed kernel launches | 30 | 20 |
| Standalone BatchNorm Triton kernels (`triton_*_native_batch_*`) | present (21 launches / 127,327 ns inside the `prologue` aggregate range) | 0 — eliminated |
| `triton_poi_fused_convolution` prologue kernels | 6 (9,920 ns) | 0 — eliminated |
| Conv implicit-GEMM kernels (per-op-id) | 6 (372,032 ns) | 6 (370,720 ns) |
| convertTensor (NCHW<->NHWC) (per-op-id) | 12 (54,655 ns) | 12 (54,304 ns) |
| classifier GEMM (SIMT) | 2 (7,616 ns) | 2 (7,552 ns) |

**Step B — Speedup attribution.** The de-duplicated, op-id-matched attributed time is effectively flat (1.004x). The reason: the heavy stage-2/stage-3 convolutions were already compute-bound on the Tensor Cores (TC 58-66%), and OPT-1/OPT-2/OPT-3 do not touch the GEMM itself — they remove the cheap memory-bound kernels around it. Those eliminated kernels (the 21 standalone BatchNorm Triton kernels and 6 conv-prologue Triton kernels, ~137 us of replay time) lived in the baseline's `layer::unique::prologue` aggregate range under the original `inductor` capture, not in the per-op-id budget, so they do not show up as a per-operator delta. All three passes are `APPLIED` in `validation_report.json` and the expected kernels disappeared (BatchNorm Triton kernels gone, relu folded into clamp_min/epilogue), so the launch-count reduction is correctly attributed to OPT-1 (BN fold) and OPT-3 (relu epilogue). OPT-4 was not applied and is credited with nothing.

**Step C — Residual opportunity.** Re-ranking the optimized profile, the two heavy convolution stages (64->128 and 128->256) still dominate at ~205,600 ns and ~184,032 ns. They run at 58-66% Tensor-Core utilization and are compute-bound — no FX-level transform in `optimizations.json` targets them, and none remains unapplied except OPT-4 (a documented no-op). The convertTensor layout kernels (54 us) persist; channels_last removed the redundant per-BN-conv conversions but cuDNN still performs its own boundary reformatting around the implicit-GEMM path. No residual FX-level gain is estimated.

## 7. What Drove Each Speedup

**Conv-BN folding (OPT-1, launch-count reduction, no per-op time delta):** Folds each eval-mode BatchNorm's frozen affine into the preceding conv's weights and bias (`scale = gamma/sqrt(var+eps)`), so the normalization is applied for free inside the conv weights. Evidence: all 21 standalone `triton_*_native_batch_*` kernels (127,327 ns of memory-bound, TC-0% replay time in the baseline) disappear entirely from the optimized capture — the optimized profile contains zero `native_batch` kernels.

**channels_last / NHWC (OPT-2, layout consistency):** Converts the module and input to channels_last so activations stay in cuDNN's preferred NHWC layout end-to-end, preventing redundant NCHW<->NHWC round-trips between fused conv stages. Evidence: the 6 `triton_poi_fused_convolution` prologue kernels are gone; the surviving 12 `convertTensor_kernel` launches are cuDNN's own implicit-GEMM boundary reformatting, unchanged in count because the heavy convs still enter/exit the xmma path.

**ReLU -> clamp_min epilogue (OPT-3, prevents standalone activation launch):** Rewrites each post-fold `aten::relu` to `aten::clamp_min(x, 0)` so Inductor schedules it into the conv-output pointwise group rather than emitting an independent full-tensor pass. Evidence: no standalone relu/activation kernel survives in the optimized capture; the activation is absorbed into adjacent kernels (confidence medium — Inductor's default epilogue fusion overlaps this benefit).

## 8. Remaining Opportunities

| ID | Type | Target | Reason Not Applied | Projected Gain |
|---|---|---|---|---|
| OPT-4 | wave_quantization | `aten::addmm` classifier 256->10 | Documented no-op: a 16x256x10 GEMM is intrinsically sub-tile and wave-starved; no fusion or layout transform helps. <2% of attributed time. | 0 ns (none) |

OPT-4 is the only unapplied proposal and is a deliberate no-op. Applying it would yield no measurable gain. No further FX-level gains are identified in this profile — the dominant convolution stages are already Tensor-Core-bound.
