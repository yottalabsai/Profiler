# Optimization Report — DepthwiseSepConv

**This optimization achieved ≈1.08× net speedup on per-operator-attributed kernel time (B=16, NVIDIA RTX PRO 6000 Blackwell).** The headline number is modest and deliberately conservative: the one transformation that landed (bf16 pointwise GEMMs, OPT-1) delivered a clear *efficiency* win — achieved occupancy on the 1×1 convolutions rose from ~8% to ~31% and the kernels moved onto the bf16 tensor-core path — but that gain was largely offset by ~110 µs of new fp32↔bf16 cast and channels-last copy traffic, and the single highest-value proposal (conv-BN folding, OPT-2) **did not apply** to the graph. The detailed accounting below explains why a clean total-vs-total ratio is not directly available for this workload and what the defensible per-operator comparison shows.

> **Timing caveat (applies throughout).** All durations come from Nsight Compute (ncu) application-mode replay. ncu serializes kernels and adds counter-collection overhead, so absolute values are **2–5× longer than real wall-clock** and are meaningful only *relative to each other within the same capture*. Never read these as production latency.

---

## 1. Hardware Context

| Field | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Architecture | Blackwell (sm_120) |
| PyTorch | 2.11.0+cu128 (CUDA 12.8) |
| Baseline compile mode | `inductor` (built-in layer-dedup backend) |
| Optimized compile mode | `depthwise_sep_conv_opt` (custom `@register_backend`) |
| Batch size | 16 |
| Input | `[16, 32, 56, 56]` → `[16, 256, 56, 56]` |
| Iteration count | warmup 2 / measure 2 (ncu replay — relative timing only) |
| Captures | baseline 2026-05-29T22:15Z · optimized 2026-05-29T22:32Z (17 min apart, same GPU) |

---

## 2. Operator Summary (baseline)

Durations are ncu-replay relative. The baseline was produced by the built-in **layer-dedup** backend, which emits a synthetic NVTX aggregate `layer::unique::prologue` wrapping the compiled unique-partition forward. **That aggregate (33 kernels) and the per-aten-op attributions (32 kernels) are disjoint kernel sets** — verified: 0 shared kernel IDs — i.e. they are two independent measurements of the same logical forward (one fused-compiled, one library-kernel). Summing them double-counts the forward pass, so the table below lists the prologue separately and the "Time (%)" column is normalized over the raw 818 µs sum only to show relative weight.

| Operator | Time (%) | Duration (ns) | Kernels | Bottleneck Class |
|---|---|---|---|---|
| `layer::unique::prologue` *(synthetic aggregate — see note)* | 53.9% | 440,866 | 33 | Mixed (compiled forward) |
| `aten::_native_batch_norm_legit_no_training` | 13.4% | 109,984 | 12 | Memory-bound (DRAM 74%, TC 0%) |
| `aten::cudnn_convolution` (depthwise+pointwise group) | 6.3% | 51,233 | 6 | Memory-bound (DRAM 63%, TC 0%) |
| `aten::cudnn_convolution` op_id 53 — pointwise 128→256 | 4.8% | 39,072 | 1 | Latency-bound GEMM (occ **8.3%**, TC 45%) |
| `aten::cudnn_convolution` op_id 26 — pointwise 64→128 | 4.7% | 38,656 | 1 | Latency-bound GEMM (occ **8.4%**, TC 45%) |
| `aten::cudnn_convolution` op_id 45/18 — pointwise 32→64 | 5.4% | 44,161 | 2 | Latency-bound GEMM (occ **8.2%**, TC 39%) |
| `aten::convolution` (bare) | 2.4% | 20,000 | 2 | Memory-bound (TC 0%) |
| `aten::cudnn_convolution` op_id 49/22 — depthwise 128 | 3.4% | 28,192 | 2 | Memory-bound (DRAM 71%, TC 0%) |
| `aten::cudnn_convolution` op_id 10/37 — pointwise 32→64 | 2.7% | 21,984 | 2 | Latency-bound GEMM (occ **8.2%**, TC 19%) |
| `aten::cudnn_convolution` op_id 41/14 — depthwise 64 | 1.8% | 14,624 | 2 | Memory-bound (DRAM 57%, TC 0%) |
| `aten::cudnn_convolution` op_id 6/33 — depthwise 32 | 1.1% | 9,312 | 2 | Memory-bound (DRAM 46%, TC 0%) |

**Two root causes** (from `optimizations.json` analysis):
1. **Pointwise 1×1 convs are latency-bound, not throughput-bound.** They run as Ampere-tuned `cutlass_80 s1688gemm` TF32 kernels on a Blackwell device, pinned at ~8% achieved occupancy (224 regs/thread) with only 16–20% eligible cycles — the SM is starved, not saturated.
2. **BatchNorm + ReLU6 are unfused, pure-bandwidth Triton kernels** (DRAM 74–84%, TC 0%), because the convs went to library kernels that cannot accept a Triton epilogue. The 256-channel `hardtanh` variant alone writes up to 472 MB.

---

## 3. Reading the Metrics

Only metrics that drive this workload's bottlenecks:

- **`achieved_occupancy` (%)** — fraction of the SM's warp slots kept resident. The pointwise GEMMs sit at **~8%**, the dominant signal here: the kernel is latency-bound because too few warps are live to hide memory/instruction latency. Anything < ~30% on a GEMM means the kernel is leaving the machine idle. Post-optimization these convs reach ~31%.
- **`tensor_core_active_pct` (%)** — fraction of cycles the tensor cores were issuing. **`0.0` (not null) means the GEMM ran on the FP32/SIMT path with tensor cores idle** — the highest-ROI signal available. A `null` value (seen on this workload's `aten::copy_` kernels) is expected for non-GEMM kernels and is *not* a problem. Baseline pointwise convs show 19–45% (TF32 tensorop); the optimized bf16 GEMMs show 36.2%.
- **`dram_throughput_pct` (%)** — DRAM bandwidth utilization. The BatchNorm/ReLU6 kernels run at 74–84% — they are bandwidth-bound and the only way to speed them up is to *eliminate the pass*, which is exactly what conv-BN folding (OPT-2) was meant to do.
- **`sm_throughput_pct` (%)** — compute-pipe utilization. Low (~25%) on the bandwidth-bound BN/copy kernels, consistent with memory-bound classification.

---

## 4. Optimizations Applied

Status from `profiler_output/validation_report.json`.

| ID | Type | Target Operators | Hardware Evidence | Confidence | Status |
|---|---|---|---|---|---|
| OPT-1 | dtype_promotion (bf16 pointwise 1×1) | `aten.convolution.default` weight `[Cout,Cin,1,1]` | cutlass_80 s1688 TF32 GEMM, occ ~8%, 224 regs/thread, eligible-cycles 16–20% | medium | **APPLIED** (3 convs → bf16, Cout=64,128,256) |
| OPT-2 | fusion (conv-BN fold) | `aten.convolution` → `aten._native_batch_norm_legit_no_training` | 12 standalone BN+ReLU6 Triton kernels, DRAM 74–84%, TC 0%, up to 472 MB written | high | **NOT_APPLIED** (no conv→BN pair matched in graph — graceful) |
| OPT-3 | memory_layout (channels_last) | conv nodes + input placeholder | interior `triton_poi_fused_convolution_0` relayout copies, ~0% cache reuse | medium | **APPLIED** (input→NHWC eager-side; FX strip-copies a no-op — no redundant copies found) |
| OPT-4 | fusion (depthwise ReLU6 epilogue) | `aten.convolution` `[C,1,3,3]` → `aten.hardtanh` | one full DRAM round-trip per depthwise stage | medium | **NOT_APPLIED** (no depthwise→hardtanh pair matched — graceful) |

Two of four passes degraded gracefully (warning + unchanged graph, no crash). Their non-application is the central finding of this run — see §6 and §8.

---

## 5. Implementation Notes

# Implementation Notes — DepthwiseSepConv Optimized Backend

Backend name (for `--compile-backend`): **`depthwise_sep_conv_opt`**
Workload: `depthwise_separable_conv_optimized.py` (exposes `get_model_and_input()`).
Target: torch 2.11.0+cu128, RTX PRO 6000 Blackwell sm_120, `compile_mode = "inductor"`.

## Backend Architecture

All graph passes run at **Aten IR** inside `_aten_inner_compile`, installed via
`compile_fx`'s `inner_compile` hook (Strategy D, Rule 9). `compile_fx` owns AOTAutograd,
the decomposition table, the boxed calling convention, and the partitioner; we only swap
the leaf compiler `compile_fx_inner` and run the four passes just before it. Apply order
follows the DAG `OPT-1 -> {OPT-2, OPT-3, OPT-4}`, `OPT-2 -> OPT-4`.

| Pass | Method | Reason |
|---|---|---|
| OPT-1 dtype_promotion (bf16 pointwise 1×1 convs) | `_aten_inner_compile` (`_pass_bf16_pointwise_conv`) | Casts the 1×1 pointwise (`groups==1`, weight `[Cout,Cin,1,1]`) conv operands to bf16 so Inductor lowers them to an autotuned bf16 tensor-core GEMM for sm_120 instead of the prebuilt sm_80 cutlass s1688 (TF32) kernel pinned at ~8% occupancy; paired with `max_autotune_gemm`. Output re-cast to fp32. |
| OPT-2 conv-BN fold | `_aten_inner_compile` (`_pass_fold_conv_bn`) | Bakes the inference BN per-channel affine into the preceding conv weight/bias (numerically exact), eliminating the 12 standalone DRAM-bound `_native_batch_norm_legit_no_training` + hardtanh Triton kernels. Reads real param tensors via threaded `real_inputs`. |
| OPT-3 channels_last layout | `get_model_and_input()` (primary) + `_aten_inner_compile` (`_pass_strip_layout_copies`) | Eager-side `.to(memory_format=channels_last)` on model + input is the primary lever so conv/GEMM kernels run native NHWC and Inductor drops the per-block permute kernels; the graph pass strips any residual redundant channels_last copy nodes. |
| OPT-4 depthwise ReLU6 epilogue fusion | config (`max_autotune` + `max_autotune_conv_backends='TRITON,ATEN'`) + `_aten_inner_compile` (`_pass_mark_depthwise_relu6_fusion`) | Config lever lets the scheduler lower the depthwise conv to a Triton template and fuse the clamp(0,6) + folded-bias epilogue, removing one DRAM round-trip per depthwise stage; the graph pass detects+tags the (depthwise conv → hardtanh) pairs. Detection-only graph mutation. |

Non-graph levers (in `get_model_and_input()`): OPT-3 channels_last only. dtype promotion
(OPT-1) is deliberately a graph pass (not an eager `model.half()`) so it can target *only*
the pointwise convs and leave the memory-bound depthwise convs in fp32.

## Key Design Decisions

**OPT-1 before OPT-2 (buffer dtype).** OPT-1 casts the pointwise conv operands to bf16,
so OPT-2's folded weight/bias buffers for those convs are materialized at bf16 (matching
the conv runtime dtype); the depthwise conv stays fp32 and its folded buffers are fp32.
OPT-2 unwraps the OPT-1 cast nodes (`aten.to.dtype` / `aten._to_copy`) to recover the
*real* fp32 parameter values for the fold math (computed in fp32, then cast to the runtime
dtype), and detects whether the conv ran bf16 by inspecting whether its weight arg is a
cast node — emitting a fp32 re-cast after the folded bf16 conv to preserve the downstream
dtype contract.

**OPT-2 before OPT-4 (fusion enablement).** The ReLU6 (hardtanh) can only fuse into the
conv epilogue once BN has been folded out; before OPT-2 the BN node sits between the conv
and the hardtanh and blocks epilogue fusion. OPT-4 therefore runs last and walks back
through any OPT-1 fp32 re-cast to confirm the hardtanh's producer is a depthwise conv.

**OPT-4 is config-driven, not node surgery.** Forcing a Triton conv template + epilogue
fusion is an Inductor scheduler decision, not an FX rewrite. The graph pass only detects
and tags the depthwise→hardtanh pairs (`conv.meta['fuse_relu6_epilogue']`) so the contract
is logged/asserted; `max_autotune_conv_backends='TRITON,ATEN'` keeps the ATEN library
kernel as a fallback when the Triton depthwise conv loses autotune (the depthwise kernel is
already NHWC and competitive).

**Flat compile path (no dedup).** The three `DWSepBlock`s have distinct channel counts
(32→64, 64→128, 128→256), so `UniqueSubgraphRegistry.build_partition_equivalence_map()`
returns no duplicates and the backend takes the flat `_compile_with_aten_passes(gm, ...)`
path, preserving cross-block Inductor fusion. The dedup branch is retained for reuse if the
model grows to repeated blocks.

**Graceful degradation.** Every pass is wrapped: OPT-2 (high confidence) assumes the
pattern exists and logs a warning + returns `gm` unchanged on a genuine exception; OPT-1/3/4
(medium) additionally no-op with a warning when their pattern is absent. No pass can crash
the compile.

---

## 6. Before/After Results

Both captures share **batch size 16** and the **same GPU** 17 minutes apart (< 6 h) — no cross-session caveat required.

**Why no single total-vs-total ratio.** The two profiles were produced by *different backends* with *different attribution structure*:
- **Baseline (dedup backend)** emits a `layer::unique::prologue` synthetic aggregate (33 kernels) **plus** a disjoint 32-kernel per-aten-op set — two independent measurements of the same forward. Raw sum = 818,084 ns is inflated by this double-measurement (strategist flags `fused_kernel_double_count`, `warm_up_inflation`).
- **Optimized (flat backend)** has no prologue aggregate; its 26 operators are a single attributed kernel set, raw sum = 350,529 ns.

Comparing the raw sums (818 µs → 350 µs) would falsely report ~2.3×. The defensible comparison excludes the synthetic prologue aggregate from the baseline, leaving the per-aten-op attributed work on both sides:

| Bucket (semantic role) | Baseline (ns) | Optimized (ns) | Δ |
|---|---|---|---|
| Pointwise 1×1 conv (→ bf16 Triton GEMM in optimized) | 143,873 *(cudnn TF32)* | 60,833 *(bf16, TC 36%)* | **−83,040** |
| Depthwise 3×3 conv | 52,128 | 63,872 | +11,744 |
| `aten::convolution` / `cudnn_convolution` bare aggregate | 71,233 | *(folded into above)* | — |
| BatchNorm + ReLU6 (`_native_batch_norm_legit_no_training`) | 109,984 | 115,520 | +5,536 |
| **`aten::copy_` (fp32↔bf16 casts + channels_last copies)** | 0 | **110,304** | **+110,304 (new)** |
| **Total (per-op attributed, excl. synthetic prologue)** | **377,218** | **350,529** | **−26,689 → ≈1.08×** |

> Bucket-level matching is approximate: the dedup backend split pointwise work across named `cudnn_convolution` ops *and* a bare `aten::convolution` aggregate (71 µs), while the flat backend routes all pointwise GEMMs through one `aten::convolution` entry (60.8 µs). Treat the per-bucket rows as directional and the **Total** as the headline.

**Step B — Speedup attribution.** A gain is credited to a pass only if (1) `status == APPLIED`, (2) the expected counter moved, and (3) the operator improved.

- **Pointwise convs improved and OPT-1 is APPLIED with the expected counter shift** → the 83 µs reduction on the pointwise bucket **is attributed to OPT-1**. Evidence: the cutlass_80 TF32 s1688 kernels (occ ~8%, 224 regs/thread) were replaced by `triton_tem_fused_convolution` bf16 templates at occ ~31% / TC 36.2%, flanked by explicit `bfloat16_copy_kernel` casts.
- **BatchNorm did not improve (110 µs → 116 µs) and OPT-2 is NOT_APPLIED** → no speedup is attributed to conv-BN folding. The BN+ReLU6 kernels remain standalone, bandwidth-bound Triton passes — exactly the cost OPT-2 was designed to remove.
- **`aten::copy_` overhead (+110 µs) is the cost side of OPT-1 + OPT-3**: the fp32↔bf16 boundary casts (OPT-1 re-casts pointwise operands and outputs) and the channels_last `.to(memory_format)` conversions. The FX `strip_layout_copies` pass (OPT-3) found no redundant *interior* copies to remove, so these casts persist and nearly cancel the pointwise GEMM win.
- **OPT-4 NOT_APPLIED** → depthwise convs were not fused with their ReLU6 epilogue (depthwise bucket actually rose slightly, +11.7 µs, consistent with no fusion + channels_last relayout).

**Net:** a real ~83 µs architectural win on the pointwise GEMMs, eroded to a ~27 µs (≈1.08×) net gain by ~110 µs of new cast/copy traffic, with the largest *safe* opportunity (BN folding, ~110 µs of pure-bandwidth work) left entirely on the table.

---

## 7. What Drove Each Speedup

**bf16 pointwise convolution promotion (OPT-1, ≈2.4× on the pointwise bucket, 143.9 µs → 60.8 µs):** Casting the 1×1 pointwise convs to bf16 plus `max_autotune_gemm` routed Inductor away from the prebuilt Ampere `cutlass_80 s1688gemm` TF32 kernel and onto an sm_120-autotuned bf16 tensor-core GEMM template (`triton_tem_fused_convolution`). The decisive counter change is achieved occupancy rising from **~8% to ~31%** (the TF32 kernel's 224 regs/thread were the occupancy limiter) with tensor cores now active at **36.2%** — the latency-bound starvation identified as root cause #1 was relieved.

*(No other proposed transformation contributed a measured speedup: OPT-2, OPT-3's interior-copy removal, and OPT-4 either did not apply or were no-ops, per §4 and §6.)*

---

## 8. Remaining Opportunities

The two highest-leverage and unrealized proposals dominate residual gain.

| ID | Type | Target | Reason Not Applied | Projected Gain |
|---|---|---|---|---|
| OPT-2 | conv-BN fold | `aten.convolution` → `_native_batch_norm_legit_no_training` (12 nodes) | Pass matched no conv→BN pair as the backend saw the graph (BN consumed via `getitem`, or conv users ≠ 1 after OPT-1 cast insertion) — degraded gracefully | ~110,000 ns (13.4% of total); eliminates the 12 bandwidth-bound BN+ReLU6 kernels, incl. the 472 MB 256-ch hardtanh write |
| OPT-4 | depthwise ReLU6 epilogue fusion | `aten.convolution` `[C,1,3,3]` → `aten.hardtanh` | No depthwise→hardtanh pair matched; depends on OPT-2 folding BN out first (BN node still sits between conv and hardtanh) | ~25,000 ns (3.1%); removes one DRAM round-trip per depthwise stage |
| OPT-3 | channels_last (interior copies) | conv nodes + layout-change boundaries | Eager-side NHWC cast applied, but FX `strip_layout_copies` found no redundant *interior* copies; instead new boundary `aten::copy_` casts appeared (+110 µs) | ~40,000 ns (4.9%) *if* the boundary casts can be hoisted/eliminated — currently net-negative |
| OPT-1 (residual) | dtype boundary | pointwise cast nodes | Applied, but per-conv fp32↔bf16 re-casts persist as `bfloat16_copy_kernel` | Keep activations in bf16 across consecutive stages (single up-front cast) to remove ~most of the 110 µs copy overhead |

**Highest priority: fix OPT-2 so conv-BN folding actually fires.** It is high-confidence, numerically exact in inference, and targets ~110 µs of pure-bandwidth work that the current run leaves untouched — by itself a larger prize than the bf16 win already banked. The likely fix is matching the BN through its `getitem(bn, 0)` tuple-return and relaxing the `len(conv.users) == 1` guard now that OPT-1 inserts a cast consumer on the conv output. Once BN is folded, OPT-4's depthwise epilogue fusion is unblocked. Pairing OPT-2 + OPT-4 + eliminating the redundant bf16 casts (keep activations in bf16 between stages) would plausibly push the workload past **1.4–1.6×** on per-operator-attributed time.

---

## Reproduction

```bash
# Baseline capture (built-in dedup backend)
python3 nvidia/scripts/run_workload.py \
    --workload examples/depthwise_separable_conv/depthwise_separable_conv.py \
    --output-prefix profiler_output/depthwise_separable_conv \
    --inductor-debug-dir profiler_output/depthwise_separable_conv_inductor_debug \
    --correlation-pass --warmup-iters 2 --measure-iters 2
nsys profile --trace=cuda,nvtx --output=profiler_output/depthwise_separable_conv \
    python3 nvidia/scripts/run_workload.py \
    --workload examples/depthwise_separable_conv/depthwise_separable_conv.py \
    --output-prefix profiler_output/depthwise_separable_conv \
    --inductor-debug-dir profiler_output/depthwise_separable_conv_inductor_debug \
    --warmup-iters 2 --measure-iters 2
# (ManifestBuilder → AttributionEngine → ncu replay [sudo] → build_profile → profile.json)

# Optimized capture (custom backend)
#   run the same two phases on depthwise_separable_conv_optimized.py
#   with --compile-backend depthwise_sep_conv_opt → profile_optimized.json
```

Validate the backend before re-profiling: `pytest test_depthwise_separable_conv_optimized.py`.
ncu requires sudo on this system (`ERR_NVGPUCTRPERM` otherwise); pass `PYTHONPATH=$(pwd)` through `sudo env`.
