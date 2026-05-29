# Optimization Report — `depthwise_separable_conv`

**On cleanly-attributed compute, the FX backend `depthwise_separable_conv_opt` achieved a 1.08× speedup at B=16 on an NVIDIA RTX PRO 6000 Blackwell Server Edition** — driven almost entirely by lowering the smaller pointwise (1×1) convolutions to cuBLAS GEMMs. See §6 for why the raw total-duration ratio (3.85×) overstates this and should not be quoted.

---

## 1. Hardware Context

| Field | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Architecture | Blackwell (GB202-class, sm_120 — assumed; see note) |
| PyTorch | 2.11.0+cu128 |
| Baseline compile mode | `inductor` (built-in dedup backend) |
| Optimized compile mode | `depthwise_separable_conv_opt` (custom `@register_backend`) |
| Batch size | 16 (input `[16, 32, 56, 56]`, FP32) |
| Iteration count | 2 warmup / 2 measure (ncu replay — **relative timing only**) |

> **Caveat on all durations:** values come from ncu application-mode replay, which inflates kernel time 2–5× vs. wall clock. Treat every `ns` figure as *relative*, never as real execution latency.
>
> **Architecture note:** `knowledge/hardware-limits.md` was unavailable during proposal; the strategist treated the device as GB202/sm_120 (~188 SMs). `warp_cycles_per_instruction` is null throughout, so the Blackwell latency-bound signal used was `eligible_cycles_pct < 20%`.

---

## 2. Operator Summary (baseline)

Sorted by share of attributed time. Note the first two rows are **NVTX-range / bucket aggregates** that overlap the per-instance conv rows below them — they do not represent additional distinct work (see §6).

| Operator | Time (%) | Duration (ns) | Kernels | Bottleneck Class |
|---|---|---|---|---|
| `layer::unique::prologue` *(aggregate)* | 50.9% | 437,850 | 30 | Mixed warm-up bundle (BN + relu6 + repack + convs) |
| `aten::cudnn_convolution` *(aggregate bucket)* | 18.7% | 160,511 | 9 | Memory-bound (mem 45.7%, occ 40.9%) |
| pointwise 1×1, 128→256 (×2) | 11.2% | 96,352 | 2 | **Latency-bound GEMM** (occ 8.3%, 228 reg/thr) |
| pointwise 1×1, 64→128 (×2) | 6.3% | 53,952 | 2 | **Latency-bound GEMM** (occ 8.3%) |
| `aten::convolution` (NCHW↔NHWC repack) | 2.5% | 21,536 | 2 | Layout overhead (sm 9.6%, TC 0%) |
| depthwise 3×3, 128ch (×2) | 3.8% | 33,087 | 2 | Memory-bound (mem 73.5%, occ 86%) |
| pointwise 1×1, 32→64 (×2) | 3.2% | 27,616 | 2 | Latency-bound GEMM (occ 8.3%) |
| depthwise 3×3, 64ch (×2) | 2.0% | 17,632 | 2 | Memory-bound (mem 61%, occ 82%) |
| depthwise 3×3, 32ch (×2) | 1.4% | 11,776 | 2 | Memory-bound (mem 47%, occ 72%) |

**Bottleneck headline:** the pointwise 1×1 convs (lowered to the implicit-GEMM `Kernel2`) run at a pathological **8.3% occupancy** with 224–228 registers/thread and 81.92 KB dynamic shared memory — the single largest addressable inefficiency. Depthwise 3×3 convs are already near the memory-bandwidth ceiling (73% mem throughput) and are not GEMM-addressable.

---

## 3. Reading the Metrics

Only the metrics that drove this workload's bottlenecks:

- **`achieved_occupancy` (8.3%)** — fraction of the SM's warp slots kept resident. 8.3% means the kernel occupies ~1 of 12 warp schedulers' capacity; the SM stalls waiting on long-latency instructions with no other warps to hide behind. This is *the* signal for the pointwise convs. Above ~50% is healthy for a GEMM.
- **`eligible_cycles_pct` (16–28%)** — share of active cycles with at least one warp ready to issue. Below ~20% on Blackwell indicates latency-bound execution (used here because `warp_cycles_per_instruction` is null). The 256-channel pointwise convs sit at ~16%.
- **`registers_per_thread` (224–228)** — extreme register pressure caps how many warps can co-reside, directly causing the 8.3% occupancy. A balanced cuBLAS GEMM tile uses far fewer.
- **`memory_throughput_pct` (47–74% on depthwise)** — the depthwise 3×3 convs are bandwidth-bound; there is no compute headroom to reclaim, so they were correctly left alone.
- **`tensor_core_active_pct`** — on the pointwise GEMMs this reads 39–45% (Tensor Cores partially engaged but starved by occupancy). Where it reads `0.0` on depthwise/BN kernels, that is **expected** (non-GEMM kernels) and is not a bottleneck signal.

---

## 4. Optimizations Applied

| ID | Type | Target Operators | Hardware Evidence | Confidence | Status |
|---|---|---|---|---|---|
| OPT-2 | Inference conv-BN folding | BN chain on all 6 convs (prologue + `aten::convolution`) | Standalone full-tensor BN normalize passes (`triton_poi_fused__native_batch_*`), one read+write per BN | high | **APPLIED** (folded into 7 convs) |
| OPT-1 | 1×1 conv → `mm`/`addmm` | 6 pointwise convs (op_id 26,53,18,45,37,10) | `Kernel2` at 8.3% occ, 228 reg/thr, 81.92 KB smem, eligible 16% | high | **APPLIED** (3 sites → `addmm`, M=50176) |
| OPT-3 | channels_last layout | depthwise convs + `aten::convolution` repack | `triton_poi_fused_convolution_0` repack kernels at 9.5% SM, 0% TC | medium | **NOT_APPLIED** (handled eager-side; no residual copy nodes to strip) |

OPT-3's graph pass found nothing to strip because the layout was forced eager-side in `get_model_and_input()`; the pass degraded gracefully (INFO no-op), exactly as designed.

---

## 5. Implementation Notes

# Implementation Notes — depthwise_separable_conv_opt

Custom `torch.compile()` backend for `DepthwiseSepConv` (three MobileNet-style
depthwise-separable blocks, channel progression 32 -> 64 -> 128 -> 256, spatial
56x56, FP32, RTX PRO 6000 Blackwell, torch 2.11.0+cu128).

Backend name (`@register_backend`): **`depthwise_separable_conv_opt`**

## Backend Architecture

All graph passes operate on the **decomposed Aten IR**, installed via Inductor's
`post_grad_custom_pre_pass` hook (the torch 2.11 FX injection point; the
`aot_autograd` `fw_compiler` path is broken on 2.11). The backend delegates
AOTAutograd + lowering to `compile_fx`. Apply order respects the DAG
`OPT-2 -> {OPT-1, OPT-3}`: BN fold first, then 1x1->GEMM, then layout.

| Pass | Method | Reason |
|---|---|---|
| OPT-2 conv-BN folding (`_pass_fold_conv_bn`) | `post_grad_custom_pre_pass` (Aten IR) | Detects the decomposed inference-BN affine chain (`sub -> mul(rstd) -> mul(gamma) -> add(beta)`) on a conv output and folds the affine into the conv weight/bias as structural graph nodes; Inductor constant-folds the weight math, deleting the standalone BN normalize/broadcast Triton kernels and letting relu6 epilogue-fuse. Must run first (prerequisite for OPT-1 and OPT-3). |
| OPT-1 1x1 conv -> GEMM (`_pass_pointwise_to_gemm`) | `post_grad_custom_pre_pass` (Aten IR) | Rewrites each 1x1/stride-1/pad-0/groups-1 `aten.convolution` into `permute->reshape->addmm/mm->reshape->permute`, routing the 8.3%-occupancy implicit-GEMM "Kernel2" onto a tiled cuBLAS GEMM that engages Blackwell Tensor Cores. Consumes the OPT-2-folded bias via `aten.addmm`. |
| OPT-3 channels_last (`_pass_strip_layout_copies` + eager-side) | `get_model_and_input()` (primary) + `post_grad_custom_pre_pass` (cleanup) | Eager-side `model/input.to(memory_format=channels_last)` makes NHWC the native layout so Inductor stops inserting per-conv `triton_poi_fused_convolution_0` pack kernels; the graph pass strips any residual redundant `clone/_to_copy(channels_last)` copies. |

Stubs: none — all three optimizations from `optimizations.json` are implemented
(OPT-2 and OPT-1 as transforming Aten-IR passes, OPT-3 as eager-side layout plus
a graph cleanup pass).

## Key Design Decisions

**Why `post_grad_custom_pre_pass` and not a Dynamo-level rewrite.** The graph a
`@register_backend` function receives is the functional Dynamo graph, not Aten IR.
`aten.convolution`, `aten.clamp_min`/`clamp_max` (relu6), and the BN decomposition
only appear after AOTAutograd. All target ops in `optimizations.json` are `aten::`
names, so every pass is installed at the Aten IR layer via Inductor's documented
post-grad pre-pass hook. The `aot_autograd(fw_compiler=...)` seam raises a boxing
assertion on torch 2.11, so it is avoided.

**Structural (FakeTensor-safe) BN fold.** Inductor traces under FakeTensorMode, so
placeholder inputs have no readable storage. The BN fold therefore never reads
weight values; it emits `aten.mul`/`aten.reshape`/`aten.sub`/`aten.add` nodes on the
existing weight/bias/BN-param placeholders. Inductor constant-folds this arithmetic
at lower time, so there is no runtime cost and no per-conv normalize kernel.

**Ordering constraint (OPT-2 before OPT-1).** OPT-1 emits `aten.addmm(bias, x, w)` to
absorb the conv bias into the GEMM epilogue. That bias only exists once OPT-2 has
folded the BN affine into the (originally bias-free) conv, so OPT-2 must precede
OPT-1. OPT-2 also simplifies the graph before OPT-3's layout pass sees it.

**`_weight_shape` walk-back.** After OPT-2, a conv's weight arg is a freshly inserted
`aten.mul(W, scale)` node with no populated `meta['val']` yet. OPT-1's pointwise guard
walks back through the mul/reshape chain to the original weight placeholder (which
carries `meta`) to read the `[C_out, C_in, 1, 1]` shape.

**`aten.permute([1,0])` instead of `aten.t`.** On torch 2.11 post-grad, `aten.t.default`
collides with an Inductor decomp/fallback assertion; `aten.permute` for the weight
transpose is equivalent and stable.

**`_repropagate_meta` after mutation.** New nodes from OPT-2/OPT-1 lack `meta['val']`,
which downstream Inductor post-grad passes (e.g. `should_prefer_unfused_addmm`) read.
A scoped `FakeTensorProp` re-run inside the active FakeTensorMode repopulates meta
before lowering.

**Dedup path retained but inactive.** Per Rule 9 the backend builds a
`UniqueSubgraphRegistry` partition equivalence map. The three blocks have different
channel widths (32/64/128/256) and are not structural duplicates, so `equiv_map` is
empty and the flat `compile_fx` path is taken (preserving cross-block Inductor fusion).
The per-rep dedup branch is kept for models with repeated identical blocks.

**Graceful degradation.** Each pass counts matches and logs INFO on application or
WARNING (no-op) when its pattern is absent; the whole chain is wrapped in a
try/except that logs a WARNING and returns the graph unchanged, so a pattern miss or
API drift never crashes the compile.

---

## 6. Before/After Results

Both profiles captured at **B=16**. Operators matched by `operator_name` and shape — the 6 pointwise 1×1 convs collapse to 6 `aten::mm` after OPT-1; the 6 depthwise 3×3 convs remain `aten::cudnn_convolution`.

> **Why not "3.85×".** The naive ratio of total attributed duration (860,312 ns → 223,485 ns = 3.85×) is **not a valid speedup.** The baseline was captured with the built-in dedup backend, which emits a `layer::unique::prologue` NVTX aggregate (437,850 ns, 50.9%) that *bundles the warm-up copies of the same convs already counted per-instance below it* — the strategist flagged this as `fused_kernel_double_count` + `warm_up_inflation`. The optimized capture used the custom backend, which attributes every kernel individually with no overlapping aggregate. Comparing the two totals compares two different accounting schemes, not two amounts of work. The only rigorous comparison is operator-matched on the cleanly-attributed compute kernels:

### Operator-matched comparison (cleanly attributed kernels only)

| Operator | Baseline (ns) | Optimized (ns) | Speedup |
|---|---|---|---|
| Pointwise 1×1, 128→256 (×2) → `aten::mm` | 96,352 | 100,735 | **0.96×** |
| Pointwise 1×1, 64→128 (×2) → `aten::mm` | 53,952 | 39,391 | **1.37×** |
| Pointwise 1×1, 32→64 (×2) → `aten::mm` | 27,616 | 20,863 | **1.32×** |
| **Pointwise subtotal (OPT-1 target)** | **177,920** | **160,989** | **1.11×** |
| Depthwise 3×3 (×6) → `aten::cudnn_convolution` | 62,495 | 62,496 | **1.00×** |
| **Matched total** | **240,415** | **223,485** | **1.08×** |

**Step B — Speedup attribution.** Per `validation_report.json`:

- **OPT-1 (APPLIED)** is credited with the pointwise gains. The kernels changed in the expected direction: the 64→128 and 32→64 convs moved off the 8.3%-occupancy `Kernel2` onto cuBLAS GEMMs and sped up 1.32–1.37×. **However**, the two largest (128→256) convs got marginally *slower* (0.96×) — their `aten::mm` still reports 8.3% occupancy and ~45% TC, i.e. the GEMM tile cuBLAS selected for M=50176, K=128, N=256 is itself latency-bound, so the substitution didn't help at that shape. Net OPT-1 effect: **1.11× on the pointwise group.**
- **OPT-2 (APPLIED)** folded BN into 7 convs. Its target kernels (standalone `triton_poi_fused__native_batch_*` BN passes) appear in the baseline bundled inside the prologue aggregate and in the optimized profile as 24 unattributed Triton kernels that returned 0 ncu rows. **Neither capture isolates their timing**, so OPT-2's wall-clock contribution cannot be measured from these profiles — only confirmed structurally (BN no longer a standalone normalize pass; relu6 epilogue-fused). It is *not* credited with measured speedup.
- **OPT-3 (NOT_APPLIED as a graph pass)** — layout was set eager-side; depthwise durations are unchanged (1.00×), consistent with the repack kernels being folded into native NHWC rather than measurably accelerating the convs.

### Step C — Residual opportunity

Re-ranking the optimized profile, the **two 128→256 `aten::mm` are now the top cost (45% of attributed time, 100,735 ns) and remain latency-bound at 8.3% occupancy** — OPT-1 relocated the bottleneck onto cuBLAS but did not resolve it at this shape. The depthwise convs (62,496 ns, 28%) are bandwidth-bound and near-optimal. No unapplied proposal in `optimizations.json` targets the residual `mm` occupancy; closing it would require GEMM tile/split-K tuning or batching the two same-shape `mm` calls — neither is an FX-level transform and both are *not yet implemented*.

---

## 7. What Drove Each Speedup

**1×1 pointwise convolution → cuBLAS GEMM (OPT-1, +1.11× on the pointwise group, mixed by shape):** rewriting each 1×1/stride-1/groups-1 conv as `permute→reshape→addmm→reshape→permute` routes the work off the implicit-GEMM `Kernel2` (8.3% occupancy, 228 reg/thread) onto a tiled cuBLAS GEMM. The hardware evidence is the disappearance of `Kernel2` and its replacement by `aten::mm` kernels; the 64→128 and 32→64 shapes gained 1.32–1.37× as occupancy pressure eased, while the 128→256 shape stayed at 8.3% occupancy and saw no gain — cuBLAS's chosen tile is latency-bound at that M/K/N.

**Conv-BN folding (OPT-2, applied, contribution not separately measurable):** folding the inference BN affine into conv weights eliminates the standalone full-tensor BN normalize passes and lets relu6 epilogue-fuse. Evidence is structural — the `triton_poi_fused__native_batch_*` kernels no longer appear as distinct attributed normalize ops — but because they fell into the prologue bundle (baseline) and into 0-row unattributed kernels (optimized), the profiler cannot assign them a wall-clock delta.

---

## 8. Remaining Opportunities

| ID | Type | Target | Reason Not Applied | Projected Gain |
|---|---|---|---|---|
| OPT-3 | channels_last layout (graph pass) | depthwise convs + repack | Layout forced eager-side in `get_model_and_input()`; no residual `clone/_to_copy` nodes for the pass to strip (graceful no-op) | ~2.4% of total (est.) — likely already realized eager-side |

All three *proposed* optimizations were exercised (OPT-1 and OPT-2 as active Aten-IR passes, OPT-3 eager-side). The meaningful residual is **not** in the proposal list: the two 128→256 `aten::mm` kernels remain latency-bound at 8.3% occupancy after substitution. Resolving them requires GEMM-level tuning (tile selection / split-K) or fusing the two identical-shape matmuls — work outside the FX-pass scope and **not yet implemented**. Realistic additional upside if that residual were closed: the 128→256 group is ~45% of attributed optimized time, so even a 1.5× on those two kernels would yield roughly a further 1.2× on the matched total.

---

## Reproduction

```bash
# Reuse the existing baseline profile.json (skip Stage 0)
# Stage 1 — propose
/propose examples/depthwise_separable_conv/profile.json

# Stage 2 — generate backend  (registered name: depthwise_separable_conv_opt)
/backend examples/depthwise_separable_conv/depthwise_separable_conv.py

# Stage 3 — validate
/validate examples/depthwise_separable_conv/depthwise_separable_conv_optimized.py

# Stage 4 — re-capture under the custom backend
/capture examples/depthwise_separable_conv/depthwise_separable_conv_optimized.py \
    --profile-name=optimized --compile-backend=depthwise_separable_conv_opt

# Stage 5 — report
/report
```

> All `ns` figures are ncu application-mode replay values (2–5× inflated vs. wall clock) and are valid only for *relative* comparison within this report.
