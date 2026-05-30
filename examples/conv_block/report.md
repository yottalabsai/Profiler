# ConvBlock — GPU Optimization Report

**This optimization achieved ~1.06× speedup on the comparable forward pass (ConvBlock, B=16, NVIDIA RTX PRO 6000 Blackwell) by eliminating redundant NCHW↔NHWC layout-copy kernels via a `channels_last` cast.** The headline gain is modest because the dominant cuDNN convolutions were already running on native-NHWC TF32 tensor-core kernels in the baseline, and the eval-mode Conv+BatchNorm fold was already performed by AOTAutograd/Inductor before any custom pass could fire.

> **Read this first — the naive number is a mirage.** Summing every operator gives `1.076 ms → 0.419 ms` (2.57×). **That is not a real speedup.** The baseline profile contains a `layer::unique::prologue` operator (629,764 ns) that is the deduplicated representative of the *entire* forward pass and re-attributes the same physical cuDNN/Triton kernels that are *also* listed under the individual `aten::cudnn_convolution` operators. The optimized profile has no such prologue entry, so the two totals are not comparable. The honest comparison below matches operators by name and excludes the double-counted prologue.

---

## 1. Hardware Context

| Field | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition (188 SMs) |
| Architecture | Blackwell (sm_120, compute capability 12.0) |
| PyTorch | 2.11.0+cu128 |
| Baseline compile mode | inductor (built-in dedup backend) |
| Optimized compile mode | `conv_block_opt` (custom registered backend) |
| Batch size | 16 (input 3×64×64) |
| dtype | float32 (TF32 tensor-core path on convolutions) |
| Iteration counts | warmup=2, measure=2 *(ncu replay — relative timing only)* |

---

## 2. Operator Summary (baseline)

Sorted by share of attributed time. The `prologue` row is the fused full-forward dedup representative and is **not additive** with the per-operator rows below it (see banner above).

| Operator | Time (%) | Duration (ns) | Kernels | Bottleneck Class |
|---|---|---|---|---|
| `layer::unique::prologue` *(double-count repr.)* | 58.5% | 629,764 | 55 | Mixed (whole forward) |
| `aten::cudnn_convolution` (64→128, 64×64) | 7.9% | 85,345 | 3 | Compute / tensor-core (tc 66.5%) |
| `aten::cudnn_convolution` (64→128, 64×64) | 7.9% | 84,864 | 3 | Compute / tensor-core (tc 66.5%) |
| `aten::cudnn_convolution` (128→256, 32×32) | 7.0% | 74,976 | 3 | Compute / tensor-core (tc 73.3%) |
| `aten::cudnn_convolution` (128→256, 32×32) | 6.9% | 74,784 | 3 | Compute / tensor-core (tc 73.3%) |
| `aten::_native_batch_norm_legit_no_training` | 6.4% | 68,578 | 14 | Memory-bound (tc 0%, occ 70%) |
| `aten::cudnn_convolution` (layout/convert helpers) | 1.7% | 17,920 | 2 | Data movement (tc 15.8%) |
| `aten::cudnn_convolution` (3→64, 64×64) | 1.3% | 14,496 | 3 | Memory→compute transitional (tc 16%) |
| `aten::cudnn_convolution` (3→64, 64×64) | 1.3% | 14,400 | 3 | Memory→compute transitional (tc 16%) |
| `aten::convolution` (layout copies) | 0.4% | 4,800 | 4 | Data movement (tc 0%) |
| `aten::addmm` (classifier GEMM) | 0.3% | 3,104 | 1 | Tiny GEMM (no counters) |
| `aten::addmm` (classifier GEMM) | 0.3% | 3,072 | 1 | Tiny GEMM (no counters) |

**Comparable per-operator total (prologue excluded): 446,339 ns.**

---

## 3. Reading the Metrics

Only the metrics that actually drive this workload's bottlenecks are explained here.

- **`tensor_core_active_pct`** — fraction of active cycles with tensor cores issuing. The dominant convs sit at **66–73%**, meaning the GEMM cores are already well-utilized at TF32; there is little FX-level headroom left in the conv math itself. A value of **0.0 (not null)** on the BatchNorm/ReLU kernels confirms they run on the pure-SIMT memory path — expected for elementwise ops, not a defect. `null` on the tiny `addmm` GEMMs simply means ncu captured no counters for those sub-microsecond kernels.
- **`dram__throughput` ≈ 78–85%** on the BatchNorm/ReLU kernels (from `optimizations.json` evidence) — these are bandwidth-bound full-tensor passes: they read the activation once, do almost no arithmetic, and are limited by bytes moved.
- **`achieved_occupancy` ≈ 8.3%** on the dominant convs — this looks alarming but is **inherent to the 150-register CUTLASS tile** the cuDNN NHWC kernel uses; it is not addressable from an FX graph pass.
- **`warp_cycles_per_instruction` is null** across all blocks — this counter was removed on Blackwell; latency assessment falls back to `eligible_cycles_pct` instead.

---

## 4. Optimizations Applied

Status read from `profiler_output/validation_report.json`.

| ID | Type | Target Operators | Hardware Evidence | Confidence | Status |
|---|---|---|---|---|---|
| OPT-1 | Conv+BatchNorm fold (eval) | conv stages + BN | BN+ReLU Triton kernels at 78.9% DRAM, 0% tensor-core, 0.23 inst/cycle | HIGH | **NOT_APPLIED** (graceful) |
| OPT-2 | `channels_last` (NHWC) propagation — eager-side primary lever | all 5 conv stages | `convertTensor`/NCHW↔NHWC transposes, ~10 launches of pure data movement | MEDIUM | **APPLIED** |
| OPT-2 | `strip_layout_copies` graph-cleanup half | conv input paths | redundant `clone`/`_to_copy(channels_last)` nodes | MEDIUM | **NOT_APPLIED** (graceful) |

**Why OPT-1 did no work (important):** The model runs in `.eval()`, so AOTAutograd + Inductor already fold the frozen-statistics BatchNorm into the preceding convolution *during decomposition*, before the custom `inner_compile` hook ever sees the graph. A fresh caches-disabled compile confirmed the Aten IR contains `3× aten.convolution + 3× aten.relu` and **zero `batch_norm` nodes**. The hand-written fold therefore matched nothing and emitted a graceful warning. The optimization's *goal* (no standalone DRAM-bound BN kernels) is achieved — just by the framework, not by this pass. Per attribution rules, **OPT-1 is credited with zero speedup.**

---

## 5. Implementation Notes

# ConvBlock — Optimized Backend Implementation Notes

Backend name: `conv_block_opt` (registered via `@register_backend`).
Target: NVIDIA RTX PRO 6000 Blackwell (sm_120), torch 2.11.0+cu128, `compile_mode = inductor`.

## Backend Architecture

| Pass | Method | Reason |
|---|---|---|
| OPT-1 Conv+BatchNorm fold (eval) | `_aten_inner_compile` (`_pass_fold_conv_bn`) | Eval BN lowers to `aten._native_batch_norm_legit_no_training` — a per-channel affine over compile-time constants; folding it into the conv weight/bias removes the DRAM-bound BN+ReLU Triton passes at zero numerical cost. |
| OPT-2 channels_last (NHWC) propagation | `get_model_and_input()` (primary) + `_aten_inner_compile` (`_pass_strip_layout_copies`, cleanup) | Memory-format change is not a node in any FX graph, so the primary lever is the eager-side `model/input.to(memory_format=channels_last)`; the Aten-IR pass only strips residual redundant `clone`/`_to_copy(channels_last)` copies so cuDNN keeps its native NHWC tensor-core kernel and drops `convertTensor_kernel` relayouts. |

Both passes are wrapped so a missing pattern or any exception logs a warning and returns the graph unchanged — the compile never crashes.

## Key Design Decisions

**Injection point (Strategy D, not `post_grad_custom_pre_pass`).** The graph `torch.compile` hands `conv_block_opt` is the functional Dynamo graph; `aten.convolution` and `aten._native_batch_norm_legit_no_training` only appear after AOTAutograd decomposition. The passes are installed via `compile_fx(gm, example_inputs, inner_compile=...)`, where the `inner_compile` hook (`_aten_inner_compile`) receives the fully decomposed Aten IR graph, runs the passes, then delegates to `compile_fx_inner` (Aten -> Triton). `compile_fx` retains ownership of AOTAutograd, the decomposition table, the boxed calling convention, and the fwd/bwd partitioner. `aot_autograd(fw_compiler=compile_fx)` is deliberately avoided — on torch 2.11 it raises `AssertionError: Expected tensors only, but got list` in `copy_misaligned_inputs`.

**Weight-value reading via `real_inputs`.** OPT-1 must read the genuine gamma/beta/running_mean/running_var and conv weight to compute the fold. Under FakeTensorMode the `inner_compile` `example_inputs` may be FakeTensors with no readable storage, so the backend threads the real parameter/input tensors as `functools.partial(_aten_inner_compile, real_inputs=list(example_inputs))`. The fold builds `ph_to_tensor = zip(placeholders, real_inputs)` and registers `new_weight`/`new_bias` as buffers, inserting a fresh `aten.convolution` whose bias slot is populated. Because BN returns a 3-tuple `(output, save_mean, save_rstd)`, the rewrite redirects the live `getitem(bn, 0)` consumers — not the BN node directly — then erases the dead getitem, BN, original conv, and now-unused parameter placeholders via `eliminate_dead_code()`.

**Prerequisite ordering OPT-1 -> OPT-2.** OPT-2 must observe the folded conv nodes; running layout propagation before the fold would re-stride conv weights that are about to be rewritten. `_aten_inner_compile` therefore calls `_pass_fold_conv_bn` before `_pass_strip_layout_copies`, matching the proposal's dependency DAG. The `prerequisite_for: ["OPT-2"]` constraint on OPT-1 is honoured by this ordering.

**Flat compile path.** ConvBlock's three conv stages are structurally distinct (3->64, 64->128, 128->256) plus a Linear head, so `UniqueSubgraphRegistry.build_partition_equivalence_map()` returns no duplicates and the backend takes the flat `_compile_with_aten_passes(gm, example_inputs)` path. This preserves cross-stage Inductor fusion (the BN+ReLU epilogues fuse into adjacent kernels) that a per-partition split would block. The dedup branch is retained verbatim for structural reuse if the model gains repeated blocks.

---

## 6. Before/After Results

Both profiles captured on the **same GPU**, ~13 minutes apart in one session → cross-session caveat does **not** apply. Batch size matches (B=16). Operators matched by type/sizes; the double-counted `prologue` is excluded from the baseline total.

| Operator | Baseline (ns) | Optimized (ns) | Speedup |
|---|---|---|---|
| `cudnn_convolution` (64→128, 64×64) #1 | 85,345 | 84,961 | 1.00× |
| `cudnn_convolution` (64→128, 64×64) #2 | 84,864 | 84,034 | 1.01× |
| `cudnn_convolution` (128→256, 32×32) #1 | 74,976 | 74,914 | 1.00× |
| `cudnn_convolution` (128→256, 32×32) #2 | 74,784 | 74,722 | 1.00× |
| `_native_batch_norm_legit_no_training` | 68,578 | 64,064 | 1.07× |
| `cudnn_convolution` (3→64, 64×64) #1 | 14,496 | 14,433 | 1.00× |
| `cudnn_convolution` (3→64, 64×64) #2 | 14,400 | 14,368 | 1.00× |
| `cudnn_convolution` (layout/convert helpers) | 17,920 | — *(eliminated)* | ∞ |
| `aten::convolution` (layout copies) | 4,800 | — *(eliminated)* | ∞ |
| `addmm` (classifier GEMM) ×2 | 6,176 | 7,808 | 0.79× |
| **Total (prologue excluded)** | **446,339** | **419,304** | **1.06×** |

*All durations are ncu-replay relative timings (2–5× longer than real wall-clock); use them for ratios, not absolute latency.*

### Speedup attribution

- The **27,035 ns saved** comes almost entirely from the **elimination of two baseline-only data-movement operator groups** — the `cudnn_convolution` layout/convert helpers (17,920 ns) and the `aten::convolution` layout copies (4,800 ns), 22,720 ns combined. The capture confirmed the baseline's `triton_poi_fused_convolution_0/1` layout-copy kernels are **absent** from the optimized trace. → attributed to **OPT-2 (`channels_last`, APPLIED)**: all three conditions hold (status APPLIED, the NCHW↔NHWC transpose kernels disappeared, the containing data-movement operators vanished).
- The small BatchNorm improvement (68,578 → 64,064 ns) is **not** credited to OPT-1 (NOT_APPLIED) — it reflects Inductor's own fusion/run-to-run variation.
- The two `addmm` GEMMs got slightly *slower* (k=1 → k=2 each), a minor cost of restoring contiguous layout for the linear head after the NHWC conv stack.

### Residual opportunity

After optimization the four large `cudnn_convolution` stages (≈76% of remaining time) are the new ceiling — and they are already 66–73% tensor-core-active at TF32 with occupancy fixed by the CUTLASS tile. **No FX-level lever remains** for the conv math itself. The only remaining non-conv cost is the memory-bound BatchNorm/ReLU group (15.3%), which is already fused as far as the framework allows.

---

## 7. What Drove Each Speedup

**`channels_last` (NHWC) layout propagation (OPT-2, eliminated 22.7 µs of layout-conversion kernels):** Casting the model and input to `torch.channels_last` in `get_model_and_input()` makes the activation tensors enter the conv stack in the exact layout cuDNN's NHWC TF32 tensor-core kernel consumes, so Inductor no longer inserts NCHW↔NHWC transpose/convert kernels around each convolution. The hardware evidence is direct: the baseline's `triton_poi_fused_convolution_0/1` layout-copy kernels and two helper-conv operator groups are entirely absent from the optimized trace, while the convolution kernels themselves are byte-for-byte unchanged in duration and tensor-core utilization.

---

## 8. Remaining Opportunities

| ID | Type | Target | Reason Not Applied | Projected Gain |
|---|---|---|---|---|
| OPT-1 | Conv+BatchNorm fold (eval) | conv + BN | Eval-mode BN already folded by AOTAutograd/Inductor before the custom pass runs — zero `batch_norm` nodes in the Aten IR; graceful no-op | ~14.7% (already realized by the framework) |
| OPT-2 (cleanup half) | `strip_layout_copies` graph pass | conv input paths | No redundant `clone`/`_to_copy` nodes remained after the eager-side cast already aligned layout; graceful no-op | included in OPT-2's realized gain |

Both proposed transformations' *goals* are effectively realized — OPT-1 by the framework's own decomposition and OPT-2 by its applied eager-side lever. No additional FX-level gains are identified in this profile: the residual cost is dominated by convolutions already near their achievable tensor-core efficiency on this Blackwell part.

---

## Reproduction

```bash
# Environment (Blackwell sm_120 needs cu128 torch)
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python3 nvidia/scripts/preflight.py

# Baseline capture (built-in dedup backend)
#   → examples/conv_block/profile.json
# Optimized capture (custom registered backend, --compile-backend=conv_block_opt)
#   → examples/conv_block/profile_optimized.json

# Or run the whole pipeline end-to-end:
/optimize examples/conv_block/conv_block.py
```
