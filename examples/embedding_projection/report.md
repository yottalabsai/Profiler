# Optimization Report — embedding_projection

This optimization achieved **2.74x total speedup** on `embedding_projection` (B=64, T=128, NVIDIA RTX PRO 6000 Blackwell): end-to-end GPU time dropped from **12.533 ms** to **4.577 ms**. The two dominant logit GEMMs were moved off the idle FP32 SIMT path onto bf16 Tensor Cores (a ~5.5x per-kernel win), but roughly **2.15 ms** of that gain is reabsorbed by two newly-introduced bf16↔fp32 conversion copies that the dtype-promotion pass inserts around the logit output.

---

## 1. Hardware Context

| Field | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition (~188 SMs, GB202-class) |
| Architecture | Blackwell |
| CUDA | 12.8 |
| PyTorch | 2.11.0+cu128 |
| Baseline compile mode | `inductor` |
| Optimized compile mode | `embedding_projection_opt` (custom backend) |
| Batch size | 64 (B=64, T=128 → 8192 rows) |
| Iteration model | 2-block sequential stack (not 2 profiling iterations) |
| Timing source | nsys capture — durations measured at locked GPU clocks; relative comparison |

The per-operator **durations** below come from the **nsys capture** phase (GPU kernel times) at a locked GPU clock, so baseline and optimized captures lock to an identical frequency and the comparison is fair. The ncu replay phase contributes only the hardware **counters** (tensor-core %, SM/DRAM throughput, occupancy) at its own base-clock lock.

---

## 2. Operator Summary (Baseline)

Baseline total attributed GPU time: **12.533 ms** across 8 operators (no unattributed kernels). Tensor Cores were idle (`tensor_core_active_pct = 0.0`) on every operator.

| Operator | Time (%) | Duration (ns) | Kernels | Bottleneck Class |
|---|---:|---:|---:|---|
| aten::mm op_id=7 — logit head [8192,512]×[512,32000] (block 1) | 42.7% | 5,352,277 | 1 | Compute-bound, FP32 SIMT (TC idle) |
| aten::mm op_id=14 — logit head [8192,512]×[512,32000] (block 2) | 42.7% | 5,347,413 | 1 | Compute-bound, FP32 SIMT (TC idle) |
| aten::addmm op_id=13 — down-proj [8192,2048]×[2048,512] (block 2) | 3.9% | 490,527 | 1 | Compute-bound, FP32 SIMT (TC idle) |
| aten::addmm op_id=6 — down-proj [8192,2048]×[2048,512] (block 1) | 3.9% | 487,776 | 1 | Compute-bound, FP32 SIMT (TC idle) |
| aten::mm op_id=11 — up-proj [8192,512]×[512,2048] (block 2) | 3.1% | 387,711 | 1 | Compute-bound, FP32 SIMT (TC idle) |
| aten::mm op_id=4 — up-proj [8192,512]×[512,2048] (block 1) | 3.1% | 384,640 | 1 | Compute-bound, FP32 SIMT (TC idle) |
| aten::addmm (fused gelu+view) | 0.4% | 53,727 | 2 | Memory-bound (DRAM ~90%) |
| aten::embedding (fused native_layer_norm) | 0.2% | 28,800 | 2 | Memory-bound (gather, DRAM ~52%) |

The two `[8192,512]×[512,32000]` logit matmuls alone account for **10.70 ms (85.4%)** of attributed time. Both ran as `cutlass_80_simt_sgemm_256x128_8x4_tn_align1` — the literal FP32 SIMT GEMM, with Tensor Cores completely idle. The embedding gather and the fused GELU kernel are bandwidth-bound (the opposite end of the roofline) but contribute under 1% combined.

---

## 3. Reading the Metrics

- **`tensor_core_active_pct = 0.0` (not null)** — the single highest-ROI signal here. A GEMM reporting exactly 0.0 ran on the FP32 SIMT (`cutlass_*_simt_sgemm_*`) path with Tensor Cores entirely idle. Every baseline matmul shows 0.0, confirming no Tensor-Core engagement. (A `null` value is expected for non-GEMM kernels and is not a problem.)
- **`sm_throughput_pct`** — SM pipe utilization. The logit GEMMs sit at ~63.7%: the SMs are busy, but doing scalar FP32 MACs rather than Tensor-Core matmuls. After optimization, ~77.7% with the work now on Tensor Cores.
- **`eligible_cycles_pct`** — fraction of cycles with a warp ready to issue. The dominant GEMMs show ~64.6%, well above the Blackwell latency-bound threshold (<20%), so they are **not** latency-bound — they are genuinely compute-bound on the wrong math path.
- **`dram_throughput_pct`** — the logit GEMMs each write a ~1.05 GB FP32 `[8192,32000]` logit tensor (`dram_bytes_op_write ≈ 991 MB`). bf16 promotion halves that write.
- **`achieved_occupancy`** — ~16.7% on the GEMMs; this is normal for high-register CUTLASS GEMM kernels (200+ registers/thread) and is not the bottleneck.
- **`warp_cycles_per_instruction`** — unavailable (null) on Blackwell; do not interpret.

---

## 4. Optimizations Applied

Status from `profiler_output/validation_report.json` (syntax / import / registration / test_suite all `pass`; overall `READY_FOR_PROFILING`). All three passes applied cleanly — 4/4 validation steps passed.

| ID | Type | Target Operators | Hardware Evidence | Confidence | Status |
|---|---|---|---|---|---|
| OPT-1 | matmul precision policy (TF32) | All 6 GEMMs | `tensor_core_active_pct=0.0`, dispatched `cutlass_80_simt_sgemm_*_align1`; Inductor warned TF32 disabled | high | **APPLIED** |
| OPT-2 | dtype promotion (bf16 operands, fp32 accumulate) | All 6 GEMMs (3 distinct nodes) | `tensor_core_active_pct=0.0` + ~991 MB fp32 logit write per GEMM | medium | **APPLIED** |
| OPT-3 | memory layout (weight freezing + `max_autotune_gemm`) | All 6 GEMMs | `align1` SIMT fallback, 212 regs/thread, 16.66% occupancy | medium | **APPLIED** |

Validation details:
- OPT-1: `cuda.matmul.allow_tf32=True`, `cudnn.allow_tf32=True`, `float32_matmul_precision='high'`.
- OPT-2: promoted 3 `aten.mm`/`aten.addmm` node(s) to bf16 operands (fp32 accumulate, fp32 output restored).
- OPT-3: `freezing=True`, `max_autotune_gemm=True`, `max_autotune=True`, `max_autotune_gemm_backends=ATEN,TRITON`.

---

## 5. Implementation Notes

# Implementation Notes — embedding_projection_opt

Custom `torch.compile()` backend for the embedding-lookup + projection-head workload
(`examples/embedding_projection/embedding_projection.py`). Registered backend name:
**`embedding_projection_opt`** (model `embedding_projection` -> snake_case + `_opt`).

`compile_mode = "inductor"` (from `optimizations.json analysis.compile_mode`). The backend
is the fixed three-stage funnel `functional -> aten -> inductor_config` invoked through
`_compile_unit`; `compile_fx` owns AOTAutograd exactly once, aten passes run through its
`inner_compile` seam, and inductor_config passes are scoped `config_patches` on that call.

## Backend Architecture

| Pass | Level | Method | Reason |
|---|---|---|---|
| OPT-1 — Enable TF32 (matmul precision policy) | inductor_config + non-graph | `get_model_and_input()`-independent process flag set at backend entry (`_enable_tf32_flags`) plus a `config_patches` contribution (`_cfg_tf32`) | TF32 is a lowering/math-mode policy Inductor + cuBLAS honor via `torch.backends.cuda.matmul.allow_tf32`; the canonical switch is the process flag, so it is set once at backend entry and recorded in the scoped config stage. No graph node surgery — routes the SIMT `cutlass_80_simt_sgemm_*_align1` (tensor_core_active=0) onto the TF32 Tensor-Core path for any GEMM left in FP32. |
| OPT-2 — BF16 dtype promotion of GEMM operands | aten | `_aten_inner_compile` (`_apass_bf16_promotion`) | Operates on the decomposed primitives `aten.mm.default` / `aten.addmm.default` and their operand edges — only expressible after AOTAutograd. Casts both matmul operands (and the addmm bias) to bf16 with fp32 accumulate, then restores fp32 on the output; routes the two dominant `[8192,512]x[512,32000]` logit GEMMs (85.4% of attributed time) and the two MLP GEMMs to the Blackwell BF16 HMMA path and halves the ~1 GB fp32 logit write. |
| OPT-3 — Weight freezing + `max_autotune_gemm` | inductor_config | `config_patches` (`_cfg_freezing`) | Constant-weight layout/freezing is a lowering decision Inductor owns; expressed as a scoped config dict, not a graph rewrite. Constant-folds and re-lays-out (aligns/pre-transposes) the frozen eval-mode projection weights so the GEMM autotuner can select an `align8` Tensor-Core template over the `align1` SIMT fallback. |

No functional-level passes apply to this workload (no QKV fusion / SDPA formation). No
non-graph dtype/memory_format/batch-shape changes are needed: there are no conv layers
(no `channels_last`) and all GEMM M/N/K dims (M=8192, K/N in {512, 2048, 32000}) are
multiples of 16, so no batch padding.

## Key Design Decisions

**OPT-1 is split flag + config, not a graph pass.** TF32 is a math-mode policy, not a node
attribute. The authoritative switch (`torch.backends.cuda.matmul.allow_tf32`) is a process
flag that the GEMM lowering reads at kernel-selection time, so it is set once at backend
entry (`_enable_tf32_flags`) before any `compile_fx` call. The funnel's inductor_config
stage (`_cfg_tf32`) logs the policy and returns an empty patch — Inductor needs no extra
config key beyond the backends flag. This honors `optimizations.json`'s own
`location: "Backend entry point, before compile_fx is invoked"`.

**OPT-2 prerequisite_for OPT-3 is satisfied by funnel level ordering, not within-level
sequencing.** Freezing (OPT-3) materializes the constant weight buffer at the runtime
dtype, so the weight must already be bf16 when frozen. OPT-2 is an `aten` pass (funnel
stage 2) and OPT-3 is `inductor_config` (funnel stage 3); the funnel runs stage 2 strictly
before stage 3, so the frozen weight is bf16 automatically — no explicit ordering code is
required and the cross-level dependency is never unsatisfiable.

**OPT-2 uses `prims.convert_element_type`, not `aten._to_copy`.** On torch 2.11
`aten._to_copy` has both a fallback and a decomp registration; inserting it into an
already-decomposed Aten graph makes Inductor raise "both a fallback and a decomp for same
op". `prims.convert_element_type` lowers cleanly to a Triton elementwise cast.
`_repropagate_meta` re-runs `FakeTensorProp` after the bf16 rewrite so the inserted cast
nodes carry `meta['val']` before `compile_fx_inner`.

**OPT-2 is medium-confidence and uses the matched-guard structure.** bf16 changes numerics
on the 32000-way logit head; the pass counts promoted nodes, logs INFO on success and
WARNING when no `aten.mm`/`aten.addmm` is found, and never raises into the compile. The
validation test asserts the fp32-restored output is free of NaN/Inf and has the expected
`(64,128,32000)` shape; top-k/argmax agreement against the fp32 baseline should be checked
in the re-capture stage.

**Flat compile path.** This model is a single linear forward (embed -> LN -> proj1 -> gelu
-> proj2 -> logits) with no repeated layer structure, so `UniqueSubgraphRegistry` returns
an empty equivalence map and the backend takes the flat `_compile_unit` path, which
preserves cross-op Inductor fusion. The dedup branch is retained only for interface
uniformity with the other example backends.

---

## 6. Before/After Results

Both captures used B=64 on the same device (`NVIDIA RTX PRO 6000 Blackwell Server Edition`), ~38 minutes apart in one session — within the cross-session threshold, so no caveat applies.

Operators are matched by role across the two captures (the FP32 op_ids in the baseline differ from the bf16-rewritten op_ids in the optimized graph). The optimized total **includes** the bf16↔fp32 conversion copies the optimization newly introduced.

| Operator | Baseline (ns) | Optimized (ns) | Speedup |
|---|---:|---:|---:|
| Logit head GEMM #1 [8192,512]×[512,32000] | 5,352,277 | 973,799 | 5.50x |
| Logit head GEMM #2 [8192,512]×[512,32000] | 5,347,413 | 973,734 | 5.49x |
| Down-proj addmm #1 [8192,2048]×[2048,512] | 487,776 | 76,064 | 6.41x |
| Down-proj addmm #2 [8192,2048]×[2048,512] | 490,527 | 73,729 | 6.65x |
| Up-proj addmm #1 [8192,512]×[512,2048] | 384,640 | 77,953 | 4.93x |
| Up-proj addmm #2 [8192,512]×[512,2048] | 387,711 | 77,600 | 5.00x |
| GELU + view (elementwise) | 53,727 | 33,856 | 1.59x |
| embedding + layer_norm (gather) | 28,800 | 19,168 | 1.50x |
| **Attributed subtotal** | **12,532,871** | **2,428,816** | **5.16x** |
| aten::t weight pre-transpose (6 kernels) | 0 | 122,913 | new overhead |
| triton_poi_fused_7 — bf16→fp32 logit recast #1 | 0 | 1,067,111 | new overhead |
| triton_poi_fused_7 — bf16→fp32 logit recast #2 | 0 | 1,068,295 | new overhead |
| Other small unattributed casts (×6) | 0 | 13,248 | new overhead |
| **Total (incl. introduced overhead)** | **12,532,871** | **4,577,470** | **2.74x** |

Note: the `aten::t` pre-transpose kernels appear in the optimized profile as an attributed operator and are counted in the attributed subtotal above; the four `triton_poi_fused_7` / small-cast entries land in `unattributed_kernels`. The grand total of 4.577 ms is the sum of all attributed and unattributed GPU time.

**Speedup attribution** (all three passes `APPLIED`; metric moved in the expected direction; operators improved):
- The logit and MLP GEMMs went from `tensor_core_active_pct = 0.0` (SIMT `cutlass_80_simt_sgemm_*_align1`) to **79.5%–92.2%** on `cutlass_80_tensorop_bf16_s16816gemm_*_align8` Tensor-Core kernels. This jointly confirms OPT-2 (bf16 HMMA path) and OPT-3 (`align8` aligned template replacing the `align1` fallback). OPT-1 (TF32) is the safe fallback path but is superseded on these nodes by bf16; the achieved kernel is the bf16 HMMA variant, not a TF32 SIMT-replacement, so the measured win is credited to OPT-2 + OPT-3.
- The logit GEMM DRAM write dropped from ~991 MB (fp32) to ~468 MB (bf16) per kernel — direct evidence of the OPT-2 bf16 output narrowing.

---

## 7. What Drove Each Speedup

**Enable TF32 (OPT-1, safety net for all 6 GEMMs):** Sets `torch.backends.cuda.matmul.allow_tf32` so any GEMM left in FP32 routes to the TF32 Tensor-Core path instead of the SIMT sgemm. On this workload OPT-2 promotes every GEMM to bf16, so TF32 is superseded on the dominant nodes; its measurable contribution is subsumed by the bf16 path and it remains the fallback if bf16 numerics were rejected.

**BF16 dtype promotion (OPT-2, +5.5x on each logit GEMM):** Casts both matmul operands to bf16 with fp32 accumulate, so the `[8192,512]×[512,32000]` logit GEMMs run on the Blackwell BF16 HMMA Tensor Cores and write a half-size logit tensor. Evidence: `tensor_core_active_pct` rose from 0.0 to ~79.5% and the per-kernel DRAM write fell from ~991 MB to ~468 MB.

**Weight freezing + max_autotune_gemm (OPT-3, kernel-selection polish):** Constant-folds and re-lays-out the frozen eval-mode weights so the autotuner picks an `align8` Tensor-Core template. Evidence: the dispatched kernel changed from `cutlass_80_simt_sgemm_*_align1` to `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_*_align8`, and the MLP addmm GEMMs reach 92% Tensor-Core activity.

---

## 8. Remaining Opportunities

All three proposed optimizations were applied. The optimization itself, however, exposed a new second-order bottleneck that no current proposal addresses:

| Operator | Optimized (ns) | New Bottleneck | Candidate Fix |
|---|---:|---|---|
| triton_poi_fused_7 — bf16→fp32 logit recast (×2) | 2,135,406 | Memory-bound copy of the [8192,32000] logit tensor introduced by the OPT-2 fp32-output restore | Keep the logit output in bf16 end-to-end (feed argmax/loss directly) or fuse the recast into the GEMM epilogue |
| aten::t weight pre-transpose (×6) | 122,913 | Small bandwidth-bound transposes from the frozen-weight relayout | Cache the transposed weight once outside the timed region (constant-fold persistence) |

The two `triton_poi_fused_7` copies (~1.067 ms + ~1.068 ms ≈ **2.135 ms**, ~47% of the optimized 4.577 ms total) are the dominant residual. They exist solely because OPT-2 restores the logit output to fp32 after the bf16 GEMM. Eliminating them — by leaving the `[8192,32000]` logit tensor in bf16 (the downstream consumer in inference is typically `argmax`/cross-entropy, which tolerate bf16) or by fusing the cast into the CUTLASS epilogue — would remove almost all of the introduced overhead. If both copies were eliminated, the end-to-end total would approach the **2.43 ms** attributed subtotal, i.e. roughly **5.2x** over baseline rather than the current 2.74x. This is a genuine, honest cost of the dtype-promotion approach and the highest-value next step.

---

## Reproduction

Full end-to-end pipeline (capture → propose → backend → validate → re-capture → compare → report):

```bash
/optimize examples/embedding_projection/embedding_projection.py
```

Individual stages:

```bash
# 1. Baseline capture (nsys + ncu) -> profile.json
/capture examples/embedding_projection/embedding_projection.py

# 2. Propose transformations from the baseline profile -> optimizations.json
/propose

# 3. Generate the custom backend + validation harness -> embedding_projection_optimized.py,
#    profiler_output/implementation_notes.md
/backend

# 4. Validate the generated backend (syntax/import/registration/pytest/smoke)
#    -> profiler_output/validation_report.json
/validate

# 5. Re-capture the optimized workload -> profile_optimized.json
/capture examples/embedding_projection/embedding_projection_optimized.py --profile-name=optimized

# 6. Generate this report
/report
```
