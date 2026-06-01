# Optimization Report — `mlp_activations`

**This optimization achieved a 3.55× total speedup on MLPActivations (B=256, NVIDIA RTX PRO 6000 Blackwell Server Edition)**, by promoting the four FP32 matmuls onto bf16 tensor cores — a 5.1× win on the GEMMs themselves — of which roughly a quarter is reabsorbed by the newly-introduced bf16 dtype-cast (`aten::copy_`) kernels that the promotion requires.

---

## 1. Hardware Context

| Field | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Architecture | Blackwell |
| PyTorch | 2.11.0+cu128 |
| Baseline compile mode | `dedup-inductor` (built-in dedup + Inductor backend) |
| Optimized compile mode | `mlp_activations_opt` (custom registered backend) |
| Batch size | 256 |
| Iterations | 2 measured iterations (nsys capture — durations measured at locked GPU clocks; relative comparison) |

**Timing source.** Per-operator **durations** come from the **nsys capture** phase (real GPU kernel times). `run_workload.py` locked both captures to an identical 2175 MHz graphics / 12481 MHz memory clock, so baseline-vs-optimized durations are directly comparable. The **ncu replay** phase contributes only the hardware **counters** (tensor-core %, SM/DRAM throughput, occupancy), collected at its own base-clock lock. Both profiles were captured on the same GPU 15 minutes apart — no cross-session caveat applies.

---

## 2. Operator Summary (Baseline)

Total attributed time: **773,562 ns** across 9 operators. Sorted by Time (%).

| Operator | Time (%) | Duration (ns) | Kernels | Bottleneck Class |
|---|---|---|---|---|
| `aten::mm` (2048×512, op 9) | 15.2% | 117,439 | 1 | Tensor-cores idle (FP32 SIMT) |
| `aten::mm` (2048×512, op 19) | 14.9% | 115,168 | 1 | Tensor-cores idle (FP32 SIMT) |
| `aten::mm` (2048×2048, op 7) | 14.4% | 111,039 | 1 | Tensor-cores idle (FP32 SIMT) |
| `aten::mm` (2048×2048, op 5) | 14.3% | 110,975 | 1 | Tensor-cores idle (FP32 SIMT) |
| `aten::mm` (2048×2048, op 17) | 13.9% | 107,743 | 1 | Tensor-cores idle (FP32 SIMT) |
| `aten::mm` (2048×2048, op 15) | 13.9% | 107,391 | 1 | Tensor-cores idle (FP32 SIMT) |
| `aten::mm` (512×2048, op 3) | 5.9% | 45,728 | 1 | Tensor-cores idle (FP32 SIMT) |
| `aten::mm` (512×2048, op 13) | 5.6% | 43,519 | 1 | Tensor-cores idle (FP32 SIMT) |
| `aten::addmm` (bias + activation epilogues, fused) | 1.9% | 14,560 | 8 | Memory-bound |

**~98% of all time is in the eight `aten::mm` GEMMs, every one of which leaves the tensor cores completely idle.**

---

## 3. Reading the Metrics

Only the metrics that drive this workload's bottleneck are explained.

- **`tensor_core_active_pct` (`smsp__pipe_tensor_cycles_active`)** — On this RTX PRO 6000 Blackwell the counter *is* populated (unlike B100/B200, where it is removed). A value of **0.0 — not null** — on a GEMM is the single highest-ROI signal in profiling: it means the matmul ran on the FP32 SIMT (CUDA-core) path with the tensor cores switched off entirely. All eight baseline GEMMs read 0.0. A `null` on a non-GEMM kernel (e.g. a `copy_` cast) is expected and is **not** a problem.
- **`sm_throughput_pct`** — Fraction of peak SM issue throughput. Baseline GEMMs run at 6–23%: the SIMT path is busy but nowhere near saturating the chip.
- **`achieved_occupancy`** — Warps resident vs. max. Baseline GEMMs sit at 8–17%, register-capped (200 regs/thread on the SIMT SGEMM) and grid-starved (B=256 underfills the GPU). This is the *second-order* bottleneck that survives the tensor-core fix.
- **`memory_throughput_pct` (DRAM)** — Drives the activation epilogues: at 26–35% these are memory-bound, as expected for elementwise ops.

---

## 4. Optimizations Applied

Status from `profiler_output/validation_report.json`; everything else from `optimizations.json`.

| ID | Type | Target Operators | Hardware Evidence | Confidence | Status |
|---|---|---|---|---|---|
| OPT-1 | `aten` (bf16 promotion) | `aten.mm` / `aten.addmm` | `tensor_core_active_pct = 0.0` + `smsp__pipe_tensor_cycles_active = 0` on all 8 GEMMs (FP32 SIMT SGEMM) | high | **APPLIED** |
| OPT-2 | `inductor_config` (TF32 + max_autotune_gemm) | fp32 matmuls / GEMM templates | same idle-tensor-core signature; lower-risk lever on the same bottleneck | high | **APPLIED** |
| OPT-3 | `functional` (bias+activation epilogue fusion) | `F.linear` → activation | 8 separate activation-epilogue launches in baseline `addmm` | medium | **APPLIED** |

All three passes applied cleanly; none degraded gracefully (no WARNING/NOT_APPLIED lines).

---

## 5. Implementation Notes

# Implementation Notes — mlp_activations_opt

Custom `torch.compile()` backend for `MLPActivations` (four Linear+activation layers,
FP32, batch 256), implementing the three optimizations from `optimizations.json` through
the three-stage funnel (`functional -> aten -> inductor_config`).

Registered backend name: **`mlp_activations_opt`**
compile_mode: `dedup-inductor` (flat compile path — the four layers have distinct shapes
512->2048, 2048->2048, 2048->2048, 2048->512, so the dedup equivalence map is empty).

## Backend Architecture

| Pass | Level | Method | Reason |
|---|---|---|---|
| OPT-3 — F.linear + activation epilogue-fusion enablement | `functional` | `_run_functional_passes` (`_fpass_mark_linear_activation_epilogue`) | Each layer is one `F.linear` feeding one activation; the shared-output identity the epilogue matcher needs exists only pre-decomposition. The pass tags the linear so Inductor (with `max_autotune_gemm` from OPT-2) fuses bias+activation into the GEMM epilogue once the matmul is on a tensor-op kernel. Non-destructive annotation. |
| OPT-1 — bf16 dtype promotion on matmul operands | `aten` | `_aten_inner_compile` (`_apass_bf16_promote_matmul`) | The bottleneck is FP32 SIMT SGEMM with idle tensor cores. Casting `aten.mm`/`aten.addmm` operands to bf16 (output back to fp32) selects a tensor-op HGEMM. The op-target rewrite is cleanly expressed only on the decomposed mm/addmm nodes, so it runs at the aten seam. |
| OPT-2 — TF32 + max_autotune_gemm | `inductor_config` | `config_patches` (`_cfg_tf32_and_autotune`) + module-load front-end flags (`_enable_tf32_frontend`) | Lowering-policy decision Inductor/cuBLAS own; no graph surgery. Routes any fp32 matmul left after OPT-1 through the TF32 pipe, and enables fused-epilogue GEMM templates. |
| (whole-module dtype / memory_format / batch shape) | non-graph | — (none) | No non-graph optimization is proposed for this workload; `get_model_and_input()` returns the baseline model/input unchanged (FP32, CUDA). |

All three passes degrade gracefully: each is wrapped in try/except and logs a WARNING + no-ops if its target pattern (F.linear pair / mm-addmm node / Inductor config key) is absent.

## Key Design Decisions

**Why OPT-1 matches both `aten.mm` and `aten.addmm`.** `optimizations.json` names the
match target `torch.ops.aten.mm.default`, but every layer here is `nn.Linear(bias=True)`,
which AOTAutograd decomposes to `aten.addmm.default(bias, x, t(weight))`, not a bare `mm`.
The pass handles both overloads: for `addmm` it promotes only operands `args[1]`/`args[2]`
(the matmul inputs) and leaves the fp32 bias for the kernel's fp32 accumulator, wrapping
the whole `addmm` output in a cast back to fp32. This preserves the bias-add and downstream
activation in fp32 while engaging the tensor-core HGEMM; accuracy impact is bounded to bf16
input rounding (HGEMM accumulates in fp32 internally).

**Why OPT-2 is split between a config dict and a module-load side effect.** The Inductor
`triton.tf32` / `max_autotune_gemm` keys are scoped per `compile_fx` call via
`config_patches` (no global Inductor mutation). But the cuBLAS TF32 routing decision is
gated by the process-global `torch.backends.cuda.matmul.allow_tf32` /
`set_float32_matmul_precision` flags, which must be in effect before compilation. These are
set once at module import (`_enable_tf32_frontend()`), which is also when the backend
registers — so importing the optimized module is sufficient to put the policy in effect.
Config keys are probed with `hasattr` so an absent key on a given torch build no-ops
instead of raising.

**Why OPT-1 and OPT-2 coexist rather than being mutually exclusive in code.** They target
the identical idle-tensor-core mechanism. OPT-1 promotes operands to bf16 (largest win,
also halves DRAM traffic); OPT-2's TF32 policy is a no-op on the now-bf16 nodes and a
lower-risk fallback for any matmul OPT-1 does not promote. Keeping both registered means
the backend still engages tensor cores via TF32 even if the bf16 pass no-ops on some node.

**Why OPT-3 only annotates instead of rewriting (per-tag, not `replace_pattern`).** The
GEMM-epilogue fusion is the scheduler's job; manually rewriting `linear+activation` into a
fused custom op would fight Inductor's lowering and lose the autotuned template. The pass
verifies the single-producer/single-consumer precondition and tags the producer; the
cross-level ordering (functional tag -> aten bf16 -> Inductor epilogue fusion) is satisfied
automatically by the funnel, with no within-level sequencing needed.

**Dedup path preserved but inactive.** `UniqueSubgraphRegistry` returns an empty
equivalence map for this model (distinct layer shapes), so the backend takes the flat
compile path, which preserves cross-layer Inductor fusion. The per-rep dedup branch is
retained verbatim for models with repeated identical blocks.

## Syntax check

`python -m py_compile mlp_activations_optimized.py` — passed (see verification step).

---

## 6. Before/After Results

Both profiles share batch size 256 and the same locked clocks; comparison is fair. Operators are grouped by role because bf16 promotion + fusion restructured the graph (8 separate `aten::mm` GEMMs collapse into one fused `aten::addmm`, and new dtype-cast `aten::copy_` kernels appear). **The optimized total includes the bf16-cast overhead the optimization itself introduced.**

| Operator group | Baseline (ns) | Optimized (ns) | Speedup |
|---|---|---|---|
| GEMMs: `aten::mm` ×8 → `aten::addmm` (bf16 tensor-core, fused) | 759,002 | 147,680 | **5.14×** |
| Activation epilogues (`relu`/`gelu`/`silu`/`tanh`) | 14,560 | 12,512 | 1.16× |
| bf16 dtype casts (`aten::copy_` ×16) | 0 | 57,888 | new overhead |
| **Total** | **773,562** | **218,080** | **3.55×** |

**Speedup attribution** (all three conditions met: APPLIED status + expected counter change + operator speedup):

- The 5.14× GEMM win is attributed to **OPT-1 (bf16 promotion)**, confirmed by the kernel swap from `cutlass_80_simt_sgemm_*` (FP32 SIMT) to bf16 Triton `triton_tem_fused_addmm_*` templates, and by `tensor_core_active_pct` rising from **0.0 → 41.1%** on the fused `addmm` operator. **OPT-2 (TF32)** is a corroborating lever on the same mechanism but is a no-op once operands are bf16. **OPT-3** contributed the GEMM-template fusion (the optimized GEMM operator is `addmm`, not `mm`, carrying the bias in-kernel).
- The 57,888 ns of `aten::copy_` kernels are the **direct cost of OPT-1**: casting each fp32 operand to bf16 and each GEMM output back to fp32. This is 26.5% of optimized time — the price of the tensor-core win, and the reason the end-to-end 3.55× is well below the 5.14× compute-only figure.

### Residual opportunity detection

Re-ranking the optimized profile, the new top operator is the fused bf16 `addmm` at **67.7%** of time — now genuinely tensor-core-bound (tc=41%) but still occupancy-limited (achieved occupancy 16.5%, grid-starved at B=256). The second-largest cost is the **bf16 cast `copy_` group (26.5%)**, an artifact of round-tripping fp32↔bf16 at every layer boundary.

| Residual opportunity | Optimized cost | Projected gain |
|---|---|---|
| Eliminate fp32↔bf16 round-trip casts (keep activations in bf16 end-to-end, or feed bf16 input) | 57,888 ns (26.5%) | Up to ~1.3× further if casts removed |
| Raise GEMM occupancy (larger batch / grid) | addmm occ 16.5% | Diminishing — bounded by B=256 |

---

## 7. What Drove Each Speedup

**bf16 dtype promotion on matmul operands (OPT-1, +5.14× on the GEMMs):** Casting the `mm`/`addmm` operands to bf16 routes the matmul off the FP32 SIMT SGEMM path and onto the tensor-core HGEMM templates, which is where Blackwell's matmul throughput lives. The evidence is unambiguous: the `cutlass_80_simt_sgemm_*` kernels disappear entirely and are replaced by `triton_tem_fused_addmm_*` bf16 templates, while `tensor_core_active_pct` on the GEMM operator rises from 0.0 to 41.1% (`smsp__pipe_tensor_cycles_active` goes from zero to active).

**TF32 + max_autotune_gemm (OPT-2, corroborating, no isolated speedup):** Enabling the TF32 cuBLAS policy and `max_autotune_gemm` engages tensor cores for any fp32 matmul OPT-1 did not promote and unlocks the autotuned fused-epilogue templates. Once OPT-1 has made the operands bf16 the TF32 path is moot, so no speedup is attributed to it independently — it is the lower-risk safety net described in the implementation notes.

**bias + activation epilogue fusion (OPT-3, enabling, +1.16× on activations):** Tagging each `F.linear`→activation pair let Inductor lower the layers as fused `addmm` GEMM templates (carrying the bias in-kernel) rather than a bare `mm` plus separate epilogue, which is why the optimized GEMM operator is `aten::addmm` with the bias folded in. Evidence: the optimized GEMM operator is `addmm` (bias absorbed), and the activation launches that were 8 fused-epilogue kernels in baseline now ride alongside the tensor-core templates.

---

## 8. Remaining Opportunities

All three proposed optimizations (OPT-1, OPT-2, OPT-3) were **APPLIED** — no unapplied FX passes remain in `optimizations.json`.

The most significant *new* opportunity, exposed by the optimization rather than proposed beforehand, is the **fp32↔bf16 cast overhead** (`aten::copy_`, 26.5% of optimized time). It was introduced by OPT-1's per-operand promotion-and-recast strategy. A follow-up pass that keeps the dataflow in bf16 across layer boundaries — promoting once at the input and demoting once at the output, rather than round-tripping at every matmul — could remove most of those 57,888 ns and push the end-to-end speedup from ~3.55× toward the ~5× compute-only ceiling. This is **not yet implemented**.

---

## Reproduction

```bash
# Drive the whole pipeline:
/optimize examples/mlp_activations/mlp_activations.py

# Or run the baseline + optimized captures manually (two-phase, per CLAUDE.md):
#   Phase 1 (correlation): run_workload.py --correlation-pass
#   Phase 2 (nsys capture): nsys profile ... run_workload.py
#   Optimized re-capture adds: --compile-backend mlp_activations_opt
#   on examples/mlp_activations/mlp_activations_optimized.py
```
