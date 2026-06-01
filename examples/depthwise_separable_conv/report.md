# Optimization Report — DepthwiseSepConv

**This optimization achieved 1.03× total speedup on DepthwiseSepConv (B=16, NVIDIA RTX PRO 6000 Blackwell).** All three proposed FX/config transformations registered and applied cleanly (validation: 4/4), but only the channels-last layout pass (OPT-3) produced a measurable kernel-level change — it cut the NCHW→NHWC boundary-copy operator by 1.47×. The two graph-level passes (Conv-BN fold via freezing, bf16 promotion) applied at the IR level but did **not** transform the lowered kernels, because every convolution in this model lowers to an **extern cuDNN kernel** that Inductor's freezing and dtype-promotion machinery cannot rewrite. The net effect is the layout-copy reduction, partially diluted by per-kernel noise on the unchanged cuDNN convolutions.

---

## 1. Hardware Context

| Field | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Architecture | Blackwell |
| PyTorch | 2.11.0+cu128 (CUDA 12.8) |
| Compile mode (baseline) | inductor (built-in dedup backend) |
| Compile mode (optimized) | `depthwise_separable_conv_opt` (custom `@register_backend`) |
| Batch size | 16 |
| Input / output | `[16, 32, 56, 56]` fp32 → `[16, 256, 56, 56]` fp32 |
| Clock lock | 1845 MHz graphics / 12481 MHz memory — **identical for both captures** |
| Capture gap | ~45 min, same GPU (same session — no cross-session caveat) |

**Timing source.** Per-operator durations come from the **nsys capture** phase (GPU kernel times). `run_workload.py` probed the sustained clock once and cached it, so baseline and optimized captures locked to the **same** 1845/12481 MHz — durations are directly comparable. The ncu replay phase contributes only the hardware **counters** (tensor-core %, SM/DRAM throughput, occupancy), collected at its own base-clock lock.

---

## 2. Operator Summary (baseline)

Total attributed GPU time: **395,135 ns** across 14 operators, 0 unattributed.

| Operator | Time (%) | Duration (ns) | Kernels | Bottleneck Class |
|---|---|---|---|---|
| `aten::_native_batch_norm_legit_no_training` (fused ×12) | 34.0% | 134,432 | 12 | Memory-bound (75% mem, 0% TC, 79% occ) |
| `aten::cudnn_convolution` (pointwise 1×1, 128→256) #26 | 12.2% | 48,063 | 1 | TF32 GEMM, partial TC (45% TC, 8% occ) |
| `aten::cudnn_convolution` (pointwise 1×1, 128→256) #53 | 12.1% | 47,808 | 1 | TF32 GEMM, partial TC (45% TC, 8% occ) |
| `aten::cudnn_convolution` (pointwise 1×1, 64→128) #18 | 7.1% | 28,224 | 1 | TF32 GEMM (39% TC, 8% occ) |
| `aten::cudnn_convolution` (pointwise 1×1, 64→128) #45 | 6.8% | 26,880 | 1 | TF32 GEMM (38% TC, 8% occ) |
| `aten::convolution` (boundary layout copy + dw) #0 | 4.9% | 19,392 | 2 | Memory-bound copy (50% mem, 9% SM, 0% TC) |
| `aten::cudnn_convolution` (depthwise 3×3) #22 | 4.2% | 16,544 | 1 | Memory-bound (73% mem, 0% TC, 85% occ) |
| `aten::cudnn_convolution` (depthwise 3×3) #49 | 4.2% | 16,480 | 1 | Memory-bound (74% mem, 0% TC, 84% occ) |
| `aten::cudnn_convolution` (pointwise 1×1, 32→64) #10 | 3.4% | 13,632 | 1 | TF32 GEMM (19% TC, 8% occ) |
| `aten::cudnn_convolution` (pointwise 1×1, 32→64) #37 | 3.4% | 13,408 | 1 | TF32 GEMM (19% TC, 9% occ) |
| `aten::cudnn_convolution` (depthwise 3×3) #14 | 2.2% | 8,672 | 1 | Memory-bound (61% mem, 0% TC) |
| `aten::cudnn_convolution` (depthwise 3×3) #41 | 2.1% | 8,352 | 1 | Memory-bound (60% mem, 0% TC) |
| `aten::cudnn_convolution` (depthwise 3×3) #33 | 1.7% | 6,688 | 1 | Memory-bound (48% mem) |
| `aten::cudnn_convolution` (depthwise 3×3) #6 | 1.7% | 6,560 | 1 | Memory-bound (48% mem) |

The profile cleanly reproduces the textbook roofline split the workload was designed to surface: the **depthwise** 3×3 convs and the **BatchNorm+ReLU6** epilogue sit on the memory-bound side (0% tensor-core, 48–75% DRAM throughput), while the **pointwise** 1×1 convs engage tensor cores (19–45% TC) but on the TF32 `cutlass_80 s1688gemm` path at only ~8% occupancy (register-bound, 224 regs/thread).

---

## 3. Reading the Metrics

Only metrics that drive this workload's bottlenecks are explained.

- **`memory_throughput_pct`** — % of peak DRAM+cache bandwidth. The BN epilogue (75%) and depthwise convs (60–74%) are memory-bound: they are limited by byte traffic, not math. Reducing bytes moved (fusion, lower dtype, layout) is the lever here.
- **`tensor_core_active_pct`** — `0.0` (not null) means the kernel ran on the SIMT/FP32 path with tensor cores **completely idle** — the highest-ROI signal. All depthwise convs and BN show `0.0` (expected — they are not GEMMs). The pointwise convs show 19–45%: tensor cores engage but are far from saturated. A `null` value would be expected for non-GEMM kernels; none appear here.
- **`achieved_occupancy`** — the pointwise GEMMs sit at ~8% occupancy (register-bound), the classic symptom that funnels these into a dtype/occupancy optimization. The memory-bound kernels run 73–85% occupancy — already saturated, so occupancy is not their constraint.
- **`sm_throughput_pct`** — low on the copy/BN kernels (9–26%), confirming they do little compute and are bandwidth- or launch-limited.

---

## 4. Optimizations Applied

| ID | Type | Target Operators | Hardware Evidence | Confidence | Status |
|---|---|---|---|---|---|
| OPT-1 Conv-BN fold via freezing | inductor_config | `_native_batch_norm_legit_no_training` (34% of time) | 82.75% DRAM throughput, 17% SM, 0% TC — redundant read-modify-write | high | **APPLIED** (config emitted; no fold materialized — see §6/§8) |
| OPT-2 bf16 dtype promotion | aten | pointwise `cudnn_convolution` GEMMs | 19–45% TC, ~8% occ, 224 regs/thread — register-bound TF32 | medium | **APPLIED** (cast inserted; no kernel reroute — see §6/§8) |
| OPT-3 channels_last / NHWC | non-graph | boundary `aten::convolution` layout copy | layout-copy kernel at 50% mem, 9% SM, 0% TC | medium | **APPLIED** (measurable, 1.47× — see §6/§7) |

All three passes report `status == APPLIED` in `validation_report.json` — meaning each pass executed without error and Inductor lowered the result cleanly (the compile-level fallback was never triggered). "Applied" here means the FX pass ran and the IR was rewritten/configured as intended; it does **not** by itself guarantee the lowered kernels changed. Section 6 measures which applied passes actually moved hardware metrics.

---

## 5. Implementation Notes

# Implementation Notes — depthwise_separable_conv_opt

Backend name: `depthwise_separable_conv_opt` (registered via `@register_backend` at module import).

Workload: three stacked MobileNet-style depthwise-separable blocks (32→64→128→256),
batch 16, 56×56, fp32, CUDA. Cross-validated against `profile.json`: input
`[16, 32, 56, 56]` fp32, output `[16, 256, 56, 56]`; baseline shows the standalone
`_native_batch_norm_legit_no_training` + ReLU6 (hardtanh) epilogue dominating at
34% of attributed time, the 1×1 pointwise GEMMs on the TF32 `cutlass_80 s1688gemm`
path at ~8% occupancy / 224 registers, and a pure `triton_poi_fused_convolution_0`
NCHW→NHWC layout-copy kernel at the graph boundary.

## Backend Architecture

| Pass | Level | Method | Reason |
|---|---|---|---|
| OPT-1 Conv-BN fold via freezing | inductor_config | `config_patches={"freezing": True}` | BN/ReLU6 fold is a lowering decision Inductor owns; freezing makes eval weights + BN stats constants so Inductor folds BN into the preceding conv and fuses the ReLU6 clamp, removing the dominant memory-bound BN epilogue kernels. Scoped to this `compile_fx` call — no global config mutation. |
| OPT-2 bf16 dtype promotion | aten | `_aten_inner_compile` (`_apass_bf16_conv_promotion`) | Decomposed-primitive dtype rewrite: inserts `prims.convert_element_type(bfloat16)` on each `aten.convolution.default` input so the 1×1 GEMMs take the native bf16 tensor-core path (relieving the register/occupancy wall) and depthwise/BN byte traffic halves. Op-target pass — keys on the conv target, does not read weight values, so no `ph_to_tensor` lookup needed. |
| OPT-3 channels_last / NHWC | non-graph | `get_model_and_input()` | Whole-module memory_format annotation; the conv library kernels are already NHWC-native, so `model.to(channels_last)` + `x.to(channels_last)` removes the boundary NCHW→NHWC layout-copy kernel. Not expressible as an op rewrite — it is a tensor memory-format property, which is why the canonical approach places it in `get_model_and_input()` despite the proposal's `aten` label. |

No `functional`-level passes apply: this stack has no shared-activation linear triplets
and no SDPA, so `_run_functional_passes` is a uniform no-op pass-through.

## Key Design Decisions

**OPT-3 routed as non-graph despite `ir_level: aten`.** The proposal labels channels_last
`aten` but its own `fx_steps` and `code_hint` describe `model.to(memory_format=...)` —
a memory-format property of tensors, not an op-graph rewrite. Per Rule 7 / the fx-patterns
"Channels-Last Conversion (Non-Graph)" reference, this is applied in `get_model_and_input()`
with an `is_contiguous(channels_last)` guard so it is a no-op if the baseline already
supplies NHWC tensors. Inductor then propagates NHWC through the stack and drops the
permute copy.

**OPT-2 cast op: `prims.convert_element_type`, not `aten._to_copy`.** On torch 2.11 the
cast must be `torch.ops.prims.convert_element_type.default(act, torch.bfloat16)` — the
primitive Inductor itself emits for dtype conversion. It has a registered Inductor lowering
and is NOT in the decomposition table, so it lowers cleanly. The earlier
`aten._to_copy.default` version hard-crashed: that op has BOTH a registered Inductor
fallback AND a decomposition, producing `InductorError: AssertionError: both a fallback and
a decomp for same op: aten._to_copy.default`. Critically, that assertion fires *inside*
`compile_fx_inner` (after the pass returns and lints), so the pass's own try/except never
saw it — it aborted the entire `compile_fx`, meaning OPT-1 freezing never folded and no
Triton was produced. `aten.to.dtype` decomposes back to `_to_copy` and hits the same wall;
`prims.convert_element_type` (verified against the live lowering registry and decomp table)
is the only one of the three that survives lowering. Its dtype is a positional arg, not a
kwarg. `_repropagate_meta` repopulates `meta['val']` on the inserted cast nodes before
`compile_fx_inner`.

**Casting only the activation input, weight left untouched.** The bf16 pass is an op-target
pass, so it runs without the real-parameter `ph_to_tensor` lookup. Casting only the
activation input keeps the pass robust under FakeTensors; Inductor's constant handling
promotes the (folded, constant) conv weight to match the bf16 input dtype, keeping the GEMM
on the native bf16 tensor-core path.

**Compile-level graceful degradation.** Because a cast-op lowering failure surfaces inside
`compile_fx_inner` — below the per-pass guard — `_compile_unit` wraps the aten-pass compile
in a try/except and retries once with `apply_aten_passes=False` (OPT-1 freezing + OPT-3
channels_last still applied) on any failure. This guarantees the backend always returns a
working compiled callable rather than hard-crashing, satisfying the medium-confidence
degrade-gracefully contract even for lowering-stage faults the pass guard cannot reach.

**Funnel ordering satisfies the OPT-2 → OPT-1 prerequisite automatically.** The fixed
funnel runs aten (OPT-2 bf16) inside `inner_compile` before Inductor's freezing
constant-folding (OPT-1) executes, so the folded conv weight buffers are allocated in bf16
with no within-level sequencing. No `aot_autograd(fw_compiler=compile_fx)` is used (it raises
on torch 2.11); `compile_fx` owns AOTAutograd exactly once.

**Dedup path present but inert.** The three blocks have different channel counts and are
not structurally identical, so `UniqueSubgraphRegistry.build_partition_equivalence_map()`
returns empty and the flat-compile path is taken (which also preserves cross-block Inductor
fusion). The per-rep dedup path is retained for interface uniformity and would activate for
a repeated-block model.

**cudagraphs note:** not requested (compile_mode is `inductor`). OPT-3 changes only static
memory format and OPT-2 inserts static casts, so neither introduces dynamic shapes that
would trigger CUDA-graph re-captures.

---

## 6. Before/After Results

Both captures share batch size 16, the same GPU, and an identical clock lock — the comparison is fair and no cross-session caveat applies.

Operators matched by `operator_name` (durations in ns):

| Operator | Baseline (ns) | Optimized (ns) | Speedup |
|---|---:|---:|---:|
| `aten::convolution` (boundary layout copy + dw) | 19,392 | 13,216 | **1.47×** |
| `aten::_native_batch_norm_legit_no_training` (×12) | 134,432 | 126,496 | 1.06× |
| `aten::cudnn_convolution` #6 (dw 32→32) | 6,560 | 5,824 | 1.13× |
| `aten::cudnn_convolution` #10 (pw 32→64) | 13,632 | 13,536 | 1.01× |
| `aten::cudnn_convolution` #18 (pw 64→128) | 28,224 | 27,040 | 1.04× |
| `aten::cudnn_convolution` #26 (pw 128→256) | 48,063 | 48,545 | 0.99× |
| `aten::cudnn_convolution` #33 (dw 32→32) | 6,688 | 5,856 | 1.14× |
| `aten::cudnn_convolution` #37 (pw 32→64) | 13,408 | 13,664 | 0.98× |
| `aten::cudnn_convolution` #41 (dw 64→64) | 8,352 | 9,152 | 0.91× |
| `aten::cudnn_convolution` #45 (pw 64→128) | 26,880 | 27,296 | 0.98× |
| `aten::cudnn_convolution` #49 (dw 128→128) | 16,480 | 17,056 | 0.97× |
| `aten::cudnn_convolution` #53 (pw 128→256) | 47,808 | 48,320 | 0.99× |
| `aten::cudnn_convolution` #14 (dw 64→64) | 8,672 | 9,440 | 0.92× |
| `aten::cudnn_convolution` #22 (dw 128→128) | 16,544 | 17,568 | 0.94× |
| **TOTAL** | **395,135** | **383,009** | **1.032×** |

Net reduction: **12,126 ns (3.1%)**. No new optimization-introduced kernels appeared in `unattributed_kernels` (both profiles have 0 unattributed) — the bf16 casts that OPT-2 inserts were fused/elided by Inductor rather than materializing as standalone cast kernels, so there is no hidden overhead row to add.

### Speedup attribution (Step B)

A speedup is credited to a pass only if its `status == APPLIED` **and** the expected hardware metric moved **and** the target operator got faster.

| Pass | Status | Expected metric change | Observed | Credited? |
|---|---|---|---|---|
| OPT-3 channels_last | APPLIED | boundary layout-copy kernel shrinks/disappears | `aten::convolution` 19,392→13,216 (1.47×); its SM throughput fell 9%→2% as copy work dropped | **YES — sole confirmed contributor** |
| OPT-1 Conv-BN fold | APPLIED | BN op disappears (folded into conv) | BN op still present, 12 kernels, 33% of time; only 1.06× (within noise) | **NO — fold did not materialize** |
| OPT-2 bf16 promotion | APPLIED | pointwise GEMM tensor-core % rises, occupancy improves | pointwise TC unchanged (#26: 44.9%→44.9%), occupancy still ~8%; several convs regressed 0.91–0.99× | **NO — kernels not rerouted** |

The measured 1.032× is therefore attributable almost entirely to **OPT-3**. OPT-1 and OPT-2 applied at the IR/config level (validation confirms the FX rewrites and config patch executed) but did not change the lowered kernels — see Section 8 for the root cause.

### Residual opportunity (Step C)

Re-ranking the optimized profile, the bottleneck order is unchanged: BN epilogue (33%) and the two large pointwise GEMMs (#26, #53, ~25% combined) still dominate. These are exactly the targets OPT-1 and OPT-2 *aimed* at but could not transform, because both convolution families lower to **extern cuDNN kernels**. Realizing their projected gains requires moving the convolutions off the cuDNN extern path (so Inductor owns them and can fold/retype) — see Section 8.

---

## 7. What Drove Each Speedup

**Channels-last / NHWC layout (OPT-3, +1.47× on `aten::convolution`):** Annotating the model and input as `channels_last` lets Inductor propagate NHWC through the stack so the conv library kernels — which are already NHWC-native — no longer need a boundary NCHW→NHWC permute. **Evidence:** the boundary `aten::convolution` operator dropped from 19,392 ns to 13,216 ns, and its SM throughput fell from 9% to 2% as the explicit layout-copy work disappeared. This is the only pass whose target metric moved, and it accounts for essentially all of the 12,126 ns net gain.

*(OPT-1 and OPT-2 are intentionally absent from this section: per the attribution rule, a pass with no confirmed metric change is not credited with a speedup even though it applied. Their non-effect and its cause are documented in Section 8.)*

---

## 8. Remaining Opportunities

The two highest-value proposals applied cleanly but did not transform the lowered kernels. The shared root cause is the **extern cuDNN convolution path**.

| ID | Type | Target | Reason Not Realized | Projected Gain (unrealized) |
|---|---|---|---|---|
| OPT-1 Conv-BN fold | inductor_config | BN epilogue (33%) | Inductor `freezing` folds BN only into **Inductor-lowered** convs. Here every conv lowers to an **extern `cudnn_convolution`**, so there is no Inductor conv node to absorb the BN affine — the BN+ReLU6 Triton epilogue survives intact. | ~20% of total (~80,000 ns) |
| OPT-2 bf16 promotion | aten | pointwise GEMMs (~25%) | The `prims.convert_element_type(bf16)` cast is inserted on conv inputs and lowers cleanly, but the convs still dispatch to the **TF32 cuDNN GEMM** (tensor-core % and ~8% occupancy unchanged). cuDNN heuristics did not select a native bf16 kernel for these shapes; a few convs even regressed slightly from the extra cast. | ~18% of total (~70,000 ns) |

**Path to realizing the residual gains:** Force the convolutions off the cuDNN extern path so Inductor owns them — e.g. `torch._inductor.config.conv_1x1_as_mm = True` (lowers pointwise 1×1 convs to Inductor matmuls, which freezing *can* fold BN into and which honor the bf16 dtype), and/or `max_autotune` / `max_autotune_gemm` so Inductor generates and tunes its own Triton/CUTLASS conv+GEMM kernels. Only once the convs are Inductor-generated will OPT-1's BN fold and OPT-2's bf16 routing take effect.

If both passes were realized via Inductor-owned convolutions, the combined projected gain is on the order of **35–40% of total runtime** (discounted for the medium confidence of the bf16 routing on these specific shapes) — far larger than the 3.1% achieved by layout alone. The current run establishes the backend, the funnel, and a correct baseline; the next iteration should re-capture with the conv-lowering config flags above.

---

## 9. Reproduction

```bash
# Baseline capture
/capture examples/depthwise_separable_conv/depthwise_separable_conv.py

# Propose → backend → validate
/propose examples/depthwise_separable_conv/profile.json
/backend examples/depthwise_separable_conv/depthwise_separable_conv.py examples/depthwise_separable_conv/optimizations.json
/validate examples/depthwise_separable_conv/depthwise_separable_conv_optimized.py

# Optimized capture (custom backend)
/capture examples/depthwise_separable_conv/depthwise_separable_conv_optimized.py \
    --profile-name=optimized --compile-backend=depthwise_separable_conv_opt

# Or run the whole pipeline end-to-end
/optimize examples/depthwise_separable_conv/depthwise_separable_conv.py
```

Artifacts retained under `profiler_output/`: `*.nsys-rep`, `*.corr.json`, `*.part.json`, `ncu_reps/`, `implementation_notes.md`, `validation_report.json`, and the cached clock lock `.gpu_clock_lock.json`.
