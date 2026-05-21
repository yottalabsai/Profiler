# Optimization Report: conv_block

**Model:** ConvBlock (VGG-style three-stage CNN)
**Device:** NVIDIA RTX PRO 6000 Blackwell Server Edition
**PyTorch:** 2.11.0+cu128
**Baseline compile mode:** inductor
**Optimized backend:** `conv_block_opt` (custom `@register_backend`)
**Report date:** 2026-05-21

---

## 1. Hardware Context

| Property | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Microarchitecture | Blackwell (sm100 class) |
| PyTorch version | 2.11.0+cu128 |
| cuDNN notes | `sm80_xmma_fprop` kernel names appear on Blackwell, indicating torch 2.11+cu128 is using Ampere-generation cuDNN heuristics. Blackwell-native sm100 cuDNN kernels are not yet registered in this build. |
| Profiling tool | Nsight Compute (application-mode replay, `--replay-mode application`) |
| ncu overhead | All `duration_ns` values are inflated 2–5x by ncu counter-collection. They are used only for relative within-profile comparison, **not as wall-clock latencies**. |
| Warp cycles counter | Removed in Blackwell. `eligible_cycles_pct < 20%` is used as the latency-bound indicator throughout. `warp_cycles_per_instruction` is null in all profiles. |
| `tensor_core_active_pct` null | On Blackwell, this aggregated field is null or zero because the underlying counter was renamed. This does **not** indicate a Tensor Core bottleneck. Use `smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` per-kernel instead. |
| Batch size | B=16 throughout. All comparisons in this report are at B=16; no normalization is needed. |

---

## 2. Operator Summary (Baseline)

The baseline model is a three-stage CNN: two convolution blocks (3→64→128 channels on 64×64 spatial, then 128→256 on 32×32) followed by global average pooling and a linear classifier. The baseline compiles with `torch.compile(backend='inductor')` in NCHW layout (FP32/TF32).

Attribution uses two tiers: torch.profiler CUPTI correlation (high confidence) for `aten::cudnn_convolution_0`'s kernels, and NVTX range enclosure (medium confidence) for `layer::unique::prologue` and the explicit per-op NVTX ranges. A `fused_kernel_double_count` edge case is present: `aten::cudnn_convolution_0` and `layer::unique::prologue_0` share kernel IDs k_00020–k_00051. Each kernel fires once on the GPU but is attributed to both operators via different methods. The best unique-execution estimate comes from `layer::unique::prologue_0`.

| Operator | Kernels | Duration (ns) | % of total | Notes |
|---|---|---|---|---|
| layer::unique::prologue_0 | 39 | 359,228 | 33.1% | Best unique-execution estimate; NVTX attribution |
| aten::cudnn_convolution_0 | 17 | 287,644 | 26.5% | Double-counts kernels in prologue_0; excluded from unique budget |
| aten::cudnn_convolution op_id=33 (64→128ch) | 3 | 86,655 | 8.0% | Explicit NVTX |
| aten::cudnn_convolution op_id=12 (64→128ch) | 3 | 86,431 | 8.0% | Explicit NVTX |
| aten::cudnn_convolution op_id=39 (128→256ch) | 3 | 76,287 | 7.0% | Explicit NVTX |
| aten::cudnn_convolution op_id=18 (128→256ch) | 3 | 75,711 | 7.0% | Explicit NVTX |
| aten::_native_batch_norm_legit_no_training_0 | 14 | 67,807 | 6.3% | Fused BN-ReLU-Pool via Triton |
| aten::cudnn_convolution op_id=7 (3→64ch) | 3 | 14,752 | 1.4% | `sm80_xmma_indexed_wo_smem` |
| aten::cudnn_convolution op_id=28 (3→64ch) | 3 | 14,720 | 1.4% | `sm80_xmma_indexed_wo_smem` |

---

## 3. Reading the Metrics

- **`gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`**: DRAM bandwidth utilisation as a percentage of peak. Values above 70% indicate a bandwidth-bound kernel.
- **`smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active`**: Fraction of active cycles where the Tensor Core pipeline was busy. Zero for elementwise kernels; >60% indicates high arithmetic utilisation for GEMMs.
- **`sm__warps_active.avg.pct_of_peak_sustained_active`**: Average active warps per SM as a fraction of capacity. Values below 15% on a GEMM kernel indicate wave starvation (too few CTAs to fill all SMs simultaneously).
- **`sm__throughput.avg.pct_of_peak_sustained_elapsed`**: Fraction of peak SM throughput achieved.
- **`eligible_cycles_pct`**: Fraction of cycles where at least one warp was eligible to issue. Values below 20% on Blackwell suggest latency-bound execution (used in place of the removed `warp_cycles_per_instruction`).
- **`tensor_core_active_pct` in aggregated**: null on Blackwell — this does not describe a bottleneck. See per-kernel `smsp__pipe_tensor_cycles_active` instead.

**Attribution confidence:** `high` = torch.profiler CUPTI External-id join (most reliable). `medium` = NVTX range enclosure using CPU launch timestamp.

---

## 4. Optimizations Applied

| ID | Description | Status | Method |
|---|---|---|---|
| OPT-1 | channels_last (NHWC) layout | **APPLIED** | `get_model_and_input()` |
| OPT-2 | BF16 dtype promotion | **APPLIED** | `get_model_and_input()` + FX pass `_pass_insert_bf16_cast` |
| OPT-3 | Inductor max_autotune (wave starvation) | **APPLIED** (stub — config flags only) | Inductor config directives before `compile_fx` |
| OPT-4 | 3-channel conv padding (aligned GEMM path) | **NOT APPLIED** — graceful no-op | FX pass pattern not matched |

**Validation log (Stage 3):**
- `pass_insert_bf16_cast` (OPT-2): APPLIED — "Inserted BF16 cast on graph input placeholder"
- `pass_pad_shallow_conv` (OPT-4): NOT APPLIED — "No 3-channel F.conv2d found — pass not applied"
- OPT-1 (channels_last): APPLIED — `model.to(memory_format=torch.channels_last)` and `x.to(memory_format=torch.channels_last)` in `get_model_and_input()`
- OPT-3 (max_autotune): APPLIED — `inductor_config.max_autotune = True`, `inductor_config.coordinate_descent_tuning = True`

---

## 5. Implementation Notes

*Verbatim from `implementation_notes.md`*

### Backend Architecture

| Pass | ID | Method | Confidence | Reason |
|---|---|---|---|---|
| channels_last layout | OPT-1 | `get_model_and_input()` | high | `memory_format` is a tensor property, not visible in FX IR. Must be set before `torch.compile` traces the graph. |
| BF16 dtype promotion | OPT-2 | `get_model_and_input()` + FX pass | medium | `model.bfloat16()` sets weight dtypes before compilation. FX pass `_pass_insert_bf16_cast` inserts a `.to(bfloat16)` node after the first placeholder so runtime FP32 inputs are cast inside the compiled graph. Both halves are needed: the non-graph step fixes parameter dtypes; the FX pass handles activation dtype at graph execution time. |
| Inductor max-autotune | OPT-3 | Inductor config directives in backend (stub) | medium | Wave starvation on 64→128 and 128→256 convolutions (`sm__warps_active` = 8.3%). The root cause is large per-CTA tile selection by cuDNN heuristics. The appropriate lever is Inductor's Triton conv autotuner (`max_autotune_conv`), which searches smaller-tile variants. This is a stub: it sets `inductor_config.max_autotune = True` and `max_autotune_conv = True` before delegating to `compile_fx`, but does not rewrite any FX nodes. |
| 3-channel conv padding | OPT-4 | FX pass `_pass_pad_shallow_conv` | medium | The 3→64 channel convolution dispatches to `sm80_xmma_fprop_implicit_gemm_indexed_wo_smem` (15% TC utilisation, 26% SM throughput) because K=27 < WMMA alignment minimum. Padding input channels from 3 to 4 (K=36) satisfies alignment and enables the shared-memory-staging GEMM path. The extra zero-padded channel contributes exactly 0 to the output. |

### Key Design Decisions

**OPT-1 — non-graph placement:** `memory_format=torch.channels_last` is a tensor-level property. The FX graph sees abstract `aten.convolution.default` nodes regardless of the memory layout of their operands — layout is metadata, not a node type. Setting channels_last before `torch.compile` means Dynamo traces with NHWC-shaped fake tensors, so Inductor's shape inference is accurate and cuDNN receives NHWC data without format-conversion kernel overhead.

**OPT-2 — two-part implementation:** Parameter dtype (weights, BN scale/shift) is set before compilation so Dynamo traces with BF16 tensor shapes. The FX pass handles activation dtype at graph execution time. Without the FX pass, a caller passing a FP32 input tensor would trigger a dtype mismatch between the first placeholder and the BF16 weight parameters. BN `running_mean` and `running_var` are kept as FP32 buffers by `BatchNorm2d` regardless of model dtype.

**OPT-3 — stub classification:** The wave-starvation bottleneck is a cuDNN tile-selection choice at dispatch time, not a structural FX graph pattern. The pass sets Inductor config flags rather than rewriting nodes. Correctness is guaranteed; speedup is hardware and Triton version dependent.

**OPT-4 — pass not triggered:** With OPT-1 applied first, Dynamo traces the 3-channel conv as an NHWC operation and may produce a node target other than `F.conv2d` at the pre-Inductor `@register_backend` level. The graceful no-op path was taken correctly.

**Dedup path:** `UniqueSubgraphRegistry` found no repeated partitions in ConvBlock, so the flat compile path was used.

**Pass ordering:** OPT-1 before OPT-2 per `prerequisite_for` in `optimizations.json`: channels_last is applied first so BF16 parameters are stored in NHWC layout from the start.

### Known Limitations

- BF16 has a narrower mantissa than FP32 (7 vs 23 bits). Validate output accuracy before production deployment.
- The `sm80_xmma_fprop` kernels on Blackwell indicate torch 2.11+cu128 uses Ampere-generation cuDNN heuristics. Upgrading to a build with native Blackwell/sm100 support may resolve OPT-3 and OPT-4 bottlenecks without code changes.
- OPT-3 `max_autotune` adds significant first-compilation latency (minutes). Set `TORCHINDUCTOR_CACHE_DIR` to persist the autotuning cache.
- OPT-4 zero-padded weights must not be saved via `torch.save` for reuse with the original `state_dict` — inference-time use only.

---

## 6. Before/After Results

All values are from ncu application-mode replay at B=16. Duration figures are ncu-inflated and valid only for relative comparison within this report. They do not represent wall-clock latency.

### Primary operator comparison

| Operator | Baseline duration (ns) | Optimized duration (ns) | Change |
|---|---|---|---|
| 64→128ch convolution, call 1 (op_id=12 → op_id=7 opt) | 86,431 | 33,344 | **-61%** |
| 64→128ch convolution, call 2 (op_id=33 → op_id=20 opt) | 86,655 | 33,440 | **-61%** |
| BN-ReLU fused (dominant pointwise kernel) | ~12,600 | ~6,944 | **~-45%** |
| Full attributed prologue (unique kernel sum) | 359,228 | ~187,296 | **~-48%** |

Note on BN-ReLU total: The optimized profile restructures operator attribution — the BN operator accumulates more fused kernels explicitly (including large convolution-fused GEMM kernels) that were folded into the `layer::unique::prologue_0` NVTX range in the baseline. This re-partitioning is an attribution artifact, not a regression. The individual BN-ReLU pointwise kernels drop from 6,944–12,672 ns to 4,032–6,944 ns, consistent with BF16 halving DRAM bytes per element.

### Key hardware counter changes

| Metric | Baseline (64→128ch `Kernel`) | Optimized (64→128ch `Kernel`) | Change |
|---|---|---|---|
| sm_warps_active pct | 8.3% | 14.6–14.8% | +6.4 pp |
| smsp__pipe_tensor_cycles_active pct | 65–73% | 70.7–70.9% | Maintained |
| SM throughput pct | 57–64% | 60.6–60.7% | Maintained |
| convertTensor_kernel launches per conv call | 2 | 0 | **Eliminated** |
| nhwcToNchw `Kernel` launches | 1 per conv call (75,104 ns) | 0 | **Eliminated** |

| Metric | Baseline (BN-ReLU dominant kernel) | Optimized (equivalent BN-ReLU kernel) | Change |
|---|---|---|---|
| DRAM throughput pct | 79–84% | 60–71% | -10 to -24 pp |
| Duration of dominant BN kernel (ns) | 12,255–12,672 | ~6,944 | **~-45%** |

---

## 7. What Drove Each Speedup

### OPT-1 — channels_last: primary driver (~61% reduction on 64→128ch convolutions)

The dominant cost in the baseline `layer::unique::prologue_0` was the `Kernel` (nhwcToNchw output re-layout, k_00065, 75,104 ns) and two `convertTensor_kernel` launches per cuDNN convolution call. These three kernels accounted for 60–78% of each cuDNN convolution's total measured time and operated at 0% Tensor Core activity, 64–66% DRAM throughput, grid 1024×1×1 — purely memory-layout overhead with no arithmetic work.

With `model.to(memory_format=torch.channels_last)` applied before compilation, cuDNN receives input and weight tensors already in NHWC format. The profiler confirms the effect in `profile_optimized.json`: zero `convertTensor_kernel` launches and zero nhwcToNchw `Kernel` launches appear for any convolution operator. The 64→128ch convolution duration drops from ~86,543 ns to ~33,392 ns — a 61% reduction driven entirely by eliminating format-conversion kernels.

### OPT-2 — BF16 dtype promotion: secondary driver on BN-ReLU DRAM pressure

The baseline BN-ReLU kernels (`triton_poi_fused__native_batch_norm_legit_no_training_relu_4`, grid 16384×256, 12,255–12,672 ns) were saturating DRAM bandwidth at 79–80% throughput, reading FP32 tensors of size [16, 256, 64, 64]. Switching to BF16 halves bytes-per-element from 4 to 2, halving DRAM traffic for every pointwise kernel.

The optimized BN-ReLU kernels show DRAM throughput of 60–71% (down from 79–84%), and the equivalent 256-channel BN stage drops from ~12,600 ns to ~6,944 ns — approximately the expected 2x improvement for a bandwidth-bound kernel. BN `running_mean` and `running_var` remain FP32 as required by `_native_batch_norm_legit_no_training`'s internal accumulation.

### OPT-3 — max_autotune: marginal contribution to GEMM improvements

The wave starvation bottleneck (`sm__warps_active` 8.3% on the dominant 64→128ch `Kernel`) partially improved to 14.6–14.8% in the optimized profile. This improvement cannot be cleanly attributed to OPT-3 alone because OPT-1 (channels_last) and OPT-2 (BF16) together change the cuDNN algorithm selection path by altering input layout and dtype. The combined OPT-1+OPT-2+OPT-3 effect on the 64→128ch GEMM kernel is a ~2x duration reduction (68–70 ns → 33 ns per kernel invocation). `smsp__pipe_tensor_cycles_active` is maintained at 70–71%, confirming Tensor Core utilisation is not degraded by BF16 conversion.

OPT-4 is explicitly credited with **no speedup** — the pass logged "not applied."

---

## 8. Remaining Opportunities

### OPT-4 — 3-channel conv padding (not applied)

The pass `_pass_pad_shallow_conv` logged "No 3-channel F.conv2d found — pass not applied." The 3→64ch convolution dispatches to `sm80_xmma_fprop_implicit_gemm_indexed_wo_smem` (15% Tensor Core utilisation, 26% SM throughput) because K = 3×3×3 = 27 is below the WMMA alignment minimum. Padding input channels from 3 to 4 (K = 36) would allow cuDNN to use the shared-memory-staging GEMM path.

The likely root cause of the miss: with OPT-1 applied first, Dynamo traces the conv in NHWC and lowers it to a different Aten node target than `F.conv2d` or `aten.convolution.default` at the `@register_backend` invocation level in torch 2.11. To activate: inspect the FX graph with `TORCH_LOGS="+dynamo"` to confirm the exact target name, update the pass's target set, and re-run.

**Estimated remaining opportunity:** ~10,000–14,000 ns savings across the two 3→64ch calls. TC utilisation improvement from ~15% toward 40–50%.

### Wave starvation (OPT-3 residual)

The optimized 64→128ch `Kernel` still runs at only 14.6–14.8% `sm__warps_active`. Two levers remain:
1. **Batch doubling** to B=32 doubles the CTA grid, increasing waves toward ~25–30% SM fill.
2. **Forcing the Triton conv path** by warming the `TORCHINDUCTOR_CACHE_DIR` autotuning cache to avoid the sm80-generation cuDNN heuristic entirely.

**Estimated additional opportunity:** ~60,000 ns (~5.5% of profiled budget).

---

## 9. Reproduction Commands

```bash
# Environment setup
pip install -r requirements.txt
python3 nvidia/scripts/preflight.py

# Baseline profile — Phase A: correlation pass (no nsys)
PYTHONPATH=/home/ubuntu/Profiler python3 nvidia/scripts/run_workload.py \
    --workload examples/conv_block/conv_block.py \
    --output-prefix examples/conv_block/profiler_output/conv_block \
    --inductor-debug-dir examples/conv_block/profiler_output/conv_block_inductor_debug \
    --correlation-pass

# Baseline profile — Phase B: NVTX capture under nsys
nsys profile --trace=cuda,nvtx \
    --output=examples/conv_block/profiler_output/conv_block \
    python3 nvidia/scripts/run_workload.py \
        --workload examples/conv_block/conv_block.py \
        --output-prefix examples/conv_block/profiler_output/conv_block \
        --inductor-debug-dir examples/conv_block/profiler_output/conv_block_inductor_debug

# Baseline ncu hardware counters (requires sudo)
sudo env PYTHONPATH=/home/ubuntu/Profiler \
    /opt/nvidia/nsight-compute/2025.4.1/ncu \
    --replay-mode application --set full \
    --export examples/conv_block/profiler_output/ncu_reps/all_kernels \
    python3 nvidia/scripts/run_workload.py \
        --workload examples/conv_block/conv_block.py \
        --output-prefix examples/conv_block/profiler_output/conv_block

# Validation test suite
PYTHONPATH=/home/ubuntu/Profiler pytest \
    examples/conv_block/test_conv_block_optimized.py -v --tb=short

# Optimized re-capture (same three-phase sequence, substitute conv_block_optimized.py)
# Add --compile-backend conv_block_opt to run_workload.py calls
```

---

*All `duration_ns` values are from ncu application-mode replay and are inflated 2–5x relative to actual GPU execution time. Throughput percentages (`gpu__dram_throughput`, `sm__throughput`, `smsp__pipe_tensor_cycles_active`) are unaffected by replay overhead and are the primary comparables for bottleneck analysis. `tensor_core_active_pct = null` in the aggregated fields is a Blackwell counter-removal artifact and does not describe a performance problem.*
