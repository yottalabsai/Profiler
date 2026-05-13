# GPT-2 GPU Optimization Report

## Hardware Context

| Field | Value |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB (108 SMs) |
| Architecture | Ampere (sm80) |
| Compile Mode | inductor (layer deduplication enabled) |
| Batch Size | 4 |
| Sequence Length | 128 |
| Model | GPT-2 small — 12 transformer blocks, hidden=768, 12 heads, FFN=3072, 117M params |
| Measurement | 3 warmup + 10 measurement iterations (ncu replay — relative timing only) |

---

## Operator Summary (Baseline)

Baseline dtype: FP32. Total attributed wall time: 103,930,896 ns (ncu replay).
Layer deduplication was active: 13 partitions detected, 3 unique signatures, 10 duplicates.
ncu replayed unique partitions only (~12× replay speedup vs. full profiling).

| Operator | Time (%) | Duration (ns) | Kernels | Bottleneck Class |
|---|---|---|---|---|
| `aten::mm` | 51.93% | 53,966,937 | 336 | **tensor_core_idle** |
| `aten::addmm` | 15.68% | 16,301,364 | 456 | well_optimized (agg.) ¹ |
| `aten::_efficient_attention_forward` | 3.18% | 3,306,052 | 108 | **wave_starvation** |
| Remaining (attributed) | 29.21% | 30,356,543 | 998 | mixed |
| Unattributed | — | — | 240 | — |

¹ `aten::addmm` aggregate TC% is 0.248% (a few fused Triton post-ops inflate it slightly), but the dominant cuBLAS kernel `ampere_sgemm_128x32_nn` runs FP32 SIMT with TC=0. Practically the same problem as `aten::mm`.

**Top bottleneck: 67.61% of attributed time runs FP32 SIMT (`ampere_sgemm_*`) rather than Tensor Core HMMA.** On A100, the FP32 SIMT ceiling is 19.5 TFLOPS; the BF16 Tensor Core ceiling is 312 TFLOPS — a 16× theoretical headroom.

---

## Reading the Metrics

**`tensor_core_active_pct = 0.0` (not null):** The GEMM kernel ran but did not use Tensor Core hardware. This always means the kernel took the FP32 SIMT path. Fixing this is the highest-ROI optimization for any GEMM-heavy model.

**`tensor_core_active_pct = null`:** Normal for non-GEMM kernels (elementwise, reductions). Not a bottleneck.

**`local_memory_spills`:** Registers spilled to DRAM because the kernel exceeded the 255-register-per-thread hardware limit. At 10–100× slower than register access, even small spill counts degrade throughput significantly. The FP32 fmha kernel used 168 registers/thread and spilled 24.5 MB per replay.

**Waves/SM:** `ceil(grid_x × grid_y × grid_z / sm_count)`. For attention at B=4, seq=128, 12 heads: grid = `[2, 12, 4]` = 96 CTAs on 108 SMs = 0.89 waves. Less than 1 wave means most SMs are idle for the entire kernel.

**ncu replay timing note:** All duration values are from ncu's application-mode replay, which runs the workload 8 times (once per counter group) and is 2–5× slower than real execution. Use for relative comparison only — absolute ns values do not represent wall-clock latency.

---

## Optimizations Applied

| ID | Type | Target Operators | Hardware Evidence | Confidence | Status |
|---|---|---|---|---|---|
| OPT-1 | dtype_promotion (BF16) | `aten::mm`, `aten::addmm`, `aten::_efficient_attention_forward` | `tensor_core_active_pct = 0.0` on mm; 168 reg/thread + 24.5 MB spills on attention | high | **APPLIED** |
| OPT-2 | pretranspose_weights | `aten::mm`, `aten::addmm` | Secondary to OPT-1 | medium | NOT APPLIED ² |
| OPT-3 | algorithm_selection (max-autotune) | all GEMM ops | SM throughput 72.8% — sub-optimal tile selection | medium | **APPLIED** |

² OPT-2 ran pre-Inductor lowering in the dedup path, where `aten.t(get_attr)` nodes do not yet exist. The pass degraded gracefully (warning logged, no graph mutation). See Remaining Opportunities.

---

## Results: Before vs. After

Same batch size (B=4) — no normalization required.

### Per-kernel evidence (representative shapes)

| Operation | Shape | Baseline kernel | Baseline (ns) | Optimized kernel | Optimized (ns) | Speedup |
|---|---|---|---|---|---|---|
| FFN up-projection | [512×768] × [768×3072] | `ampere_sgemm_32x128_nn` | 197,952 | `ampere_bf16_s16816gemm_bf16_128x128_ldg8_relu_f2f_stages_64x3_nn` | 21,120 | **9.4×** |
| Attention output projection | [512×768] × [768×768] | `ampere_sgemm_64x32_sliced1x4_nn` | 54,848 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_relu_f2f_stages_64x4_nn` | 12,769 | **4.3×** |
| Flash attention | B=4, H=12, S=128 | `fmha_cutlassF_f32_aligned_64x64_rf_sm80` | 30,720 | `fmha_cutlassF_bf16_aligned_64x64_rf_sm80` | ~17,200 | **1.79×** |

### Aggregate operator speedup (attributed totals)

| Operator | Baseline (ns) | Optimized est. (ns) | Speedup | Bottleneck resolved? |
|---|---|---|---|---|
| `aten::mm` + `aten::addmm` | 70,268,301 | ~9–12 M | **6–8×** | YES — Tensor Cores active |
| `aten::_efficient_attention_forward` | 3,306,052 | ~1,850,000 | **1.79×** | PARTIAL — wave starvation remains |
| Remaining | 30,356,543 | ~25 M (est.) | ~1.2× | minor |
| **Total** | **103,930,896** | **~35–42 M** | **~2.5–3.0×** | |

---

## What Drove Each Speedup

**OPT-1: BF16 dtype promotion (+5–8× on GEMM, +1.79× on attention)**

Converting the model to BF16 rerouted every GEMM from the FP32 SIMT path to the BF16 HMMA (Tensor Core) path. On A100's Ampere architecture, BF16 GEMMs use the `s16816` HMMA instruction (16×8×16 tile), which throughputs at 312 TFLOPS vs. 19.5 TFLOPS for FP32 SIMT — a 16× peak gap. The measured per-kernel speedups (4.3× for smaller GEMMs, 9.4× for the FFN up-projection) reflect both the faster instruction and better tile matching via max-autotune.

For the attention kernel: the FP32 variant (`fmha_cutlassF_f32_aligned_64x64_rf_sm80`) required 168 registers/thread and spilled 24,468,480 bytes to local DRAM per replay pass. The BF16 variant (`fmha_cutlassF_bf16_aligned_64x64_rf_sm80`) reduced register pressure to 128/thread and eliminated all spills (0 bytes). This produced a 1.79× per-kernel speedup despite the persistent wave starvation (occupancy improved only marginally: 6.30% → 6.34%).

**OPT-3: max-autotune tile selection (+5–10% on top of OPT-1)**

With `config_patches={'max_autotune': True}`, Inductor benchmarked available CUTLASS tile configurations for GPT-2's matrix shapes. For the FFN up-projection ([512, 768] × [768, 3072]), the benchmark selected `128x128_ldg8_relu_f2f_stages_64x3` — a larger tile with 3-stage pipelining vs. the default heuristic-selected `32x128` tile. Inductor also fused `addmm + native_layer_norm` into a single Triton Template Matmul kernel (`triton_tem_fused_addmm_native_layer_norm_view_1`, TC=52.1%), eliminating a separate layer norm kernel launch per transformer block.

---

## Remaining Opportunities

| ID | Type | Target | Reason Not Applied | Projected Gain |
|---|---|---|---|---|
| OPT-2 | pretranspose_weights | `aten::mm`, `aten::addmm` | Pass runs pre-Inductor lowering in dedup path; `aten.t(get_attr)` nodes don't exist yet | ~2–4% |
| — | Batch size scaling | `aten::_efficient_attention_forward` | wave_starvation at B=4 (0.89 waves/SM); B=16 gives 3.5 waves | ~50% on attention |
| — | Flash Attention v2 | `aten::_efficient_attention_forward` | Current cutlass fmha has 6.34% occupancy; FA2 improves occupancy via online softmax | ~1.3× on attention |

**Fixing OPT-2:** Move `_pass_pretranspose` to run after Inductor lowers the graph, or pre-transpose `nn.Linear.weight` buffers at model-load time before `torch.compile` traces them (same approach as OPT-1 — modify `get_model_and_input()`). Estimated additional gain: 2–4% of total latency.

**Attention wave starvation** is structural: at B=4, seq=128, 12 heads, the attention grid has only 96 CTAs vs. 108 SMs. The fix requires either larger batch (serving decision) or a kernel that processes multiple heads per SM (Flash Attention v2 design).

---

## Reproduction Commands

```bash
cd /root/Profiler

# 1. Capture baseline profile (with layer deduplication)
PYTHONPATH=/root/Profiler python nvidia/scripts/run_workload.py \
    --workload examples/gpt2/gpt2.py \
    --warmup-iters 3 --measure-iters 10 \
    --correlation-pass \
    --output-prefix examples/gpt2/profiler_output/gpt2 \
    --inductor-debug-dir examples/gpt2/profiler_output/gpt2_inductor_debug \
    --layer-deduplicate

# 2. Analyze bottlenecks
# Output: examples/gpt2/triage.json

# 3. Run optimized workload (smoke test)
PYTHONPATH=/root/Profiler python nvidia/scripts/run_workload.py \
    --workload examples/gpt2/gpt2_optimized.py \
    --compile-backend gpt2_opt \
    --warmup-iters 1 --measure-iters 1

# 4. Run test suite
cd examples/gpt2 && PYTHONPATH=/root/Profiler python -m pytest test_gpt2_optimized.py -v

# 5. Re-profile optimized workload
PYTHONPATH=/root/Profiler python nvidia/scripts/run_workload.py \
    --workload examples/gpt2/gpt2_optimized.py \
    --compile-backend gpt2_opt \
    --warmup-iters 3 --measure-iters 10 \
    --correlation-pass \
    --output-prefix examples/gpt2/profiler_output/gpt2_optimized \
    --inductor-debug-dir examples/gpt2/profiler_output/gpt2_optimized_inductor_debug
```

**Key files:**

| File | Description |
|---|---|
| `examples/gpt2/gpt2.py` | Baseline workload (FP32, standard inductor) |
| `examples/gpt2/gpt2_optimized.py` | Optimized workload (`gpt2_opt` backend, BF16, max-autotune) |
| `examples/gpt2/profile.json` | Baseline ncu profile (103.9 ms attributed, 259 operators) |
| `examples/gpt2/profile_optimized.json` | Optimized ncu profile (174 operators) |
| `examples/gpt2/triage.json` | Bottleneck classification (top: tensor_core_idle on aten::mm) |
| `examples/gpt2/optimizations.json` | Ranked optimization proposals with hardware evidence |
| `examples/gpt2/validation_report.json` | 5-step validation results (all pass) |
| `examples/gpt2/comparison.md` | Detailed before/after hardware counter comparison |
| `examples/gpt2/profiler_output/ncu_reps/all_kernels.ncu-rep` | Baseline ncu report (224 MB) |
| `examples/gpt2/profiler_output/ncu_reps_optimized/all_kernels.ncu-rep` | Optimized ncu report (257 MB) |
