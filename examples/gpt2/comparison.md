# GPT-2 Small — Baseline vs Optimized Profile Comparison

**Workload:** GPT-2 small (117M), batch=4, seq_len=128  
**Hardware:** A100-SXM4-80GB  
**Capture:** `--warmup-iters 3 --measure-iters 10` (identical — no normalization needed)  
**Baseline:** FP32, `torch.compile(backend='inductor')`  
**Optimized:** BF16, `torch.compile(mode='max-autotune', fullgraph=True, backend='gpt2_backend')`

---

## Overall Speedup

| Metric | Baseline | Optimized | Change |
|--------|----------|-----------|--------|
| Total attributed time | 104.059 ms | 41.355 ms | −60.3% |
| Overall speedup | — | — | **2.52×** |
| Throughput (B=4 samples/sec) | 38.4 | 96.7 | +151.7% |

---

## Per-Operator Speedup (Top 10 by Baseline Time)

| Rank | Operator | Baseline (ms) | Baseline % | Optimized (ms) | Speedup | Resolved? |
|------|----------|---------------|------------|-----------------|---------|-----------|
| 1 | aten::mm_0 (FFN + proj GEMMs, 336 kernels) | 53.991 | 51.88% | (merged) | — | YES |
| 2 | aten::addmm_0 (biased linear, 456 kernels) | 16.301 | 15.67% | (merged) | — | YES |
| 1+2 | **Combined GEMM (aten::mm + aten::addmm)** | **70.292** | **67.55%** | **9.100** | **7.72×** | YES — TC active 0%→55–75% |
| 3 | aten::_efficient_attention_forward_0 | 3.313 | 3.18% | 1.800 | **1.84×** | PARTIAL — spills gone, occupancy still 6.2% |
| 4 | aten::view_0 / fused elementwise | 0.946 | 0.91% | ~0.850 | ~1.11× | N/A — already well-optimized |
| 5–10 | aten::mm [512,3072]×[3072,768] (per-layer) | 0.210 each | 0.20% each | ~0.021 each | ~10× | YES |

---

## Hardware Counter Evidence

### Combined GEMM (aten::mm/addmm baseline → aten::addmm optimized)

| Counter | Baseline | Optimized | Verdict |
|---------|----------|-----------|---------|
| Dominant kernel | `ampere_sgemm_32x32_sliced1x4_nn` | `ampere_bf16_s16816gemm_bf16_128x128_ldg8_relu_f2f_stages_64x3_nn` | BF16 TC path confirmed |
| tensor_core_active_pct | 0.0% | 55–75% | Tensor Cores engaged |
| sm_throughput_pct | 72.8% | 31–44% | 7.7× faster at lower absolute time |
| achieved_occupancy | 26.82% | 6.2–6.3% | Lower: large BF16 tiles use more regs/thread |
| registers_per_thread | 57 | 150–238 | TC HMMA encoding; no spills |
| local_memory_spills | 0 | 0 | No regression |
| DRAM throughput_pct | 3.18% | 9.9–12.8% | BF16 halved weight size |

### aten::_efficient_attention_forward

| Counter | Baseline | Optimized | Verdict |
|---------|----------|-----------|---------|
| Dominant kernel | `fmha_cutlassF_f32_aligned_64x64_rf_sm80` | `fmha_cutlassF_bf16_aligned_64x64_rf_sm80` | BF16 xformers dispatch (not native flash_fwd) |
| registers_per_thread | 168 | 128 | 24% reduction — spills eliminated |
| local_memory_spills | 24,468,480 bytes | 0 bytes | Fully resolved |
| achieved_occupancy | 6.24% | 6.20% | Still wave-starved (waves_per_sm ≈ 0.89) |
| sm_throughput_pct | 19.68% | 13.86% | Not SM-bound in either profile |
| Per-layer duration | ~30,678 ns | 17,024 ns | 1.80× per call |

---

## Pass Attribution

| Pass | Status | Attribution |
|------|--------|-------------|
| pass_bf16_cast (OPT-1) | APPLIED | Primary driver — routes all GEMMs from FP32 SIMT → BF16 Tensor Core path. ~1.9× contribution to overall 2.52× |
| pass_max_autotune (OPT-2) | APPLIED | Secondary GEMM gain — larger tile configs (128×128 vs 32×32 baseline), epilogue fusion. ~1.3× incremental |
| pass_sdpa_flash (OPT-3) | APPLIED (partial) | Attention 1.84× — spills eliminated (168→128 regs). Dispatched to xformers BF16, not native flash_fwd; occupancy gap remains |
| _pass_fuse_qkv (OPT-4) | NOT APPLIED (graceful) | HF c_attn already fuses Q/K/V at module level. No speedup, no regression |
| pass_tf32 (OPT-5) | APPLIED | Defense-in-depth; no incremental effect (all GEMMs on BF16 path) |

---

## Residual Optimization Opportunities

### 1. Flash Attention-2 native dispatch (medium priority)
- Achieved occupancy still 6.2% (predicted 30–40% with native `flash_fwd`)
- Cause: dispatch landed on `fmha_cutlassF_bf16` (xformers), not PyTorch native Flash Attention-2
- Fix: Replace xformers call with `F.scaled_dot_product_attention(q, k, v, is_causal=True)`
- Estimated gain: ~1% end-to-end → **2.59×** total

### 2. Small attention-projection GEMMs — wave starvation (low priority)
- [512,768]×[768,768] output projections (12 per pass): grid=[6,8,1], 48 blocks / 108 SMs → 0.44 waves
- `sm_throughput_pct=17.4%`, `tensor_core_active_pct=46%`
- Fix: Stream-K GEMM tiling, batched `torch.bmm`, or CUDA Graphs
- Estimated gain: ~1.5% end-to-end → **2.63×** total

### 3. Triton elementwise kernels — DRAM underutilization (low priority)
- GELU + LayerNorm: `dram_throughput_pct=13–16%`, `sm_throughput_pct=22–36%`
- Fix: `F.gelu(x, approximate='tanh')` for hardware GELU; verify persistent-reduction for LayerNorm
- Estimated gain: ~2–3% end-to-end → **2.70×** total

### Combined ceiling
| Scenario | Total Time | Speedup |
|----------|------------|---------|
| Current | 41.355 ms | 2.52× |
| + FlashAttention-2 native | ~40.2 ms | ~2.59× |
| + Projection wave fix | ~39.6 ms | ~2.63× |
| + GELU/LayerNorm tuning | ~38.5 ms | ~2.70× |
| **All three combined** | **~37.5 ms** | **~2.77×** |
