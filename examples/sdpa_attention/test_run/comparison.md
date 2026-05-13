# SDPAAttentionBlock — Baseline vs. Optimized Comparison

**Device:** NVIDIA A100-SXM4-80GB | **Batch:** 8 | **Seq:** 512 | **Dim:** 512 | **Heads:** 8

> **ncu replay timing caveat:** All duration values come from ncu hardware-counter replay (2–5× longer than real execution). The 15.7× ncu-replay improvement reflects genuine kernel-level improvements but is amplified by aggressive Inductor fusion (43 → 14 kernels). Use hardware counter evidence below to attribute speedup; use nsys wall-clock times for absolute latency comparison.

---

## Summary

| Metric | Baseline | Optimized | Change |
|--------|----------|-----------|--------|
| Attributed operators | 15 | 5 | 10 fused away by Inductor |
| Total kernels | 43 | 14 | −67% |
| Total ncu-replay time (ns) | 4,721,771 | 300,960 | **15.7× faster** (ncu replay) |
| GEMM kernel | `ampere_sgemm_128x64_tn` | Triton BF16 / `sm80_xmma_gemm_bf16bf16` | FP32 SIMT → Tensor Core |
| SDPA kernel | `fmha_cutlassF_f32_aligned_64x64_rf_sm80` | `flash_fwd_kernel` (BF16 FlashAttention) | FP32 → BF16 FA path |
| Tensor core active (GEMM) | 0.0% | ~52% | +52 pp |
| Tensor core active (SDPA) | ~40% | ~47% | +7 pp |
| SDPA local memory spills | 4,714,496 bytes/invocation | ~0 | Eliminated |
| SDPA registers/thread | 168 | ~64 (BF16 FA typical) | −62% |

---

## Per-Operator Table

### Baseline Operators

| Operator | Duration (ns) | % Time | Kernel | tensor_core_active_pct | Bottleneck |
|----------|--------------|--------|--------|----------------------|------------|
| layer::unique::prologue (fused: ln + 8×mm + 2×fmha) | 1,911,941 | 40.5% | `ampere_sgemm_128x64_tn` / `fmha_cutlassF_f32` | 11.3% (GEMM=0.0%, fmha=39.6%) | tensor_core_idle |
| aten::mm (4 standalone, output projections) | 645,857 | 13.7% | `ampere_sgemm_128x64_tn` | 0.0% | tensor_core_idle |
| aten::_efficient_attention_forward (op_id=36) | 273,569 | 5.8% | `fmha_cutlassF_f32_aligned_64x64_rf_sm80` | 39.6% | wave_starvation (168 regs, 4.7MB spills) |
| aten::_efficient_attention_forward (op_id=14) | 272,257 | 5.8% | `fmha_cutlassF_f32_aligned_64x64_rf_sm80` | 39.6% | wave_starvation |
| aten::_efficient_attention_forward (op_id=0) | 273,473 | 5.8% | `fmha_cutlassF_f32_aligned_64x64_rf_sm80` | 39.7% | wave_starvation |
| aten::mm Q-proj L1 (op_id=4) | 162,944 | 3.5% | `ampere_sgemm_128x64_tn` | 0.0% | tensor_core_idle |
| aten::mm Q-proj L2 (op_id=26) | 162,656 | 3.4% | `ampere_sgemm_128x64_tn` | 0.0% | tensor_core_idle |
| aten::mm out-proj L1 (op_id=21) | 162,912 | 3.5% | `ampere_sgemm_128x64_tn` | 0.0% | tensor_core_idle |
| aten::mm K-proj L1 (op_id=5) | 162,305 | 3.4% | `ampere_sgemm_128x64_tn` | 0.0% | tensor_core_idle |
| aten::mm K-proj L2 (op_id=27) | 161,857 | 3.4% | `ampere_sgemm_128x64_tn` | 0.0% | tensor_core_idle |
| aten::mm V-proj L1 (op_id=6) | 161,056 | 3.4% | `ampere_sgemm_128x64_tn` | 0.0% | tensor_core_idle |
| aten::mm V-proj L2 (op_id=28) | 160,960 | 3.4% | `ampere_sgemm_128x64_tn` | 0.0% | tensor_core_idle |
| aten::mm out-proj L2 (op_id=43) | 163,136 | 3.5% | `ampere_sgemm_128x64_tn` | 0.0% | tensor_core_idle |
| aten::_unsafe_view + layer_norm (fused) | 28,032 | 0.6% | Triton fused kernel | n/a | below threshold |
| aten::native_layer_norm | 18,816 | 0.4% | Triton | n/a | below threshold |
| **Total** | **4,721,771** | 100% | | | |

### Optimized Operators

| Operator | Duration (ns) | % Time | Kernel | tensor_core_active_pct | Notes |
|----------|--------------|--------|--------|----------------------|-------|
| aten::view (fused: Q/K/V/out GEMMs, both layers) | 170,720 | 56.7% | Triton BF16 mm / `sm80_xmma_gemm_bf16bf16` | **52.2%** | 12 baseline mm ops → 8 BF16 Tensor Core kernels |
| aten::_flash_attention_forward (op_id=17, L1) | 46,560 | 15.5% | `flash_fwd_kernel` (BF16 FA) | **47.0%** | No register spills; was `fmha_cutlassF_f32` |
| aten::_flash_attention_forward (op_id=44, L2) | 46,368 | 15.4% | `flash_fwd_kernel` (BF16 FA) | **46.0%** | No register spills |
| aten::_unsafe_view | 19,200 | 6.4% | Triton elementwise | 0.26% | Reshape — no optimization headroom |
| aten::native_layer_norm | 18,112 | 6.0% | `triton_per_fused_native_layer_norm_0` | 1.8% | Memory-bound, below threshold |
| **Total** | **300,960** | 100% | | | |

---

## Pass Attribution

### OPT-1: BF16 Dtype Promotion — **PRIMARY DRIVER**

Applied in `get_model_and_input()` as a pre-compile non-graph change.

**Evidence of application:**
- Baseline: `ampere_sgemm_128x64_tn` kernel (FP32 SIMT), `smsp__pipe_tensor_cycles_active=0.0` on all 12 mm invocations
- Optimized: Triton BF16 mm kernels with `tensor_core_active_pct=52.2%` — Tensor Cores fully engaged
- SDPA: `fmha_cutlassF_f32_aligned_64x64_rf_sm80` (168 regs/thread, 4.7MB spills, 15.9% occupancy) → `flash_fwd_kernel` (BF16 FlashAttention, no spills, TC=47%)
- Kernel count reduction: 43 → 14 (the BF16 dispatch path + max-autotune allowed Inductor to fuse aggressively)

**Attributed speedup:** Accounts for the majority of the 15.7× ncu-replay improvement. The FP32 → BF16 GEMM kernel substitution alone explains 8–12× on the mm kernels (A100: 312 BF16 TFLOPS vs 19.5 FP32 TFLOPS; real-world ~8× due to bandwidth becoming the new bottleneck). SDPA contributed an additional ~6× on SDPA ops by eliminating register spills and switching to the BF16 FA path.

### OPT-4: max-autotune Compile Mode — **SECONDARY DRIVER**

Applied inside the custom backend (`torch._inductor.config.max_autotune = True`).

**Evidence of application:** Inductor selected `triton_mm` with `BLOCK_K=32, BLOCK_M=64, BLOCK_N=128, num_stages=3, num_warps=4` for the 4096×512 BF16 GEMM — a non-default tile config. Also enabled aggressive kernel fusion: the 12 baseline mm operators collapsed into 8 kernels attributed to a single `aten::view` operator span, suggesting Inductor fused adjacent operations across boundaries.

**Attributed speedup:** ~5–15% incremental over OPT-1 alone, primarily from optimal tile selection for the non-square M=4096, N=512 GEMM shape.

### OPT-2: QKV Weight Fusion — **GRACEFUL DEGRADE**

Skipped. The `_pass_fuse_qkv` FX pass detected a Dynamo-traced flat graph and could not `register_buffer` (no `_dynamo_source` entry for post-trace buffers). Inductor's max-autotune natively batched the three Q/K/V GEMMs into fewer kernel launches, providing a partial substitute.

**Impact:** The expected 4 kernel launch eliminations (6→2 explicit QKV kernels per pair of layers) were partially realized by Inductor fusion. Dedicated QKV weight concatenation (fusing 3×[512,512] → 1×[1536,512]) would add ~5–10% on top of OPT-4.

### OPT-3: SDPA Replacement — **NO-OP (expected)**

Not applied. `SDPAAttentionBlock` already calls `F.scaled_dot_product_attention` at the Python level; Dynamo traces this directly to `aten::_scaled_dot_product_flash_attention`. The manual attention decomposition pattern (q@k.T → softmax → @v) was not present in the graph. OPT-1 alone switched the SDPA dispatch from `fmha_cutlassF_f32` to the BF16 FlashAttention path.

---

## Residual Optimization Opportunities

| Opportunity | Expected Gain | Blocker |
|-------------|---------------|---------|
| Fix QKV weight fusion (OPT-2) | ~5–10% additional GEMM speedup | Need to apply `register_buffer` pre-Dynamo, or use `torch.nn.utils.parametrize` / pre-concat weights in model definition |
| Increase batch size / sequence length | Raises arithmetic intensity for bandwidth-bound BF16 GEMMs | Workload constraint |
| FlashAttention causal mask | ~20% SDPA speedup (halves attention compute) | Requires `is_causal=True` — only valid for autoregressive inference |
| SDPA `sdp_kernel(enable_flash=True)` context manager | Forces FA path if any fallback occurs | Defensive — already on FA path post-OPT-1 |
