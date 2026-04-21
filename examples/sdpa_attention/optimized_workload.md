# SDPA Attention — Optimized Workload

## Overview

`sdpa_attention_optimized.py` implements a custom `torch.compile()` backend (`transformer_opt`) that applies five operator-level optimizations derived from profiling data in `optimizations.json`. The baseline model is `SDPAAttentionBlock` from `sdpa_attention.py` — a multi-head self-attention block using `F.scaled_dot_product_attention`.

Total estimated latency reduction: **55–70%** (dominated by OPT-001 tensor core activation on `aten::mm`).

---

## Quick Start

```bash
# Smoke test (uncompiled + compiled forward pass)
python sdpa_attention_optimized.py

# Verification tests
python test_sdpa_attention_optimized.py
# or: pytest test_sdpa_attention_optimized.py -v

# Profile optimized workload
operator-profiler profile sdpa_attention_optimized.py \
    --model-name SDPAAttentionOpt \
    --compile-mode transformer_opt \
    --output runs/sdpa_attention_opt

# Map kernels to operators
operator-profiler map runs/sdpa_attention_opt.manifest.json \
    --script scripts/run_workload.py \
    --ncu-sudo \
    --script-args --workload sdpa_attention_optimized.py \
                  --compile-backend transformer_opt

# Side-by-side comparison (requires two profile runs)
operator-profiler compare \
    runs/sdpa_attention_profile.json \
    runs/sdpa_attention_opt_profile.json
```

---

## Optimizations Summary

| ID | Operators | Bottleneck | Transformation | Expected Impact | Confidence |
|----|-----------|-----------|----------------|-----------------|------------|
| OPT-001 | `aten::mm` ×40, shape [4096,512]×[512,512] | 0% tensor core utilization; cuBLAS dispatches scalar SGEMM (Kernel2) because `allow_tf32=False`; 16.6% occupancy | `allow_tf32=True` globally + BF16 cast in `get_model_and_input()` | **50–60% reduction** on mm group (66.6% of runtime); occupancy 16.6% → 40–50% | HIGH |
| OPT-002 | `aten::mm` ×40, all serialized independent launches | 40 serialized kernel launches; full dispatch overhead per call; marginal tile alignment at N=512 | Horizontal QKV GEMM fusion: 3×`mm([4096,512],[512,512])` → 1×`mm([4096,512],[512,1536])` + `chunk(3)` | **10–20% reduction** on mm group (6–13% overall); launches 40 → ~13 | MEDIUM |
| OPT-003 | `aten::_efficient_attention_forward` ×10, shape [8,512,8,64] | FP32 FMHA path; 168 regs/thread; 7.57M local-memory spill accesses; 14.1% occupancy | Replace with `F.scaled_dot_product_attention` + BF16 Q/K/V cast → FlashAttention-2 (96 regs/thread, zero spills) | **30–40% reduction** on attention kernels (9–12% overall); O(N²)→O(N) memory | HIGH |
| OPT-004 | `aten::layer_norm` ×10, `triton_per_fused_native_layer_norm_0`, grid=[2048,1,1] | Single-warp CTA (32 threads); 4.79% SM throughput; IPC=0.05; every access hits DRAM | **Stub**: detection + recommendation for `max-autotune` or custom Triton kernel with BLOCK_SIZE=512 | **15–25% reduction** on layer_norm (0.2–0.3% overall) | MEDIUM |
| OPT-005 | `aten::add + aten::layer_norm` (fused) | Memory-bound at 68.2% DRAM throughput; benefits from BF16 propagation | **No-op monitor**: Inductor automatically propagates BF16 dtype after OPT-001; halves DRAM bytes | **40–50% reduction** on this kernel after OPT-001 (0.6% overall) | MEDIUM |

---

## Architecture

### Custom Backend: `transformer_opt`

```
torch.compile(model, backend="transformer_opt")
       │
       ▼
transformer_opt(gm: fx.GraphModule, example_inputs)
       │
       ├── pass_replace_sdpa()          ← OPT-003 (HIGH)
       │     Replace _efficient_attention_forward → SDPA + BF16 casts
       │
       ├── pass_fuse_qkv()              ← OPT-002 (MEDIUM)
       │     Detect 3×mm(x, W_i) triplets → single mm(x, W_fused) + chunk
       │
       ├── pass_retile_layernorm()      ← OPT-004 (MEDIUM / STUB)
       │     Detect layer_norm nodes, log recommendation
       │
       ├── pass_monitor_dtype_inheritance() ← OPT-005 (no-op)
       │     Detect add+layer_norm patterns, log BF16 inheritance status
       │
       └── compile_fx(gm, example_inputs)  ← delegate to Inductor
```

### Non-Graph Optimizations (`get_model_and_input`)

OPT-001 is applied outside the FX graph in two ways:

1. **Module-level side effect**: `torch.backends.cuda.matmul.allow_tf32 = True` is set at import time. This routes all FP32 `aten::mm` nodes to TF32 tensor core tiles without any graph surgery.

2. **`get_model_and_input()` dtype cast**: The function inspects `next(model.parameters()).dtype` and applies `model.to(torch.bfloat16)` + `x.to(torch.bfloat16)` only if the baseline model is not already BF16. This is idempotent and safe to call repeatedly.

---

## Why a Custom Backend?

A custom `@register_backend` function operating at the Aten IR level is:

- **Model-agnostic**: passes pattern-match on `aten::` ops, not `nn.Module` names
- **Composable**: each pass is an isolated function; add/remove independently
- **Observable**: all transformations logged at INFO level via Python `logging`
- **Robust**: every pass wraps its body in `try-except`; graph is returned unchanged on failure
- **Inductor-compatible**: delegates to `compile_fx` after passes, so all of Inductor's own optimizations (fusion, tiling, autotuning) still apply

---

## Key Design Decisions

### BF16 Outside the Graph (OPT-001)

Dtype is a property of tensors, not a graph operation. Casting inside an FX pass would require inserting `convert_element_type` nodes at every parameter boundary, which is brittle and conflicts with Inductor's own dtype propagation. Instead:

- `allow_tf32=True` is the zero-edit path: no graph changes, cuBLAS picks TF32 automatically
- `model.to(bfloat16)` is applied in `get_model_and_input()` so the entire model runs in BF16, and Inductor generates BF16 kernels throughout

### OPT-003 Applied Before OPT-002

The `pass_replace_sdpa` pass (OPT-003) runs before `pass_fuse_qkv` (OPT-002) because the QKV fusion pass pattern-matches on `mm` nodes. If attention replacement restructures Q/K/V first, the subsequent fusion pass sees the correct input topology.

### Defensive Pattern Matching (OPT-002)

The QKV fusion pass only fuses groups of **exactly 3** `mm` nodes sharing the same input, and only when their weight nodes are `get_attr` (parameter) nodes. This avoids incorrectly fusing non-QKV weight matrices. Groups of 2 or 4+ are logged and skipped.

### OPT-004 as a Stub

Full LayerNorm retiling requires a custom `@triton.jit` kernel registered as a `torch.library` op. Implementing that kernel inline would make this file non-portable. The pass detects the pattern and logs an actionable recommendation (`max-autotune` or custom Triton). The Triton kernel is straightforward to add as a follow-up (see Future Work).

### OPT-005 as a Monitor

The `add+layer_norm` fusion is already handled by Inductor. No action is needed beyond ensuring OPT-001 BF16 changes propagate. The pass exists to confirm that propagation occurred and to surface the fused node in logs.

---

## Comparison Against Baseline

```bash
# Profile baseline
operator-profiler profile sdpa_attention.py \
    --model-name SDPAAttention --compile-mode inductor \
    --output runs/sdpa_baseline

# Profile optimized
operator-profiler profile sdpa_attention_optimized.py \
    --model-name SDPAAttentionOpt --compile-mode transformer_opt \
    --output runs/sdpa_opt

# Compare
operator-profiler compare \
    runs/sdpa_baseline_profile.json \
    runs/sdpa_opt_profile.json
```

---

## Verification Checklist

After profiling, check the following in the resulting profile:

- [ ] **OPT-001**: `aten::mm` kernels show non-zero Tensor Core utilization (BF16 HMMA path active); `sm_throughput_pct` increases from 36.3% toward 70%+
- [ ] **OPT-001**: Achieved occupancy on mm kernels increases from 16.6% toward 40–50%
- [ ] **OPT-002**: Total `aten::mm` kernel count decreases from 40 toward ~13–14 per forward pass
- [ ] **OPT-002**: At least one `mm` kernel shows N-dim ≥1536 (fused QKV shape)
- [ ] **OPT-003**: `_efficient_attention_forward` kernel no longer present; `flash_fwd` or `scaled_dot_product_attention` appears instead
- [ ] **OPT-003**: Local memory spill accesses drop from 7.57M to ~0 for attention kernels
- [ ] **OPT-003**: Attention kernel register count drops from 168 to ~96/thread
- [ ] **OPT-004**: `triton_per_fused_native_layer_norm_0` block size shows 512 threads/CTA if Triton stub is replaced
- [ ] **OPT-005**: `triton_per_fused__unsafe_view_add_native_layer_norm_1` shows BF16 dtype in kernel signature

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `TypeError: 'module' object is not callable` at compile time | `from torch._inductor import compile_fx` imports the module | Use `from torch._inductor.compile_fx import compile_fx` (already correct in this file) |
| OPT-002: "no QKV triplet pattern found" | Baseline already fuses QKV (Inductor may pre-fuse) | Expected; check if mm count is already low in baseline profile |
| OPT-003: "_efficient_attention_forward not found" | `sdpa_attention.py` baseline already uses `F.scaled_dot_product_attention` | Expected for this workload; OPT-003 is more relevant for `transformer_block.py` using xformers |
| Large numerical diff (>0.1) after BF16 cast | BF16 has 3× lower precision than FP32 | Validate on your task metric; consider keeping weights FP32 with BF16 activations only |
| `allow_tf32` has no effect | PyTorch built without cuBLAS TF32 support, or running on Volta (sm70) | Verify with `torch.cuda.get_device_capability()` — TF32 requires sm80+ |
| `transformer_opt` not in `torch._dynamo.list_backends()` | Module not imported before compile call | Ensure `import sdpa_attention_optimized` runs before `torch.compile(backend='transformer_opt')` |

---

## Future Work

1. **OPT-004 Triton kernel**: Implement `@triton.jit layer_norm_kernel` with `BLOCK_SIZE=512`, register via `torch.library.custom_op`, and replace the stub detection with a full graph substitution pass.

2. **OPT-002 generalization**: Extend QKV fusion to handle groups of 2 (K+V or Q+KV patterns), and handle `linear` (addmm) nodes in addition to bare `mm` nodes.

3. **OPT-001 graph-level BF16**: Add an FX pass that inserts `convert_element_type` around `mm` inputs/outputs for mixed-precision workflows where full model BF16 is not acceptable.

4. **sm120 / Blackwell recompile**: The `_sm80` suffix in `fmha_cutlassF_f32_aligned_64x64_rf_sm80` indicates xformers was compiled for Ampere. If running on RTX PRO 6000 (sm120), recompile xformers and PyTorch extensions targeting `sm120` to avoid the sm80 compat fallback.

5. **Benchmarking harness**: Add a `benchmark()` function in `sdpa_attention_optimized.py` that runs `torch.utils.benchmark.Timer` against the baseline and prints a side-by-side latency table.