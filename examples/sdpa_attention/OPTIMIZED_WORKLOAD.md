# SDPA Attention Optimized Workload

## Overview

This workload applies four GPU-level optimizations to `SDPAAttentionBlock` — a
multi-head self-attention block using `F.scaled_dot_product_attention`, three
separate Q/K/V projections (`nn.Linear(512, 512, bias=False)` each), an output
projection, and pre/post LayerNorm.

The baseline profile reveals complete Tensor Core disengagement
(`smsp__pipe_tensor_cycles_active = 0.0%` across all 20 GEMM kernel launches),
FP32 SIMT dispatch on all matrix multiplications, and an sm80-compiled CUTLASS
attention kernel running under forward-compatibility on a Blackwell GPU. All
four optimizations target these root causes:

| ID | Type | Location | Confidence | Expected Impact |
|----|------|----------|------------|----------------|
| OPT-1 | BF16 dtype promotion | `get_model_and_input()` | High | ~38.7% total latency reduction |
| OPT-2 | QKV weight fusion | FX pass (manual per-rep) | High | ~6.6% additional reduction |
| OPT-3 | Flash Attention backend | `get_model_and_input()` | Medium | ~5.0% additional reduction |
| OPT-4 | Pre-transposed QKV weight | FX pass (manual per-rep) | Medium | ~0.3% additional reduction |

Combined expected impact: ~50% total latency reduction from the baseline ~0.60 ms
per forward pass (8 × 512 × 512, FP32 on NVIDIA RTX PRO 6000 Blackwell).

---

## Quick Start

```bash
# Syntax check
PYTHONPATH=/home/ubuntu/Profiler python3 -m py_compile \
    examples/sdpa_attention/sdpa_attention_optimized.py

# Run validation tests (no GPU compilation required for tests 1-3)
PYTHONPATH=/home/ubuntu/Profiler pytest \
    examples/sdpa_attention/test_sdpa_attention_optimized.py -v

# Quick forward pass sanity check
PYTHONPATH=/home/ubuntu/Profiler python3 \
    examples/sdpa_attention/sdpa_attention_optimized.py
# Expected output:
#   Output shape : torch.Size([8, 512, 512])
#   Output dtype : torch.bfloat16

# Profile — Phase 1 (correlation pass, no nsys)
PYTHONPATH=/home/ubuntu/Profiler python3 nvidia/scripts/run_workload.py \
    --workload examples/sdpa_attention/sdpa_attention_optimized.py \
    --compile-backend sdpa_attention_opt \
    --output-prefix profiler_output/sdpa_attention_opt \
    --inductor-debug-dir profiler_output/sdpa_attention_opt_inductor \
    --correlation-pass \
    --warmup-iters 2 --measure-iters 2

# Profile — Phase 2 (NVTX capture under nsys)
nsys profile --trace=cuda,nvtx \
    --output=profiler_output/sdpa_attention_opt \
    python3 nvidia/scripts/run_workload.py \
        --workload examples/sdpa_attention/sdpa_attention_optimized.py \
        --compile-backend sdpa_attention_opt \
        --output-prefix profiler_output/sdpa_attention_opt \
        --inductor-debug-dir profiler_output/sdpa_attention_opt_inductor \
        --warmup-iters 2 --measure-iters 2
```

**Cache compilation to avoid repeated overhead:**
```bash
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache_sdpa_opt
```

---

## Optimizations Table

| ID | Optimization | Target Operators | Mechanism | Expected Speedup |
|----|-------------|-----------------|-----------|-----------------|
| OPT-1 | BF16 dtype promotion | All 20 `aten::mm` launches (Q/K/V/O projections, ~66.6% of wall time), all 3 `aten::_efficient_attention_forward` launches (~18.4%) | `model.to(torch.bfloat16)` + `x.to(torch.bfloat16)` before `torch.compile`. Forces cuBLAS to BF16 Tensor Core dispatch (HMMA). Reduces register pressure from 210 to ~128-160 regs/thread, raising occupancy from 16.6% to ~25-40%. Eliminates 757,760 local_memory_spills per fmha launch. | ~38.7% total wall time reduction |
| OPT-2 | QKV weight fusion | `aten::mm` for Q, K, V projections (6 launches, 3 per block, each 128 blocks / 0.75 waves) | FX pass: detects 3 × `F.linear(x, W_*)` sharing the same post-LayerNorm input. Concatenates `W_q, W_k, W_v` into `W_qkv [1536, 512]`. Replaces with 1 × `F.linear(x, W_qkv)` + `chunk(3)`. Raises block count from 128 to ~384 (2.3 waves), eliminates 6 kernel launches. | ~6.6% additional reduction |
| OPT-3 | Flash Attention backend | `aten::_efficient_attention_forward` (3 launches, `fmha_cutlassF_f32_aligned_64x64_rf_sm80`) | `enable_flash_sdp(True)` + `enable_mem_efficient_sdp(False)` before `torch.compile`. After BF16 (OPT-1), SDPA dispatcher becomes eligible for Flash kernel. Replaces sm80-compiled CUTLASS binary (running under forward-compat on SM100 Blackwell) with a BF16-native Flash kernel. | ~5.0% additional reduction |
| OPT-4 | Pre-transposed QKV weight | The fused QKV GEMM (1 launch post-OPT-2, `x=[4096,512] × W_qkv_T=[512,1536]`) | FX pass: after QKV fusion, detects `F.linear(x, _fused_qkv_weight)` and replaces with `operator.matmul(x, _fused_qkv_weight_T)` where `_fused_qkv_weight_T=[512,1536]` is pre-computed. Converts NT GEMM to NN GEMM. Eliminates the implicit `aten.t()` in the Inductor graph. | ~0.3% additional reduction |

---

## Architecture

### Non-Graph Optimizations (`get_model_and_input()`)

Both OPT-1 and OPT-3 are applied in `get_model_and_input()` before `torch.compile`
is called. They are not expressible in the FX IR.

**OPT-1 — BF16 dtype promotion:**
```python
if next(model.parameters()).dtype != torch.bfloat16:
    model = model.to(torch.bfloat16)
if x.dtype != torch.bfloat16:
    x = x.to(torch.bfloat16)
```
Dynamo bakes tensor `dtype` into its trace specialization. Applying `.to(bfloat16)`
inside the backend (post-trace) would not change intermediate tensor dtypes. This
optimization must precede `torch.compile`.

**OPT-3 — Flash Attention backend:**
```python
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(False)
```
These are global SDPA dispatcher flags. After BF16 promotion, Flash becomes
eligible (it requires FP16 or BF16). Disabling the memory-efficient and math
backends forces the dispatcher to select Flash rather than falling back to the
sm80 CUTLASS xFormers path.

### FX Pass: `_pass_fuse_qkv` (OPT-2)

Classification: **Manual per-rep pass** — requires `register_buffer` to store
`W_qkv`, and needs actual tensor values from `partition_inputs` to compute the
concatenated weight. `replace_pattern` cannot do either.

**Detection:** Three `F.linear` nodes in the graph whose first argument (`args[0]`)
is the same node (the post-LayerNorm activation). Weight tensors are resolved from
the placeholder→tensor map built from `partition_inputs`.

**Transformation:**
1. Concatenate `W_q, W_k, W_v` → `W_qkv [1536, 512]`; register as buffer `_fused_qkv_weight`
2. Insert `get_attr("_fused_qkv_weight")` + `F.linear(x, W_qkv)` + `torch.chunk(3, dim=-1)` before the Q linear node
3. Replace uses of `q_lin, k_lin, v_lin` with `chunk[0], chunk[1], chunk[2]`
4. Erase the three original linear nodes
5. `graph.lint()` + `gm.recompile()`

### FX Pass: `_pass_pretranspose_fused_qkv` (OPT-4)

Classification: **Manual per-rep pass** — requires `register_buffer` for
`W_qkv_T`; replaces `F.linear` (which has an implicit transpose) with
`operator.matmul` using the pre-transposed buffer.

**Must run after OPT-2.** Detects `F.linear(x, get_attr("_fused_qkv_weight"))`
and replaces it with `operator.matmul(x, get_attr("_fused_qkv_weight_T"))`.

The pre-transposed buffer `W_qkv_T [512, 1536]` is computed as
`W_qkv.T.contiguous()` and stored as a named buffer. This gives cuBLAS a
contiguous B matrix in NN layout, eliminating the descriptor overhead of the
implicit transpose in NT GEMM mode.

### Dedup-Aware Backend (`sdpa_attention_opt`)

`SDPAAttentionBlock` has a single attention block — no repeated layer structure.
`UniqueSubgraphRegistry.build_partition_equivalence_map()` returns an empty map,
so the **flat compile path** is taken:

1. Build placeholder→tensor map from `example_inputs`
2. Apply `_pass_fuse_qkv` (OPT-2) to the full flat graph
3. Apply `_pass_pretranspose_fused_qkv` (OPT-4) to the full flat graph
4. Delegate to `compile_fx(gm, example_inputs)`

The dedup path (per-rep loop + `compile_fx` per unique rep) is present for
structural completeness and activates automatically if this backend is used with
a stacked-block variant of the model.

---

## Key Design Decisions

### OPT-1 as non-graph

Dynamo traces dtype as a static property baked into graph specialization. Changing
dtype after compilation triggers a full recompile. The only correct location for
BF16 promotion is `get_model_and_input()`, before `torch.compile` is called. An
FX pass that casts inputs to BF16 inside the backend would operate on an already-
traced graph and would not affect the dtype of intermediate tensors.

### OPT-2 as manual per-rep (not `replace_pattern`)

`replace_pattern` is purely structural — it cannot access actual tensor values.
QKV fusion requires reading `W_q, W_k, W_v` to compute `W_qkv = cat(...)` and
calling `gm.register_buffer()` to store the result. Both requirements make this
a manual per-rep pass.

### OPT-3 as non-graph (global SDPA flag)

`torch.backends.cuda.enable_flash_sdp()` is a global SDPA dispatcher flag, not
an FX node. The SDPA operation in the traced graph (`F.scaled_dot_product_attention`)
is a single opaque node; the backend choice is made at kernel dispatch time, not
at FX-graph construction time. The flag must be set before tracing begins so
Dynamo records the correct dispatch path.

### OPT-4 replaces F.linear with operator.matmul (not post-Inductor surgery)

At the pre-Inductor level (where `@register_backend` receives the graph), `F.linear`
is still a single `call_function` node — Inductor has not yet decomposed it into
`aten.t(get_attr) + aten.mm`. OPT-4 therefore operates at the pre-Inductor level
by replacing `F.linear(x, W_qkv)` with `operator.matmul(x, W_qkv_T)`. Inductor
sees a direct matmul with a contiguous buffer and does not introduce an `aten.t()`
node. This achieves the NN GEMM effect without requiring post-Inductor graph surgery.

### OPT-2 and OPT-4 ordering

OPT-4 depends on OPT-2: it detects `get_attr("_fused_qkv_weight")` which is
registered by OPT-2. If OPT-2 fails or finds no pattern, OPT-4 logs a warning and
returns gm unchanged. The ordering guarantee is enforced by calling `_pass_fuse_qkv`
before `_pass_pretranspose_fused_qkv` in both the flat and dedup paths.

---

## Troubleshooting

### `TypeError: 'module' object is not callable`

Wrong import for `compile_fx`. The file uses:
```python
from torch._inductor.compile_fx import compile_fx  # correct — imports the function
```
Not:
```python
from torch._inductor import compile_fx  # wrong — imports the module
```

### `_pass_fuse_qkv` reports "Weight tensors not resolved"

The QKV fusion pass builds a `placeholder→tensor` map from `partition_inputs`. If
this map is missing entries, the weight tensors cannot be resolved. Causes:
1. `partition_inputs` length does not match placeholder count — verify that
   `example_inputs` has the same number of elements as `placeholder` nodes in the graph.
2. The model has been modified after `get_model_and_input()` returns (e.g., extra
   `torch.compile` calls that change the graph structure before the backend runs).

### `_pass_fuse_qkv` reports "Pattern not found"

Three `F.linear` nodes must share the same `args[0]` node. If the baseline
model's forward modifies the input between Q/K/V calls, the shared-input condition
fails. Verify the forward method matches `SDPAAttentionBlockOpt` exactly.

### `_pass_pretranspose_fused_qkv` reports "_fused_qkv_weight not found"

OPT-4 depends on OPT-2 having registered `_fused_qkv_weight`. Check that:
1. `_pass_fuse_qkv` ran without error (check log output for `[_pass_fuse_qkv] Fused`).
2. The pass order in the backend is `_pass_fuse_qkv` before `_pass_pretranspose_fused_qkv`.

### `graph.lint()` failure after graph mutation

If `lint()` raises, the graph has dangling references. Common cause: a node was
erased before all its users were replaced. The pattern is:
```python
node.replace_all_uses_with(new_node)  # FIRST — redirect all users
gm.graph.erase_node(node)             # THEN — safe to erase
```
Both passes in this file follow this order.

### Flash SDP dispatches sm80 kernel despite OPT-3

If `enable_flash_sdp(True)` still dispatches `fmha_cutlassF_f32_*` after BF16
promotion, verify that:
1. BF16 promotion (OPT-1) actually ran — `next(model.parameters()).dtype` must
   be `torch.bfloat16` before `torch.compile` is called.
2. PyTorch 2.11+cu128 ships a Blackwell-aware Flash kernel. If not, install
   `flash-attn >= 2.6.0` separately (ships Hopper/Blackwell CUDA kernels) and
   call `F.scaled_dot_product_attention` after `torch.compile`.
3. Check `torch.backends.cuda.flash_sdp_enabled()` returns `True` at the point
   `torch.compile` is invoked.

### BF16 model produces NaN outputs

BF16 has ~3 decimal digits of precision (vs ~7 for FP32). The most common cause
of NaN in attention models after BF16 promotion is overflow in softmax when
`Q @ K^T` scores become large. Mitigations:
1. Verify `F.scaled_dot_product_attention` is used (it applies `1/sqrt(head_dim)`
   scaling internally) — do not manually scale QK before SDPA.
2. Check `torch.backends.cuda.matmul.allow_tf32 = True` is set.
3. If NaN persists, try `torch.backends.cuda.enable_math_sdp(True)` to allow
   fallback to a more numerically stable math path at reduced performance.

---

## Future Work

### OPT-4 via post-Inductor FX surgery (alternative implementation)

The current OPT-4 implementation replaces `F.linear` with `operator.matmul` at
the pre-Inductor level, which is clean and effective. An alternative is to apply
the `aten.t(get_attr) → get_attr(pre-transposed)` pattern at the post-Inductor
level (as done in `gpt2_optimized.py`'s `_pass_pretranspose`). This would require
the backend to receive the post-Inductor graph, which is not the current `@register_backend`
contract. Infrastructure needed: a custom Inductor lowering hook or a post-lowering
callback.

### Stacked attention blocks (multi-layer variant)

The dedup path in `sdpa_attention_opt` is fully implemented and activates
automatically if `UniqueSubgraphRegistry` detects repeated partitions. Testing
with a 12-layer stacked variant (e.g., `nn.ModuleList` of `SDPAAttentionBlockOpt`)
would exercise the dedup compile path and validate cross-layer compiled callable sharing.
Infrastructure needed: no new infrastructure — the dedup path is already present.

### LayerNorm-Linear fusion

Both pre-norm (`ln_pre → q/k/v_proj`) and post-norm (`out_proj → ln_post`) patterns
appear in `SDPAAttentionBlockOpt`. Fusing LayerNorm into the adjacent Linear would
eliminate one intermediate tensor write per block.
Infrastructure needed: custom Triton kernel (e.g., liger-kernel `LN-MM` fused op).

### OPT-3 verification with ncu

After profiling with the optimized workload, verify that `fmha_cutlassF_f32_aligned_64x64_rf_sm80`
no longer appears in ncu output and has been replaced by a BF16 Flash kernel
(e.g., `flash_fwd_kernel_*` from flash-attn, or `cutlass_80_*_bf16*` from
PyTorch's built-in Flash implementation). Check `smsp__pipe_tensor_cycles_active`
on the new kernel to confirm Tensor Core engagement.
