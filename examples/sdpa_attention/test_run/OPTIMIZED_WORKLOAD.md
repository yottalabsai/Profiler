# Optimized SDPAAttentionBlock Workload

## Overview

`sdpa_attention_optimized.py` implements four operator-level optimizations via a
custom `torch.compile()` backend called `sdpa_attention_opt`.  The root cause
identified in the profile is that **all 12 GEMM kernels dispatch to FP32 SIMT
(`ampere_sgemm_128x64_tn`) with `smsp__pipe_tensor_cycles_active = 0.0` on every
invocation** — Tensor Cores are completely idle.  OPT-1 (BF16 promotion) addresses
this directly; OPT-2 through OPT-4 compound the gain.

## Quick Start

```bash
# Smoke test — uncompiled forward pass (no torch.compile required)
python examples/sdpa_attention/test_run/sdpa_attention_optimized.py

# Validation tests
python examples/sdpa_attention/test_run/test_sdpa_attention_optimized.py

# Profile with optimized backend (Phase 1: correlation pass)
python nvidia/scripts/run_workload.py \
    --workload examples/sdpa_attention/test_run/sdpa_attention_optimized.py \
    --compile-backend sdpa_attention_opt \
    --output-prefix examples/sdpa_attention/test_run/profile_optimized \
    --correlation-pass

# Profile with optimized backend (Phase 2: nsys capture)
nsys profile --trace=cuda,nvtx \
    --output=examples/sdpa_attention/test_run/profile_optimized \
    python nvidia/scripts/run_workload.py \
        --workload examples/sdpa_attention/test_run/sdpa_attention_optimized.py \
        --compile-backend sdpa_attention_opt \
        --output-prefix examples/sdpa_attention/test_run/profile_optimized
```

## Optimizations Table

| ID | Name | Type | Target Ops | Expected Impact |
|----|------|------|-----------|----------------|
| OPT-1 | BF16 Dtype Promotion | Non-graph (pre-compile) | All 12 aten::mm, 3x SDPA | 35-50% total latency reduction; activates Tensor Core path |
| OPT-2 | QKV Weight Fusion | FX pass (manual per-rep) | Q/K/V mm per layer (6 kernels -> 2) | 5-15% GEMM time reduction; 3x arithmetic intensity |
| OPT-3 | SDPA Replacement | FX pass (manual per-rep) | Manual attn chain -> F.sdpa | 30-50% SDPA op improvement (5-9% total); no-op if already using F.sdpa |
| OPT-4 | max-autotune | Compile mode | All GEMM shapes | 5-15% incremental over OPT-1 for non-square GEMMs |

**Expected aggregate improvement: 40-60% total wall-time reduction on A100.**

## Architecture

### Backend Registration

The file registers `sdpa_attention_opt` via `@register_backend` from
`torch._dynamo`.  Importing the file triggers registration.  Pass
`--compile-backend sdpa_attention_opt` to `run_workload.py` to activate it.

### Dedup-Aware Backend Structure

The backend uses `UniqueSubgraphRegistry` to split the FX graph by detected layer
structure.  If repeated layers are found (e.g. two identical attention blocks),
FX passes are applied only to unique structural representatives and the compiled
callable is shared with structural duplicates — avoiding redundant compilation.

**No repeated layers detected** (single SDPAAttentionBlock): flat compile path.
FX passes run on the full graph and the result is passed to `compile_fx` directly.

**Repeated layers detected** (stacked blocks): per-rep pass loop, then
`compile_fx` per unique rep, callable shared with duplicates.

### FX Pass: OPT-2 — QKV Weight Fusion

Detects three `F.linear(x, W_q/k/v)` nodes sharing the same input `x`,
concatenates the weight tensors into `W_qkv = cat([W_q, W_k, W_v], dim=0)`,
registers it as a buffer, and replaces the three linear calls with one
`F.linear(x, W_qkv)` followed by `torch.chunk(3, dim=-1)`.

Weight tensor values are resolved by matching placeholder nodes to
`partition_inputs` (captured by running the split graph once before passes run).

### FX Pass: OPT-3 — SDPA Replacement

Anchors on the final `operator.matmul` of the manual attention pattern, walks
backwards through `softmax -> mul -> matmul -> transpose`, extracts the
pre-transposed K, and replaces the full chain with
`F.scaled_dot_product_attention(q, k, v, is_causal=False)`.

**Expected behavior for this workload:** The model already calls
`F.scaled_dot_product_attention` at the Python level; Dynamo traces it directly
to `aten::_scaled_dot_product_efficient_attention`.  The manual attention pattern
(`operator.matmul -> softmax -> operator.matmul`) will not be present in the FX
graph, so this pass will log a warning and degrade gracefully.  OPT-1 (BF16 cast)
alone handles the SDPA dispatch improvement in this case.

### OPT-4 — max-autotune

Activates `torch._inductor.config.max_autotune = True` before calling
`compile_fx`.  This triggers exhaustive tile-size and algorithm selection for each
unique GEMM shape (`M=4096, N=512` for Q/K/V/out; `M=4096, N=1536` after QKV
fusion).  The config flag is restored to its original value after compilation.

## Key Design Decisions

### BF16 Applied in get_model_and_input(), Not the Backend

Dtype is a tensor property, not an FX IR node.  Dynamo traces at the dtype
present at compile time; applying dtype inside the backend or after
`torch.compile()` has no effect on cuBLAS kernel selection.  BF16 must be set
before `torch.compile()` is called.

### Do NOT Use torch.fx.replace_pattern for OPT-2

`replace_pattern` operates purely structurally and cannot access actual tensor
values.  Concatenating `W_qkv` requires reading `W_q`, `W_k`, `W_v` from
`partition_inputs` and calling `register_buffer` on the `GraphModule`.  Only the
manual per-rep approach supports this.

### OPT-3 Is a Graceful No-Op for This Model

The profiler shows `aten::_efficient_attention_forward` and
`aten::_scaled_dot_product_efficient_attention` in the fused-with fields — this
confirms Dynamo has already traced the Python-level `F.scaled_dot_product_attention`
call directly.  The SDPA FX pass is included for completeness and to catch any
future decomposition path, but it is expected to log a warning and return the
graph unchanged for this workload.

### max-autotune Scope

`torch._inductor.config.max_autotune` is set only for the duration of the
`compile_fx` call and restored immediately after.  This avoids leaking autotune
mode into other compilation units that may be active in the same process.

## Troubleshooting

### compile_fx ImportError

```
# ALWAYS (imports the callable function)
from torch._inductor.compile_fx import compile_fx

# NEVER — imports the module -> TypeError: 'module' object is not callable
from torch._inductor import compile_fx
```

### Graph lint failure after QKV fusion

Root cause: a node was erased before its uses were replaced.  Fix: always call
`node.replace_all_uses_with(new_node)` BEFORE `gm.graph.erase_node(node)`.  The
implementation follows this ordering: replace q_lin/k_lin/v_lin uses, then erase
in a separate loop.

### Shape mismatch after QKV fusion

The guard `W_q.shape[1] == W_k.shape[1] == W_v.shape[1]` rejects fusion when
the K (input) dimensions differ (e.g. multi-query attention).  Check logs for
`[pass_fuse_qkv] Weight K-dims differ`.

### Weights not in partition_inputs

Log message: `[pass_fuse_qkv] Weight tensors not found in partition inputs`.
This means the weight nodes in the FX graph are not placeholder nodes — Inductor
may have already lifted them as constants via a different path.  The pass
degrades gracefully and skips fusion.

### SDPA pass not applied (expected)

Log message: `[pass_replace_sdpa] Manual attention pattern not found — pass not applied`.
This is the expected behavior for this workload.  OPT-1 (BF16) alone will
dispatch SDPA to the BF16 FlashAttention path.  If after OPT-1 re-profiling
still shows `fmha_cutlassF_f32`, explicitly enable Flash attention:

```python
with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    out = model(x)
```

### BF16 precision concern

`torch.bfloat16` has the same dynamic range as `float32` (8-bit exponent vs 8
bits), so attention logit overflow is not a risk.  BF16 is preferred over
`float16` on A100 for this reason — no loss scaling required.

## Future Work

### OPT-3 Full SDPA Replacement for Decomposed Graphs

If a future workload uses manual attention (e.g. `q @ k.T / math.sqrt(d)` then
softmax then `@ v`), `_pass_replace_sdpa` will apply.  To verify: run
`torch.fx.symbolic_trace(model)` and inspect the graph for `operator.matmul`
nodes before and after a `torch.softmax` node.

### LayerNorm-Linear Fusion

The stub `_pass_fuse_ln_linear_stub` (from `fx-patterns.md`) detects the
`F.layer_norm -> F.linear` chain present in this model (`ln_pre` into Q/K/V
projections, `ln_post` into output).  Full implementation requires:

1. A custom Triton kernel that keeps normalized rows in registers before issuing
   the GEMM (eliminates a full HBM round-trip per layer)
2. Registration via `torch.library` to make the custom op torch.compile-compatible
3. Expected gain: 15-30% latency reduction per LN-linear pair

### Quantization

The same backend can apply INT8 weight quantization (`torch.ao.quantization`) or
FP8 (via `transformer_engine`) after BF16 promotion.  Expected gain: additional
2x compute throughput on H100/Hopper (FP8 Tensor Cores).
