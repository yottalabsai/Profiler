# EmbeddingProjection — Optimized Workload

## Overview

This workload applies four operator-level optimizations to the baseline
`EmbeddingProjection` model (token embedding + LayerNorm + two-layer FFN + logit
projection). The optimizations are derived from ncu profiling on an NVIDIA A100
SXM4-80GB and address a single dominant bottleneck: every GEMM kernel dispatches to
`ampere_sgemm_128x64_tn` (FP32 SIMT, 0% Tensor Core utilization), accounting for
~99.7% of attributed wall time.

| # | ID | Confidence | Type | Target Operator | Expected Reduction |
|---|-----|------------|------|-----------------|--------------------|
| 1 | OPT-1 | High | Non-graph | All GEMMs (aten::mm, aten::addmm) | 50–80% total |
| 2 | OPT-2 | Medium | Stub | aten::mm logit projection (multi-call) | 6% (multi-call variant) |
| 3 | OPT-3 | Medium | FX pass | aten::mm, aten::addmm (TN→NN mode) | 2–8% additional |
| 4 | OPT-4 | Medium | Compile flag | All GEMM operators | 3–15% additional |

---

## Quick Start

```bash
# Syntax check
python -m py_compile examples/embedding_projection/embedding_projection_optimized.py

# Uncompiled smoke test
python -m pytest examples/embedding_projection/test_embedding_projection_optimized.py -v

# Compiled smoke test (1 warmup, 1 measure iteration)
PYTHONPATH=/root/Profiler python nvidia/scripts/run_workload.py \
    --workload examples/embedding_projection/embedding_projection_optimized.py \
    --compile-backend embedding_projection_opt \
    --warmup-iters 1 --measure-iters 1

# Full profiling run
PYTHONPATH=/root/Profiler python nvidia/scripts/run_workload.py \
    --workload examples/embedding_projection/embedding_projection_optimized.py \
    --compile-backend embedding_projection_opt \
    --output-prefix runs/embedding_projection_opt \
    --warmup-iters 3 --measure-iters 10
```

To compare against the baseline:

```bash
PYTHONPATH=/root/Profiler python nvidia/scripts/run_workload.py \
    --workload examples/embedding_projection/embedding_projection.py \
    --output-prefix runs/embedding_projection_baseline \
    --warmup-iters 3 --measure-iters 10
```

---

## Optimizations Table

### OPT-1: BF16 Dtype Promotion (High Confidence)

**Target ops:** all aten::mm and aten::addmm nodes (logit projection, FFN up-proj, FFN down-proj)

**Where applied:** `get_model_and_input()` — before `torch.compile()`, as a non-graph transformation

**Mechanism:** `model.to(torch.bfloat16)` casts all parameters and buffers (embedding
table, LayerNorm, Linear weights) to BF16 before Dynamo traces the graph. Dynamo sees
a monomorphic BF16 graph from the start. Token IDs remain `int64` (embedding lookup
requires integer indices).

**Impact:** Routes all GEMMs from `ampere_sgemm_128x64_tn` (FP32 SIMT, ~19.5 TFLOPS
theoretical peak) to `sm80_xmma_gemm_bf16bf16` (HMMA Tensor Core, ~312 TFLOPS peak).
The logit projection alone is 37.7% of attributed time at 97.7% SM throughput. Expected
4–8x speedup on large compute-bound GEMMs → 50–80% total wall-time reduction.

**Why non-graph:** Dtype is a tensor property. Casting inside the compiled region inserts
explicit `aten.to()` nodes that inhibit Inductor's kernel selection. Must precede compile.

---

### OPT-2: Batch Logit Projection (Medium Confidence — Stub)

**Target ops:** Multiple aten::mm calls sharing the same logit weight matrix

**Where applied:** Backend FX pass (detection stub, no graph mutation)

**Mechanism:** The profiling data describes 6 independent logit projection calls in a
multi-layer variant of the model (op_ids 50, 60, 70, 80, 90, 100), each dispatching a
separate GEMM of shape [8192, 512] x [512, 32000]. The baseline `EmbeddingProjection`
model in this workload calls `self.logits` once per forward pass, so the multi-call
pattern is absent.

The stub detects if the pattern exists in any graph passed to this backend (e.g. a
stacked-layer variant) and logs a warning explaining the required transformation:
concatenate activations along the batch axis, run one GEMM, then chunk the output.

**Impact:** ~6% total time reduction for a 6-call variant (eliminates 5 kernel launches,
allows cuBLAS to select a wider tile for the expanded M dimension).

**Why stub:** Implementing this as a general FX pass requires identifying independent
activation tensors without cross-dependencies. The baseline model does not exhibit the
pattern; adding structural assumptions would make the pass brittle.

---

### OPT-3: Pre-transposed Weights (Medium Confidence — Full FX Pass)

**Target ops:** aten::mm, aten::addmm nodes where the weight argument is wrapped in
aten::t() (TN GEMM mode)

**Where applied:** Backend FX pass `_pass_pretranspose_weights()`, applied to the
Inductor-lowered graph (post-Dynamo, where F.linear is decomposed into
`addmm(bias, x, t(weight))`)

**Mechanism:** Detects the pattern `mm(x, t(get_attr('weight')))` or
`addmm(bias, x, t(get_attr('weight')))` for weights >= 1 MB (BF16 bytes). Pre-transposes
the weight tensor, registers a contiguous buffer, inserts a `get_attr` node pointing to
it, patches the mm/addmm args, and erases the orphaned `aten.t()` node. This switches
cuBLAS from TN (transposed-normal) to NN (normal-normal) GEMM mode.

**Eligible weights after OPT-1:**
- `logits.weight`: [32000, 512] → [512, 32000], 33.6 MB BF16
- `proj1.weight`: [2048, 512] → [512, 2048], 2.1 MB BF16
- `proj2.weight`: [512, 2048] → [2048, 512], 2.1 MB BF16
- `embed.weight` is not a Linear weight and does not appear in mm/addmm patterns

**Impact:** 2–8% total time reduction. For the large logit projection (waves_per_sm=297,
fully compute-bound) the tile efficiency gain is modest but non-zero. Smaller FFN weights
benefit proportionally more but contribute < 3% of total time each.

**Why FX pass:** The transformation requires registering new buffer nodes in the compiled
graph — cannot be done as a simple `replace_pattern` pass.

---

### OPT-4: max-autotune Mode (Medium Confidence)

**Target ops:** All GEMM operators

**Where applied:** `config_patches={"max_autotune": True}` forwarded to `compile_fx`
inside the backend

**Mechanism:** Enables Inductor's cuBLAS algorithm search and Triton GEMM tile
autotuning. For the logit projection ([8192, 512] x [512, 32000] in BF16), the SM is
97.7% saturated but `eligible_cycles_pct` is only 59%, meaning ~41% of cycles are stalled
on pipeline hazards. Autotuning selects pipeline stages that better hide these stalls.

**Impact:** 3–15% additional reduction on large compute-bound GEMMs, applied on top of
OPT-1. Conservative 8% total time estimate given 99.7% GEMM workload share.

**Note:** First compilation will take several minutes for a model of this size. Set
`TORCHINDUCTOR_CACHE_DIR` to avoid recompilation:

```bash
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache/embedding_projection_opt
```

---

## Architecture

### FX Pass Structure

```
get_model_and_input()
  └── OPT-1: model.to(bfloat16)     [non-graph, before compile]

torch.compile(model, backend='embedding_projection_opt')
  └── embedding_projection_opt backend
        ├── UniqueSubgraphRegistry(gm).build_partition_equivalence_map()
        │     EmbeddingProjection has no repeated layers → flat path
        ├── OPT-3: _pass_pretranspose_weights(gm)   [manual FX pass]
        ├── OPT-2: _pass_batch_logit_projection_stub(gm)   [detection only]
        └── compile_fx(gm, example_inputs, config_patches={"max_autotune": True})
              └── OPT-4: max-autotune Inductor compilation
```

### Dedup Awareness

`UniqueSubgraphRegistry` splits the FX graph by detected layer structure. For
`EmbeddingProjection` (single-path sequential model), no structural duplicates exist and
the flat compile path is taken. The dedup path is present for correctness if this backend
is reused with a stacked-layer variant of the model.

### Backend Registration

The backend is registered as `embedding_projection_opt` via `@register_backend` and is
available through `torch.compile(model, backend='embedding_projection_opt')`. It can also
be passed to `run_workload.py` via `--compile-backend embedding_projection_opt`.

---

## Key Design Decisions

### BF16 in get_model_and_input(), not the backend

Dynamo traces dtype as a static property. If `model.to(bfloat16)` is called inside the
compiled region (e.g. in a backend pass), the graph contains explicit `aten.to()` nodes
that prevent Inductor from selecting the HMMA kernel path. The cast must happen before
`torch.compile()` is called so the traced graph is monomorphic BF16 from the start.

### OPT-2 as a stub

The baseline `EmbeddingProjection.forward()` calls `self.logits` exactly once, producing
a single `aten.mm` node in the FX graph. The profile's multi-call pattern (op_ids 50–100)
refers to a hypothetical stacked variant. Implementing OPT-2 as a full pass against the
baseline graph would be a no-op; the stub correctly handles both cases and provides
actionable guidance when the pattern is detected.

### Weight threshold of 1 MB for OPT-3

The runtime overhead of `aten.t()` (which is a view op, effectively free) is negligible
compared to the GEMM. The benefit of TN→NN switching comes from cuBLAS tile selection,
not from eliminating a transpose kernel. For small weights (< 1 MB) this effect is below
measurement noise. The 1 MB threshold ensures only large logit and FFN weights are
pre-transposed, avoiding unnecessary memory duplication.

### compile_fx import

```python
# CORRECT — imports the callable function
from torch._inductor.compile_fx import compile_fx

# WRONG — imports the module, causes TypeError: 'module' object is not callable
from torch._inductor import compile_fx
```

---

## Troubleshooting

### TypeError: 'module' object is not callable

The `compile_fx` import is wrong. Ensure the import reads:
```python
from torch._inductor.compile_fx import compile_fx
```

### Graph lint failure after _pass_pretranspose_weights

This occurs if `node.args` is mutated before `replace_all_uses_with` is called. The
pass follows the correct mutation order: patch `node.args` in-place (no replacement
needed for the mm node itself), erase `t_node` only after verifying `len(t_node.users) == 0`,
then call `gm.graph.lint()` after all mutations.

### OPT-3 pass logs "Pattern not found"

This pass targets the post-Dynamo Inductor-lowered graph, where `F.linear` is decomposed
into `addmm(bias, x, t(weight))`. If you are running the pass on a pre-Inductor graph
(e.g. from eager-mode tracing with `torch.fx.symbolic_trace`), the pattern will not
match. The backend calls this pass on the graph delivered to the backend function, which
is already Inductor-lowered.

### NaN output after BF16 cast

BF16 has lower dynamic range than FP32 (max ~65504). If the logit projection output
contains values outside this range, they will saturate. For the logit projection with
vocab size 32000, logit values are typically in [-10, 10] range and do not overflow BF16.
If NaNs appear, add `torch.backends.cuda.matmul.allow_tf32 = True` and verify the
embedding table weights are initialized in a reasonable range.

### max-autotune compilation timeout

OPT-4 (`max-autotune`) benchmarks multiple cuBLAS algorithm candidates at compile time.
For the logit projection ([8192, 512] x [512, 32000]), expect 3–8 minutes on first
compile. Use `TORCHINDUCTOR_CACHE_DIR` to cache:
```bash
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache/embedding_projection_opt
```

---

## Future Work

### OPT-2: Full batched logit projection implementation

Requires model-level restructuring: accumulate hidden states from each FFN block into a
list, stack along the batch dimension, call `self.logits` once on the stacked tensor, then
chunk the output back. This is a model-source modification, not a general FX pass, because
correctly identifying independence between mm nodes requires semantic knowledge of the
model's control flow.

### Fused embedding + LayerNorm (already optimal)

The baseline profile shows `triton_red_fused_embedding_native_layer_norm_0` at only 0.13%
of total time with 80.7% achieved occupancy. No further optimization is warranted; this
kernel is already near the memory bandwidth ceiling.

### FP8 logit projection (requires H100/Hopper)

FP8 (E4M3) on Ampere (A100) is not supported. The `sm89_xmma_gemm_e4m3` kernel path
requires H100 or newer. This optimization is a post-Hopper migration target.

### Dynamic shape support for autoregressive decoding

For single-token autoregressive decoding, GEMM shapes collapse to [1, 512] x [512, 32000]
— severely wave-starved (waves_per_sm ≈ 1). In that regime, speculative decoding,
continuous batching, or a dedicated decode-path kernel (e.g. marlin, awq_marlin) are more
impactful than dtype promotion. Compile with `dynamic=True` to avoid retracing.
