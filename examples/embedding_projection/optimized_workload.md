# EmbeddingProjection — Optimized Workload

Custom `torch.compile()` backend (`transformer_opt`) implementing 5 operator-level optimizations derived from `ncu` profiling of the baseline `EmbeddingProjection` workload.

---

## Overview

The profiler identified a single dominant bottleneck: **100% of GEMM operators use cuBLAS Kernel2**, an opaque FP32 code path with 212 registers/thread and 16.7% achieved occupancy. Tensor core utilization is 0% across all 20 `aten::mm` dispatches. Collectively these kernels represent 91.6% of wall time (~44 ms of 48 ms).

This workload implements targeted FX graph surgery to replace those kernels with Triton HMMA GEMMs via BF16 precision routing, while also addressing dispatch overhead and downstream bandwidth bottlenecks.

---

## Quick Start

```bash
# Run optimized workload
python scripts/run_workload.py \
    --workload scripts/embedding_projection_optimized.py \
    --compile-backend transformer_opt

# Profile with ncu
operator-profiler map manifest.json \
    --script scripts/run_workload.py \
    --ncu-sudo \
    --ncu-env PYTHONPATH=/repo \
    --script-args --workload scripts/embedding_projection_optimized.py \
                  --compile-backend transformer_opt

# Baseline comparison
python scripts/run_workload.py \
    --workload scripts/embedding_projection.py \
    --compile-mode inductor

# Smoke test
python test_embedding_projection_optimized.py

# Syntax check
python -m py_compile embedding_projection_optimized.py
```

---

## Optimizations

| ID | Pass | Target Operators | Bottleneck | Transformation | Expected Impact | Confidence |
|----|------|-----------------|------------|----------------|-----------------|------------|
| OPT-1 | `pass_insert_bf16_casts` | `aten::mm [8192,512]×[512,32000]` (×10) | Kernel2, 0% TC, 16.7% occ, 212 regs/thread | Insert `aten._to_copy(bf16)` on both mm inputs | 4–8× latency reduction; 85% wall time freed | **High** |
| OPT-2 | `pass_insert_bf16_casts` | `aten::mm [8192,512]×[512,2048]` (×10) | Same Kernel2 fallback; SM-bound (DRAM only 5.3%) | Same BF16 cast; tiling hint 128×128×32 via max-autotune | ~3 ms freed / 10 iterations; occ → 50%+ | **High** |
| OPT-3 | `pass_batch_sequential_mm` | Both mm shapes (10 sequential dispatches each) | 10 launches/shape × ~5 µs dispatch = ~50 µs overhead | Fuse to single `aten::bmm`; 10 → 1 kernel launch | 30–50 µs eliminated; better wave count | Medium |
| OPT-4 | `pass_propagate_bf16_pointwise` | `triton_poi_fused_addmm_gelu_view_1` (×10) | DRAM 90.3% peak; 67 MB reads (FP32 element size) | BF16 dtype propagation halves DRAM traffic | ~50% reduction; 208 µs → ~104 µs | Medium |
| OPT-5 | `pass_detect_embedding_quant` (stub) | `triton_red_fused_embedding_native_layer_norm_0` (×10) | L2 hit rate 11.2%; scatter-reads defeat cache | INT8 embedding weight quantization (detection only) | ~4 µs; ~50% embedding table size | Low |

---

## Architecture

### FX Graph Passes

Each optimization is a standalone function with signature `pass_*(gm: fx.GraphModule) -> fx.GraphModule`. Passes operate at the Aten IR level — they are model-agnostic and do not reference `EmbeddingProjection` class internals.

```
Baseline FX Graph (FP32, Kernel2 GEMMs)
         │
         ▼
pass_insert_bf16_casts         ← OPT-1/2: route mm → HMMA
         │
         ▼
pass_propagate_bf16_pointwise  ← OPT-4: DRAM halving on pointwise
         │
         ▼
pass_batch_sequential_mm       ← OPT-3: dispatch fusion
         │
         ▼
pass_detect_embedding_quant    ← OPT-5: stub recommendation
         │
         ▼
compile_fx(max_autotune=True)  ← Triton GEMM tile search + cache
         │
         ▼
Compiled callable (HMMA GEMMs, fused pointwise in BF16)
```

### Backend Registration

```python
@register_backend
def transformer_opt(gm: fx.GraphModule, example_inputs) -> Callable:
    gm = pass_insert_bf16_casts(gm)
    gm = pass_propagate_bf16_pointwise(gm)
    gm = pass_batch_sequential_mm(gm)
    gm = pass_detect_embedding_quant(gm)
    return compile_fx(gm, example_inputs, config_patches={"max_autotune": True})
```

Use with:
```python
torch.compile(model, backend="transformer_opt")
```

---

## Why a Custom Backend

| Approach | Pros | Cons |
|----------|------|------|
| `model.to(bfloat16)` only | Simple | Does not address dispatch overhead or pointwise dtype propagation |
| `torch.compile(mode='max-autotune')` only | Automatic tile search | Does not insert BF16 casts if model is FP32; Kernel2 persists |
| Custom backend (this file) | Full control; model-agnostic; observable; composable | Requires FX graph familiarity |

The custom backend is the minimal intervention that reliably breaks the Kernel2 path on any GEMM shape, regardless of module structure.

---

## Key Design Decisions

### BF16 applied both in `get_model_and_input()` and as an FX pass

`model.to(torch.bfloat16)` ensures the embedding table, layer norm parameters, and linear weights are already BF16 before tracing. The FX pass `pass_insert_bf16_casts` then inserts explicit `aten._to_copy(bfloat16)` nodes on mm inputs as a belt-and-suspenders measure — in case inductor widens intermediate activations back to FP32 during lowering.

`get_model_and_input()` checks `next(model.parameters()).dtype` before casting to remain idempotent if the baseline is updated to return a BF16 model.

### token_ids stay int64

Embedding index lookups require integer indices. Only the dense weight parameters and activations are cast to BF16.

### Pattern matching is structural, not name-based

All passes identify nodes by `node.target` (e.g. `torch.ops.aten.mm.default`) rather than module names or string heuristics, making them robust to `torch.compile` graph renaming.

### Defensive error handling

Every pass wraps its body in `try/except Exception` and calls `logger.warning(...)` on failure, then returns the unmodified `gm`. This ensures a failing pass degrades to a no-op rather than crashing the compilation pipeline.

### OPT-5 is a stub

INT8 embedding quantization requires registering a custom `torch.ops` dequantize kernel via `torch.library`. The stub pass detects the pattern and logs a structured TODO so it remains visible in profiling logs without silently being skipped.

---

## Comparison Against Baseline

```bash
# Baseline wall time (~48 ms, Kernel2, 0% TC)
python scripts/run_workload.py \
    --workload scripts/embedding_projection.py \
    --compile-mode inductor \
    --measure-iters 100

# Optimized wall time (target: <12 ms, HMMA, ≥50% occ)
python scripts/run_workload.py \
    --workload scripts/embedding_projection_optimized.py \
    --compile-backend transformer_opt \
    --measure-iters 100
```

### What to look for in the optimized profile

| Metric | Baseline | Target (Optimized) |
|--------|----------|--------------------|
| `smsp__pipe_tensor_cycles_active` | 0 | > 0 on all mm kernels |
| `achieved_occupancy` | 16.7% | ≥ 50% |
| `launch__registers_per_thread` | 210–212 | ~80 |
| `l2_hit_rate` on mm kernels | 84.6% | ≥ 85% |
| `dram_throughput` on addmm+gelu | 90.3% | ~45% |
| Total mm kernel count | 20 dispatches | ≤ 10 (if OPT-3 active) |
| Kernel name | `Kernel2` (cuBLAS opaque) | `triton_mm_*` |

---

## Verification Checklist

```
[ ] python -m py_compile embedding_projection_optimized.py
[ ] python -c "import embedding_projection_optimized"
[ ] 'transformer_opt' in torch._dynamo.list_backends()
[ ] pytest test_embedding_projection_optimized.py -v  (all tests pass)
[ ] python embedding_projection_optimized.py  (prints ✓ lines, no CUDA errors)
[ ] ncu profile: smsp__pipe_tensor_cycles_active > 0 on aten::mm kernels
[ ] ncu profile: kernel names contain 'triton', not 'Kernel2'
[ ] ncu profile: achieved_occupancy ≥ 50% on GEMM kernels
[ ] ncu profile: dram_throughput on addmm+gelu < 50%
[ ] Numerical sanity: output shape (64, 128, 32000), no NaN/Inf
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'embedding_projection'`
Add the workload directory to `PYTHONPATH`:
```bash
export PYTHONPATH=/path/to/scripts:$PYTHONPATH
```

### `TypeError: 'module' object is not callable` on `compile_fx`
Ensure the import is:
```python
from torch._inductor.compile_fx import compile_fx   # ← function
# NOT: from torch._inductor import compile_fx        # ← module
```

### BF16 casts inserted but Kernel2 still appears in profile
The model may still be in FP32 at trace time. Verify:
```python
print(next(model.parameters()).dtype)  # should be torch.bfloat16
```
If not, the `get_model_and_input()` cast did not run before `torch.compile`. Check that `transformer_opt` backend is active (not `inductor` default).

### `pass_batch_sequential_mm` logs "cannot verify shapes"
Shape meta is not available at symbolic trace time on all torch versions. This is a medium-confidence pass — the graph still benefits from OPT-1/2 without it.

### CUDA out of memory after BF16 cast
BF16 halves parameter memory but the embedding table for VOCAB_SIZE=32000 is still ~32 MB. If OOM occurs, reduce `BATCH_SIZE` in `embedding_projection.py`.

### `--script-args` flag parse error with operator-profiler
All `operator-profiler map` flags must appear **before** `--script-args`:
```bash
# CORRECT
operator-profiler map manifest.json \
    --script scripts/run_workload.py \
    --ncu-sudo \
    --ncu-env PYTHONPATH=/repo \
    --script-args --workload scripts/embedding_projection_optimized.py \
                  --compile-backend transformer_opt
```

---

## Future Work

| Item | Pass | Priority |
|------|------|----------|
| INT8 embedding quantization | `pass_detect_embedding_quant` → full impl | Low (0.2% wall time) |
| Flash-decode for large batch inference | New pass: replace mm+softmax+mm with SDPA | Medium |
| Inductor `use_mixed_mm` flag | Fallback if BF16 cast still routes to Kernel2 | Low |
| Persistent kernel config cache | Set `TORCHINDUCTOR_CACHE_DIR` for warm restarts | Low |
| Triton autotune annotation | Explicit `BLOCK_M=128,N=128,K=32` hint for [8192,512]×[512,2048] | Medium |