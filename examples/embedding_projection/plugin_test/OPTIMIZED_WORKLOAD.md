# EmbeddingProjection — Optimized Workload

## Overview

The baseline EmbeddingProjection workload ran all 30 cuBLAS GEMM nodes (20x `aten::mm`,
10x `aten::addmm`) on the FP32 SIMT path, producing `tensor_core_active_pct=0.0` and
`registers_per_thread=212` across every attributed kernel. The dominant operator —
`aten::mm [8192,512]x[512,32000]` (logit projection) — consumed **85% of total profiled
time** at 63.7% SM throughput. Moving that single GEMM to the BF16 Tensor Core HMMA path
is the highest-leverage change in this workload.

Three optimizations from `optimizations.json` are implemented:

| ID | Title | Confidence | Where Applied |
|----|-------|------------|---------------|
| OPT-1 | BF16 dtype cast | HIGH | `get_model_and_input()` (eager) |
| OPT-2 | TF32 flags | HIGH | `get_model_and_input()` (eager) |
| OPT-3 | Pre-transposed weights FX pass | MEDIUM | Custom `torch.compile` backend |

OPT-4 (max-autotune) and OPT-5 (tied-weight untying) are handled inline:
OPT-5's tied-weight check is embedded in `get_model_and_input()` before the BF16 cast;
OPT-4 is available as a `mode=` argument to `torch.compile` and can be enabled by changing
the compile call in `get_model_and_input()`.

---

## Quick Start

### Run the optimized workload directly

```bash
cd /home/ubuntu/Profiler

# Smoke test (uncompiled forward pass)
python examples/embedding_projection/plugin_test/embedding_projection_optimized.py

# Run via the profiler runner (1 warmup, 10 measure iters)
python nvidia/scripts/run_workload.py \
    --workload examples/embedding_projection/plugin_test/embedding_projection_optimized.py \
    --compile-backend embedding_projection_opt \
    --warmup-iters 3 \
    --measure-iters 10
```

### nsys capture

```bash
nsys profile \
    --trace=cuda,nvtx \
    --output=runs/embedding_projection_opt \
    python nvidia/scripts/run_workload.py \
        --workload examples/embedding_projection/plugin_test/embedding_projection_optimized.py \
        --compile-backend embedding_projection_opt \
        --warmup-iters 3 \
        --measure-iters 10
```

### Run the validation tests

```bash
cd /home/ubuntu/Profiler
python -m pytest examples/embedding_projection/plugin_test/test_embedding_projection_optimized.py -v
```

### Syntax check only

```bash
python3 -m py_compile examples/embedding_projection/plugin_test/embedding_projection_optimized.py
```

---

## Optimizations Table

| ID | Optimization | Target Operators | Mechanism | Expected Speedup |
|----|-------------|-----------------|-----------|-----------------|
| OPT-2 | TF32 flags | All 30 cuBLAS GEMMs | `allow_tf32=True` routes FP32 GEMMs through HMMA Tensor Cores with 10-bit mantissa | 2–3x on GEMMs, ~2–2.5x total |
| OPT-1 | BF16 dtype cast | All 30 cuBLAS GEMMs | Casts weights and activations to bfloat16; forces cuBLAS `sm80_xmma_gemm_bf16bf16`; 31.25 MB BF16 table fits in A100 L2 (40 MB) vs 62.5 MB FP32 | 3–5x on GEMMs, 3–4x total |
| OPT-3 | Pre-transposed weights | `aten::mm` / `aten::addmm` with large weight matrices | Eliminates `CUBLAS_OP_T` transpose flag at runtime; pre-stores `W.t().contiguous()` as a graph buffer | 5–10% on GEMMs, ~4–9% total |

Combined (OPT-1 + OPT-2 + OPT-3): estimated **3–5x total workload speedup** on A100 vs. baseline FP32 SIMT.

---

## Architecture

### Non-graph optimizations (`get_model_and_input()`)

**OPT-2** and **OPT-1** cannot be expressed as FX graph passes because they modify tensor
storage dtype before the graph is traced. They are applied eagerly in `get_model_and_input()`:

1. TF32 flags are set with `torch.backends.cuda.matmul.allow_tf32 = True` and
   `torch.backends.cudnn.allow_tf32 = True`.
2. The tied-weight check compares `model.logits.weight.data_ptr()` against
   `model.embed.weight.data_ptr()`. If they alias, an explicit `clone()` breaks the tie
   before the BF16 cast, preventing shared-storage issues in the pre-transpose buffers.
3. The model is cast to `bfloat16` via `model.to(torch.bfloat16)`.
4. `token_ids` stays `int64` — `nn.Embedding.forward()` requires integer indices.

### FX pass (`pass_pretranspose_weights`)

**OPT-3** runs inside the `embedding_projection_opt` backend at graph-compile time (once,
before Inductor's own passes):

1. Walk the frozen FX graph for `aten.t.default(get_attr)` nodes.
2. For each such node where the underlying tensor exceeds 1 MB:
   - Call `weight.t().contiguous()` eagerly (once at compile time, not at runtime).
   - Register the result as a buffer on `gm` via `gm.register_buffer(buf_name, weight_t)`.
   - Insert a `get_attr` node pointing to the new buffer.
   - Call `node.replace_all_uses_with(new_attr_node)` then `gm.graph.erase_node(node)`.
3. Call `gm.graph.lint()` then `gm.recompile()` after all mutations.

The pass is MEDIUM confidence: if Inductor already selects the same cuBLAS algorithm
regardless of `OP_T` vs `OP_N` for these specific shapes, the gain may be near zero.
The pass degrades gracefully — if no eligible patterns are found it logs a warning and
returns `gm` unchanged.

### Backend

```
embedding_projection_opt(gm, example_inputs)
  └── pass_pretranspose_weights(gm)     # OPT-3 FX pass
  └── compile_fx(gm, example_inputs)   # Standard Inductor
```

The backend is registered with `@torch._dynamo.register_backend` and is selected via
`torch.compile(model, backend="embedding_projection_opt")`.

---

## Key Design Decisions

### Why BF16 cast is outside the graph

`nn.Embedding` is not FX-traceable in the same way as `nn.Linear`. Its weight matrix
participates in a CUDA gather kernel, not a GEMM — and its input is integer-typed.
Changing the embedding weight dtype requires modifying the `nn.Module` state before
tracing, not during graph execution. Attempting to insert dtype-cast nodes in the FX
graph for the embedding table would require understanding the gather semantics at the
Aten IR level, which is fragile. The eager path is simpler and reliable.

### Why the tied-weight check is critical

When `logits.weight` and `embed.weight` share the same storage (tied embeddings), calling
`gm.register_buffer("logits_weight_pretransposed", logits_weight.t().contiguous())` creates
a new tensor — but if the original parameter had been cast via the shared pointer, the
buffer may diverge from the embedding table. The explicit clone before BF16 cast ensures
both the embedding table and the projection weight are independent, well-defined BF16
tensors.

### Why OPT-3 uses a 1 MB threshold

Small weight matrices (bias tensors, LayerNorm weights) generate `aten.t` nodes too.
Pre-transposing a 512-element bias vector adds memory pressure and compile overhead with
no measurable benefit. The 1 MB threshold corresponds approximately to a 512×256 FP32
matrix — well below the smallest projection weight in this model (512×512 = 1 MB exactly
at FP32, 512 KB at BF16). After the BF16 cast, the 512×512 weight is exactly at the
threshold, so in practice only the larger weights (512×2048 = 4 MB, 512×32000 = 62.5 MB
at FP32 / 31.25 MB at BF16) will be pre-transposed. This can be adjusted by changing
`_MIN_WEIGHT_BYTES` at the top of the file.

### Why OPT-4 (max-autotune) is not included by default

`max-autotune` adds 2–5 minutes of cold-start compile time and its benefit is shape-
specific. For the dominant GEMM (`[8192, 512] x [512, 32000]`) the expected gain is
10–20% on `aten::addmm` (7.8% of total), translating to ~1–2% overall. This is outside
the compile-time cost budget for a general profiling workload. Enable it by changing the
`torch.compile` call:

```python
model = torch.compile(model, backend="embedding_projection_opt",
                      mode="max-autotune", fullgraph=False)
```

---

## Troubleshooting

### `TypeError: 'module' object is not callable`

Wrong import for `compile_fx`. Must import the function, not the module:

```python
# WRONG
from torch._inductor import compile_fx

# CORRECT
from torch._inductor.compile_fx import compile_fx
```

### `graph.lint()` assertion failure after pre-transpose pass

Usually caused by erasing a node that still has users, or inserting a node after the
`output` node. Check that `node.replace_all_uses_with(new_node)` is called **before**
`gm.graph.erase_node(node)`. The pass wraps everything in `try/except` and returns the
unmodified graph on failure.

### Output dtype is float32 instead of bfloat16

This can happen if `model.to(torch.bfloat16)` was not applied (e.g., `current_dtype`
check short-circuited). Verify with:

```python
model, _ = get_model_and_input()
raw = model._orig_mod if hasattr(model, "_orig_mod") else model
print(next(raw.parameters()).dtype)  # should print torch.bfloat16
```

### `AssertionError: CUDA required`

No GPU available in the current environment. Verify with `torch.cuda.is_available()`.

### Backend not found in `torch._dynamo.list_backends()`

The optimized module must be imported before checking backend registration. The
`@register_backend` decorator runs at import time:

```python
import examples.embedding_projection.plugin_test.embedding_projection_optimized
import torch
print(torch._dynamo.list_backends())  # embedding_projection_opt should appear
```

### First run is slow (compilation overhead)

`torch.compile` traces and compiles the graph on the first forward call. Allow 2–3 warmup
iterations before measuring. The profiler runner's `--warmup-iters 3` flag handles this.

---

## Future Work

| Stub | Infrastructure Needed |
|------|-----------------------|
| OPT-4: max-autotune | Change `mode=` argument; no new code needed — just a config change |
| LayerNorm-Linear fusion | Custom Triton kernel that keeps normalized rows in registers before issuing GEMM, eliminating one DRAM round-trip per LN→Linear pair |
| Quantized int8 logit projection | INT8 weight quantization with FP16 scale factors for the dominant 512×32000 GEMM; requires `torch.ao.quantization` or `bitsandbytes` |
| KV-cache aware batching | For autoregressive decode, batch the logit projection across decode steps rather than per-token — changes the GEMM shape from (B·T, 512) to (B, 512) per step |
