# GPT-2 Optimized Workload

## Overview

This document describes the optimizations implemented in `gpt2_optimized.py` for the GPT-2 small (117M) workload. All optimizations are derived from profiling feedback in `optimizations.json`, which identified that 67.55% of wall-clock time was consumed by GEMM operations (`aten::mm`, `aten::addmm`) running on the FP32 SIMT path with zero Tensor Core utilization.

Five optimizations are implemented in dependency order:

| ID | Name | Where Applied | Confidence |
|----|------|--------------|------------|
| OPT-5 | TF32 global flags | Module load time | HIGH |
| OPT-1 | BF16 model cast | `get_model_and_input()` | HIGH |
| OPT-3 | BF16 SDPA dispatch | Module load time + cascades from OPT-1 | HIGH |
| OPT-4 | QKV fusion FX pass | `gpt2_backend` (FX pass) | MEDIUM |
| OPT-2 | max-autotune compile | `get_model_and_input()` | HIGH |

Expected combined speedup: **3x–6x** end-to-end wall-clock vs FP32 baseline.

---

## Quick Start

```bash
# Install dependencies
pip install transformers

# Run the optimized workload (downloads GPT-2 weights ~500 MB on first run)
# First compile is slow: max-autotune autotuning takes 60-180 s
python scripts/run_workload.py examples/gpt2/gpt2_optimized.py \
    --warmup-iters 3 --measure-iters 10

# Profile with nsys
nsys profile --trace=cuda,nvtx --output=gpt2_opt \
    python scripts/run_workload.py examples/gpt2/gpt2_optimized.py \
        --warmup-iters 3 --measure-iters 10

# Run verification tests (no autotuning; fast)
python -m pytest test_gpt2_optimized.py -v
```

---

## Optimizations Table

| ID | Name | Target Operators | Baseline Bottleneck | Expected Impact |
|----|------|-----------------|---------------------|-----------------|
| OPT-5 | TF32 global flag | Any residual FP32 mm/addmm | `tensor_core_idle` | 1.0x–1.05x incremental; defense-in-depth |
| OPT-1 | BF16 model cast | aten::mm (336 kernels, 51.88%), aten::addmm (456 kernels, 15.67%) | `tensor_core_idle`: 0.0% Tensor Core cycles across all 792 GEMM kernels. ampere_sgemm path = 19.5 TFLOPS peak | 3x–5x end-to-end. Shifts cuBLAS to sm80_xmma_gemm_bf16 (HMMA) at 312 TFLOPS peak |
| OPT-3 | BF16 SDPA dispatch | aten::_efficient_attention_forward (108 kernels, 3.18%) | `wave_starvation`: fmha_cutlassF_f32 uses 168 regs/thread, collapses occupancy to 6.24%, spills 24.5 MB to DRAM | 3x–5x on attention; 1.05x–1.1x end-to-end (attention share grows after OPT-1 reduces GEMM dominance) |
| OPT-4 | QKV fusion check | aten::mm nodes sharing the same input ([512,768]x[768,768]) | `wave_starvation (secondary)`: waves_per_sm=1.778 on Q/K/V projection GEMMs | 1.03x–1.08x incremental if unfused; safe no-op for HF c_attn |
| OPT-2 | max-autotune compile | All GEMM and fused elementwise kernels | Heuristic tile sizes at 72-74% SM throughput | 1.2x–1.5x incremental over OPT-1; autotuned tiles for each unique GEMM shape |

---

## Architecture

### Non-Graph Optimizations (applied in `get_model_and_input()`)

These run before `torch.compile()` so Inductor traces the already-optimized dtype graph.

**OPT-5 — TF32 flags** (set at module load time):
```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```
Routes any residual FP32 GEMMs through Tensor Core hardware (TF32 = 156 TFLOPS vs 19.5 TFLOPS FP32 SIMT). No-op if flags are already set (PyTorch >= 1.12 default).

**OPT-1 — BF16 cast**:
```python
model = model.to(torch.bfloat16)          # cast before compile
model = torch.compile(model, ...)          # Inductor traces BF16 weight nodes
```
BF16 is preferred over FP16 for GPT-2 because BF16 has the same exponent range as FP32 (8 bits), making it safe for the large activation magnitudes in transformer FFN layers without loss scaling. `input_ids` stays `int64` — embedding lookup is a gather operation, not a GEMM.

**OPT-3 — SDPA backend selection** (set at module load time):
```python
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(False)
```
With BF16 inputs on A100 (sm80), PyTorch dispatches `F.scaled_dot_product_attention` to the FlashAttention-2 kernel automatically. The explicit `enable_flash_sdp` call enforces this selection and prevents fallback to the reference math path.

**OPT-2 — max-autotune compile**:
```python
model = torch.compile(
    model,
    backend="gpt2_backend",
    mode="max-autotune",
    fullgraph=True,
)
```
Runs the Triton autotuner over tile configurations (128x256, 64x256, 256x128, ...) and selects the empirically fastest for each unique GEMM shape. GPT-2 small has approximately 4 unique GEMM shapes. FX graph cache is enabled (`_inductor_config.fx_graph_cache = True`) to avoid re-autotuning on subsequent runs.

### FX Backend (`gpt2_backend`)

The backend is registered with `@register_backend` and wraps Inductor. It detects whether the model has repeated layers (GPT-2 has 12 structurally identical transformer blocks) and takes one of two paths:

**Dedup path (12-layer GPT-2):**
1. `UniqueSubgraphRegistry` splits the graph by layer using `split_module`.
2. `_pass_fuse_qkv` is applied only to the unique representative partition (e.g., `submod_0` for transformer block structure).
3. The same pass is propagated to all 11 structural duplicates.
4. Each unique representative is compiled with `compile_fx` using its actual partition inputs (captured via forward-pre hooks).
5. Duplicate partitions share the compiled callable from their representative.

**Flat path (no repeated layers):**
- `_pass_fuse_qkv` is applied to the full flat graph.
- The graph is passed directly to `compile_fx`.

### OPT-4 — QKV Fusion FX Pass

The pass scans the Aten IR graph for groups of exactly three `aten.mm` nodes that share the same input activation tensor. If found, it:
1. Extracts the three weight tensors (handling both bare `get_attr` and `t(get_attr(...))` patterns).
2. Concatenates them along the output dimension: `[K, N] × 3 → [K, 3N]`.
3. Registers the fused weight as a buffer.
4. Replaces the three `mm` nodes with one `mm` + `chunk(3, dim=-1)`.

For standard HuggingFace GPT-2, `GPT2Attention` uses `self.c_attn = Conv1D(3*n_embd, n_embd)` which fuses Q/K/V into a single projection at the module level. The Inductor-traced graph will show one `mm` node (or `addmm`) with a `[768, 2304]` weight, not three separate `[768, 768]` nodes. The pass detects zero matching groups and exits cleanly as a no-op.

---

## Key Design Decisions

### Why BF16 is applied in `get_model_and_input()`, not in the FX backend

`torch.compile()` traces the model's dtype graph at compile time. If weights are cast to BF16 after `compile()`, Inductor's traced graph contains FP32 weight nodes and the cuBLAS HMMA path is never selected. The cast must precede `torch.compile()`.

### Why `input_ids` stays `int64`

Embedding lookup (`aten::embedding`) is an integer gather — it reads rows from the weight matrix using integer indices. The dtype of `input_ids` does not affect whether the weight matrix is BF16; the embedding output dtype follows the weight dtype. There is no GEMM on `input_ids` itself.

### Why OPT-4 is MEDIUM confidence

The profile evidence (waves_per_sm=1.778 on `[512,768]x[768,768]` GEMMs at triage rank 149+) is real. However, HuggingFace GPT-2 already fuses Q/K/V at the module level via `c_attn`. The Inductor-traced graph must be inspected at runtime to confirm whether separate Q/K/V mm nodes appear. The pass is implemented as a correct, safe no-op when the pattern is absent.

### Why `enable_math_sdp(False)` is set explicitly

With `enable_flash_sdp(True)` and `enable_math_sdp(False)`, PyTorch's SDPA dispatcher raises an error if FlashAttention is unavailable rather than silently falling back to the slow math path. This makes configuration issues visible during development. On A100 with BF16 inputs and PyTorch >= 2.0, FlashAttention is always available.

### Why `_capture_partition_inputs` is needed

`split_module` decomposes the flat graph into partition submodules. Each partition takes a subset of the original inputs. Using the top-level `example_inputs` for all partitions would pass tensors with incorrect shapes to Inductor, causing shape inference failures. Forward-pre hooks on the split graph capture the real per-partition inputs in a single dry-run forward pass.

---

## Troubleshooting

**`TypeError: 'module' object is not callable` at compile time**

Cause: `from torch._inductor import compile_fx` imports the module, not the function.

Fix: Always import `from torch._inductor.compile_fx import compile_fx`.

**`gm.graph.lint()` fails after a pass**

Cause: A node was erased while live uses remain, or `replace_all_uses_with` was called after `erase_node`.

Fix: Always call `node.replace_all_uses_with(new_node)` BEFORE `gm.graph.erase_node(node)`. Take a snapshot of nodes with `list(gm.graph.nodes)` before iterating.

**`AssertionError: sm_major < 8` on non-Ampere hardware**

Cause: `get_model_and_input()` asserts `sm_major >= 8` for BF16 Tensor Core support.

Fix: Remove the assertion or replace with a warning to run on older hardware (BF16 GEMMs will still execute but without Tensor Core acceleration).

**max-autotune compilation hangs or is very slow**

Cause: First-run autotuning tests ~100 tile configurations per unique GEMM shape. GPT-2 small has ~4 unique shapes.

Fix: Enable the FX graph cache (already enabled in this file). After the first run, recompilation is near-instant. Alternatively use `mode="reduce-overhead"` for faster but untuned compilation.

**`fullgraph=True` raises a graph break error**

Cause: HuggingFace modeling code contains Python control flow that depends on tensor values, or uses unsupported operations.

Fix: Switch to `dynamic=True`:
```python
model = torch.compile(
    model,
    backend="gpt2_backend",
    mode="max-autotune",
    fullgraph=True,
    dynamic=True,
)
```
Or remove `fullgraph=True` to allow graph breaks (Inductor compiles each subgraph independently; cross-operator fusion opportunities may be reduced).

**OPT-4 pass registers a buffer but dtype mismatch occurs**

Cause: The fused QKV weight buffer is created from `torch.cat(weights, ...)` where weights are BF16. If OPT-1 was not applied before the backend is called, the buffer will be FP32.

Fix: Ensure `model.to(torch.bfloat16)` is called before `torch.compile()` in `get_model_and_input()`.

---

## Future Work

| Pass | Status | Infrastructure Required |
|------|--------|------------------------|
| LayerNorm-Linear fusion | Not implemented | Custom Triton kernel that keeps LayerNorm normalized rows in registers before issuing the following GEMM, eliminating the DRAM round-trip between the two operations |
| Persistent GEMM kernels | Not implemented | Triton persistent kernel for small GEMMs (waves_per_sm < 2) — reduces launch overhead for the 36 attention projection GEMMs |
| KV cache for autoregressive inference | Not implemented | Requires model-level changes to `GPT2Wrapper.forward()` to accept and return a KV cache state; the current workload is prefill-only |
| INT8 weight quantization | Not implemented | `torch.ao.quantization` or Inductor's built-in INT8 support; would improve memory bandwidth for memory-bound attention projection GEMMs at long sequence lengths |
