# GPT-2 Optimized Workload

## Overview

This workload applies three GPU-level optimizations to GPT-2 small (117M parameters,
12 transformer decoder blocks, hidden=768, heads=12, ffn_dim=3072).

The baseline profile reveals that 67.6% of total wall time is spent in FP32 SIMT GEMM
kernels (`ampere_sgemm_*`) with zero Tensor Core activity. All three optimizations
target this class of inefficiency:

| ID | Type | Location | Confidence | Expected Impact |
|----|------|----------|-----------|----------------|
| OPT-1 | BF16 dtype promotion | `get_model_and_input()` | High | ~50-60% total latency reduction |
| OPT-2 | Pre-transposed weight buffers | FX pass (manual per-rep) | Medium | ~3-8% additional reduction |
| OPT-3 | max-autotune GEMM tile selection | `compile_fx` options | Medium | ~5-15% additional reduction on GEMM ops |

---

## Quick Start

```bash
# Install dependencies
pip install transformers
PYTHONPATH=/root/Profiler python3 -c "from nvidia.operator_profiler.fx import UniqueSubgraphRegistry; print('OK')"

# Syntax check
cd /root/Profiler && python3 -m py_compile examples/gpt2/gpt2_optimized.py

# Run tests (no GPU compilation required for tests 1-3)
cd /root/Profiler && PYTHONPATH=/root/Profiler pytest examples/gpt2/test_gpt2_optimized.py -v

# Profile the optimized workload (Phase 1 — correlation pass)
PYTHONPATH=/root/Profiler python3 nvidia/scripts/run_workload.py \
    --workload examples/gpt2/gpt2_optimized.py \
    --output-prefix /tmp/gpt2_opt \
    --inductor-debug-dir /tmp/gpt2_opt_inductor \
    --correlation-pass

# Profile the optimized workload (Phase 2 — NVTX capture)
nsys profile --trace=cuda,nvtx --output=/tmp/gpt2_opt \
    python3 nvidia/scripts/run_workload.py \
        --workload examples/gpt2/gpt2_optimized.py \
        --output-prefix /tmp/gpt2_opt \
        --inductor-debug-dir /tmp/gpt2_opt_inductor

# Quick forward pass sanity check
PYTHONPATH=/root/Profiler python3 examples/gpt2/gpt2_optimized.py
# Expected output:
#   Output shape : torch.Size([4, 128, 768])
#   Output dtype : torch.bfloat16
```

**Note on compile time:** OPT-3 (`max-autotune`) benchmarks multiple CUTLASS/cuBLAS tile
configurations at compile time. First compilation of GPT-2 small takes approximately
5-15 minutes. Set `TORCHINDUCTOR_CACHE_DIR` to a persistent path to avoid recompilation
across runs:

```bash
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache_gpt2
```

---

## Optimizations Table

| ID | Optimization | Target Operators | Mechanism | Expected Speedup |
|----|-------------|-----------------|-----------|-----------------|
| OPT-1 | BF16 dtype promotion | `aten::mm` (336 kernels, 51.93%), `aten::addmm` (456 kernels, 15.68%), `aten::_efficient_attention_forward` (108 kernels, 3.18%) | Replaces FP32 SIMT kernels (`ampere_sgemm_*`) with BF16 Tensor Core kernels (`sm80_xmma_gemm_bf16bf16`). Eliminates 24.5 MB register spills in fmha_cutlassF_f32 kernel. A100 BF16 Tensor Core peak is 312 TFLOPS vs ~19.5 TFLOPS SIMT FP32. | ~50-60% total wall time reduction |
| OPT-2 | Pre-transposed weights | `aten::mm` (336 kernels), `aten::addmm` (456 kernels) | Detects `aten.t(get_attr)` nodes in the FX graph; pre-transposes weight buffers at graph-rewrite time; patches mm/addmm to use the contiguous transposed buffer. Eliminates per-call implicit DRAM reshape in cuBLAS TN dispatch. Applies only to weights with K >= 512. | ~3-8% additional reduction |
| OPT-3 | max-autotune kernel selection | `aten::mm` (336 kernels), `aten::addmm` (456 kernels) | Passes `{"max_autotune": True}` to `compile_fx`. Inductor benchmarks CUTLASS and cuBLAS algorithm variants for each unique (M,N,K) shape at compile time and caches the winner. Most impactful for 512x768x3072 FFN shapes where multiple competitive tile configurations exist. | ~5-15% additional reduction on GEMM ops |

**Combined expected impact:** ~55-70% total latency reduction (from baseline 103.9 ms).
Post-optimization target: ~30-45 ms for batch=4, seq_len=128 on A100 SXM4-80GB.

---

## Architecture

### Non-Graph Optimizations (`get_model_and_input()`)

OPT-1 lives entirely in `get_model_and_input()`. Dynamo traces dtype as a static
property at compile time — changing dtype after compilation triggers a full recompile.
The optimization is applied before `torch.compile` is called:

```python
model = model.to(torch.bfloat16)  # all parameters including wte/wpe embeddings
# input_ids remains int64 — embedding lookup requires integer indices
```

`torch.backends.cuda.matmul.allow_tf32 = True` is set at module import time as a
belt-and-suspenders complement to OPT-1 (no-op after BF16 promotion, but useful if
OPT-1 is disabled).

### FX Pass: `_pass_pretranspose` (OPT-2)

This pass operates on the **post-Dynamo, Inductor-lowered graph** where `F.linear` has
already been decomposed into `aten.t(get_attr('weight')) + aten.mm` or `aten.addmm`.
The detection pattern is:

```
aten.addmm(bias_node, x_node, aten.t(get_attr('weight')))
aten.mm(x_node, aten.t(get_attr('weight')))
```

For each matched node:
1. Retrieve the weight tensor via `gm.get_parameter(weight_node.target)`
2. Skip if `weight.shape[-1] < 512` (memory overhead threshold)
3. Compute `W_T = weight.t().contiguous()` and register as a named buffer
4. Insert a `get_attr` node for the transposed buffer immediately before the mm node
5. Patch the mm/addmm `.args` to use the new buffer node directly (no `aten.t()`)
6. Erase the orphaned `aten.t()` node if it has no remaining users
7. Call `gm.graph.lint()` then `gm.recompile()` after all mutations

### Dedup-Aware Backend (`gpt2_opt`)

GPT-2's 12 transformer blocks are structurally identical. `UniqueSubgraphRegistry`
splits the FX graph by layer structure and groups partitions by graph signature.

**If no equivalence map** (no repeated layers):
- Flat compile path — apply `_pass_pretranspose` to the full flat graph, then
  delegate to `compile_fx(gm, example_inputs, options={"max_autotune": True})`.
  This path preserves cross-layer Inductor fusion opportunities.

**If equivalence map present** (GPT-2's normal case — 12 repeated blocks):
- Dedup path — capture per-partition inputs via `_capture_partition_inputs`.
- Apply `_pass_pretranspose` to each unique representative only.
- Compile each unique rep with `compile_fx(..., options={"max_autotune": True})`.
- Share the compiled callable with all structural duplicates by patching `.forward`.
- Return `lambda *args: registry.split(*args)`.

For GPT-2 small, the dedup path reduces max-autotune compilation time from ~12x
(once per layer) to ~1x (once per unique signature).

---

## Key Design Decisions

### OPT-1 as non-graph (not FX pass)

BF16 promotion modifies tensor `dtype`, which is a static property baked into Dynamo's
trace specialization. Dynamo must see BF16 inputs and parameters at trace time to
generate BF16-native ops. Applying `.to(bfloat16)` inside the backend (post-Dynamo)
would operate on an already-traced graph and would not affect the dtype of any
intermediate tensors. The only correct location is `get_model_and_input()`, before
`torch.compile` is called.

### OPT-2 as manual per-rep (not `replace_pattern`)

`torch.fx.subgraph_rewriter.replace_pattern` cannot call `register_buffer` on the
module — it operates purely structurally and has no access to tensor values. Pre-
transposing a weight requires:
1. Reading the actual weight tensor to compute `W_T = weight.t().contiguous()`
2. Calling `gm.register_buffer(...)` to store the result

These requirements make it a manual per-rep pass.

### OPT-3 via `compile_fx` options (not graph mutation)

max-autotune is an Inductor compilation mode, not an FX graph transformation. No
graph nodes are modified. The option is forwarded as `options={"max_autotune": True}`
to `compile_fx`. This is equivalent to `torch.compile(model, mode="max-autotune")`.

### GPT-2 QKV fusion not applied

GPT-2's combined QKV projection (`c_attn: nn.Linear(768, 2304)`) is already fused
as a single 512x2304 GEMM in the HuggingFace implementation. The three-separate-mm
QKV fusion pattern does not appear in this model's graph. This matches the note in
`optimizations.json`:

> "GPT-2's combined QKV projection (c_attn: nn.Linear(768, 2304)) is already fused
> as a single 512x2304 GEMM."

### OPT-2 prerequisite: OPT-1 must run first

`_pass_pretranspose` registers `W_T = weight.t().contiguous()` as a buffer. If OPT-1
has not run, `weight` is FP32 and `W_T` is a FP32 buffer, but the mm inputs are also
FP32 — this is consistent and safe. However, after OPT-1 makes the model BF16, the
weights are already BF16 when the backend is called (Dynamo traces with BF16 weights),
so `W_T` is naturally BF16. The prerequisite is satisfied by placing OPT-1 in
`get_model_and_input()` and OPT-2 in the backend.

---

## Troubleshooting

### `TypeError: 'module' object is not callable`

Wrong import. The file must use:
```python
from torch._inductor.compile_fx import compile_fx  # correct — imports the function
```
Not:
```python
from torch._inductor import compile_fx  # wrong — imports the module
```

### `torch.fx.graph.lint()` failure after graph mutation

If `lint()` raises, the graph has dangling references. Common cause: a node was erased
before all its users were replaced. Always call `node.replace_all_uses_with(new_node)`
**before** `gm.graph.erase_node(node)`. The `_pass_pretranspose` implementation
checks `len(t_node.users) == 0` before erasing `t_node`.

### `_pass_pretranspose` reports "Pattern not found"

This pass targets the post-Dynamo Inductor-lowered graph where `F.linear` has been
decomposed to `aten.t(get_attr) + aten.mm`. If the pass reports no matches:
1. Verify `torch.compile(model, backend="gpt2_opt")` is being used (not eager mode).
2. Check that model weights have `K >= 512` (GPT-2 smallest projections are 768-dim,
   all above the threshold).
3. Enable `TORCH_LOGS="+inductor"` to inspect the graph before the pass runs.

### `OutOfMemoryError` after OPT-2

OPT-2 doubles the memory footprint of each Linear weight matrix by storing both the
original and the transposed copy. For GPT-2 small, the additional memory is
approximately 12 layers × (768×768 + 768×3072 + 3072×768 + 768×2304) × 2 bytes
(BF16) ≈ 540 MB. If GPU memory is tight, disable OPT-2 by removing the
`_pass_pretranspose(gm)` calls from the backend function.

### max-autotune compile takes too long

First compilation with `max_autotune=True` benchmarks multiple kernel candidates and
can take 5-15 minutes for GPT-2 small. Set a persistent cache directory:
```bash
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache_gpt2
```
Subsequent runs with identical shapes reuse cached kernel selections.

### BF16 model produces NaN outputs

GPT-2 LayerNorm uses accumulation in higher precision internally, but downstream
operations in BF16 can produce NaN if inputs have very large magnitudes. Verify:
1. `torch.backends.cuda.matmul.allow_tf32 = True` is set.
2. HuggingFace model was loaded with `from_pretrained` (not randomly initialized).
3. Input `input_ids` are within `[0, vocab_size)` range.

---

## Future Work

### SDPA Replacement (`_pass_replace_sdpa`) — Not Applied

GPT-2 in HuggingFace PyTorch 2.x already calls `F.scaled_dot_product_attention`
internally (since transformers 4.36+). If the baseline profile shows
`aten::_efficient_attention_forward`, this indicates SDPA is already active. A manual
SDPA replacement pass would detect `softmax(qk_matmul * scale) @ v` patterns — if
the HuggingFace version does not use SDPA, this pass would provide significant
additional speedup for the attention operator (currently 3.18% of wall time).

Infrastructure needed: no additional infrastructure — the pass is available in
`knowledge/fx-patterns.md` Pattern 2 and can be added to the per-rep loop.

### LayerNorm-Linear Fusion — Requires Custom Triton Kernel

GPT-2 has the pattern `LayerNorm → Linear` in every block (pre-norm architecture).
Fusing these into a single kernel would eliminate one intermediate tensor write.

Infrastructure needed: custom Triton kernel (e.g., liger-kernel's `LN-MM` fused op,
or `flash-attn`'s `flash_attn_interface.layer_norm_linear`).

### RoPE Detection — Not Applicable to GPT-2

GPT-2 uses learned positional embeddings (`wpe`), not RoPE. This stub applies to
LLaMA/Mistral variants.
