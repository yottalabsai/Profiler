# TransformerStack Optimized Workload

## Overview

This workload applies four operator-level optimizations to a GPT-2-style 8-layer
TransformerStack (HIDDEN=512, N_HEADS=8, FFN_DIM=2048, SEQ_LEN=128, BATCH=4) running
on NVIDIA A100-SXM4-80GB. All optimizations are derived from ncu profiling evidence.

The dominant bottleneck is FP32 GEMMs with zero Tensor Core utilization
(`smsp__pipe_tensor_cycles_active = 0.0` on both `ampere_sgemm_128x32_tn` for
linear projections and `ampere_sgemm_128x128_nn` for attention matmuls). The
BF16 cast alone is expected to deliver a 4-6x speedup on GEMM-bound operations,
accounting for ~41.9% of total attributed wall time.

## Quick Start

```bash
# 1. Run the optimized workload with the custom backend
python nvidia/scripts/run_workload.py \
    --workload examples/transformer_stack/transformer_stack_optimized.py \
    --compile-backend transformer_stack_opt \
    --warmup-iters 3 \
    --measure-iters 10

# 2. Run validation tests (no GPU required beyond CUDA device detection)
cd /root/Profiler
python examples/transformer_stack/test_transformer_stack_optimized.py

# 3. Syntax check only
python -m py_compile examples/transformer_stack/transformer_stack_optimized.py
```

## Optimizations Table

| ID    | Type            | Target Ops                                      | Confidence | Expected Impact                            | Location               |
|-------|-----------------|--------------------------------------------------|------------|--------------------------------------------|------------------------|
| OPT-1 | BF16 dtype cast | aten::mm (432 launches), aten::bmm (144 launches) | HIGH       | 4-6x GEMM speedup; ~41.9% total time reduction | `get_model_and_input()` |
| OPT-3 | SDPA replacement | 2 BMM + softmax per layer (24 kernels/fwd) | MEDIUM     | 3 kernels → 1 FlashAttention kernel; ~3-6% incremental | Source-level (class override) |
| OPT-2 | QKV weight fusion | aten::mm × 3 per layer (24 launches/fwd) | HIGH       | 3 launches → 1; waves/SM 0.09 → 0.27 per attn block; ~2-4% incremental | FX pass `pass_fuse_qkv` |
| OPT-4 | max-autotune    | aten::mm (all GEMM shapes: 512×512, 512×2048)   | MEDIUM     | 2-5% incremental; Triton autotuner sweeps tile configs | `get_model_and_input()` Inductor config |

**Application order:** OPT-1 → OPT-3 (source) → OPT-2 (FX pass) → OPT-4 (compile config)

OPT-2 must follow OPT-1: the fused weight buffer is registered in BF16; applying
QKV fusion before BF16 cast would create an FP32 fused weight that mismatches BF16
activations at runtime.

## Architecture

### Dedup-Aware Backend

This model has 8 structurally identical `TransformerLayer` partitions. The backend
uses `UniqueSubgraphRegistry` to split the FX graph by layer and group partitions by
structural signature:

```
UniqueSubgraphRegistry splits gm → 8 partitions
    modules_0  (unique representative)
    modules_1 … modules_7  (duplicates)
```

FX passes are applied only to `modules_0`, then propagated to `modules_1…7`. This
avoids redundant graph surgery and reduces compilation overhead.

### Pass Classification

| Pass | Category | Reason |
|------|----------|--------|
| BF16 cast | Non-graph | dtype is a tensor property invisible in Aten IR; applied in `get_model_and_input()` |
| SDPA replacement | Source-level (not an FX pass) | Inserting SDPA at the Python level lets Dynamo trace it as a single op before Inductor lowering; avoids fragile pattern matching against the decomposed softmax (exp+sum+div) |
| QKV fusion | Manual per-rep | `register_buffer` required to store fused weight; `replace_pattern` cannot handle the `cat()` that creates a new tensor |
| max-autotune | Non-graph (compile config) | Inductor config flag, not a graph transformation |

### Backend Control Flow

```
transformer_stack_opt(gm, example_inputs)
├── UniqueSubgraphRegistry(gm)         → splits 8 layers
├── build_partition_equivalence_map()  → {modules_1: modules_0, …, modules_7: modules_0}
├── equiv_map non-empty → dedup path
│   ├── for each unique rep (modules_0):
│   │   ├── pass_fuse_qkv(rep_mod)              # QKV fusion
│   │   └── pass_fuse_ln_linear_stub(rep_mod)   # detection-only stub
│   ├── for each duplicate:
│   │   └── pass_fuse_qkv(dup_mod)              # propagate
│   ├── _capture_partition_inputs(registry.split, example_inputs)
│   └── compile_fx(rep_mod, partition_inputs)   # unique rep only
│       dup_mod.forward = compiled              # share callable
└── return lambda *args: registry.split(*args)
```

If `equiv_map` is empty (no repeated layers — e.g. model has been restructured),
the backend falls back to flat compile applying passes to the full graph.

## Key Design Decisions

### OPT-1: BF16 in get_model_and_input(), not the backend

dtype is a tensor property resolved at cuBLAS kernel selection time, below the
FX IR level. Dynamo traces dtype as a static property: changing dtype after
compilation forces a recompile. The check `next(model.parameters()).dtype != torch.bfloat16`
guards against redundant application if the baseline already casts.

### OPT-3: Source-level class override, not FX pattern matching

The optimizations.json recommends source-level SDPA replacement as "strongly
preferred" over the FX alternative. The FX alternative is fragile because
Inductor decomposes `softmax` into `exp + sum + div` before FX passes run —
`replace_pattern` never sees a `softmax` node. Rather than write a brittle
multi-node chain matcher, this file overrides `SelfAttention` with
`_SelfAttentionSDPA`, which calls `F.scaled_dot_product_attention` directly.
Dynamo then traces it as the native op `aten.scaled_dot_product_attention`,
which SDPA dispatcher routes to FlashAttention-2 or efficient-attention on A100.

### OPT-2: Manual per-rep (not FxPassRunner.apply_pass)

`FxPassRunner.apply_pass` uses `replace_pattern`, which cannot handle
transformations that create new tensors via `register_buffer`. QKV fusion
requires registering a new fused weight tensor, so it is implemented as a
manual `pass_fuse_qkv` function applied in the per-rep loop.

### OPT-4: max_autotune config applied early

`torch._inductor.config.max_autotune = True` must be set before `compile_fx`
is called inside the backend. Setting it in `get_model_and_input()` before
`torch.compile` is invoked ensures the config is active when Inductor launches.

### Stub: LayerNorm-Linear fusion

Each TransformerLayer has two LayerNorm→Linear chains. Fusing these would
eliminate intermediate DRAM writes between LayerNorm and GEMM, but requires
a custom Triton kernel (e.g. liger-kernel `FusedLinearCrossEntropy` or a
hand-written LN+MM epilogue). The stub `pass_fuse_ln_linear_stub` detects the
pattern and logs a warning without modifying the graph.

## Troubleshooting

### TypeError: 'module' object is not callable
**Cause:** Wrong compile_fx import.
```python
# WRONG — imports the module
from torch._inductor import compile_fx

# CORRECT — imports the callable function
from torch._inductor.compile_fx import compile_fx
```

### gm.graph.lint() raises "Node used before defined"
**Cause:** Node insertion after a node that appears after its consumer.
**Fix:** Always use `inserting_after(anchor)` where `anchor` is the last
node in the group being replaced. Check that `ordered` (sorted by graph
position) places the anchor after all mm nodes being fused.

### pass_fuse_qkv: "No 3-mm groups found"
**Cause:** Pattern not matched. Common reasons:
- OPT-3 (SDPA) was not applied — the SDPA projection path uses view+transpose
  wrapping that may reorder which `mm` nodes share the same input.
- The model was traced with SDPA path but the QKV projections still produce
  3 separate mm nodes consuming the same `ln1` output — verify by printing
  `input_to_mms` keys.
- Graph was split by UniqueSubgraphRegistry; each partition only contains one
  layer's nodes, so the mm group is 3 (not 24). Verify `len(mm_nodes) >= 3`.

### Shape mismatch after QKV fusion
**Cause:** Weight ordering in the fused buffer doesn't match expected Q/K/V order.
**Fix:** The `ordered` list is sorted by position in `gm.graph.nodes`, which
reflects the order in which q_proj, k_proj, v_proj appear in the traced graph.
For `_SelfAttentionSDPA`, the order is q → k → v (matching the source code).
If the order changes (e.g. due to Dynamo reordering), the split slices will
produce Q from the wrong weight. Verify by comparing output of compiled vs.
uncompiled model with a fixed input.

### max-autotune warning: "could not set max_autotune"
**Cause:** torch._inductor.config API may differ across PyTorch versions.
**Fix:** Check `dir(torch._inductor.config)` for the correct attribute name.
In some versions it is `max_autotune_gemm_backends` instead of `max_autotune_gemm`.

### UniqueSubgraphRegistry: "0 duplicate partitions, flat compile path"
**Cause:** The model was not split into per-layer partitions — either the
layer_graph_splitter did not detect NVTX ranges, or the model uses a single
forward without per-layer structure.
**Fix:** The flat path still applies `pass_fuse_qkv` to the full graph and
delegates to Inductor. Functionality is preserved; only the dedup optimization
is skipped.

## Future Work

| Stub / Optimization | Infrastructure Required |
|---------------------|------------------------|
| LayerNorm-Linear fusion (`pass_fuse_ln_linear_stub`) | Custom Triton kernel fusing LN normalization epilogue into GEMM prologue (e.g. liger-kernel `LigerLayerNorm` + `LigerFusedLinear`, or triton-flash-attn LN+MM kernel) |
| Rotary position embedding (RoPE) fusion | This model does not use RoPE, but future variants may; would require a custom Triton kernel fusing RoPE application into the Q/K projection |
| Persistent kernel for LayerNorm | At SEQ_LEN=128, BATCH=4, LayerNorm grids are small; a persistent Triton kernel with stream-K scheduling could improve SM utilization for the `triton_per_fused_native_layer_norm` kernels observed in the profile |
| torch.compile with fullgraph=True | Add `fullgraph=True` to the torch.compile call to prevent graph breaks at layer boundaries; requires removing any Python-level control flow that Dynamo cannot trace |
