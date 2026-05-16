# MLPActivations — Optimized Workload

## Overview

The baseline `mlp_activations.py` runs FP32 GEMMs through the cuBLAS 'Kernel2' SIMT SGEMM path on an NVIDIA RTX PRO 6000 Blackwell Server Edition (84 SMs). Every one of the 20 GEMM kernel launches shows `smsp__pipe_tensor_cycles_active = 0.0` — Tensor Cores are completely idle while consuming 98.4% of total forward-pass time. Register pressure of 200-210 regs/thread caps warp occupancy to 8-17%, preventing latency hiding.

`mlp_activations_optimized.py` implements four optimizations (one stub) targeting this bottleneck:

| Opt | Type | Where Applied | Confidence | Expected Gain |
|-----|------|---------------|------------|---------------|
| OPT-1 | BF16 dtype promotion | `get_model_and_input()` | High | ~49.2% latency reduction |
| OPT-2 | max_autotune GEMM tuning | `get_model_and_input()` (inductor config) | Medium | ~5.0% additional |
| OPT-3 | TF32 enable | Alternative to OPT-1 (not applied) | Medium | ~35.2% (if OPT-1 absent) |
| OPT-4 | reduce-overhead CUDA graphs | torch.compile mode at call site | Medium | ~3.4% additional |
| OPT-5 | Batch-padding output projections | FX stub (detect-only) | Low | ~2.4% (not implemented) |

## Quick Start

### Run optimized workload under nsys

```bash
# From project root
PYTHONPATH=/home/ubuntu/Profiler \
nsys profile \
    --trace=cuda,nvtx \
    --output=examples/mlp_activations/profiler_output/mlp_activations_opt \
    --force-overwrite=true \
    python3 nvidia/scripts/run_workload.py \
        --workload examples/mlp_activations/mlp_activations_optimized.py \
        --compile-backend mlp_activations_opt \
        --warmup-iters 2 --measure-iters 2 \
        --output-prefix examples/mlp_activations/profiler_output/mlp_activations_opt \
        --inductor-debug-dir examples/mlp_activations/profiler_output/mlp_activations_opt_inductor_debug
```

### Activate OPT-4 (CUDA graphs / reduce-overhead)

Pass `--compile-mode reduce-overhead` to `run_workload.py`. Note that `reduce-overhead` and `max-autotune` are mutually exclusive; GEMM autotuning (OPT-2) is skipped in this mode.

```bash
PYTHONPATH=/home/ubuntu/Profiler \
nsys profile --trace=cuda,nvtx \
    --output=examples/mlp_activations/profiler_output/mlp_activations_opt_cudagraph \
    python3 nvidia/scripts/run_workload.py \
        --workload examples/mlp_activations/mlp_activations_optimized.py \
        --compile-backend mlp_activations_opt \
        --compile-mode reduce-overhead \
        --warmup-iters 3 --measure-iters 3 \
        --output-prefix examples/mlp_activations/profiler_output/mlp_activations_opt_cudagraph
```

### Run validation tests

```bash
PYTHONPATH=/home/ubuntu/Profiler \
pytest examples/mlp_activations/test_mlp_activations_optimized.py -v
```

### Quick smoke test

```bash
PYTHONPATH=/home/ubuntu/Profiler \
python3 examples/mlp_activations/mlp_activations_optimized.py
# Expected output:
# Output shape : torch.Size([256, 512])
# Output dtype : torch.bfloat16
```

## Optimizations Table

| ID | Target Operators | Bottleneck | Transformation | Expected Impact |
|----|-----------------|------------|----------------|-----------------|
| OPT-1 | All `aten::mm` / `aten::addmm` (8 shapes) | FP32 SIMT SGEMM, 0% Tensor Core, 200-210 regs/thread, 8-17% occupancy | Cast model + input to `bfloat16` in `get_model_and_input()` before `torch.compile`. Forces cuBLAS to select BF16 HMMA TC path. | ~49.2% latency reduction (879,835 ns) |
| OPT-2 | `aten::mm` [256x2048 @ 2048x512] (op_id=9,19) and [256x512 @ 512x2048] (op_id=3,13) | Sub-4-wave SM dispatch, eligible_cycles_pct=12% (warp scheduler stalled 88% of cycles) | Set `torch._inductor.config.max_autotune = True`, `max_autotune_gemm = True`, backends=TRITON,CUTLASS | ~5.0% additional (89,600 ns) |
| OPT-3 | All `aten::mm` (6 shapes) | Same as OPT-1 — alternative for FP32 accuracy requirements | `torch.backends.cuda.matmul.allow_tf32 = True` (not applied; superseded by OPT-1) | ~35.2% (630,000 ns) — only if OPT-1 not used |
| OPT-4 | All kernels (~12 launches) | CPU-side cuLaunchKernel overhead ~60-120 µs per forward pass | `torch.compile(..., mode='reduce-overhead')` — CUDA graph capture | ~3.4% additional (60,000 ns) |
| OPT-5 | `aten::mm` [256x2048 @ 2048x512] (op_id=9,19) | M=256 → 176 thread blocks on 84 SMs, 2.1 blocks/SM, eligible_cycles_pct=12% | Pad M from 256→384 before mm, slice output back to [256,512] (stub only) | ~2.4% additional (43,500 ns) |

## Architecture

### Backend: `mlp_activations_opt`

The backend is registered via `@register_backend` from `torch._dynamo`. It wraps `compile_fx` from `torch._inductor.compile_fx` (the callable function, not the module).

```
torch.compile(model, backend="mlp_activations_opt")
         │
         ▼
  mlp_activations_opt(gm, example_inputs)
         │
         ├── UniqueSubgraphRegistry(gm)
         │        └── build_partition_equivalence_map() → {} (empty for flat MLP)
         │
         ├── [flat path] _stub_pass_pad_output_projections(gm)
         │        └── OPT-5: detect output-projection nodes, log info, return gm unchanged
         │
         └── compile_fx(gm, example_inputs)  ← Inductor
```

### Non-graph path (in `get_model_and_input()`)

```
get_model_and_input()
    │
    ├── torch._inductor.config.max_autotune = True     ← OPT-2
    ├── torch._inductor.config.max_autotune_gemm = True
    │
    ├── model.to(torch.bfloat16)                       ← OPT-1
    └── x.to(torch.bfloat16)                           ← OPT-1
```

### Dedup handling

`MLPActivations` is a sequential four-layer MLP. `UniqueSubgraphRegistry.build_partition_equivalence_map()` detects no repeated subgraphs and returns `{}`. The flat compile path is always taken. The dedup path is present for structural completeness (identical interface to `conv_block_optimized.py` and `sdpa_attention_optimized.py`).

## Key Design Decisions

### 1. All primary optimizations are non-graph

OPT-1 (BF16), OPT-2 (max_autotune), OPT-3 (TF32), and OPT-4 (CUDA graphs) are all applied via dtype casts, `torch._inductor.config` flags, or `torch.compile` mode arguments — none require FX graph surgery. This is intentional: the MLP has no fused activation patterns, no repeated QKV projections, and no BatchNorm to fold. The graph structure is already near-optimal; the bottleneck is purely the cuBLAS dispatch path.

### 2. OPT-1 vs OPT-3 are mutually exclusive

Per `optimizations.json` global_notes: "Apply OPT-1 (BF16) for maximum throughput gain (~49% reduction). Apply OPT-3 (TF32) as a less invasive alternative that preserves FP32 external API (~35% reduction). Do not stack both."

This file implements OPT-1 (higher impact). To switch to OPT-3, remove the `model.to(bfloat16)` and `x.to(bfloat16)` calls and add:
```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

### 3. OPT-2 config is set before torch.compile, not inside the backend

`torch._inductor.config` is process-global state. The autotuning config must be set before Dynamo triggers Inductor compilation, which happens at the first `torch.compile` invocation. Setting it inside `get_model_and_input()` guarantees it is active before the caller invokes `torch.compile`. The backend function `mlp_activations_opt` runs inside Dynamo's compilation pipeline and delegates to `compile_fx` — by that point `max_autotune` is already set.

### 4. OPT-4 is activated at the torch.compile call site, not in the backend

The compile `mode` argument (`"max-autotune"`, `"reduce-overhead"`) is Dynamo-level configuration that routes to different Inductor lowering strategies. The backend cannot set it retroactively. When using `run_workload.py`, pass `--compile-mode reduce-overhead` to activate OPT-4. The default in this file's `__main__` block is `"max-autotune"` to demonstrate OPT-2.

### 5. OPT-5 is a low-confidence stub — detect-only, never transforms

Batch-padding for [256x2048 @ 2048x512] GEMMs requires `node.meta['tensor_meta']` for shape matching. This metadata is populated only after Inductor's shape propagation pass (post-lowering). The `@register_backend` function receives the pre-Inductor FX graph where `tensor_meta` is absent on most nodes. A complete implementation requires a post-Inductor FX pass hook that is outside the current backend scope. The stub logs diagnostic information to confirm the detection path works when shape metadata becomes available.

### 6. Weight node detection note

At the pre-Inductor FX IR level (where `@register_backend` receives the graph), all `nn.Module` parameters are lifted to `placeholder` nodes. There are no `get_attr` nodes for weights in this flat graph. Weight tensors are in `example_inputs`, matched positionally to placeholder nodes. Any future FX pass that reads weight values must build `ph_to_tensor = {ph: t for ph, t in zip(placeholders, example_inputs)}`.

## Troubleshooting

### `TypeError: 'module' object is not callable`
```
# Wrong — imports the module
from torch._inductor import compile_fx

# Correct — imports the callable function
from torch._inductor.compile_fx import compile_fx
```

### `AssertionError` in `_try_get_metadata_from_dynamo`
If a future pass calls `gm.register_buffer()`, strip Dynamo source-map metadata first:
```python
gm.meta.pop("dynamo_compile_id", None)
if hasattr(gm, "_param_name_to_source"):
    del gm._param_name_to_source
gm.register_buffer("my_buffer", tensor)
```
See `sdpa_attention_optimized.py` `_pass_fuse_qkv` for the full pattern.

### `gm.graph.lint()` fails after node mutation
Ensure mutations follow this order:
1. `node.replace_all_uses_with(new_node)` — before erase
2. `gm.graph.erase_node(node)` — after replace
3. `gm.graph.lint()` — after all mutations
4. `gm.recompile()` — after lint

### max_autotune significantly increases compile time
The first `torch.compile` call with `max_autotune=True` benchmarks multiple Triton and CUTLASS configs per GEMM shape. For this workload (4 unique shapes), expect 2-10 minutes compile time on first run. Subsequent runs use the Inductor cache. Set `TORCHINDUCTOR_CACHE_DIR` to a persistent location to avoid re-tuning.

### BF16 precision loss
The tanh final activation bounds outputs to (-1, 1). BF16 has 7 mantissa bits (~0.78% relative error vs FP32's 23 bits). For inference serving, BF16 precision is generally acceptable. If accumulation precision is required, use OPT-3 (TF32) instead of OPT-1.

## Future Work

### OPT-5: Full batch-padding implementation
Requires a post-Inductor FX pass. Infrastructure needed:
1. Register a custom Inductor lowering pass via `torch._inductor.lowering.register_lowering` or post-graph hook
2. At that level, `node.meta['tensor_meta']` is populated and shapes can be verified
3. Insert `aten.constant_pad_nd(x, [0,0,0,128], 0.0)` before the target `mm` node
4. Insert `aten.slice.Tensor(out, 0, 0, 256)` after the `mm` node
5. Expected gain: ~2.4% (~43,500 ns) for the two [256x2048 @ 2048x512] projections

### GQA-style weight sharing detection
If the model is extended to use grouped-query attention or weight tying, `UniqueSubgraphRegistry` can detect the repeated structure automatically and the dedup path (already in the backend) will activate without code changes.

### Persistent GEMM autotuning cache
Configure `torch._inductor.config.cache_dir` to a project-specific directory so max_autotune results persist across Python process restarts, reducing repeated compile latency to near-zero.
