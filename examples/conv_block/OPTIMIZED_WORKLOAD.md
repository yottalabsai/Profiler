# ConvBlock Optimized Workload

## Overview

`conv_block_optimized.py` applies four operator-level optimizations to the
baseline VGG-style `ConvBlock` model, derived from ncu profiling on an A100-SXM4-80GB
running FP32 in NCHW layout.

The optimizations are applied in order at the `nn.Module` level before
`torch.compile()` is called — no custom FX graph passes are needed because the
transformations either (a) modify tensor metadata (memory format, dtype) or (b)
mutate the `nn.Module` tree directly (BN fold).

| ID | Name | Target Op | Mechanism | Confidence |
|---|---|---|---|---|
| OPT-3 | BN fold | `aten::_native_batch_norm_legit_no_training` (23.9%) | Fold BN weight/bias/running-stats into preceding Conv2d; remove BN from model | High |
| OPT-1 | channels_last | `aten::cudnn_convolution` (66.6%) | Eliminate cuDNN convertTensor_kernel; route to NHWC-native cuDNN path | High |
| OPT-2 | BF16 | `aten::cudnn_convolution`, `aten::addmm` (71.6% combined) | Route conv to sm80_xmma_gemm_bf16bf16 HMMA path; reduce register pressure from 238 to ~120 regs/thread | High |
| OPT-4 | max-autotune + TF32 | All conv and GEMM ops | Exhaustive cuDNN/cuBLAS algorithm search; TF32 fallback | Medium |

**Conservative expected wall-time reduction:** ~50.9% (2.1 ms → ~1.0 ms at profiled input size)

---

## Quick Start

### Run optimized workload directly

```bash
cd /root/Profiler
PYTHONPATH=/root/Profiler python3 examples/conv_block/conv_block_optimized.py
```

### Profile optimized workload (Phase 1 — correlation pass)

```bash
PYTHONPATH=/root/Profiler python3 nvidia/scripts/run_workload.py \
    --workload examples/conv_block/conv_block_optimized.py \
    --output-prefix runs/conv_block_opt \
    --inductor-debug-dir runs/conv_block_opt_inductor \
    --correlation-pass
```

### Profile optimized workload (Phase 2 — nsys capture)

```bash
nsys profile --trace=cuda,nvtx --output=runs/conv_block_opt \
    python3 nvidia/scripts/run_workload.py \
        --workload examples/conv_block/conv_block_optimized.py \
        --output-prefix runs/conv_block_opt \
        --inductor-debug-dir runs/conv_block_opt_inductor \
        --warmup-iters 3 --measure-iters 10
```

### Run tests

```bash
PYTHONPATH=/root/Profiler pytest examples/conv_block/test_conv_block_optimized.py -v
```

### Compare baseline vs. optimized

```bash
# Baseline
PYTHONPATH=/root/Profiler python3 nvidia/scripts/run_workload.py \
    --workload examples/conv_block/conv_block.py \
    --output-prefix runs/conv_block_base \
    --warmup-iters 3 --measure-iters 10

# Optimized (use same warmup/measure counts for fair comparison)
PYTHONPATH=/root/Profiler python3 nvidia/scripts/run_workload.py \
    --workload examples/conv_block/conv_block_optimized.py \
    --output-prefix runs/conv_block_opt \
    --warmup-iters 3 --measure-iters 10
```

---

## Optimizations Table

| ID | Step | Location | Target Op | Expected Latency Reduction |
|---|---|---|---|---|
| OPT-3 | 1 | `get_model_and_input()` pre-compile | `aten::_native_batch_norm_legit_no_training` (985,505 ns, 70 kernels) | 591,303 ns conservative (14.4%) |
| OPT-1 | 2 | `get_model_and_input()` pre-compile | `aten::cudnn_convolution` convertTensor overhead | 329,139–548,565 ns (8–13.3%) |
| OPT-2 | 3 | `get_model_and_input()` pre-compile | `aten::cudnn_convolution` + `aten::addmm` | 1,177,766 ns conservative (28.6%) |
| OPT-4 | 4 | `torch.compile(mode='max-autotune')` | All conv + GEMM | 3–8% estimated |

**Combined conservative estimate:** 2,098,242 ns / 50.9% wall-time reduction.

---

## Architecture

### Why no FX pass is needed

`torch.compile()` backends receive an FX graph **after** TorchDynamo has traced
the model.  For ConvBlock there are three categories of optimization:

1. **Pre-compile `nn.Module` mutation (OPT-3, BN fold):** TorchDynamo traces
   the *mutated* model, so the BN nodes never appear in the FX graph at all.
   The `fold_all_bn()` function walks the `nn.Module` tree before compile and
   replaces each `BatchNorm2d` with `nn.Identity`, absorbing its parameters
   into the preceding `Conv2d`.

2. **Tensor metadata (OPT-1 channels_last, OPT-2 BF16):** `memory_format` and
   `dtype` are tensor properties that TorchDynamo does not trace as graph
   operations.  They must be set before `torch.compile()` is called so that
   Inductor sees the correct metadata when selecting cuDNN/cuBLAS kernels.

3. **Compile mode (OPT-4 max-autotune):** The `mode='max-autotune'` argument is
   passed at the `torch.compile()` call site in the profiling driver.

### Backend structure

The `conv_block_opt` backend registered with `@register_backend` follows the
dedup-aware structure mandated by the project:

- It constructs a `UniqueSubgraphRegistry` to detect repeated layer structure.
- For ConvBlock (no repeated layers), `equiv_map` is empty and the backend takes
  the **flat compile path**, calling `compile_fx(gm, example_inputs)` directly.
  This preserves cross-layer Inductor fusion opportunities.
- For hypothetical models with repeated layers that happen to use this backend,
  the dedup path compiles each unique representative partition once and shares the
  compiled callable with structural duplicates.

### Optimization application order

```
get_model_and_input():
  1. baseline ConvBlock().to("cuda").eval()
  2. fold_all_bn(model)              # OPT-3: fold BN into Conv2d, remove BN modules
  3. model.to(channels_last)         # OPT-1: NHWC layout
  4. x.to(channels_last)
  5. model.to(bfloat16)              # OPT-2: BF16 dtype
  6. x.to(bfloat16)
  7. allow_tf32 = True               # OPT-4 partial: TF32 fallback
  8. return model, x

torch.compile(model, backend="conv_block_opt", mode="max-autotune")  # OPT-4 full
  → conv_block_opt backend
    → UniqueSubgraphRegistry(gm)
    → equiv_map = {} (no repeated layers)
    → flat path: compile_fx(gm, example_inputs)
```

---

## Key Design Decisions

### OPT-3 as nn.Module mutation, not FX pass

`aten::_native_batch_norm_legit_no_training` returns a **3-tuple**
`(output, save_mean, save_invstd)`.  The `replace_pattern` API in
`torch.fx.subgraph_rewriter` cannot match tuple-returning patterns because the
output node matching requires a single return value.  The manual FX alternative
(detecting the BN call node and re-wiring its tuple outputs) is fragile — it
depends on the exact decomposition Inductor applies.

Folding BN into Conv at the `nn.Module` level before tracing is both simpler
and more reliable: TorchDynamo never generates BN nodes, so there is nothing to
match or remove in the FX graph.

### OPT-1/OPT-2 as tensor metadata, not FX pass

Memory format and dtype are tensor properties that `torch.compile()` reads from
the concrete input tensors.  Applying them inside an FX pass would require
inserting `aten.to()` nodes into the graph and re-running Inductor's dtype
propagation — complex and fragile.  Setting them before `torch.compile()` lets
Dynamo capture the correct metadata on the first trace.

### Fold arithmetic in FP32

BN running statistics (`running_mean`, `running_var`) accumulate in FP32 during
training.  The fold formula involves a square root and division; truncating
these to BF16 before the fold introduces unnecessary error.  The fold is done in
FP32, then the result is cast back to `conv.weight.dtype` (FP32 at fold time)
before storing.  The subsequent `model.to(bfloat16)` then casts the folded
weights atomically.

### max-autotune compilation time

`mode='max-autotune'` increases compilation time significantly (minutes vs.
seconds for the default mode) because Inductor benchmarks all available cuDNN
and cuBLAS algorithms for each unique shape.  It is appropriate for production
inference with stable input shapes but should not be used during iterative
development.  The first compile is slow; subsequent runs use the Inductor cache.

---

## Troubleshooting

### `TypeError: 'module' object is not callable` at `compile_fx()`

Cause: `from torch._inductor import compile_fx` imports the *module*, not the
callable function.

Fix:
```python
# WRONG
from torch._inductor import compile_fx

# CORRECT
from torch._inductor.compile_fx import compile_fx
```

### `AssertionError: CUDA required`

The workload requires a CUDA GPU.  Check:
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

### BN fold does not eliminate all BN modules

`fold_all_bn()` handles two cases: `nn.Sequential` children where Conv and BN
are adjacent entries, and direct attributes of a non-Sequential module where an
attribute named after a BN follows one named after a Conv.  If a custom module
stores Conv and BN in a non-standard layout (e.g. a dict, a list), the fold will
miss it.

Debug: after calling `get_model_and_input()`, check:
```python
import torch.nn as nn
model, _ = get_model_and_input()
remaining = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.BatchNorm2d)]
print(remaining)  # should be empty
```

If any BN modules remain, inspect the model structure with `print(model)` and
extend `fold_all_bn()` to handle the specific layout.

### `graph.lint()` failure after FX mutations

If a future FX pass is added that mutates the graph, always:
1. Snapshot nodes with `list(gm.graph.nodes)` before iterating.
2. Call `node.replace_all_uses_with(new_node)` BEFORE `gm.graph.erase_node(node)`.
3. Call `gm.graph.lint()` after ALL mutations, then `gm.recompile()`.

### Output shape mismatch after BN fold

If the folded model produces a different output shape, check that no BN module
is being used for its normalization effect on spatial dimensions (not applicable
for standard `BatchNorm2d`, which is channel-wise).  Verify with:
```python
base_model, base_x = baseline_get_model_and_input()
opt_model, opt_x   = get_model_and_input()
print(base_model(base_x).shape, opt_model(opt_x).shape)
```

---

## Future Work

The following optimizations are not implemented because they require
infrastructure not yet available or provide marginal benefit for this model:

- **Per-layer autotuning cache warm-up stub:** The `max-autotune` first-compile
  overhead could be reduced by pre-warming the Inductor cache from a saved
  compilation artifact.  Requires `torch._inductor.config.cache_dir` to be set
  and the cache serialized between runs.

- **FP8 quantization (post-training):** A100 does not support FP8 natively;
  H100 (Hopper, sm_90) does.  For future H100 deployment: quantize conv weights
  to `torch.float8_e4m3fn` and use `torch._scaled_mm` for the GEMM path.

- **cuDNN frontend v8 persistent kernel:** For the large 256-channel stage-3
  conv (128×256, 16×16 spatial), a persistent cuDNN kernel that fuses conv +
  BN-folded bias + ReLU into a single kernel can be selected by setting
  `torch.backends.cudnn.benchmark = True`.  `max-autotune` partially covers
  this but does not guarantee persistent kernel selection.
