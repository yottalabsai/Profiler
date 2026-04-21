# ConvBlock Optimized Workload

Custom `torch.compile()` backend implementing 5 operator-level optimizations
derived from NCU profiling of the baseline `conv_block.py` workload.

---

## Overview

The baseline ConvBlock profile (FP32, batch=16, 64×64 spatial) shows four
dominant cost buckets:

| Operator | Wall time share | Key pathology |
|---|---|---|
| `aten::cudnn_convolution` | 81.9% | `convertTensor_kernel` overhead (FP32→TF32 coercion), low-occupancy GEMM |
| `aten::batch_norm` | 15.6% | 7 Triton kernels per call including redundant reductions |
| `aten::addmm` | 1.4% | `gemmSN_TN_kernel` (4 thread blocks, 0% tensor core) |
| `aten::conv2d` | 1.1% | Degenerate bias-scatter kernel reading 12 bytes |

This workload applies targeted FX graph transformations and runtime flags to
address each pathology.

---

## Quick Start

```bash
# Smoke test (no profiler needed)
python conv_block_optimized.py

# Full test suite
python test_conv_block_optimized.py

# Profile (requires operator-profiler + ncu)
operator-profiler profile conv_block_optimized.py \
    --model-name ConvBlock \
    --compile-mode inductor \
    --output runs/conv_block_optimized

operator-profiler map runs/conv_block_optimized.manifest.json \
    --script scripts/run_workload.py \
    --ncu-sudo \
    --script-args --workload conv_block_optimized.py \
                  --compile-backend convblock_opt

# Side-by-side comparison
python scripts/run_workload.py --workload conv_block.py --compile-backend inductor
python scripts/run_workload.py --workload conv_block_optimized.py --compile-backend convblock_opt
```

---

## Optimization Summary

| ID | Target operators | Pathology | Transformation | Expected gain | Confidence |
|---|---|---|---|---|---|
| OPT-1 | All 30 `aten::cudnn_convolution` | `convertTensor_kernel` fires 2× per conv call (FP32→TF32 coercion), 60 launches, 222 µs | Cast model + input to FP16 in `get_model_and_input()` via `model.to(torch.float16)` | −222 µs (10.5% of total) | High |
| OPT-2 | Conv stages 2 & 3 (64→128, 128→256) | cuDNN implicit GEMM at 8.3% warp occupancy (150 regs/thread, grid 512–1024 blocks) | `cudnn.benchmark = True` + `max_autotune=True`; Triton conv substitution is a future-work stub | −12–20% estimated | Medium |
| OPT-3 | `aten::batch_norm` (all 3 BN nodes) | Inductor decomposes to 7 Triton kernels including 2 redundant reduction passes; DRAM at 67.6% | Constant-fold BN to `x * scale + bias_eff` in FX graph; inductor fuses with ReLU into 1 triton_poi | −260 µs (12.3%); 40% DRAM reduction | High |
| OPT-4 | `aten::conv2d` (stage 1 bias-scatter) | Two degenerate triton_poi kernels; `convolution_1` reads only 12 bytes from DRAM, dominated by launch latency | Absorb conv bias into adjacent BN bias constant; zero out conv bias | −23 µs (1.1%); 20 kernel launches eliminated | High |
| OPT-5 | `aten::addmm` (linear classifier, 10 nodes) | `gemmSN_TN_kernel`: 4 thread blocks, 0.2% SM throughput, 0% tensor core | Pad weight N-dim to multiple of 16 (10→16); slice output post-GEMM | 2–4× per kernel; −15–20 µs absolute | Medium |

**Total estimated savings: ~23–25% wall time reduction** (after OPT-1 and OPT-3 dominate).

---

## Architecture

### FX Graph Passes

Each optimization is an independent function with signature
`(gm: fx.GraphModule) -> fx.GraphModule`.  Passes operate at the Aten IR
level and are model-agnostic — they match on `torch.ops.aten.*` targets, not
on PyTorch module class names.

```
get_model_and_input()          ← non-graph: FP16 cast, cudnn.benchmark
         │
         ▼
torch.compile(model, backend="convblock_opt")
         │
         ▼
convblock_opt(gm, example_inputs)
    │
    ├─ pass_absorb_conv_bias_into_bn   [OPT-4]
    ├─ pass_fold_bn_constants          [OPT-3]
    ├─ pass_pad_linear_weights         [OPT-5]
    ├─ pass_cudnn_autotune_stub        [OPT-2 — detection + logging only]
    │
    └─ compile_fx(gm, example_inputs) ← inductor backend
```

### Pass Order Rationale

`pass_absorb_conv_bias_into_bn` must run **before** `pass_fold_bn_constants`
because absorption updates the `bn_bias` buffer.  If the BN folder runs
first, it reads the old (pre-absorption) bias and the conv bias never gets
folded.

### Backend Registration

```python
from torch._dynamo import register_backend

@register_backend
def convblock_opt(gm, example_inputs):
    ...
    return compile_fx(gm, example_inputs)
```

The `@register_backend` decorator makes `"convblock_opt"` available as a
`torch.compile(backend=...)` string and in `torch._dynamo.list_backends()`.

---

## Why a Custom Backend

| Approach | Tradeoff |
|---|---|
| Edit model source | Couples optimization to a specific architecture; breaks when the model class changes |
| Post-hoc `state_dict` manipulation | Cannot eliminate kernel launches; operates on weights, not graph |
| Custom FX passes + inductor delegate | Model-agnostic, operates at Aten IR, each pass is independently removable, inductor handles codegen |

The custom backend approach matches the `operator-profiler` workflow: profile
→ identify bottleneck → write targeted pass → re-profile to verify.  Passes
degrade gracefully (log + skip) if the pattern isn't present, making them
safe to apply across model variants.

---

## Key Design Decisions

### FP16 Outside the FX Graph (OPT-1)

`model.to(torch.float16)` and `x.to(torch.float16)` are applied in
`get_model_and_input()` rather than as an FX pass.  Dtype is a property of
tensors returned by `get_attr` nodes, not an operation in the graph.
Inserting a `to(dtype)` node before every placeholder would work but is
fragile across dynamo trace modes.  The autocast approach is idempotent and
composable with `torch.autocast`.

### Idempotency Guards

Both non-graph optimizations check current state before applying:

```python
# Only cast if not already half-precision
if next(model.parameters()).dtype not in (torch.float16, torch.bfloat16):
    model = model.to(torch.float16)
```

This prevents silent double-casting if the baseline is later updated to
include its own precision changes.

### BN Constant Folding Strategy (OPT-3)

Rather than using `torch.fx.subgraph_rewriter.replace_pattern` (which
requires an exact structural match on the pattern graph), the pass walks
nodes directly and resolves `get_attr` constants from `gm.named_buffers()`
and `gm.named_parameters()`.  This is more robust to inductor's
decomposition variants (`_native_batch_norm_legit_no_training`,
`batch_norm`, etc.).

### Defensive Error Handling

Every pass wraps its body in `try/except Exception`:

```python
try:
    # pattern detection and surgery
    gm.graph.lint()
    gm.recompile()
except Exception as exc:
    logger.warning("pass_X failed: %s — skipping", exc)
return gm
```

A pass that fails to match (e.g. because inductor already folded the
operator) silently returns the unmodified graph rather than crashing the
compilation pipeline.

### OPT-2 as Stub

The cuDNN implicit GEMM occupancy problem (OPT-2) requires either:
- `cudnn.benchmark = True` to select a lower-register algorithm (applied as a flag), or
- A custom Triton convolution kernel with occupancy-aware block shapes

The Triton path is left as a stub (`pass_cudnn_autotune_stub`) that detects
the conv nodes and logs a TODO.  `cudnn.benchmark` alone may resolve the
occupancy issue for standard tile shapes; re-profiling after OPT-1 and OPT-3
will determine whether the additional Triton investment is warranted.

---

## Comparison Against Baseline

```bash
# Baseline
python scripts/run_workload.py \
    --workload conv_block.py \
    --compile-backend inductor \
    --warmup-iters 5 --measure-iters 20

# Optimized
python scripts/run_workload.py \
    --workload conv_block_optimized.py \
    --compile-backend convblock_opt \
    --warmup-iters 5 --measure-iters 20
```

Key metrics to compare in the resulting profiles:

- **Total wall time** — target: 23–25% reduction
- **`convertTensor_kernel` launches** — target: 0 (was 60)
- **Triton kernels per BN call** — target: 1 (was 7)
- **`triton_poi_fused_convolution_*` launches** — target: 0 (was 20)
- **`gemmSN_TN_kernel` tensor core %** — target: >0% (was 0%)
- **cuDNN conv warp occupancy** — target: >8.3% (was 8.3%)

---

## Verification Checklist

After profiling the optimized workload, confirm:

- [ ] `convertTensor_kernel` does not appear in kernel timeline (OPT-1)
- [ ] Total `aten::cudnn_convolution` duration is ≥10% lower
- [ ] `aten::batch_norm` dispatches ≤2 Triton kernels per call (OPT-3)
- [ ] No `triton_poi_fused_convolution_0` or `_1` kernels (OPT-4)
- [ ] `gemmSN_TN_kernel` grid is `[2,2,1]` or larger — or replaced by a tensor-core kernel (OPT-5)
- [ ] `cudnn.benchmark` log line appears at startup
- [ ] `convblock_opt backend: all passes complete` appears in logs
- [ ] No `pass_* failed` warnings in logs
- [ ] Output shape is `(16, 10)` and no NaN/Inf values

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'conv_block'`**  
Ensure `conv_block.py` is on `PYTHONPATH` or in the same directory as
`conv_block_optimized.py`.

**`TypeError: 'module' object is not callable` at compile time**  
Wrong import: `from torch._inductor import compile_fx` imports the module.
Correct: `from torch._inductor.compile_fx import compile_fx`.

**`convblock_opt` not in `torch._dynamo.list_backends()`**  
The module must be imported before `torch.compile` is called.  Import
`conv_block_optimized` explicitly before constructing the compiled model.

**`pass_fold_bn_constants: could not resolve constant tensors`**  
Inductor may have already folded or decomposed BN differently.  This is a
warning, not an error — the pass skips gracefully and inductor's default BN
handling applies.

**FP16 overflow / NaN outputs**  
Some Conv+BN weight initializations can overflow FP16 range.  Try
`torch.bfloat16` instead by changing `torch.float16` to `torch.bfloat16` in
`get_model_and_input()`.  BF16 has the same exponent range as FP32 and is
less prone to overflow.

**`--script-args` parse error with operator-profiler**  
`--script-args` uses `nargs=REMAINDER` and must be the **last** flag.  Place
all `operator-profiler map` flags (`--ncu-sudo`, `--ncu-env`, etc.) before
`--script-args`.

---

## Future Work

- **OPT-2 full implementation**: Register a Triton convolution kernel
  (`triton.autotune` with block shapes targeting <128 registers/thread) as an
  FX substitution for the two high-register cuDNN kernels in stages 2 and 3.

- **Fuse Conv+BN+ReLU into a single Triton kernel**: After BN constant
  folding, the `conv → mul → add → relu` chain is a candidate for a single
  fused Triton pointwise kernel, eliminating all intermediate tensor
  materializations.

- **Multi-stage batched addmm**: If the 10 `addmm` nodes in the classifier
  head are topologically independent, stack them into a single `aten::bmm`
  to give cuBLAS a larger effective tile and improve wave occupancy beyond
  what dimension padding alone achieves.

- **NHWC layout propagation**: Switching convolutions to channels-last layout
  (`model = model.to(memory_format=torch.channels_last)`) eliminates the
  remaining NCHW→NHWC permutations that cuDNN inserts for HMMA dispatch.
  This is complementary to OPT-1 and can be implemented as a pre-pass that
  inserts `aten.contiguous(memory_format=NHWC)` on conv input/weight nodes.