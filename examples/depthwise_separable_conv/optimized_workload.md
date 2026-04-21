# DepthwiseSepConv Optimized Workload

## Overview

`depthwise_sep_conv_optimized.py` wraps the baseline `depthwise_separable_conv.py`
(MobileNet-style 3-block DWSepConv, 32→64→128→256 channels, 56×56 spatial) with a
custom `torch.compile()` backend (`transformer_opt`) that applies five operator-level
optimisations derived from Blackwell (RTX PRO 6000 / GB202) profiling data.

The profile revealed two dominant pathologies: 51.8% GPU idle time from kernel launch
overhead (130 CUDA API calls serialised by CPU dispatch), and 8% occupancy on the
pointwise 1×1 convolutions due to cuDNN's 73.7 KB/block shared-memory allocation
preventing more than 1–2 blocks/SM. Together, these account for ~85% of addressable
wall time.

---

## Quick Start

```bash
# Syntax check
python -m py_compile depthwise_sep_conv_optimized.py

# Smoke test (eager BF16 + compilation warm-up)
python depthwise_sep_conv_optimized.py

# Verification tests
python test_depthwise_sep_conv_optimized.py

# Profile (requires operator-profiler and ncu access)
operator-profiler profile depthwise_sep_conv_optimized.py \
    --model-name DepthwiseSepConvOpt \
    --compile-mode transformer_opt \
    --output runs/dsc_opt

operator-profiler map runs/dsc_opt.manifest.json \
    --script scripts/run_workload.py \
    --ncu-sudo \
    --ncu-env PYTHONPATH=/repo \
    --script-args --workload depthwise_sep_conv_optimized.py \
                  --compile-backend transformer_opt

# Side-by-side baseline comparison
operator-profiler profile depthwise_separable_conv.py \
    --model-name DepthwiseSepConvBaseline \
    --compile-mode inductor \
    --output runs/dsc_baseline
```

---

## Optimisations Summary

| ID | Priority | Confidence | Target Ops | Implementation | Expected Impact |
|----|----------|------------|-----------|----------------|-----------------|
| OPT-001 | 1 | HIGH | All 130 kernel launches | `mode='reduce-overhead'` (CUDA Graphs) in `get_model_and_input()` | 40–50% latency reduction; ~1.75 ms wall time recovered |
| OPT-002 | 2 | HIGH | 3× `aten::cudnn_convolution` (1×1 pointwise, Kernel2) | FX pass: reshape → `aten::mm` → reshape | 25–35% on 1×1 kernels; occupancy 8% → 50%+; TC utilisation → 60%+ |
| OPT-003 | 3 | MEDIUM | 3× `aten::cudnn_convolution` (3×3 depthwise, groups=C) | Stub FX pass: detection + warning | Requires custom Triton kernel; ~20% depthwise latency reduction when implemented |
| OPT-004 | 4 | MEDIUM | All `conv → BN → hardtanh` chains | Stub FX pass: chain detection + warning | Requires Triton epilog kernel; ~30–40% DRAM reduction, ~10% BN/act latency reduction |
| OPT-005 | 5 | HIGH | All conv inputs/weights (FP32 → BF16) | `model.to(bfloat16)` + `x.to(bfloat16)` in `get_model_and_input()` | 40–80% TC throughput gain on pointwise; 50% activation memory reduction; 25–35% net kernel time |

---

## Architecture

### Custom Backend: `transformer_opt`

Registered via `@torch._dynamo.register_backend`. Invoked as:

```python
torch.compile(model, backend="transformer_opt", mode="reduce-overhead")
```

The backend receives the post-Dynamo FX graph (`GraphModule`) and runs five
sequential passes before handing the modified graph to Inductor (`compile_fx`):

```
Dynamo trace
    └─► transformer_opt backend
            ├── pass_cuda_graphs()        # OPT-001: logging / verification
            ├── pass_conv1x1_as_mm()      # OPT-002: 1×1 conv → MM (full rewrite)
            ├── pass_depthwise_triton_stub() # OPT-003: depthwise detection (stub)
            ├── pass_conv_bn_relu6_fusion()  # OPT-004: chain detection (stub)
            └── pass_annotate_bf16()      # OPT-005: BF16 propagation check
                    └─► compile_fx (Inductor) → cuDNN / Triton kernels
```

### Pass Details

**`pass_cuda_graphs` (OPT-001)**
No-op at the FX level. CUDA Graph capture is activated by `mode='reduce-overhead'`,
which wraps the compiled callable in a CUDA Graph replayer after the first warm-up
call. The pass logs confirmation that static shapes are satisfied.

**`pass_conv1x1_as_mm` (OPT-002)**
Walks the FX graph for `aten::convolution` nodes where the weight shape has spatial
dims `(1, 1)` and `groups=1`. Replaces each with:

```
input (B, C_in, H, W)
  └─► reshape([-1, C_in])           # (B*H*W, C_in)
  └─► mm(·, weight.reshape(C_out, C_in).t())  # (B*H*W, C_out)
  └─► reshape([B, C_out, H, W])     # restore spatial dims
```

Inductor then lowers `aten::mm` to a Triton GEMM with autotune tile configs
(default 64×64 or 128×128), bypassing cuDNN's monolithic 73.7 KB/block kernel.
Spatial shape is recovered from `node.meta["val"]` when available; falls back to
dynamic reshape otherwise.

**`pass_depthwise_triton_stub` (OPT-003)**
Detects `aten::convolution` nodes with `kernel_size=(3,3)` and `groups==C_out`
(depthwise pattern). Emits a `logger.warning` with a TODO note for each detected
node. No graph mutation. Full implementation requires a custom Triton kernel with
`float4` vectorised loads and an optional BN+ReLU6 epilog.

**`pass_conv_bn_relu6_fusion` (OPT-004)**
Walks the graph for `conv → batch_norm → hardtanh(0, 6)` chains with single
consumers at each stage. Logs a `WARNING` per chain with the specific node names.
No graph mutation. Full implementation requires an Inductor fusion pass that emits
a single Triton kernel with inline BN scaling and `tl.clamp`.

**`pass_annotate_bf16` (OPT-005)**
Reads `node.meta["val"].dtype` for all conv nodes. Logs `WARNING` if any conv
output is still FP32 (indicates the BF16 cast in `get_model_and_input()` did not
propagate). Logs `INFO` confirming BF16 when correct.

---

## Why a Custom Backend?

The standard approach (`torch.compile(model, mode='max-autotune')`) leaves two
addressable bottlenecks on the table:

1. **Kernel launch overhead**: Inductor does not automatically apply CUDA Graphs for
   inference in all cases. Combining `backend='transformer_opt'` with
   `mode='reduce-overhead'` makes this explicit and auditable.

2. **1×1 conv dispatch path**: `torch._inductor.config.conv_1x1_as_mm = True` exists
   but is a global flag. The FX pass version is scoped to a single compilation unit,
   is visible in the graph, and can be selectively applied (e.g., skip depthwise).

The custom backend is **model-agnostic**: passes detect patterns at the Aten IR level
rather than relying on module names or layer indices.

---

## Key Design Decisions

### BF16 Cast Outside the Graph (OPT-005)
`model.to(bfloat16)` is applied in `get_model_and_input()` before `torch.compile()`.
Dtype is a tensor property, not a graph operation — inserting `aten::to` nodes inside
the FX pass would require tracking all weight get_attr nodes and all input placeholders,
and would interact poorly with Dynamo's dtype propagation. Casting before compilation
is simpler, reliable, and gives Inductor full visibility for TC dispatch selection.

The pass checks `next(model.parameters()).dtype` before casting to avoid redundant
work if the baseline is updated to use BF16 directly.

### CUDA Graphs via `mode='reduce-overhead'` (OPT-001)
`torch.cuda.make_graphed_callables()` gives finer-grained capture control but
requires an explicit warm-up call outside the compile path and doesn't compose
naturally with other `torch.compile()` backends. `mode='reduce-overhead'` is the
correct Inductor-integrated path: it uses CUDA Graph capture internally after
warm-up with zero code changes to the forward pass.

Constraint satisfied: this model has static shapes (B=16, HW=56×56) and no Python
control flow in the forward path.

### Stub Passes for Medium-Confidence Optimisations
OPT-003 and OPT-004 are stubs rather than full implementations because both require
custom Triton kernels:

- OPT-003 (depthwise vectorised): TC engagement is structurally impossible for
  depthwise convolutions (the per-channel GEMM is rank-deficient). Gains come from
  `float4` loads and reduced memory latency — this requires writing and testing a
  Triton kernel, not an FX graph rewrite.

- OPT-004 (conv-BN-ReLU6 epilog): Fusing the BN scaling and `tl.clamp` into the
  conv output store requires an Inductor-level fusion pass operating on the
  post-lowering graph, after `aten::convolution` has been lowered to a specific
  backend kernel. The current FX graph pass runs pre-lowering where this is not yet
  actionable.

Both stubs emit actionable `WARNING` log lines with specific `TODO` instructions for
the implementer.

### Defensive Error Handling
Every pass is wrapped in `try-except Exception`. A pass failure logs a `WARNING` and
returns the unmodified `GraphModule`, allowing the remaining passes and Inductor to
proceed. This prevents a single pattern-matching failure from breaking the entire
compilation pipeline.

---

## Baseline Comparison

```bash
# Run baseline
python -c "
import torch
from depthwise_separable_conv import get_model_and_input
model, x = get_model_and_input()
model = torch.compile(model, mode='inductor')
with torch.no_grad():
    for _ in range(3): y = model(x)  # warm up
import time
t0 = time.perf_counter()
for _ in range(100):
    with torch.no_grad(): y = model(x)
torch.cuda.synchronize()
print(f'Baseline: {(time.perf_counter()-t0)*10:.2f} ms/iter')
"

# Run optimised
python -c "
import torch
from depthwise_sep_conv_optimized import get_model_and_input
model, x = get_model_and_input()
with torch.no_grad():
    for _ in range(3): y = model(x)  # warm up + CUDA Graph capture
import time
t0 = time.perf_counter()
for _ in range(100):
    with torch.no_grad(): y = model(x)
torch.cuda.synchronize()
print(f'Optimised: {(time.perf_counter()-t0)*10:.2f} ms/iter')
"
```

---

## Verification Checklist

After profiling the optimised workload, check the following in ncu / operator-profiler output:

- [ ] **Kernel count**: < 130 launches (CUDA Graph reduces dispatch overhead; MM nodes
  may reduce pointwise kernel count)
- [ ] **GPU idle fraction**: < 20% (baseline 51.8%; CUDA Graphs target ≤ 5% between
  graph-captured kernels)
- [ ] **Pointwise conv (formerly Kernel2)**:
  - [ ] No `cudnn_convolution` entries for 1×1 kernels (replaced by Triton `mm`)
  - [ ] Achieved occupancy > 30% (baseline 8.1–8.6%)
  - [ ] TC utilisation > 40% (baseline 18.7–45.3% FP32; BF16 target 60%+)
  - [ ] Shared memory/block < 20 KB (baseline 73.7 KB)
- [ ] **Depthwise conv** (`conv2d_c1_k1_nhwc`):
  - [ ] TC utilisation still 0% (structurally impossible; verify no regression)
  - [ ] DRAM throughput maintained or improved
- [ ] **Dtype**: all conv kernels show BF16 inputs (check `--metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum` for tensor widths)
- [ ] **Output correctness**: compare `y` shapes and spot-check values vs. FP32 baseline
  (allow ~1e-2 absolute tolerance for BF16 accumulation error)

---

## Troubleshooting

**`TypeError: 'module' object is not callable` at compile time**
The `compile_fx` import must come from the submodule, not the package:
```python
# WRONG
from torch._inductor import compile_fx
# CORRECT
from torch._inductor.compile_fx import compile_fx
```

**`transformer_opt` not in `torch._dynamo.list_backends()`**
Import the optimised module before checking: the `@register_backend` decorator
runs at import time, not at module definition time.

**BF16 warning: "conv node output is FP32"**
The BF16 cast in `get_model_and_input()` should propagate if applied before
`torch.compile()`. Check that `get_baseline_model_and_input()` is not re-casting
the model to FP32 internally. If using `torch.autocast`, ensure the context manager
wraps the forward call, not just the model construction.

**CUDA Graph capture fails with `CUDAGraphTreeManager` error**
`mode='reduce-overhead'` requires that the model has no Python-level control flow
inside `forward()`. The baseline `DepthwiseSepConv.forward()` is a pure sequential
chain — if capture fails, check for any in-place operations or `.item()` calls that
trigger CPU-GPU synchronisation.

**Pattern not matched in `pass_conv1x1_as_mm`**
If Dynamo traces the model *after* cuDNN dispatch (e.g., when called with
`backend='eager'`), `aten::cudnn_convolution` may appear instead of
`aten::convolution`. The pass checks a set of known conv targets; add
`torch.ops.aten.cudnn_convolution.default` to `_node_is_conv()` if needed.

**`--script-args` parse error in operator-profiler map**
`--script-args` uses `nargs=REMAINDER` and must be the last flag on the command
line. All `operator-profiler map` flags (`--ncu-sudo`, `--ncu-env`, etc.) must
appear before `--script-args`.

---

## Future Work

### OPT-003: Custom Triton Depthwise Kernel
```python
# TODO: implement in triton_kernels/depthwise_conv3x3.py
@triton.jit
def depthwise_conv3x3_float4_kernel(
    x_ptr, w_ptr, out_ptr, bn_mean_ptr, bn_var_ptr, bn_weight_ptr, bn_bias_ptr,
    B, C, H, W,
    BLOCK_C: tl.constexpr, BLOCK_HW: tl.constexpr,
):
    # float4 vectorised loads (128-bit transactions)
    # inline BN scaling epilog
    # tl.clamp(0.0, 6.0) for ReLU6
    ...

# Register as custom lowering:
from torch._inductor.lowering import register_lowering
@register_lowering(torch.ops.aten.convolution.default)
def depthwise_conv_lowering(x, w, bias, stride, padding, dilation, transposed, output_padding, groups):
    if groups == x.shape[1] == w.shape[0]:  # depthwise check
        return call_triton_depthwise_kernel(x, w, ...)
    return fallback_lowering(x, w, ...)
```

### OPT-004: Conv-BN-ReLU6 Epilog Fusion
Register as a post-lowering Inductor fusion pass:
```python
from torch._inductor.pattern_matcher import register_graph_pattern, PatternMatcherPass

@register_graph_pattern(conv_bn_relu6_pattern, pass_dict=...)
def fuse_conv_bn_relu6(match, ...):
    # Emit single Triton kernel with BN + tl.clamp epilog
    ...
```

### Autotuned Tile Sizes for OPT-002 MM
After applying the 1×1 conv → MM rewrite, `mode='max-autotune'` will sweep Triton
tile configs. The best config for the three pointwise shapes (32→64, 64→128, 128→256
at B×H×W = 16×56×56 = 50176 rows) is likely `BLOCK_M=128, BLOCK_N=128, BLOCK_K=32`.
Profile with `torch._inductor.config.max_autotune = True` to confirm.