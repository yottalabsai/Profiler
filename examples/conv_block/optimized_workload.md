# ConvBlock Optimized Workload

Custom `torch.compile()` backend implementing 5 operator-level optimizations
derived from NCU/nsys profiling of the baseline `conv_block.py` workload on an
NVIDIA A100-SXM4-40GB.

---

## Overview

The baseline ConvBlock profile (FP32, batch=16, 64x64 spatial) reveals four
dominant cost buckets:

| Operator | Wall-time share | Pathology |
|---|---|---|
| `aten::cudnn_convolution` (64ch, 128ch groups) | 64.55% | NCHW implicit-GEMM kernel: 238 regs/thread, 24% occupancy — occupancy starvation, not compute bound |
| `aten::_native_batch_norm_legit_no_training` | 24.76% | DRAM-bound Triton kernel (65% DRAM throughput, 70 launches) — constant arithmetic that can be folded |
| `aten::convolution` (bias-add) | 1.54% | 20 standalone bias-scatter kernels from wave-starved Triton |
| `aten::addmm` (linear head) | 5.01% | `ampere_sgemm_32x128_tn` SIMT path, 0% Tensor Core utilisation, 1 wave across 108 SMs |

The five optimizations together target ~51.4% of baseline wall time.

---

## Quick Start

```bash
# Navigate to the workload directory
cd /home/ubuntu/Profiler/examples/conv_block

# Smoke test (no profiler required, ~10 seconds)
python conv_block_optimized.py

# Run the 4-test validation suite
python test_conv_block_optimized.py
# or with pytest
python -m pytest test_conv_block_optimized.py -v

# Profile baseline vs. optimized (requires operator-profiler + ncu with sudo)
operator-profiler profile conv_block.py \
    --model-name ConvBlock --compile-mode inductor \
    --output runs/conv_block_baseline

operator-profiler profile conv_block_optimized.py \
    --model-name ConvBlock --compile-mode inductor \
    --output runs/conv_block_optimized

# Side-by-side timing comparison
python scripts/run_workload.py \
    --workload examples/conv_block/conv_block.py \
    --compile-backend inductor \
    --warmup-iters 5 --measure-iters 20

python scripts/run_workload.py \
    --workload examples/conv_block/conv_block_optimized.py \
    --compile-backend conv_block_opt \
    --warmup-iters 5 --measure-iters 20
```

---

## Optimizations Table

| ID | Operators targeted | Pathology | Transformation | Estimated gain | Confidence |
|---|---|---|---|---|---|
| OPT-1 | All `aten::cudnn_convolution` (64ch, 128ch, 256ch groups, 64.55% combined) | NCHW layout forces cuDNN to select high-register-count GEMM variant (238 regs/thread, 24% occupancy). 3ch conv already auto-selects NHWC kernel (112 regs, 34% occupancy) as proof. | Non-graph: `model.to(memory_format=torch.channels_last)` + `x.to(memory_format=torch.channels_last)` in `get_model_and_input()` before `torch.compile()`. Inductor propagates NHWC layout through full graph. | ~885,000 ns / **19.4%** of total | **High** |
| OPT-2 | `aten::_native_batch_norm_legit_no_training` (24.76%, 70 launches) + `aten::convolution` bias-add (1.54%, 20 launches) | BN in inference mode is a fixed affine transform: constant per-channel scale and bias that can be pre-computed and absorbed into the preceding conv's weight and bias, eliminating the kernel entirely. | FX pass `pass_fold_bn_into_conv()`: compute `W_folded = W_conv * (gamma / sqrt(var + eps)).view(-1,1,1,1)` and `b_folded = (b_conv - mean) * scale + beta`, register as buffers, rewire conv, erase BN node. | ~1,202,000 ns / **25.0%** of total (eliminates 90 launches) | **High** |
| OPT-3 | `aten::addmm` (linear classifier, 5.01% combined, 10 launches) + all conv ops as secondary beneficiary | All addmm dispatch to `ampere_sgemm_32x128_tn` (SIMT, 0% Tensor Core). On Ampere, BF16 dtype guarantees routing to `sm80_xmma_gemm_bf16bf16` (HMMA Tensor Core). | Non-graph: `model.to(torch.bfloat16)` + `x.to(torch.bfloat16)` in `get_model_and_input()`, applied AFTER OPT-1 (channels_last) and AFTER the BN fold runs at compile time so that folded buffers are cast together. | ~46,000 ns / **1.0%** of total | **High** |
| OPT-4 | All `aten::cudnn_convolution` (complementary to OPT-1) | After channels_last, cuDNN may have multiple NHWC algorithm candidates per shape; default heuristic does not always choose the fastest for batch=16 and fixed spatial dims. | Non-graph: `torch.backends.cudnn.benchmark = True` in `get_model_and_input()`. Triggers one-time cuDNN algorithm search cached per (shape, dtype, device). Assumes fixed input shapes. | ~206,000 ns / **4.5%** of total (on top of OPT-1) | **Medium** |
| OPT-5 | `aten::addmm` (M=16, N=10, K=256 — 1 CTA on 108 SMs, sm_throughput 0.58%) | Structural wave starvation: 160 output elements / 108 SMs = 0.009 CTAs/SM. BF16 alone cannot fix under-parallelism. split-K decomposes K=256 across up to 16 SMs. | Compile-time: `torch.compile(model, backend='conv_block_opt', mode='max-autotune')` at call site. Also sets `torch.backends.cuda.matmul.allow_tf32 = True` for any residual FP32 paths. | ~69,000 ns / **1.5%** of total | **Medium** |

**Cumulative estimated improvement: ~51.4% wall-time reduction** (non-additive;
OPT-2 dominates due to BN elimination, OPT-1 second).

---

## Architecture

### Dependency Order

```
get_model_and_input()
    OPT-1: model.to(channels_last)            ← must be first (layout before dtype)
    OPT-4: cudnn.benchmark = True              ← before first inference call
    OPT-5 (partial): allow_tf32 = True        ← global flag, zero cost
    OPT-3: model.to(bfloat16)                 ← after channels_last, before compile
         │
         ▼
torch.compile(model, backend='conv_block_opt', mode='max-autotune')  ← OPT-5
         │
         ▼
conv_block_opt(gm: fx.GraphModule, example_inputs)
    OPT-2: pass_fold_bn_into_conv(gm)         ← FX pass: BN elimination
         │
         ▼
    compile_fx(gm, example_inputs)            ← Inductor backend
```

### FX Pass: pass_fold_bn_into_conv (OPT-2)

Detection target: `aten._native_batch_norm_legit_no_training.default` nodes
whose first argument is an `aten.convolution.default` node.

Transformation steps (graph surgery):

1. Walk `list(gm.graph.nodes)` (snapshot — never mutate while iterating live).
2. For each `(conv_node, bn_node)` pair, retrieve `gamma`, `beta`,
   `running_mean`, `running_var`, and `eps` via `_get_param_or_buffer()`.
3. Compute `W_folded` and `b_folded` on CPU in FP32 for numerical precision.
4. Register `W_folded` and `b_folded` as buffers on `gm` with unique names.
5. Insert `get_attr` nodes for the buffers using `gm.graph.inserting_before()`.
6. Insert a new `aten.convolution.default` node with folded weight and bias
   using `gm.graph.inserting_after()`.
7. Walk `bn_node.users`: replace `getitem(bn_node, 0)` uses with `new_conv_node`,
   erase the getitem node. Erase `bn_node` and old `conv_node`.
8. After all pairs: `gm.graph.eliminate_dead_code()` → `gm.graph.lint()` → `gm.recompile()`.

Effect on the kernel timeline:
- `triton_poi_fused__native_batch_norm_legit_no_training_relu_4` → eliminated
- `triton_poi_fused_convolution_0` (bias-add) → eliminated
- The trailing ReLU becomes a standalone node that Inductor will fuse into the
  conv epilogue or next pointwise pass at no extra kernel launch cost.

### Backend Registration

```python
from torch._dynamo import register_backend
from torch._inductor.compile_fx import compile_fx  # function, not module

@register_backend
def conv_block_opt(gm: fx.GraphModule, example_inputs) -> Callable:
    gm = pass_fold_bn_into_conv(gm)
    return compile_fx(gm, example_inputs)
```

The `@register_backend` decorator makes `"conv_block_opt"` available as a
`torch.compile(backend=...)` string and in `torch._dynamo.list_backends()`.

---

## Key Design Decisions

### Why OPT-1 and OPT-3 are non-graph

Memory format (`channels_last`) and dtype (`bfloat16`) are tensor properties,
not operations visible in the Aten IR graph. Applying them before `torch.compile()`
allows Inductor's layout propagation pass to see NHWC from the start and
propagate it through the full graph — more effective than inserting
format-conversion nodes in the FX backend.

### Why BN fold runs in the FX backend (not in get_model_and_input)

`pass_fold_bn_into_conv` operates on the Aten IR graph produced by Dynamo
tracing, which exposes the exact `aten._native_batch_norm_legit_no_training`
and `aten.convolution` nodes with their parameter `get_attr` connections. At
the `nn.Module` level, the Conv2d and BatchNorm2d are separate sub-modules
with no direct connection visible to pre-compile Python code.

### Why BN fold precedes BF16 cast

The fold arithmetic is done in FP32 to preserve numerical accuracy of the
scale/bias computation (`gamma / sqrt(var + eps)`). The model-level BF16 cast
(`model.to(bfloat16)`) then casts all parameters including the folded buffers
together in a single operation, which is efficient and ensures the folded
buffers are in the same dtype as the conv weights they correspond to.

### Idempotency guards

All non-graph optimizations check current state before applying:

```python
# OPT-1: only if not already channels_last
if not first_param.is_contiguous(memory_format=torch.channels_last):
    model = model.to(memory_format=torch.channels_last)

# OPT-3: only if not already reduced precision
if next(model.parameters()).dtype not in (torch.bfloat16, torch.float16):
    model = model.to(torch.bfloat16)
```

This prevents redundant conversions if the baseline is later updated.

### OPT-5 at call site, not inside the backend

`mode='max-autotune'` is a `torch.compile()` option that instructs Inductor
to run a broader tile-config search before emitting Triton kernels. It must be
set at the `torch.compile()` call site, not inside the custom backend, because
the custom backend receives the already-configured `gm` and has no way to
re-trigger Inductor's autotuning search from within its callback.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'conv_block'`**

`conv_block_optimized.py` imports from `conv_block`. Both files must be in the
same directory, and that directory must be on `PYTHONPATH` or you must run
from that directory.

```bash
cd /home/ubuntu/Profiler/examples/conv_block
python conv_block_optimized.py
```

**`TypeError: 'module' object is not callable` at compile time**

Wrong import form. This always means:
```python
# WRONG — imports the module object, not the function
from torch._inductor import compile_fx

# CORRECT — imports the callable function
from torch._inductor.compile_fx import compile_fx
```

**`conv_block_opt` not in `torch._dynamo.list_backends()`**

The `@register_backend` decorator runs at import time. The module must be
imported before `torch.compile()` is called:
```python
import conv_block_optimized  # triggers registration
compiled = torch.compile(model, backend="conv_block_opt")
```

**`gm.graph.lint()` raises `RuntimeError: use of dead node`**

A node was erased before all its users were replaced. The `pass_fold_bn_into_conv`
implementation handles this by: (1) replacing all `getitem` users of `bn_node`
before erasing them, (2) calling `conv_node.replace_all_uses_with(new_conv_node)`
before `erase_node(conv_node)`. If you extend the pass, always call
`replace_all_uses_with()` before `erase_node()`.

**`[pass_fold_bn_into_conv] No Conv→BN patterns found`**

This warning is logged when the Aten IR graph does not contain a
`_native_batch_norm_legit_no_training` node directly following a
`convolution` node. Possible causes:
- The model was traced in training mode (BN uses a different op in training).
- Inductor already fused the BN into a pointwise kernel before the backend ran.
- The `compile_mode` is not `"inductor"` (e.g., `"eager"` produces no FX graph).

The pass skips gracefully — compilation continues with the unfused graph.

**NaN or Inf in output after BN fold**

The folded scale `gamma / sqrt(var + eps)` can produce large values if
`running_var` is near zero (undertrained BN). This is very rare for
pre-trained or eval-mode models. Check:
```python
# Inspect running variance of each BN layer
for name, m in model.named_modules():
    if isinstance(m, torch.nn.BatchNorm2d):
        print(name, m.running_var.min().item())
```
If any variance is <1e-4, increase the BN `eps` parameter before compilation.

**max-autotune compilation is slow (30 s – 5 min)**

This is expected on first run. Inductor benchmarks all tile configurations.
For production, serialize the compiled model:
```python
# Serialize
torch._inductor.aot_compile(model, (x,), options={"max_autotune": True})
# or use torch.export + ExecuTorch for deployment
```

---

## Future Work

- **OPT-2 full Triton conv**: Register a Triton convolution kernel with
  occupancy-aware block shapes (<128 regs/thread) as an FX substitution for
  the cuDNN implicit-GEMM kernels in conv stages 2 and 3. Requires validating
  correctness across all tile shapes at batch=16 and verifying that Inductor's
  layout propagation does not re-insert format conversions.

- **Conv+BN+ReLU fused Triton epilogue**: After BN fold, the
  `conv → mul → add → relu` chain is a candidate for a single fused Triton
  pointwise kernel that eliminates all intermediate tensor materializations.
  Can be implemented as a custom Triton kernel registered via
  `torch._custom_ops` or `torch.library`.

- **Batch padding for addmm wave starvation**: Padding the batch dimension
  from 16 to 64 would produce `ceil(64*10 / tile) >= 4` CTAs, raising SM
  coverage from 0.9% to ~3.7%. This trades latency for throughput and is only
  appropriate in batched-inference serving scenarios where the caller controls
  batch size. Implement in `get_model_and_input()` with a corresponding
  `out[:BATCH_SIZE]` slice after the forward pass.
