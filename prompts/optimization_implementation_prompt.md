# PyTorch FX Graph Optimization Prompt Template

You are a senior PyTorch systems engineer specializing in FX graph transformations and torch.compile() backend development. Your task is to generate a custom torch.compile() backend that implements operator-level optimizations derived from profiling feedback.

## Task

Given:
1. A baseline workload module (`workload.py`) with `get_model_and_input()` interface
2. An `OPTIMIZATIONS.json` file describing operator-level bottlenecks and transformations

Generate:
- A new `workload_optimized.py` file with a custom `transformer_opt` backend
- A test script (`test_workload_optimized.py`)
- Documentation (`OPTIMIZED_WORKLOAD.md`)

## Input Format

### workload.py Structure
```python
# Must expose:
def get_model_and_input() -> tuple[model, input_tensor]:
    """Return (uncompiled model on CUDA, input tensor on CUDA)."""
    model = YourModel().to("cuda").eval()
    x = torch.randn(..., device="cuda")
    return model, x
```

### OPTIMIZATIONS.json Structure
```json
{
  "Operator-Level Optimizations": [
    {
      "Operators": "aten::op_name (op_id X, Y, Z)",
      "Bottleneck": "Detailed performance issue description",
      "Transformation": "Specific FX graph transformation to apply",
      "Impact": "Expected latency/throughput/memory improvements",
      "Confidence": "High|Medium|Low"
    },
    ...
  ]
}
```

## Output Requirements

### 1. workload_optimized.py
**Structure (production-ready, ~400 lines):**

```python
"""
workload_optimized.py — [Model Name] with custom torch.compile() backend.

Implements [N] operator-level optimizations via FX graph passes:
  1. [Optimization 1] — [brief description]
  2. [Optimization 2] — [brief description]
  ...

To profile with optimizations:
    python scripts/run_workload.py \\
        --workload scripts/workload_optimized.py \\
        --compile-backend {model_name}_opt
"""

from __future__ import annotations

import logging
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
from torch._dynamo import register_backend
from torch._inductor.compile_fx import compile_fx   # import the function, not the module
from nvidia.operator_profiler.fx import UniqueSubgraphRegistry, FxPassRunner

# Import baseline workload
from scripts.workload import get_model_and_input as get_baseline_model_and_input

DEVICE = "cuda"
BATCH_SIZE = ...   # preserve original constants

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ============================================================================
# Pass utilities
# ============================================================================

def _capture_partition_inputs(
    split_gm: fx.GraphModule,
    example_inputs: list,
) -> dict[str, list]:
    """Capture real input tensors for each partition via forward-pre hooks."""
    partition_inputs: dict[str, list] = {}
    hooks = []
    for name, submod in split_gm.named_children():
        if isinstance(submod, fx.GraphModule):
            def _hook(mod, args, _name=name):
                partition_inputs[_name] = list(args)
            hooks.append(submod.register_forward_pre_hook(_hook))
    with torch.no_grad():
        split_gm(*example_inputs)
    for h in hooks:
        h.remove()
    return partition_inputs

# ============================================================================
# replace_pattern-compatible passes (FxPassRunner applies these automatically)
# ============================================================================
# Use for: pure functional patterns — no tuple outputs, no register_buffer,
# no control flow. FxPassRunner.apply_pass() handles unique-rep application
# and propagation to all structural duplicates.

def _[opt_pattern](x, ...):
    """Pattern to match."""
    ...

def _[opt_replacement](x, ...):
    """Replacement subgraph."""
    ...

# ============================================================================
# Manual per-rep passes (applied in the per-rep loop below)
# ============================================================================
# Use for: tuple outputs (BN fold), decomposed ops (SDPA — softmax is
# decomposed before passes run), or register_buffer needed.

def _pass_[optimization_name](gm: fx.GraphModule) -> fx.GraphModule:
    """
    [What this pass does]
    Pattern: [FX graph pattern it detects]
    Transformation: [How it transforms the graph]
    Effect: [Expected kernel/performance impact]
    """
    try:
        # Pattern detection and transformation
        for node in list(gm.graph.nodes):
            if node.op == "call_function" and node.target == torch.ops.aten.[...]:
                pass
        gm.graph.lint()
        gm.recompile()
    except Exception as e:
        logger.warning(f"[_pass_{optimization_name}] Failed: {e}")
    return gm

# ============================================================================
# Backend Registration
# ============================================================================

@register_backend
def {model_name}_opt(gm: fx.GraphModule, example_inputs) -> Callable:
    """
    Custom torch.compile() backend implementing all FX graph optimizations.

    Dedup path: applies passes only to structurally unique layer partitions,
    then propagates to duplicates. Falls back to flat compile for models with
    no repeated layers.
    """
    logger.info("{model_name}_opt backend: starting")
    registry = UniqueSubgraphRegistry(gm)
    equiv_map = registry.build_partition_equivalence_map()

    if not equiv_map:
        # No repeated layers — flat compile preserves cross-layer Inductor fusion
        logger.info("{model_name}_opt: no repeated layers, flat compile path")
        gm = _pass_[opt1](gm)
        gm = _pass_[opt2](gm)
        logger.info("{model_name}_opt: delegating to Inductor")
        return compile_fx(gm, example_inputs)

    logger.info(f"{model_name}_opt: {len(equiv_map)} duplicate partitions, dedup path")
    runner = FxPassRunner(registry)

    # replace_pattern-compatible passes (FxPassRunner propagates to duplicates)
    runner.apply_pass(_[opt_pattern], _[opt_replacement])

    # Manual passes: apply to each unique rep, then propagate to duplicates
    for rep_name, rep_mod in registry.unique_reps:
        _pass_[opt1](rep_mod)
        for _, dup_mod in registry.duplicates_of(rep_name):
            _pass_[opt1](dup_mod)

    # Compile unique reps with real partition inputs; share callable with duplicates
    partition_inputs = _capture_partition_inputs(registry.split, example_inputs)
    for rep_name, rep_mod in registry.unique_reps:
        compiled = compile_fx(rep_mod, partition_inputs.get(rep_name, example_inputs))
        rep_mod.forward = compiled
        for _, dup_mod in registry.duplicates_of(rep_name):
            dup_mod.forward = compiled

    logger.info("{model_name}_opt: all passes applied, returning compiled split graph")
    return lambda *args: registry.split(*args)

# ============================================================================
# Workload Interface
# ============================================================================

def get_model_and_input() -> tuple:
    """
    Workload interface for run_workload.py.

    Applies non-graph optimizations (dtype, layout, batch shape) on top of
    the baseline. FX graph passes run inside the backend above.
    """
    assert torch.cuda.is_available(), "CUDA required"
    model, x = get_baseline_model_and_input()

    # Non-graph optimizations — check before applying to avoid redundant transforms
    if next(model.parameters()).dtype != torch.bfloat16:
        model = model.to(torch.bfloat16)
        x = x.to(torch.bfloat16)

    if x.shape[0] < 64:
        pad = 64 - x.shape[0]
        x = torch.nn.functional.pad(x, (0, 0, 0, pad))

    return model, x

if __name__ == "__main__":
    m, x = get_model_and_input()
    with torch.no_grad():
        y = m(x)
    print(f"Output shape: {y.shape}")
```

**Key Principles:**

1. **Defensive error handling** — Each pass includes try-except, logs warnings, continues gracefully
2. **Isolated passes** — Each optimization is a separate function, can be added/removed independently
3. **Approximate pattern matching** — Passes degrade gracefully if patterns don't match exactly
4. **Logging** — All major operations logged at INFO level for observability
5. **Model-agnostic** — FX passes work at Aten IR level, not hardcoded to specific modules
6. **Complementary dtype/shape** — Non-graph optimizations (BF16, padding) applied in `get_model_and_input()` only when not already present in the baseline; always inspect baseline state before applying

### 2. test_workload_optimized.py
**Requirements:**

```python
"""
test_workload_optimized.py — Verification tests for optimized workload.

Validates:
  1. Module imports
  2. Backend registration
  3. Model and input shapes/dtypes
  4. Uncompiled forward pass
"""

def test_import():
    """Module imports successfully."""
    
def test_backend_registration():
    """Backend is registered with torch._dynamo."""
    
def test_get_model_and_input():
    """Model/input creation with correct shapes/dtypes."""
    
def test_forward_pass():
    """Uncompiled forward pass completes without error."""
```

### 3. OPTIMIZED_WORKLOAD.md
**Sections:**

- **Overview** — What optimizations are implemented
- **Quick Start** — Copy-paste commands to run it
- **Six Optimizations Table** — Summarize each with target ops and expected impact
- **Architecture** — Explain FX passes and backend approach
- **Why Custom Backend** — Model-agnostic, robust, maintainable
- **Key Design Decisions** — BF16 outside graph, token padding, pattern matching, defensive passes
- **Comparison Against Baseline** — How to run side-by-side profiling
- **Verification Checklist** — What to check in the resulting profile
- **Troubleshooting** — Common issues and fixes
- **Future Work** — Stubs and next steps

## Implementation Guidelines

### Pass Classification (Do This First)

Before writing any code, classify each optimization into one of three categories:

| Category | Criteria | Implementation |
|---|---|---|
| **`replace_pattern`-compatible** | Pure functional: no tuple outputs, no `register_buffer`, no control flow | `_pattern_fn` + `_replacement_fn`; call `runner.apply_pass()` |
| **Manual per-rep** | Tuple outputs, decomposed ops, or `register_buffer` needed | `def _pass_name(gm) -> gm`; apply in per-rep loops |
| **Non-graph** | dtype, memory_format, batch shape | `get_model_and_input()` only |

Canonical classifications:
- **`replace_pattern`-compatible:** QKV fusion, tanh→GELU substitution
- **Manual per-rep:** SDPA (softmax is decomposed into exp+sum+div before passes run — `replace_pattern` never sees a softmax node), BN fold (`aten._native_batch_norm_legit_no_training` returns a 3-tuple — `replace_pattern` cannot match tuple-returning patterns), pre-transposed weights (`register_buffer` needed)
- **Non-graph:** BF16 cast, channels_last, batch padding

### For Each Optimization in OPTIMIZATIONS.json

**HIGH confidence optimizations:**
- Implement as full FX pass
- Match pattern reliably (e.g., 3× mm nodes with same input)
- Test pattern detection logic carefully

**MEDIUM confidence optimizations:**
- Implement as FX pass with defensive error handling
- Pattern matching may be approximate
- Log warnings if pattern doesn't match

**LOW confidence optimizations (or require custom kernels):**
- Implement as stub pass with detection
- Log warning: "Pattern detected but not applied — requires [Triton kernel / custom op]"
- Note as TODO for future work

### Pattern Detection Strategies

**By operator type:**
```python
if node.op == "call_function" and node.target == torch.ops.aten.mm.default:
    ...
```

**By graph structure:**
```python
# Build input_node → [consumers] map
input_to_consumers = {}
for node in gm.graph.nodes:
    for input_node in node.all_input_nodes:
        if input_node not in input_to_consumers:
            input_to_consumers[input_node] = []
        input_to_consumers[input_node].append(node)

# Find groups of 3+ mm nodes with same input
for input_node, consumers in input_to_consumers.items():
    mm_nodes = [n for n in consumers if n.target == torch.ops.aten.mm.default]
    if len(mm_nodes) >= 2:
        # Fuse mm_nodes
```

**By subgraph pattern:**
```python
from torch.fx.subgraph_rewriter import replace_pattern

def pattern_func(x, y, z):
    a = torch.ops.aten.mm.default(x, y)
    b = torch.ops.aten.softmax.default(a, -1, False)
    return torch.ops.aten.mm.default(b, z)

def replacement_func(x, y, z):
    return torch.nn.functional.scaled_dot_product_attention(x, y, z)

replace_pattern(gm, pattern_func, replacement_func)
```

### Graph Manipulation

**Insert node:**
```python
with gm.graph.inserting_after(node):
    new_node = gm.graph.call_function(torch.ops.aten.op.default, (arg1, arg2))
```

**Replace uses:**
```python
old_node.replace_all_uses_with(new_node)
```

**Erase node:**
```python
gm.graph.erase_node(old_node)
```

**Register buffer:**
```python
gm.register_buffer('buffer_name', tensor_value)
with gm.graph.inserting_before(node):
    buffer_node = gm.graph.get_attr('buffer_name')
```

**Validate graph:**
```python
gm.graph.lint()
gm.recompile()
```

## Optimization Mapping Examples

### 1. BF16 Precision Casting
- **Location:** `get_model_and_input()`
- **Implementation:** Check `next(model.parameters()).dtype` first; apply `model.to(torch.bfloat16)` + `x.to(torch.bfloat16)` only if not already BF16
- **Why check:** Baseline may already cast; idempotent but wasteful and could mask intent
- **Not in FX pass** because dtype is tensor property, not graph operation
- **Effect:** Routes all GEMMs to BF16 Tensor Core path

### 2. QKV Projection Fusion
- **Pattern:** 3× `mm(x, W_i)` nodes with identical input `x`
- **FX pass:** `pass_fuse_qkv()`
- **Algorithm:**
  1. Detect 3 mm nodes with same input
  2. Extract weights: `W_q, W_k, W_v = [gm.get_parameter(node.target) for node in ...]`
  3. Fuse: `W_fused = torch.cat([W_q, W_k, W_v], dim=0)`
  4. Register: `gm.register_buffer('fused_qkv_weight', W_fused)`
  5. Replace: single `mm(x, W_fused)` → `chunk(3, dim=1)`
- **Effect:** 3 kernels → 1 kernel, Waves/SM 0.33 → 0.68

### 3. FlashAttention (SDPA) Replacement
- **Pattern:** `mm(Q, K^T) → div → softmax → mm(attn, V)`
- **FX pass:** `pass_replace_sdpa()`
- **Algorithm:** Use `torch.fx.subgraph_rewriter.replace_pattern` with pattern function
- **Effect:** 3 kernels → 1 FlashAttention kernel, 60% DRAM reduction

### 4. Consistent GELU Activation
- **Pattern:** `relu` in `mm → relu → mm` (FFN context)
- **FX pass:** `pass_normalize_gelu()`
- **Algorithm:**
  1. For each `aten.relu` node
  2. Check if producer is `mm` and consumers are `mm`
  3. If yes: replace with `gelu(approximate='tanh')`
- **Effect:** Consistent activation kernel, 10–15% faster

### 5. Pre-transposed Weights
- **Pattern:** `mm(x, aten.t(W))` where W is large parameter
- **FX pass:** `pass_pretranspose_weights()`
- **Algorithm:**
  1. Find `aten.t()` nodes fed into `mm`
  2. For large weights (K ≥ 512): pre-store `W.T.contiguous()`
  3. Replace `aten.t(W)` with `get_attr` to pre-transposed buffer
- **Effect:** gemmSN_TN → gemmSN_NN, 14880ns → 7000–9000ns

### 6. Token Padding
- **Location:** `get_model_and_input()`
- **Implementation:** Check `x.shape[0] < 64` before padding; only pad if batch dim is below tile size
- **Why check:** Baseline may already pad or use a batch size ≥ 64
- **Detail:** Pad input `[B, D]` → `[64, D]`
- **Effect:** Waves/SM 0.01 → 0.17 for elementwise kernels

### 7. LayerNorm-Linear Fusion (Stub)
- **Pattern:** `layer_norm → mm`
- **FX pass:** `pass_fuse_ln_linear()` — detection only
- **Status:** Stub; full implementation requires custom Triton kernel
- **Log:** "LN-Linear fusion detected but not applied — requires custom Triton kernel"

## Known Implementation Notes

### compile_fx import
`from torch._inductor import compile_fx` imports the **module** `torch._inductor.compile_fx`, not the callable function. This causes `TypeError: 'module' object is not callable` at compile time. Always import from the submodule:
```python
# WRONG — imports the module, not the function
from torch._inductor import compile_fx

# CORRECT
from torch._inductor.compile_fx import compile_fx
```

### operator-profiler map --script-args
`--script-args` in the `map` command uses `nargs=argparse.REMAINDER`, which means it must be the **last** flag on the command line. Any option-style arguments (e.g. `--workload`, `--compile-backend`) must follow it without interruption, otherwise argparse misinterprets them as flags for the `map` command itself.

```bash
# WRONG — --ncu-env after --script-args causes parse error
operator-profiler map manifest.json \
    --script scripts/run_workload.py \
    --script-args --workload scripts/workload_optimized.py \
    --ncu-env PYTHONPATH=/repo   # ← this breaks --script-args parsing

# CORRECT — all map flags before --script-args
operator-profiler map manifest.json \
    --script scripts/run_workload.py \
    --ncu-sudo \
    --ncu-env PYTHONPATH=/repo \
    --script-args --workload scripts/workload_optimized.py \
                  --compile-backend transformer_opt
```

---

## Testing Strategy

**Before profiling:**
1. Syntax check: `python -m py_compile workload_optimized.py`
2. Module import: `python -c "import scripts.workload_optimized"`
3. Backend registration: Verify in `torch._dynamo.list_backends()`
4. Smoke test: `python test_workload_optimized.py`
5. Compiled forward pass: `python scripts/run_workload.py --workload scripts/workload_optimized.py --compile-backend transformer_opt --warmup-iters 1 --measure-iters 1`

**During profiling:**
- Check kernel count reduction (due to fusion)
- Verify QKV: 1 kernel vs. 3
- Verify Attention: 1 FlashAttention kernel vs. 3-kernel chain
- Check FFN down-proj latency: ~7000–9000ns vs. baseline 14880ns
- Verify tensor core utilization for GEMM kernels (BF16 active)

## Tone & Style

- **Technical but accessible** — Explain FX concepts without over-simplifying
- **Defensive** — Always assume pattern matching may fail; handle gracefully
- **Observable** — Log all major operations; use `logger.info()` extensively
- **Modular** — Each pass is self-contained; can be added/removed independently
- **Production-ready** — Error handling, validation, comprehensive docstrings

## Final Checklist

- [ ] Passes classified as `replace_pattern`-compatible vs. manual-per-rep vs. non-graph
- [ ] `UniqueSubgraphRegistry` and `FxPassRunner` imported and used in backend
- [ ] `if not equiv_map:` flat fallback path present for models with no repeated layers
- [ ] `_capture_partition_inputs` used when compiling per-partition (not original `example_inputs`)
- [ ] All HIGH confidence optimizations implemented as full passes
- [ ] All MEDIUM confidence optimizations implemented with error handling
- [ ] All LOW confidence optimizations implemented as stubs (detection + warning)
- [ ] Backend registration with `@register_backend` decorator
- [ ] `get_model_and_input()` applies non-graph optimizations (BF16, padding)
- [ ] Test script validates import, backend, shapes, forward pass
- [ ] Documentation covers all optimizations, quick start, architecture, troubleshooting
- [ ] All code is defensive (try-except, logging, graceful degradation)
- [ ] Syntax validated: `python -m py_compile workload_optimized.py`
- [ ] Model-agnostic: FX passes work at Aten IR level, not tied to specific modules
