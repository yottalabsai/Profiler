---
name: backend
description: Generate a production-ready workload_optimized.py with a custom torch.compile() backend implementing the transformations in optimizations.json. Each optimization becomes a named FX graph pass. Also generates a test script and documentation. Pass workload.py and optimizations.json.
---

# /backend — Custom torch.compile() Backend Generation

Generates `{workload}_optimized.py` implementing a custom Torch Dynamo backend with FX graph passes derived from `optimizations.json`. This is the code that, when profiled, should show the speedups predicted in `/propose`.

## Usage

```
/backend workload.py optimizations.json
/backend workload.py optimizations.json --profile=profile.json   # extra shape validation
```

## What It Generates

1. **`{workload}_optimized.py`** — Custom backend with FX graph passes (~300–500 lines)
2. **`test_{workload}_optimized.py`** — 4-test validation suite
3. **`OPTIMIZED_WORKLOAD.md`** — Documentation with quick-start commands

## Output Structure

```python
"""
{workload}_optimized.py — {ModelName} with custom torch.compile() backend.

Implements N operator-level optimizations via FX graph passes:
  1. pass_name — brief description
  ...

To profile:
    python scripts/run_workload.py --workload {workload}_optimized.py \
        --compile-backend {model_name}_opt
"""

from torch._inductor.compile_fx import compile_fx  # function, not module
from torch._dynamo import register_backend
import torch.fx as fx
from nvidia.operator_profiler.fx import UniqueSubgraphRegistry, FxPassRunner

# replace_pattern-compatible pass functions (traced by FxPassRunner)
def _pattern_fn(...): ...
def _replacement_fn(...): ...

# Manual per-rep pass functions (applied to each subgraph individually)
def _pass_manual(gm: fx.GraphModule) -> fx.GraphModule: ...

# Utility: capture real input tensors for each partition (see fx-patterns.md)
def _capture_partition_inputs(split_gm, example_inputs): ...

# Backend registration
@register_backend
def {model_name}_opt(gm: fx.GraphModule, example_inputs) -> Callable:
    registry = UniqueSubgraphRegistry(gm)
    equiv_map = registry.build_partition_equivalence_map()

    if not equiv_map:
        # No repeated layers — flat compile preserves cross-layer Inductor fusion
        gm = _pass_manual(gm)
        return compile_fx(gm, example_inputs)

    runner = FxPassRunner(registry)
    runner.apply_pass(_pattern_fn, _replacement_fn)   # propagates to duplicates

    for rep_name, rep_mod in registry.unique_reps:    # manual passes per unique rep
        _pass_manual(rep_mod)
        for _, dup_mod in registry.duplicates_of(rep_name):
            _pass_manual(dup_mod)

    partition_inputs = _capture_partition_inputs(registry.split, example_inputs)
    for rep_name, rep_mod in registry.unique_reps:    # compile unique reps; share with dups
        compiled = compile_fx(rep_mod, partition_inputs.get(rep_name, example_inputs))
        rep_mod.forward = compiled
        for _, dup_mod in registry.duplicates_of(rep_name):
            dup_mod.forward = compiled

    return lambda *args: registry.split(*args)

# Workload interface (with non-graph optimizations)
def get_model_and_input() -> tuple[nn.Module, torch.Tensor]: ...
```

## Critical Rules

### compile_fx Import (Most Common Bug)
```python
# ALWAYS — imports the callable function
from torch._inductor.compile_fx import compile_fx

# NEVER — imports the module, causes TypeError at runtime
from torch._inductor import compile_fx
```

### Weight Node Detection in Inductor-Traced Graphs
Inductor wraps parameters as `t(get_attr('weight'))` in the FX graph. The bare `get_attr` pattern does not work. Always look through the `aten.t()` wrapper:

```python
if (node.target == torch.ops.aten.t.default
        and node.args[0].op == 'get_attr'):
    param_name = node.args[0].target
    weight = gm.get_parameter(param_name)
```

### Graph Mutation Discipline
Every FX pass must:
1. Iterate over `list(gm.graph.nodes)` — snapshot, not live iterator
2. Call `replace_all_uses_with()` BEFORE `erase_node()`
3. Call `gm.graph.lint()` after all mutations
4. Call `gm.recompile()` after `lint()`
5. Wrap everything in `try/except` with `logger.warning` on failure

### Pass Structure by Confidence Level

| Confidence | Implementation | On Pattern Miss |
|---|---|---|
| `high` | Full detection + transformation | Raises exception, caught by try/except |
| `medium` | Approximate detection + graceful fallback | `logger.warning` + return gm unchanged |
| `low` | Detection stub only | Always returns gm unchanged; logs warning |

### Canonical FX Patterns
Use `knowledge/fx-patterns.md` implementations for:
- QKV fusion (`pass_fuse_qkv`)
- SDPA replacement (`pass_replace_sdpa`)
- BN fold (`pass_fold_bn`)
- Pre-transposed weights (`pass_pretranspose_weights`)
- Activation substitution (`pass_tanh_to_gelu`)

### Dedup-Aware Backend Structure
ALWAYS use `UniqueSubgraphRegistry` + `FxPassRunner`. Check `equiv_map = registry.build_partition_equivalence_map()` — if empty, fall back to flat compile; if non-empty, use the dedup compile path. Use `FxPassRunner.apply_pass` for `replace_pattern`-compatible passes (pure functional, no tuple outputs, no `register_buffer`); use explicit per-rep loops for manual passes (SDPA, BN fold, pre-transposed weights). See Pass Taxonomy in `knowledge/fx-patterns.md`.

### Non-Graph Optimizations
Apply in `get_model_and_input()`, NOT in the backend function:
- BF16/FP16 dtype cast: check `next(model.parameters()).dtype` before applying
- `channels_last` layout: check `is_contiguous(memory_format=...)` before applying
- Batch padding: check `x.shape[0] < tile_size` before padding

### compile_mode Handling
- `"inductor"` → standard FX pass backend (the normal case)
- `"eager"` → FX passes DO NOT run (no graph is traced); warn user; propose adding `torch.compile(model, backend='inductor')` to the workload
- `"cudagraphs"` → FX passes run at first capture; flag dynamic-shape passes as risky

## Pass Application Order

In the backend function, apply passes in this order:
1. Fusion passes (QKV, SDPA, BN fold) — must run at original graph structure
2. Layout passes (pre-transposed weights) — after fusion locks the weight nodes
3. Algorithm passes (activation substitution) — independent of the above
4. All ordering must respect `prerequisite_for[]` from `optimizations.json`

High-confidence passes always run before medium-confidence to minimize graph corruption risk.

## Test Script: 4 Required Tests

```python
def test_import():           # module loads without error
def test_backend_registration():  # '{backend_name}' in torch._dynamo.list_backends()
def test_get_model_and_input():   # shapes/dtypes match expected
def test_forward_pass():     # uncompiled forward, no NaN/Inf, no exception
```

## Common Failure Modes and Fixes

| Error | Cause | Fix |
|---|---|---|
| `TypeError: 'module' object is not callable` | Wrong `compile_fx` import | Use `from torch._inductor.compile_fx import compile_fx` |
| `Graph is not valid` | Missed `gm.graph.lint()` after mutation | Add lint+recompile after EVERY graph change |
| `Expected tensors on same device` | Graph pass created CPU tensor | Ensure all new tensors/buffers are moved to `gm.device` |
| `Compiled output identical to baseline` | Backend not registered | Verify `@register_backend` decorator; call `torch._dynamo.reset()` before recompiling |
| `AssertionError: shape mismatch` | Batch padding applied but output not sliced | Slice output: `out[:original_batch_size]` |
| `AttributeError: 'NoneType' has no attribute 'target'` | Node graph traversal hit placeholder | Add `if node is None: continue` guards |
| `TypeError` or shape error in registry split | Partition input shapes don't match original model inputs | Use `_capture_partition_inputs` to capture real partition inputs; never pass original `example_inputs` directly to `compile_fx` for partitions |

## Validation After Generation

After writing the file, the backend-engineer agent runs:
```bash
python -m py_compile {output_file}
```

If syntax check fails, the agent fixes errors before reporting completion.

Run `/validate {output_file}` next to run the full 5-step validation suite before profiling.
