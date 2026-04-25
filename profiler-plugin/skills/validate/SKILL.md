---
name: validate
description: Systematically validate a generated workload_optimized.py before profiling. Runs 5 steps in order: syntax check, import check, backend registration, pytest suite, compiled smoke test. Reports which FX passes applied vs. degraded gracefully. Saves wasted ncu replay time on broken code.
---

# /validate — Backend Validation

Runs the complete 5-step validation suite on a generated backend before running the expensive ncu profiling pipeline. A failed validation here saves minutes of wasted profiling time.

## Usage

```
/validate workload_optimized.py
/validate workload_optimized.py --backend-name=my_model_opt   # explicit backend name
/validate workload_optimized.py --numerical                    # also run numerical check
```

## The 5-Step Sequence

Steps run in order. If a step fails, diagnosis is reported and later steps are skipped.

### Step 1: Syntax
```bash
python -m py_compile workload_optimized.py
```
Catches: unclosed brackets, bad indentation, typos in keywords.

### Step 2: Import
```bash
python -c "import workload_optimized"
```
Catches: missing imports, wrong `compile_fx` import path, circular imports.

### Step 3: Backend Registration
```bash
python -c "import torch; import workload_optimized; assert '{backend}' in str(torch._dynamo.list_backends())"
```
Catches: missing `@register_backend` decorator, import that didn't execute the decorator.

### Step 4: Test Suite
```bash
python -m pytest test_workload_optimized.py -v --tb=short
```
Four tests must all pass:
- `test_import` — module loads
- `test_backend_registration` — backend found in dynamo
- `test_get_model_and_input` — correct shapes and dtypes
- `test_forward_pass` — uncompiled forward runs clean (no NaN/Inf)

### Step 5: Compiled Smoke Test
```bash
python nvidia/scripts/run_workload.py \
    --workload workload_optimized.py \
    --compile-backend {backend_name} \
    --warmup-iters 1 \
    --measure-iters 1
```
Catches: runtime errors that only appear during torch.compile (graph lint failures, device mismatches, shape errors from padding).

## Reading the Pass Application Log

After Step 5, the stdout contains `logger.info` and `logger.warning` lines from each FX pass. These tell you exactly which passes applied:

```
INFO  [pass_channels_last] Applied — convertTensor_kernel eliminated
INFO  [pass_bf16_dtype]    Applied — model and input converted to BF16
WARNING [pass_fuse_qkv]    Pattern not found — pass not applied
```

A `WARNING ... Pattern not found` is NOT a failure — passes are designed to degrade gracefully. It means that optimization will not contribute to the speedup. Re-check whether the FX graph actually contains the expected pattern (Inductor may have fused or decomposed it).

## Common Failures and Fixes

### `TypeError: 'module' object is not callable`
**Cause:** Wrong `compile_fx` import.
**Fix:** Change `from torch._inductor import compile_fx` to `from torch._inductor.compile_fx import compile_fx`.

### `Graph is not valid` (torch.fx GraphError)
**Cause:** A graph mutation was not followed by `gm.graph.lint()`.
**Fix:** Add `gm.graph.lint(); gm.recompile()` after every block of graph mutations.

### `Expected all tensors to be on the same device`
**Cause:** A graph pass created a new tensor (e.g., from `weight.T.contiguous()`) on CPU.
**Fix:** Move the tensor to the correct device: `tensor = tensor.to(gm.device)` or use `gm.register_buffer(name, tensor.cuda())`.

### Compiled output numerically identical to uncompiled
**Cause:** Backend was not invoked (old compiled version cached by dynamo).
**Fix:** Add `torch._dynamo.reset()` before the compiled forward call. Then re-run the smoke test.

### `AssertionError` in `test_get_model_and_input`
**Cause:** Batch padding in `get_model_and_input()` changed the expected batch size.
**Fix:** Either update the test assertion to the padded shape, or slice the output `out[:original_batch_size]` before shape checks.

### `RuntimeError: Expected ... but got ...` shape error in Step 5
**Cause:** Padding applied in `get_model_and_input()` but not accounted for in output slicing.
**Fix:** Slice the output tensor back to original batch size before returning from `get_model_and_input()` or after the forward pass.

## Numerical Correctness (Optional)

If `--numerical` is passed, also checks that the optimized output is numerically close to the baseline:

```
BF16 tolerance: atol=1e-2, rtol=1e-2
```

This test is EXPECTED TO FAIL if dtype promotion (BF16) was applied — BF16 produces different values than FP32, and that difference is correct by design. The test passes when the difference is within BF16 precision range.

## After Validation

- **All 5 steps PASS** → ready to profile with `/capture workload_optimized.py --profile-name=optimized`
- **Any step FAILS** → fix the issue, then re-run `/validate` before profiling

Do not run the ncu pipeline on a backend that fails Step 3 or later — ncu replay takes minutes and will produce meaningless metrics if the backend crashes.
