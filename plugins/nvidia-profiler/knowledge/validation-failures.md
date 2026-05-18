# Validation Failure Reference

## Non-Obvious Failures

### TypeError: 'module' object is not callable (Step 2)
**Symptom:** Step 2 import fails with `TypeError: 'module' object is not callable`
**Cause:** Wrong `compile_fx` import form — imports the module instead of the function
**Fix:** Replace `from torch._inductor import compile_fx` with `from torch._inductor.compile_fx import compile_fx`
**Diagnostic:** `grep -n "from torch._inductor import compile_fx" {file_path}`

### Output numerically identical to baseline (Step 4 — compiled test)
**Symptom:** `test_compiled_forward_pass` passes but no logger output is captured (`caplog.records` is empty)
**Cause:** `torch._dynamo` cached a previous compilation and is returning the cached (unoptimized) result — the custom backend is never invoked
**Fix:** Add `torch._dynamo.reset()` before `torch.compile()` in the test to clear the dynamo cache
**Diagnostic:** Check that `test_backend_registration` confirms the backend name before the compiled test runs

### NaN in output (Step 4 — compiled test)
**Symptom:** `test_compiled_forward_pass` output contains NaN
**Cause:** BF16 overflow — inputs to the BF16 cast were outside FP32→BF16 safe range
**Fix:** Check that activations are normalized before dtype promotion; consider casting only weights (not activations) to BF16

---

## Common Failures

### ModuleNotFoundError (Step 2)
**Cause:** PYTHONPATH does not include the project root
**Fix:** Run with `PYTHONPATH={project_root}` or `pip install -e .` from project root

### Backend not found in torch._dynamo (Step 3)
**Cause:** `@register_backend` decorator is missing, or the module import failed silently before registration ran
**Fix:** Confirm `@register_backend` decorator is present; re-run Step 2 import check to rule out silent failure

### test_get_model_and_input shape check fails (Step 4)
**Cause:** Batch padding in `get_model_and_input()` changed the tensor shape from what the test expects
**Fix:** Update the shape assertion in the test to match the padded shape, or verify padding logic is correct

### Graph is not valid (Step 4 — compiled test)
**Cause:** FX graph mutation performed without calling `gm.graph.lint()` afterward
**Fix:** Add `gm.graph.lint()` after all graph mutations, then `gm.recompile()`
**Diagnostic:** Search file for node erasure or insertion sites missing a subsequent `lint()` call

### Expected tensors on same device (Step 4 — compiled test)
**Cause:** A new tensor (e.g., from `register_buffer`) was created on CPU inside an FX pass
**Fix:** Ensure all `register_buffer` tensors are moved to CUDA: `self.register_buffer('name', tensor.cuda())`
