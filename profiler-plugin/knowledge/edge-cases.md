# Attribution Edge Cases

The 8 edge cases handled by `attribution_engine.py`, written for users reading `profile.json` directly. Each entry describes the data signature, impact on metric validity, and recommended action.

---

## Edge Case 1: Clock Domain Mismatch

**What it looks like in profile.json:**
- `kernel.start_ns` values are negative, absurdly large (> 10^18), or do not cluster near the `capture_timestamp_utc` epoch
- `duration_ns` values are normal (nanoseconds range) while `start_ns` values are implausible

**Cause:** nsys and ncu use different clock domains (GPU hardware counter vs. wall clock). When absolute timestamps are cross-referenced between the two tools, the offset can be enormous.

**Impact:** Absolute timestamps are unreliable. The profiler already avoids this by matching kernels via `(kernel_name, invocation_index)` rather than absolute timestamps.

**Action:** Use `duration_ns` for all timing comparisons. Never compute time deltas using `start_ns - start_ns` across different kernel records. The `/compare` skill normalizes on `duration_ns` only.

---

## Edge Case 2: CUDA Graph Replay

**What it looks like:**
- `capture_metadata.compile_mode == "cudagraphs"`
- `kernel_count` for each operator is a multiple of `measure_iters × graph_replay_count`
- Many kernels share identical `kernel_name` values in sequence (the replayed graph)

**Cause:** CUDA Graphs capture the entire forward pass as a single GPU graph and replay it on each iteration. Each replay re-executes every kernel in the original trace.

**Impact:**
- Kernel counts are inflated (each graph replay adds one set of kernels)
- `duration_ns` per kernel is the duration of a single graph replay invocation — this IS the real execution time
- Attribution is harder: NVTX ranges may not fire during replay (only during the capture pass)

**Action:** Divide kernel counts by `measure_iters` to get per-forward-pass counts. Treat `duration_ns` as valid. For attribution analysis, compare against an `inductor` baseline.

---

## Edge Case 3: Multi-Stream Overlap

**What it looks like:**
- A single `OperatorRecord` contains `kernels[]` with multiple distinct `stream_id` values
- `aggregated.total_duration_ns` = sum of all kernel durations (includes parallel streams)

**Cause:** PyTorch may dispatch GEMM (stream 7) and elementwise (stream 0) kernels for the same operator in parallel on different CUDA streams.

**Impact:** `total_duration_ns` overestimates wall time by the amount of stream overlap. An operator with two 1ms kernels on different streams appears to take 2ms, but the real wall time is 1ms.

**Action:** When `len(set(k.stream_id for k in op.kernels)) > 1`, note that `total_duration_ns` overestimates wall time. The actual wall time is `max(k.end_ns for k in op.kernels) - min(k.start_ns for k in op.kernels)` (this is not computed in `profile.json`). Use kernel-level `duration_ns` for bottleneck identification.

---

## Edge Case 4: JIT Warm-Up Inflation

**What it looks like:**
- `kernel_count` is NOT a multiple of `measure_iters`
- One or more kernels have `duration_ns` that is > 10× the median duration for kernels of the same name
- These outlier kernels typically appear early in the stream (low `start_ns` relative to others)

**Cause:** PyTorch JIT compilation (cuDNN algorithm search, Inductor compilation, Triton kernel compilation) happens during the first few iterations. These compilation kernels appear in the nsys trace alongside the actual compute kernels.

**Impact:** `aggregated.kernel_count` and potentially `total_duration_ns` are inflated by compilation overhead. The outlier compilation kernel duration can dominate the measurement.

**Action:** The `ManifestBuilder` attempts to exclude warm-up kernels using a duration outlier heuristic (> 10× median). If the `warnings[]` array in `profile.json` contains `"warm_up_kernel_excluded: ..."`, the pipeline already handled this. If you see suspiciously high `kernel_count` (e.g., 31 kernels instead of 30), the extra kernel is likely a JIT artifact.

---

## Edge Case 5: Async Kernel Launch

**What it looks like:**
- Host-side Python `time.time()` measurements around an operator are longer than `duration_ns` in `profile.json`
- The gap between `kernel.start_ns` and the host-side timestamp is large

**Cause:** CUDA kernel launches are asynchronous. The CPU returns immediately after dispatching; the GPU executes later. nsys captures GPU-side `start_ns`/`end_ns` from CUPTI, not host-side dispatch times.

**Impact:** None for `profile.json` analysis. GPU-side CUPTI timestamps in `duration_ns` are authoritative. Host-side timing (e.g., from `torch.cuda.Event`) is subject to CPU overhead.

**Action:** Always use `duration_ns` from `profile.json`. Ignore any host-side timing from outside the profiler. The `/analyze` skill and `/compare` skill are built on `duration_ns` exclusively.

---

## Edge Case 6: Dynamic Shapes

**What it looks like:**
- Multiple `OperatorRecord` entries with the same `operator_name` but different `call_index` values
- Wide variance in `aggregated.total_duration_ns` across these entries (> 50% coefficient of variation)
- `aggregated.kernel_count` differs between entries of the same operator

**Cause:** A model using dynamic shapes may encounter different input shapes at different call indices. Each unique shape triggers a different Dynamo trace, producing different kernels.

**Impact:**
- Metrics from one trace cannot be extrapolated to another
- Batch padding optimizations are risky (pad to the wrong tile size for some shapes)
- Wave occupancy calculations are only valid for the specific `grid_dim` observed

**Action:** Flag the `dynamic_shapes` edge case. Analyze each `call_index` separately. Avoid padding optimizations. Check whether the model can be compiled with `torch.compile(dynamic=False)` to fix shapes.

---

## Edge Case 7: Fused Kernel Multi-NVTX

**What it looks like:**
- `kernel.is_fused == true` on one or more `KernelRecord` entries
- `kernel.attribution.all_enclosing_ranges` contains more than one NVTX range
- The same kernel appears under multiple `OperatorRecord` entries (it is shared between operators)

**Cause:** Inductor generates "fused" Triton kernels that implement multiple `aten::` operators in a single GPU kernel. The kernel's NVTX range spans multiple operator boundaries.

**Impact:**
- The kernel's `duration_ns` represents the combined cost of multiple operators
- Attributing it entirely to one operator double-counts it in the time budget
- `aggregated.total_duration_ns` for a fused operator already accounts for this (it uses `is_fused` to avoid double-counting)

**Action:** When `is_fused == true`, do not add this kernel's duration to a manual total. The `aggregated` fields already handle this correctly. For optimization targeting: the fused kernel's metrics represent the combined bottleneck; you cannot optimize individual sub-operators independently without un-fusing.

---

## Edge Case 8: ncu Replay Timing

**What it looks like:**
- This applies to ALL data in `profile.json` — it is not detectable from the data itself
- `duration_ns` values in `profile.json` are typically 2–5× longer than the same kernels measured by nsys

**Cause:** `ncu --replay-mode kernel` instruments each kernel individually with performance counters. The instrumentation overhead is significant. The `duration_ns` stored in `profile.json` is the ncu-instrumented duration, not the real execution duration.

**Impact:** Absolute `duration_ns` values CANNOT be used as wall-clock latency estimates. A kernel that takes 10μs in real execution may show 30–50μs in `profile.json`.

**Action:**
- Use `duration_ns` for RELATIVE comparisons only (between operators, or before/after optimization)
- A 2× speedup in ncu-replayed durations corresponds to a real 2× speedup (within ~10% measurement noise)
- For absolute wall-clock latency, use `torch.utils.benchmark` or nsys timeline view (not `profile.json`)
- The `/compare` skill and `/report` skill always include this caveat
