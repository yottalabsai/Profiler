# profile.json Schema Version Guide

Current version: `1.0`. All fields documented in `nvidia/operator_profiler/schema/profile.py`.

## Version Detection

```python
import json

with open('profile.json') as f:
    profile = json.load(f)

assert profile.get('schema_version') == '1.0'
```

## Top-level Structure

```json
{
  "schema_version": "1.0",
  "capture_metadata": { ... },
  "operators": [ ... ],
  "unattributed_kernels": [ ... ],
  "warnings": [ ... ]
}
```

## `capture_metadata` Fields

- `model_name` (str)
- `torch_version` (str)
- `cuda_version` (str | null)
- `compile_mode` (str: "eager" | "inductor" | "cudagraphs")
- `nsys_report_path` (str | null)
- `ncu_report_path` (str | null)
- `capture_timestamp_utc` (str, ISO 8601)
- `device_name` (str | null)

## `aggregated` Fields

- `total_duration_ns` (int)
- `kernel_count` (int)
- `dominant_kernel_id` (str | null)
- `total_dram_bytes_read` (int | null)
- `total_dram_bytes_written` (int | null)
- `memory_throughput_pct` (float | null)
- `dram_throughput_pct` (float | null)
- `mem_busy_pct` (float | null)
- `l1_hit_rate` (float | null)
- `l2_hit_rate` (float | null)
- `sm_throughput_pct` (float | null)
- `tensor_core_active_pct` (float | null)
- `achieved_occupancy` (float | null)
- `warp_cycles_per_instruction` (float | null)
- `eligible_cycles_pct` (float | null)
- `total_executed_instructions` (int)
- `total_issued_instructions` (int)
- `ipc_active` (float | null)
- `avg_threads_per_warp` (float | null)
- `registers_per_thread` (float | null)
- `local_memory_spills` (int | null)
- `dynamic_smem_per_block` (float | null)
