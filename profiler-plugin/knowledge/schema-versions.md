# profile.json Schema Version Guide

Compatibility reference for reading `profile.json` files across schema versions.

---

## Version Detection

```python
import json

with open('profile.json') as f:
    profile = json.load(f)

schema_version = profile.get('schema_version', None)

if schema_version == '1.0':
    # Current — all fields as documented
    pass
elif schema_version is None:
    # Pre-v1.0 — apply compatibility mapping below
    pass
```

---

## v1.0 (Current)

All field names are snake_case. All fields documented in `nvidia/operator_profiler/schema/profile.py`.

**Top-level structure:**
```json
{
  "schema_version": "1.0",
  "capture_metadata": { ... },
  "operators": [ ... ],
  "unattributed_kernels": [ ... ],
  "warnings": [ ... ]
}
```

**`capture_metadata` fields:**
- `model_name` (str)
- `torch_version` (str)
- `cuda_version` (str | null)
- `compile_mode` (str: "eager" | "inductor" | "cudagraphs")
- `nsys_report_path` (str | null)
- `ncu_report_path` (str | null)
- `capture_timestamp_utc` (str, ISO 8601)
- `device_name` (str | null)

**`aggregated` fields:**
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

---

## Pre-v1.0 (Legacy)

No `schema_version` field. Produced by older builds of the profiler.

**Field name changes (old → new):**

| Old Field | New Field | Location |
|---|---|---|
| `captureMetadata` | `capture_metadata` | top-level |
| `modelName` | `model_name` | capture_metadata |
| `torchVersion` | `torch_version` | capture_metadata |
| `cudaVersion` | `cuda_version` | capture_metadata |
| `compileMode` | `compile_mode` | capture_metadata |
| `captureTimestampUtc` | `capture_timestamp_utc` | capture_metadata |
| `deviceName` | `device_name` | capture_metadata |
| `operatorId` | `operator_id` | OperatorRecord |
| `operatorName` | `operator_name` | OperatorRecord |
| `callIndex` | `call_index` | OperatorRecord |
| `isFused` | `is_fused` | OperatorRecord |
| `fusedWith` | `fused_with` | OperatorRecord |
| `totalDurationNs` | `total_duration_ns` | aggregated |
| `kernelCount` | `kernel_count` | aggregated |
| `dominantKernelId` | `dominant_kernel_id` | aggregated |
| `achievedOccupancy` | `achieved_occupancy` | aggregated |
| `meanAchievedOccupancy` | `achieved_occupancy` | aggregated (renamed in v0.9) |
| `tensorCoreActivePct` | `tensor_core_active_pct` | aggregated |
| `warpCyclesPerInstruction` | `warp_cycles_per_instruction` | aggregated |
| `smThroughputPct` | `sm_throughput_pct` | aggregated |
| `memoryThroughputPct` | `memory_throughput_pct` | aggregated |
| `dramThroughputPct` | `dram_throughput_pct` | aggregated |
| `l1HitRate` | `l1_hit_rate` | aggregated |
| `l2HitRate` | `l2_hit_rate` | aggregated |

**Fields absent in pre-v1.0:**
- `unattributed_kernels` — did not exist; skip if missing
- `eligible_cycles_pct` — added in v0.8
- `dynamic_smem_per_block` — added in v0.9
- `ipc_active` — added in v0.9

**Compatibility reading function:**
```python
def read_aggregated_safe(agg: dict, field: str):
    """Read an aggregated field, handling both v1.0 and pre-v1.0 names."""
    COMPAT = {
        'total_duration_ns': ['totalDurationNs', 'total_duration_ns'],
        'achieved_occupancy': ['achievedOccupancy', 'meanAchievedOccupancy', 'achieved_occupancy'],
        'tensor_core_active_pct': ['tensorCoreActivePct', 'tensor_core_active_pct'],
        'sm_throughput_pct': ['smThroughputPct', 'sm_throughput_pct'],
        'dram_throughput_pct': ['dramThroughputPct', 'dram_throughput_pct'],
        'memory_throughput_pct': ['memoryThroughputPct', 'memory_throughput_pct'],
        'l1_hit_rate': ['l1HitRate', 'l1_hit_rate'],
        'l2_hit_rate': ['l2HitRate', 'l2_hit_rate'],
        'warp_cycles_per_instruction': ['warpCyclesPerInstruction', 'warp_cycles_per_instruction'],
    }
    for name in COMPAT.get(field, [field]):
        if name in agg and agg[name] is not None:
            return agg[name]
    return None
```

---

## Future Versions

Planned additions for v1.1+:
- `capture_metadata.batch_size` — explicit batch size (currently encoded in `model_name` by convention)
- `capture_metadata.warmup_iters` / `measure_iters` — stored alongside profile for replay validation
- `operators[*].aggregated.arithmetic_intensity` — computed FLOP/byte ratio (requires FLOP count from FLOPs profiler)
- `operators[*].aggregated.theoretical_occupancy` — max occupancy given `registers_per_thread` and `dynamic_smem_per_block`

When reading a file with `schema_version == "1.1"` or higher, check for these fields before using them; they may or may not be populated depending on profiler version.
