---
name: analyze
description: Parse a profile.json file and produce a triage table of GPU bottlenecks ranked by optimization opportunity. Classifies operators as tensor_core_idle, compute_bound, memory_bound, wave_starved, latency_bound, register_pressure, or layout_overhead. Outputs triage.json.
---

# /analyze — GPU Bottleneck Triage

Reads `profile.json` (from `/capture` or provided directly) and classifies every operator by its primary performance bottleneck. Outputs `triage.json` consumed by `/propose`.

## Usage

```
/analyze profile.json
/analyze profile.json --verbose       # include raw metric values in output
```

## What It Does

1. **Schema validation** — checks `schema_version`, warns on pre-v1.0 fields
2. **Time budget** — computes per-operator wall-time percentage (not stored in profile.json, must be derived as `op.total_duration_ns / sum(all ops)`)
3. **Bottleneck classification** — applies the decision tree below for each operator above 1% threshold
4. **Architecture lookup** — maps `device_name` to SM count and hardware limits for Waves/SM calculation
5. **Edge case flags** — detects the 8 attribution edge cases and warns which metrics to trust

## Bottleneck Classification Decision Tree

Applied in priority order (first match wins):

### tensor_core_idle (Highest Priority)
- `tensor_core_active_pct == 0.0` AND operator is a GEMM (`linear`, `mm`, `matmul`, `conv`, `bmm`)
- **Meaning:** GEMM kernel ran but Tensor Cores were idle — FP32 SIMT path
- **Fix:** Cast inputs/weights to BF16 or FP16; routes to Tensor Core hardware path
- **Potential gain:** 2–16× faster for large GEMMs

> Note: `tensor_core_active_pct == null` is NOT a bottleneck — it means the counter is unavailable for this kernel type (elementwise ops) or was removed on this architecture (Blackwell). Only `== 0.0` on a GEMM kernel is a problem.

### compute_bound
- `sm_throughput_pct > 70` AND `memory_throughput_pct < 40`
- **Meaning:** SM compute units are saturated; the kernel is near peak FLOPS
- **Fix:** Optimize tile size (`max-autotune`), ensure Tensor Cores are active, algorithm selection

### memory_bound
- `dram_throughput_pct > 60` AND `sm_throughput_pct < 30`
- OR `l1_hit_rate < 20` AND `l2_hit_rate < 50` (cache pollution)
- **Meaning:** HBM bandwidth is the bottleneck; data movement exceeds compute
- **Fix:** Operator fusion (eliminate intermediate materialization), weight layout optimization

### wave_starvation
- `achieved_occupancy < 20%`
- Compute **Waves/SM** = `ceil(grid_x × grid_y × grid_z / sm_count)` (SM count from hardware-limits.md)
  - `waves_per_sm < 0.5` → severe starvation
  - `0.5 ≤ waves_per_sm < 2.0` → moderate starvation
- **Meaning:** Not enough parallel work to keep all SMs busy; most SMs sit idle
- **Fix:** Increase batch size, batch padding, QKV fusion

### latency_bound
- `warp_cycles_per_instruction > 20` (warp stalls waiting for memory)
- OR on Blackwell (where this counter is removed): `eligible_cycles_pct < 20`
- **Meaning:** Warps stall on memory latency; insufficient occupancy to hide it
- **Fix:** Increase occupancy (padding), reduce register pressure

### register_pressure
- `registers_per_thread > 128` → register file pressure limits occupancy
- `local_memory_spills > 0` → **CRITICAL** — register spills to DRAM (10–100× slower)
- **Fix:** `torch.compile(mode='reduce-overhead')`, kernel rewrite

### layout_overhead
- `convertTensor_kernel` appears in any kernel within this operator
- **Meaning:** cuDNN coerces NCHW ↔ NHWC memory layout per operator call
- **Fix:** `model.to(memory_format=torch.channels_last)` at model initialization

## Architecture-Specific Notes

### Blackwell (B100, B200, RTX PRO 6000)
- `warp_cycles_per_instruction` counter was removed — `null` is expected and correct
- Use `eligible_cycles_pct < 20` instead as the latency-bound indicator
- Ridge point is very high (563–814 FLOP/byte) — most models are memory-bound on Blackwell unless compute-intensive

### Hopper (H100, H200)
- Tensor Cores require BF16/FP16 and grid alignment for WGMMA path
- `sm90_xmma_gemm_bf16bf16_*` in kernel name = Tensor Cores active (good)
- `gemmSN_NN_*` or `gemmSN_TN_*` = FP32 SIMT (Tensor Cores idle)

### Ampere (A100, A10G)
- `allow_tf32 = True` is a lower-cost alternative to BF16 on Ampere (3-bit mantissa sacrifice, Tensor Cores enabled)
- Tile size matters more for occupancy: minimum 64×64×16 for BF16 HMMA

## The 8 Attribution Edge Cases

The analysis flags conditions that affect metric reliability:

| Flag | Meaning | Impact |
|---|---|---|
| `ncu_replay_timing` | All durations come from ncu replay (2–5× longer than real) | Use for relative comparison only |
| `multi_stream_overlap` | Multiple stream_ids in one operator | total_duration_ns overestimates wall time |
| `warm_up_inflation` | kernel_count not divisible by measure_iters | Extra JIT-compilation kernels inflate count |
| `cuda_graph_replay` | compile_mode == cudagraphs | Kernel counts are replay counts, not execution counts |
| `fused_kernel_double_count` | is_fused == true for some operators | Duration is shared; don't add to total |
| `dynamic_shapes` | High kernel duration variance within same operator | Different shapes triggered different trace paths |
| `high_unattributed` | >10% kernels unattributed | Some operator costs are invisible in the profile |
| `clock_domain` | Negative or absurd start_ns values | Use duration_ns only, not absolute timestamps |

## Output: triage.json

```json
{
  "profile_summary": {
    "model_name": "ConvBlock",
    "device": "NVIDIA A100-SXM4-80GB",
    "compile_mode": "inductor",
    "total_attributed_ns": 2122456,
    "unattributed_kernel_count": 2,
    "total_kernel_count": 162
  },
  "time_budget": [
    {
      "operator_id": "aten::cudnn_convolution_0",
      "operator_name": "aten::cudnn_convolution",
      "pct": 81.9,
      "total_duration_ns": 1740320,
      "kernel_count": 90,
      "bottleneck_class": "layout_overhead",
      "priority": 1,
      "evidence": {
        "dominant_kernel_name": "convertTensor_kernel",
        "tensor_core_active_pct": 0.0,
        "achieved_occupancy": 8.2
      }
    }
  ],
  "edge_case_flags": ["ncu_replay_timing"],
  "architecture_notes": "A100 SXM5: 108 SMs, ridge point 156 FLOP/byte BF16.",
  "optimization_ceiling": "Top 2 operators account for 97.5% of attributed time."
}
```

The output is written to `triage.json` in the same directory as `profile.json`. Feed it to `/propose` for optimization proposals.
