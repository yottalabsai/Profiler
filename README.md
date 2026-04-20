# Operator Profiler

**Know exactly why your GPU workload is slow.**

Operator Profiler captures real NVIDIA hardware counter data (`nsys` + `ncu`) and attributes it to individual PyTorch operators, giving you per-operator metrics: DRAM throughput, cache efficiency, occupancy, compute utilization, and more. Use these metrics to identify and optimize bottlenecks instead of guessing.

---

## Quick Start

For a complete walkthrough — baseline capture, hardware metric analysis, FX graph optimizations, and before/after comparison with real measured numbers — see **[examples/example.md](examples/example.md)**.

The example walks through a TransformerBlock workload on an NVIDIA RTX PRO 6000 Blackwell GPU, achieving a **6.3× per-sample speedup** purely from profiler-guided changes.

---

## Why this profiler?

Most profiling tools tell you *what* ran. Operator Profiler tells you *why it was slow* and *which operators to fix first*.

### Hardware-level attribution

Rather than relying on PyTorch's built-in timing hooks, Operator Profiler:

1. Runs your workload under `nsys` to capture the full CUDA kernel timeline and NVTX operator annotations
2. Replays each NVTX range under `ncu` to collect hardware counters — DRAM bandwidth, L1/L2 hit rates, tensor-core utilization, warp occupancy, arithmetic intensity, and more
3. Attributes every CUDA kernel back to the PyTorch operator that launched it via a confidence-ranked chain:
   - **NVTX enclosure** (`medium` confidence) — kernel falls within an `aten::` NVTX range emitted by `emit_nvtx`
   - **Kernel name heuristic** (`low` confidence) — Triton kernel name parsed to infer fused aten ops

---

## Pipeline

```
[Capture]          [Map]              [Output]
  nsys    ──────►  ncu replay  ──────►  profile.json
  NVTX             attribution          per-operator hardware
  annotation       (kernel→aten op)     metrics
```

1. **`profile`** — runs the target script under `nsys` and builds a mapping manifest (kernel → NVTX ranges)
2. **`map`** — replays each NVTX range under `ncu` to collect hardware counters, attributes kernels to operators, and produces the final `profile.json`

---

## Requirements

| Requirement | Notes |
|---|---|
| Python ≥ 3.10 | |
| PyTorch | Any version with `torch.fx` and `torch.autograd.profiler.emit_nvtx` |
| NVIDIA Nsight Systems (`nsys`) | On `PATH` for capture |
| NVIDIA Nsight Compute (`ncu`) | On `PATH` for range replay; **requires sudo** on most systems |
| CUDA GPU | Required for capture and replay |

---

## Installation

```bash
pip install -e .
```

`torch` must be installed separately with the correct CUDA variant for your driver. See [pytorch.org](https://pytorch.org) for the appropriate install command.

---

## CLI Reference

### `profile` — capture

Runs the target script under `nsys` and builds a mapping manifest.

```bash
operator-profiler profile model.py \
    --model-name MyModel \
    --output runs/my_run \
    --warmup-iters 2
```

| Flag | Default | Description |
|---|---|---|
| `script` | — | Python script to profile |
| `--model-name` | `model` | Human-readable label stored in the manifest |
| `--output` | `profile` | Path prefix; `nsys` appends `.nsys-rep`, manifest gets `.manifest.json` |
| `--compile-mode` | `eager` | `eager` / `inductor` / `cudagraphs` |
| `--warmup-iters` | `2` | Warm-up iterations before the annotated capture run |
| `--nsys-executable` | `nsys` | Path to `nsys` binary |

Outputs: `<output>.nsys-rep`, `<output>.manifest.json`

---

### `map` — ncu range replay + attribution

Replays each NVTX range under `ncu`, attributes kernels to operators, and produces an operator-attributed profile with hardware metrics.

```bash
operator-profiler map runs/my_run.manifest.json \
    --script model.py \
    --output runs/profile.json \
    --device-name "A100 SXM4 80GB"
```

| Flag | Default | Description |
|---|---|---|
| `manifest` | — | Path to `.manifest.json` from `profile` |
| `--script` | required | Same script used for capture |
| `--script-args` | `[]` | Arguments forwarded to the script |
| `--output` | `profile.json` | Output path for `OperatorAttributedProfile` JSON |
| `--ncu-executable` | `ncu` | Path to `ncu` binary |
| `--ncu-sudo` | disabled | Prefix `ncu` with `sudo -E`; required on most Linux systems to access GPU performance counters |
| `--ncu-env KEY=VAL` | `[]` | Extra env vars forwarded under `sudo` (e.g. `PYTHONPATH=/path/to/repo`); needed because `sudo` drops the environment |
| `--device-name` | auto | GPU name (stored in metadata for reference) |

Output: `profile.json` — an `OperatorAttributedProfile` with per-operator hardware metrics.

---

## Example Workloads

Six ready-to-run workloads are in `scripts/workloads/`. Each exposes a `get_model_and_input()` function compatible with `scripts/run_workload.py`:

| Workload | What it covers |
|---|---|
| `transformer_block` | Attention + FFN + LayerNorm — the reference workload for `examples/example.md` |
| `conv_block` | Conv2d + BatchNorm + ReLU |
| `mlp_activations` | Deep MLP with multiple activation types |
| `sdpa_attention` | Multi-head SDPA (routes to FlashAttention-2 under Inductor) |
| `depthwise_separable_conv` | Depthwise + pointwise convolutions |
| `embedding_projection` | Embedding lookup + linear projection |

To profile all six in one batch:

```bash
python scripts/run_all_profiles.py
```

Results land in `runs/<workload_name>/`.

---

## Output: `profile.json`

All data is serialized as JSON using Pydantic v2 models.

```jsonc
{
  "schema_version": "1.0",
  "capture_metadata": {
    "model_name": "MyModel",
    "torch_version": "2.3.0",
    "compile_mode": "eager",
    "device_name": "A100 SXM4 80GB",
    "capture_timestamp_utc": "2026-04-07T12:00:00+00:00"
  },
  "operators": [
    {
      "operator_id": "aten::mm_0",
      "operator_name": "aten::mm",
      "call_index": 0,
      "is_fused": false,
      "kernels": [
        {
          "kernel_id": "k_0_0",
          "kernel_name": "ampere_sgemm_128x64_tn",
          "stream_id": 0,
          "device_id": 0,
          "start_ns": 1000000,
          "end_ns": 15321000,
          "duration_ns": 14321000,
          "metrics": {
            "raw": {
              "dram_bytes_read": 1073741824,
              "dram_bytes_written": 67108864,
              "achieved_occupancy": 0.87,
              "tensor_core_active_pct": 91.2,
              "l1_hit_rate": 0.34,
              "l2_hit_rate": 0.71
            }
          }
        }
      ],
      "aggregated": {
        "total_duration_ns": 14321000,
        "kernel_count": 1,
        "dominant_kernel_id": "k_0_0",
        "total_dram_bytes_read": 1073741824,
        "total_dram_bytes_written": 67108864,
        "achieved_occupancy": 0.87,
        "tensor_core_active_pct": 91.2,
        "l1_hit_rate": 0.34,
        "l2_hit_rate": 0.71
      }
    }
  ],
  "unattributed_kernels": [],
  "warnings": []
}
```

The profile contains hardware metrics for every operator: DRAM throughput, cache hit rates, occupancy, compute utilization (Tensor Cores), and instruction throughput. You can then use these metrics to reason about bottlenecks and optimization strategies — either manually, or by passing the profile to Claude or another tool for analysis.

---

## Python API

### Capture helpers

```python
from operator_profiler.capture.nvtx_capture import NvtxCapture

# Warm-up + annotated capture in one context manager
with NvtxCapture(warmup_iters=2, warmup_fn=lambda: model(x)):
    output = model(x)
```

### Loading a profile

```python
from operator_profiler.schema.profile import OperatorAttributedProfile

profile = OperatorAttributedProfile.model_validate_json(
    open("runs/profile.json").read()
)

# Find the slowest operators
operators_by_duration = sorted(
    (op for op in profile.operators if op.aggregated),
    key=lambda op: op.aggregated.total_duration_ns,
    reverse=True
)

for op in operators_by_duration[:5]:
    agg = op.aggregated
    print(f"{op.operator_name}: {agg.total_duration_ns / 1e6:.2f} ms, "
          f"occupancy={agg.achieved_occupancy or 'N/A':.1%}, "
          f"dram_read={agg.total_dram_bytes_read / 1e9 or 0:.1f} GB")
```

---

## Optimization Workflow

Once you have a profile, use the provided prompt templates to identify and implement optimizations:

### 1. Identify Optimizations (`optimization_proposal_prompt.md`)

Pass your `profile.json` to this prompt along with your baseline `workload.py`. It analyzes hardware metrics and produces a structured list of operator-level optimizations with:
- Specific operators and bottleneck analysis (e.g., "Waves/SM=0.11 indicates kernel launch starvation")
- Concrete FX graph transformations (fusion, elimination, kernel substitution)
- Impact estimates (latency, throughput, memory improvements)

### 2. Implement Optimizations (`optimization_implementation_prompt.md`)

Pass the optimization recommendations and your workload to this prompt. It generates:
- A custom `torch.compile()` backend with FX graph passes that implement the optimizations
- A test script to verify the optimized workload
- Before/after documentation

See **[examples/example.md](examples/example.md)** for a full walkthrough of this workflow with real measured improvements (6.3× speedup on a TransformerBlock).

---

## Version

`0.1.0` — see `operator_profiler/__init__.py`.
