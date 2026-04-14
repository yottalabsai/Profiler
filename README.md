# Operator Profiler

**Know exactly why your GPU workload is slow — and what to do about it.**

Operator Profiler captures real NVIDIA hardware counter data (`nsys` + `ncu`) and attributes it to individual PyTorch operators. It then applies LLM-powered analysis (via Claude) to reason about your workload's bottlenecks *relative to itself*, giving you workload-aware, actionable optimization guidance instead of generic threshold checks.

---

## Quick Start

For a complete walkthrough — baseline capture, bottleneck diagnosis, FX graph optimizations, and before/after comparison with real measured numbers — see **[example.md](example.md)**.

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

### Roofline analysis

For each operator, Operator Profiler computes arithmetic intensity and positions it against your GPU's roofline model — classifying whether it's **compute-bound**, **memory-bound**, or **latency-bound** using real GPU specs (A100, H100, RTX 4090/5090, and more).

### LLM-powered diagnosis

The `DiagnosisAgent` goes beyond static thresholds. It uses Claude to reason about each operator's metrics *relative to the rest of the workload*:

- An operator at 40% occupancy is alarming if the model median is 75%, unremarkable if the median is 35%
- Bottleneck labels reflect how each operator compares to its peers — not generic rules
- Produces a `bottleneck_classification` field (`compute_bound`, `memory_bound`, `latency_bound`, `unknown`) with full reasoning in the agent's internal trace

This prompt-engineered analysis runs as a post-pass over all operators after hardware metrics are collected, so it has global context across the entire workload before making any call.

After diagnosis, use `prompt.md` as a reusable template to ask Claude for operator-level optimization recommendations based on your `profile.json`. The resulting structured recommendations (like those in `OPTIMIZATIONS.json`) map hardware evidence to concrete code transformations, ready to implement as FX graph passes or dtype/shape changes. See **[example.md](example.md)** for the full optimization workflow in action.

---

## Pipeline

```
[Capture]          [Map]              [Output]
  nsys    ──────►  ncu replay  ──────►  profile.json
  NVTX             attribution          per-operator hardware
  annotation       + DiagnosisAgent     metrics + bottleneck
                   (Claude)             classification
```

1. **`profile`** — runs the target script under `nsys` and builds a mapping manifest (kernel → NVTX ranges)
2. **`map`** — replays each NVTX range under `ncu` to collect hardware counters, attributes kernels to operators, and runs DiagnosisAgent to produce the final `profile.json`

---

## Requirements

| Requirement | Notes |
|---|---|
| Python ≥ 3.10 | |
| PyTorch | Any version with `torch.fx` and `torch.autograd.profiler.emit_nvtx` |
| NVIDIA Nsight Systems (`nsys`) | On `PATH` for capture |
| NVIDIA Nsight Compute (`ncu`) | On `PATH` for range replay; **requires sudo** on most systems |
| CUDA GPU | Required for capture and replay |
| `anthropic` Python SDK | Required for LLM-powered diagnosis (optional — falls back to roofline heuristics) |

---

## Installation

```bash
# Core install
pip install -e .

# With LLM diagnosis (recommended)
pip install -e ".[llm]"
```

Set your Anthropic API key to enable LLM-powered bottleneck analysis:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
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

### `map` — ncu range replay + LLM diagnosis

Replays each NVTX range under `ncu`, attributes kernels to operators, and produces an operator-attributed profile. Runs DiagnosisAgent by default if `ANTHROPIC_API_KEY` is set.

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
| `--device-name` | auto | GPU name (used for roofline specs lookup) |
| `--diagnose` / `--no-diagnose` | enabled | Run DiagnosisAgent for LLM-powered bottleneck classification |

Output: `profile.json` — an `OperatorAttributedProfile` with per-operator hardware metrics and bottleneck classifications.

---

## Example Workloads

Six ready-to-run workloads are in `scripts/workloads/`. Each exposes a `get_model_and_input()` function compatible with `scripts/run_workload.py`:

| Workload | What it covers |
|---|---|
| `transformer_block` | Attention + FFN + LayerNorm — the reference workload for `example.md` |
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
      "attribution_confidence": "medium",
      "kernels": [
        {
          "kernel_name": "ampere_sgemm_128x64_tn",
          "metrics": {
            "duration_ns": 14321000,
            "dram_bytes_read": 1073741824,
            "dram_bytes_write": 67108864,
            "achieved_occupancy": 0.87,
            "tensor_core_active_pct": 91.2,
            "arithmetic_intensity": 42.3,
            "l1_hit_rate": 0.34,
            "l2_hit_rate": 0.71
          }
        }
      ],
      "aggregated": {
        "total_duration_ns": 14321000,
        "kernel_count": 1,
        "total_dram_bytes_read": 1073741824,
        "mean_achieved_occupancy": 0.87,
        "mean_tensor_core_active_pct": 91.2,
        "bottleneck_classification": "compute_bound"
      }
    }
  ],
  "unattributed_kernels": [],
  "warnings": []
}
```

The `bottleneck_classification` field is set by the DiagnosisAgent when available, or by the roofline heuristic as a fallback. The agent has access to the full model-wide metric distribution (median arithmetic intensity, median occupancy, GPU ridge point) before classifying any single operator.

---

## Python API

### Capture helpers

```python
from operator_profiler.capture.nvtx_capture import NvtxCapture

# Warm-up + annotated capture in one context manager
with NvtxCapture(warmup_iters=2, warmup_fn=lambda: model(x)):
    output = model(x)
```

### DiagnosisAgent

```python
from operator_profiler.agents import DiagnosisAgent
from operator_profiler.aggregator.profile_builder import build_profile

# Pass to build_profile for LLM-powered bottleneck classification
agent = DiagnosisAgent()
profile = build_profile(
    manifest=manifest,
    operator_records=operator_records,
    unattributed_kernels=unattributed,
    model_name="MyModel",
    torch_version="2.3.0",
    device_name="A100 SXM4 80GB",
    diagnosis_agent=agent,
)
```

### Loading a profile

```python
from operator_profiler.schema.profile import OperatorAttributedProfile

profile = OperatorAttributedProfile.model_validate_json(
    open("runs/profile.json").read()
)

# Find the worst bottlenecks
memory_bound = [
    op for op in profile.operators
    if op.aggregated and op.aggregated.bottleneck_classification == "memory_bound"
]
memory_bound.sort(key=lambda op: op.aggregated.total_duration_ns, reverse=True)

for op in memory_bound[:5]:
    print(f"{op.operator_id}: {op.aggregated.total_duration_ns / 1e6:.2f} ms")
```

---

## Supported GPUs

Roofline specs are built in for:

| GPU | Peak Compute (TFLOPS FP16) | Peak Bandwidth (GB/s) |
|---|---|---|
| H100 SXM5 80GB | 1,979 | 3,350 |
| A100 SXM4 80GB | 312 | 2,000 |
| A100 PCIe 80GB | 312 | 1,935 |
| RTX 5090 | 839 | 1,792 |
| RTX 4090 | 83 | 1,008 |
| RTX 3090 | 36 | 936 |
| RTX 5070 | 244 | 672 |
| RTX 5070 Laptop | 198 | 448 |

Custom specs can be added to `operator_profiler/aggregator/roofline.py:KNOWN_GPU_SPECS`.

---

## Environment Variables

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic API key for `DiagnosisAgent` LLM-powered bottleneck classification |

---

## Version

`0.1.0` — see `operator_profiler/__init__.py`.
