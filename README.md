# Operator Profiler

**Know exactly why your GPU workload is slow.**

Operator Profiler captures real NVIDIA hardware counter data (`nsys` + `ncu`) and attributes it to individual PyTorch operators, giving you per-operator metrics: DRAM throughput, cache efficiency, occupancy, compute utilization, and more. Use these metrics to identify and optimize bottlenecks instead of guessing.

---

## Quick Start

Three complete examples are in `examples/`, each with a baseline profile, optimized backend, validation test suite, and before/after comparison:

| Example | Model | GPU | Speedup |
|---|---|---|---|
| [`examples/conv_block/`](examples/conv_block/) | VGG-style ConvBlock | A100 SXM4-80GB | **2.93×** measured |
| [`examples/gpt2/`](examples/gpt2/) | GPT-2 small (117M) | A100 SXM4-80GB | **~2.5–3.0×** estimated |
| [`examples/embedding_projection/`](examples/embedding_projection/) | Embedding + FFN + logit projection | A100 SXM4-80GB | **~2–5×** estimated |

Each directory contains `profile.json`, `*_optimized.py`, `report.md`, `OPTIMIZED_WORKLOAD.md`, `test_*.py`, and `validation_report.json`. Four additional baseline workloads (`mlp_activations`, `sdpa_attention`, `depthwise_separable_conv`, `lstm_sequence_encoder`) are included and will have results in the future.

---

## Why this profiler?

Most profiling tools tell you *what* ran. Operator Profiler tells you *why it was slow* and *which operators to fix first*.

### Hardware-level attribution

Rather than relying on PyTorch's built-in timing hooks, Operator Profiler:

1. Runs your workload under `nsys` to capture the full CUDA kernel timeline and NVTX operator annotations
2. Replays the workload under `ncu` in application-mode replay to collect hardware counters — DRAM bandwidth, L1/L2 hit rates, tensor-core utilization, warp occupancy, arithmetic intensity, and more
3. Attributes every CUDA kernel back to the PyTorch operator that launched it via a confidence-ranked chain:
   - **torch.profiler correlation** (`high` confidence) — kernel matched to an `aten::` op via CUPTI `EXTERNAL_ID` links from an optional `--correlation-pass` run
   - **NVTX enclosure** (`medium` confidence) — kernel falls within an `aten::` NVTX range emitted by `emit_nvtx`
   - **Inductor fusion enrichment** (`medium` confidence) — when `--inductor-debug-dir` is set, unattributed Triton fused kernels are matched to their fused `aten::` ops via Inductor debug artifacts
   - **Unattributed** — kernels with no match are collected in `unattributed_kernels[]`; nothing is silently dropped

### Layer deduplication

For models with repeated structure (e.g., transformer blocks), the profiler automatically detects structurally identical FX subgraphs, compiles only one representative per equivalence class, and propagates hardware metrics to all duplicates.

`UniqueSubgraphRegistry` splits the FX graph by structural signature (op kinds + targets), tags each partition as `layer::unique::<label>` or `layer::duplicate::<label>` in the NVTX trace, and writes a `.part.json` equivalence map during capture. Pass `--partition-map <prefix>.part.json` to `map` to skip duplicate-partition kernels during ncu replay and propagate metrics by positional index.

GPT-2's 12 identical transformer blocks go from ~30–45 min ncu replay time to ~3–5 min.

---

## Pipeline

```
[Capture]              [Map]                   [Output]
  nsys        ──────►  attribution +  ──────►  profile.json
  (optional:           ncu application          per-operator hardware
   correlation         replay                   metrics
   pass)
```

1. **`profile`** — runs the target script under `nsys` and builds a mapping manifest (kernel → NVTX ranges); layer deduplication is always active; requires `--inductor-debug-dir` for Inductor fusion enrichment; pass `--correlation-pass` to also run a `torch.profiler` capture for high-confidence attribution
2. **`map`** — runs `ncu` in application-replay mode to collect hardware counters, attributes kernels to operators, and produces the final `profile.json`; pass `--partition-map` to enable dedup-aware replay

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

## Claude Code Plugin

The profiler ships a Claude Code plugin that automates the full optimization workflow inside your Claude Code session. Pass a workload file and get a profiled, optimized, validated PyTorch backend — no manual pipeline steps required.

For full documentation — agents, hooks, knowledge base, and per-skill reference — see **[plugins/nvidia-profiler/README.md](plugins/nvidia-profiler/README.md)**.

### Prerequisites

- [Claude Code](https://claude.ai/code) installed
- `nsys` and `ncu` available (see [Requirements](#requirements))
- A CUDA GPU

### Installation

In any Claude Code session:

```
/plugin marketplace add yottalabsai/Profiler
/plugin install nvidia-profiler@profiler-plugins
```

This registers the plugin globally — no need to clone the repository first.

---

## CLI Reference

### `profile` — capture

Runs the target script under `nsys` and builds a mapping manifest.

```bash
operator-profiler profile model.py \
    --model-name MyModel \
    --output runs/my_run \
    --warmup-iters 2 \
    --inductor-debug-dir /tmp/torch_compile_debug
```

| Flag | Default | Description |
|---|---|---|
| `script` | — | Python script to profile |
| `--model-name` | `model` | Human-readable label stored in the manifest |
| `--output` | `profile` | Path prefix; `nsys` appends `.nsys-rep`, manifest gets `.manifest.json` |
| `--compile-mode` | `eager` | `eager` / `inductor` / `cudagraphs` |
| `--warmup-iters` | `2` | Warm-up iterations before the annotated capture run |
| `--nsys-executable` | `nsys` | Path to `nsys` binary |
| `--inductor-debug-dir` | required | Directory of Inductor trace artifacts (`output_code.py` files) written when `TORCH_COMPILE_DEBUG=1` is set |
| `--correlation-pass` | disabled | Run a `torch.profiler` pre-capture pass to build a `HIGH`-confidence CUPTI correlation map |

Outputs: `<output>.nsys-rep`, `<output>.manifest.json`, `<output>.part.json` (partition equivalence map)

---

### `map` — ncu range replay + attribution

Runs `ncu` in application-replay mode, attributes kernels to operators, and produces an operator-attributed profile with hardware metrics.

```bash
operator-profiler map runs/my_run.manifest.json \
    --script model.py \
    --output runs/profile.json \
    --partition-map runs/my_run.part.json \
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
| `--partition-map` | `None` | Path to `.part.json` from capture; enables dedup-aware ncu replay — skips duplicate-partition kernels and propagates metrics from unique representatives |
| `--device-name` | auto | GPU name (stored in metadata for reference) |

Output: `profile.json` — an `OperatorAttributedProfile` with per-operator hardware metrics.

---

## Example Workloads

Seven ready-to-run workloads are in `examples/`. Each exposes a `get_model_and_input()` function compatible with `nvidia/scripts/run_workload.py`:

| Workload | What it covers | Status |
|---|---|---|
| `conv_block` | Conv2d + BatchNorm + ReLU | Complete — profiled, optimized, validated |
| `gpt2` | GPT-2 small (117M), 12 transformer blocks | Complete — profiled, optimized, validated |
| `embedding_projection` | Embedding lookup + LayerNorm + FFN + logit projection | Complete — profiled, optimized, validated |
| `mlp_activations` | Deep MLP with multiple activation types | Baseline only |
| `sdpa_attention` | Multi-head SDPA (routes to FlashAttention-2 under Inductor) | Baseline only |
| `depthwise_separable_conv` | Depthwise + pointwise convolutions | Baseline only |
| `lstm_sequence_encoder` | LSTM sequence encoder | Baseline only |

To profile all seven in one batch:

```bash
python nvidia/scripts/run_all_profiles.py
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
    "compile_mode": "inductor",
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
              "dram__bytes_read.sum": 1073741824,
              "dram__bytes_write.sum": 67108864,
              "dram__throughput.avg.pct_of_peak_sustained_elapsed": 7.19,
              "sm__throughput.avg.pct_of_peak_sustained_elapsed": 82.4,
              "smsp__inst_executed.sum": 12345678
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

Raw metric names use ncu's namespaced format (`counter__metric.aggregation`). The `aggregated` block contains pre-computed Python fields derived from the raw counters.

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

The Claude Code plugin (see above) automates this entire workflow via `/optimize workload.py`. The manual steps below are available if you want finer control over individual stages.

Once you have a profile, use the provided prompt templates to identify and implement optimizations:

### 1. Identify Optimizations (`optimization_proposal_prompt.md`)

Pass your `profile.json` to this prompt along with your baseline `workload.py`. It analyzes hardware metrics and produces a structured list of operator-level optimizations with:
- Specific operators and bottleneck analysis (e.g., "Waves/SM=0.11 indicates kernel launch starvation")
- Concrete FX graph transformations (fusion, elimination, kernel substitution)
- Impact estimates (latency, throughput, memory improvements)

### 2. Implement Optimizations (`optimization_implementation_prompt.md`)

Pass the optimization recommendations and your workload to this prompt. It generates:
- A custom `torch.compile()` backend with FX graph passes routed across three IR levels: `functional` passes (fusion, SDPA) run before `compile_fx` on the Dynamo graph; `aten` passes (dtype casts, BN fold) run inside `_aten_inner_compile` targeting `torch.ops.aten.*` nodes; `inductor_config` passes are scoped `config_patches` on `compile_fx`. The three stages are composed as `_compile_unit` — see `plugins/nvidia-profiler/knowledge/fx-patterns.md` for the canonical implementation.
- A test script to verify the optimized workload
- Before/after documentation

See **[`examples/conv_block/`](examples/conv_block/)** for a complete walkthrough — `report.md` for the summary including before/after hardware counter evidence, and `OPTIMIZED_WORKLOAD.md` for implementation details.

---

## Version

`0.1.0` — see `pyproject.toml`.
