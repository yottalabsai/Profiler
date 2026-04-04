# Operator Profiler

A GPU hardware profiling and optimization toolkit that correlates CUDA kernel metrics with PyTorch operators and uses an LLM-driven loop to suggest and apply graph rewrites.

---

## Overview

Operator Profiler runs a multi-stage pipeline:

```
[Capture]  →  [Map]  →  [Report]  →  [Rewrite → Verify → Profile]  →  [Summarize]
  nsys          ncu        CLI         LLM planner + FX executor         diff report
```

1. **Capture** — profiles a script under `nsys`, collecting NVTX ranges and CUDA kernel timelines. Optionally collects Inductor provenance for compile-mode runs.
2. **Map** — replays each NVTX range under `ncu` to collect hardware counters (DRAM bandwidth, occupancy, tensor-core utilization, etc.) and attributes every kernel to a PyTorch operator.
3. **Aggregate & Report** — aggregates kernel metrics per operator, classifies bottlenecks using a roofline model, and prints a ranked table.
4. **Rewrite** — an LLM-backed planner (`ThetaPlanner`) generates `RewritePlan` JSON objects targeting the worst bottleneck. A `HybridExecutor` applies the plan to a `torch.fx.GraphModule`, then a `VerificationGate` checks numerical equivalence before lowering to Inductor.
5. **Summarize** — computes before/after diffs, renders Markdown/HTML/Rich dashboards, and exposes per-node explanations.

---

## Features

- **Full attribution chain**: provenance → NVTX enclosure → name heuristic fallback, with confidence scores (`high` / `medium` / `low`).
- **Eight edge-case handlers**: clock-domain mismatch, CUDA graph replay, multi-stream, JIT warm-up inflation, async kernel launch, dynamic shapes, fused kernels, and ncu-replay timing skew.
- **Roofline analysis**: classifies each operator as compute-bound or memory-bound against known GPU peak specs (A100, H100, RTX 4090/5090 and more).
- **LLM optimization loop**: beam search with explore/refine strategies, LLM-ranked memory retrieval, `VerifierAgent`-guided repair retries, and progressive baseline advancement.
- **Rich reporting**: Markdown, self-contained HTML, and `rich` CLI dashboards with per-node provenance tables.
- **Pure-Python schema**: all data contracts are Pydantic v2 models — easy to inspect, serialize, and extend.

---

## Requirements

| Requirement | Notes |
|---|---|
| Python ≥ 3.10 | |
| PyTorch | Any version with `torch.fx` and `torch.autograd.profiler.emit_nvtx` |
| NVIDIA Nsight Systems (`nsys`) | On `PATH` for capture |
| NVIDIA Nsight Compute (`ncu`) | On `PATH` for range replay |
| CUDA GPU | Required for capture and replay; unit tests run without GPU |
| `anthropic` Python SDK | Required only for `ThetaPlanner` / LLM features |
| `rich` ≥ 13.0 | Optional; required for `--format rich` in `summarize` |

---

## Installation

```bash
# Core install (profiling + attribution + aggregation)
pip install -e .

# With rich dashboard support
pip install -e ".[rich]"

# Development (adds pytest + pytest-mock)
pip install -e ".[dev]"

# LLM planner features
pip install anthropic
```

Set `ANTHROPIC_API_KEY` in your environment before using `ThetaPlanner`:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## CLI Reference

The package installs a single entry point: `operator-profiler`.

### `profile` — capture

Runs the target script under `nsys` and builds a mapping manifest.

```bash
operator-profiler profile model.py \
    --model-name MyModel \
    --output runs/my_run \
    --compile-mode inductor \
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
| `--no-provenance` | off | Disable Inductor provenance sidecar (inductor mode only) |

Outputs: `<output>.nsys-rep`, `<output>.manifest.json`

---

### `map` — ncu range replay + attribution

Runs `ncu` range replays for each NVTX range in the manifest and emits an operator-attributed profile.

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
| `--device-name` | auto | GPU name (used for roofline specs lookup) |

Output: `<output>.json` (an `OperatorAttributedProfile`)

---

### `report` — console summary

Prints a ranked operator table to stdout.

```bash
operator-profiler report runs/profile.json --top 20 --sort duration
```

| Flag | Default | Description |
|---|---|---|
| `profile` | — | Path to `OperatorAttributedProfile` JSON |
| `--top` | `20` | Number of operators to display |
| `--sort` | `duration` | Sort key: `duration` / `dram_read` / `dram_write` / `kernel_count` |

Example output:

```
======================================================================
  Operator-Attributed Profile
  Model:   MyModel
  Device:  A100 SXM4 80GB
  Mode:    inductor
======================================================================

Rank  Operator                                  Duration ms    % Total  Kernels  Bottleneck       Confidence
---------------------------------------------------------------------------------------------------------
1     aten::mm_0                                      14.321      42.3%       3  memory_bound     high
2     aten::layer_norm_0                               8.102      23.9%       2  compute_bound    high
...
```

---

### `summarize` — optimization report

Generates a full before/after optimization report from a completed loop run.

```bash
operator-profiler summarize \
    --before runs/profile_before.json \
    --after  runs/profile_after.json \
    --loop-result runs/loop_result.json \
    --memory  runs/memory.json \
    --format  html \
    --output  runs/summary.html \
    --top-n   5
```

| Flag | Default | Description |
|---|---|---|
| `--before` | required | Before-optimization profile JSON |
| `--after` | required | After-optimization profile JSON |
| `--loop-result` | required | `LoopResult` JSON from `OptimizationLoop.run()` |
| `--memory` | required | `OptMemoryStore` JSON |
| `--format` | `rich` | `rich` (console) / `markdown` / `html` |
| `--output` | stdout / auto | Output file path |
| `--top-n` | `5` | Top bottlenecks to include in the report |

---

### `explain` — per-node explanation

Prints a natural-language explanation for one operator node.

```bash
operator-profiler explain \
    --node aten__mm_0 \
    --before runs/profile_before.json \
    --after  runs/profile_after.json \
    --loop-result runs/loop_result.json
```

`--node` accepts `aten::mm_0` or the shell-safe alias `aten__mm_0` (double underscores replace `::`).

---

## Python API

### Running the optimization loop

```python
import torch
import torch.fx as fx
from operator_profiler.planner import (
    ThetaPlanner, PlannerConfig,
    OptimizationMemory,
    OptimizationLoop, LoopConfig, LoopResult,
    BeamSearch,
)
from operator_profiler.rewriter import HybridExecutor, ExecutorConfig
from operator_profiler.agents import VerifierAgent

# 1. Trace your model
gm: fx.GraphModule = fx.symbolic_trace(model)

# 2. Get a baseline profile (run capture + map pipeline, or use your own)
initial_profile = ...  # OperatorAttributedProfile loaded from JSON

# 3. Build the loop components
planner = ThetaPlanner(PlannerConfig(model="claude-sonnet-4-6"))
memory  = OptimizationMemory()
search  = BeamSearch(beam_width=3)

def profiler_fn(gm: fx.GraphModule):
    # Re-run the map pipeline on the rewritten graph and return a new profile
    ...

loop = OptimizationLoop(
    planner=planner,
    memory=memory,
    search=search,
    profiler_fn=profiler_fn,
    config=LoopConfig(n_iterations=5, speedup_threshold=1.05),
    verifier_agent=VerifierAgent(),
)

# 4. Run
result: LoopResult = loop.run(gm, initial_profile, example_inputs=[...])
print(f"Best speedup: {result.best_speedup:.3f}x")
```

### Capture helpers

```python
from operator_profiler.capture.nvtx_capture import NvtxCapture

# Warm-up + annotated capture in one context manager
with NvtxCapture(warmup_iters=2, warmup_fn=lambda: model(x)) as ctx:
    output = model(x)
```

### Applying a rewrite plan manually

```python
from operator_profiler.rewriter import HybridExecutor, ExecutorConfig, RewritePlan

plan = RewritePlan.model_validate({
    "plan_version": "1.0",
    "ops": [
        {"op": "fuse", "id": "f0", "nodes": ["add_1", "relu_1"], "strategy": "inductor_fuse"}
    ]
})

executor = HybridExecutor(gm, plan, ExecutorConfig(atol=1e-4, rtol=1e-4))
rewritten_gm, verification_results = executor.run()
```

### Generating a diff report

```python
from operator_profiler.summarizer import compute_diff, render_markdown, render_html

diff = compute_diff(before=before_profile, after=after_profile, plan=best_plan, top_n=5)
print(render_markdown(SummaryReport(diff=diff, rules=[], lessons_learned=[], ...)))
```

---

## Architecture

### Module layout

```
operator_profiler/
├── capture/               # Stage 1 — nsys + NVTX + provenance collection
│   ├── nsys_runner.py     # subprocess wrapper for nsys profile
│   ├── nvtx_capture.py    # NvtxCapture context manager (warm-up + emit_nvtx)
│   ├── provenance_reader.py
│   ├── provenance_collector.py
│   └── cuda_graph_capture.py
│
├── mapper/                # Stage 2 — NVTX→kernel attribution
│   ├── manifest_builder.py    # Builds MappingManifest from nsys export
│   ├── attribution_engine.py  # Confidence fallback chain; 8 edge-case handlers
│   ├── interval_tree.py       # GPU-timestamp interval tree for enclosure queries
│   ├── nsys_export.py         # nsys SQLite export helper
│   ├── ncu_runner.py          # ncu range-replay subprocess wrapper
│   ├── ncu_parser.py          # Parses ncu CSV → KernelMetrics
│   └── range_replay.py        # Orchestrates per-range ncu replays
│
├── aggregator/            # Stage 3 — metrics aggregation
│   ├── metric_aggregator.py   # Additive + duration-weighted mean strategies
│   ├── roofline.py            # Arithmetic intensity + bottleneck classification
│   └── profile_builder.py     # Assembles OperatorAttributedProfile
│
├── rewriter/              # Stage 4 — FX graph rewriting
│   ├── dsl.py             # RewritePlan DSL (fuse/reorder/change_layout/buffer_sharing)
│   ├── executor.py        # HybridExecutor — applies plan to GraphModule
│   ├── verification.py    # VerificationGate — numerical equivalence check
│   ├── provenance.py      # ProvenanceTracker — node lineage across rewrites
│   ├── lowering.py        # Lowers rewritten graph to Inductor
│   └── ops/               # Individual op implementations
│
├── planner/               # Stage 5 — LLM-backed optimization planner
│   ├── planner.py         # ThetaPlanner — Anthropic API + tool_choice JSON output
│   ├── loop.py            # OptimizationLoop — beam search outer loop
│   ├── search.py          # BeamSearch — explore/refine strategy partitioning
│   ├── memory.py          # OptimizationMemory — Jaccard + LLM-ranked retrieval
│   ├── system_prompt.py   # LLM system prompt + METRIC_RULES
│   └── schema.py          # BeamState, GraphPattern, MemoryEntry, SearchCandidate
│
├── summarizer/            # Stage 6 — reporting
│   ├── diff.py            # compute_diff — before/after ProfileDiff
│   ├── markdown.py        # render_markdown
│   ├── html.py            # render_html (self-contained)
│   ├── dashboard.py       # RichDashboard + LiveProgressDashboard
│   ├── explain.py         # explain_node — natural-language per-node explanation
│   ├── provenance.py      # ProvenanceRow — joined operator/kernel/metrics/plan view
│   └── rules.py           # entries_to_rules — OptimizationMemory → OptimizationRule
│
├── agents/                # LLM-backed agents
│   ├── diagnosis.py       # DiagnosisAgent — bottleneck classification
│   ├── verifier.py        # VerifierAgent — repair hints from VerificationResult
│   ├── rule.py            # RuleAgent — causal explanations for OptimizationRule
│   └── curator.py         # MemoryCuratorAgent — memory deduplication + compaction
│
├── schema/                # Shared Pydantic v2 data contracts
│   ├── profile.py         # OperatorAttributedProfile (top-level output schema)
│   ├── manifest.py        # MappingManifest (capture → map handoff)
│   └── metrics.py         # NCU metric policy table + aggregation ops
│
├── utils/
│   ├── subprocess_utils.py
│   ├── clock_sync.py
│   └── validation.py
│
└── cli/                   # CLI entry points
    ├── __init__.py        # main() dispatcher
    ├── profile_cmd.py
    ├── map_cmd.py
    ├── report_cmd.py
    ├── summarize_cmd.py
    └── explain_cmd.py
```

### Attribution confidence fallback chain

```
1. Inductor provenance (INDUCTOR_PROVENANCE=1)  →  confidence: HIGH
2. NVTX range enclosure (interval tree)          →  confidence: MEDIUM
3. Kernel name heuristic (Triton op patterns)   →  confidence: LOW
4. CUDA graph manifest lookup                   →  confidence: LOW
5. Unattributed                                 →  confidence: UNATTRIBUTED
```

### Rewrite DSL operations

| Op | Description |
|---|---|
| `fuse` | Fuse ≥2 FX nodes using `inline`, `custom_op`, or `inductor_fuse` strategy |
| `reorder` | Move a node before or after another node in the graph |
| `change_layout` | Insert a memory-format conversion (e.g., NCHW → NHWC) before a node |
| `buffer_sharing` | Alias two tensors' storage to eliminate a copy |

### Supported GPU roofline specs

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

## Data Schemas

All data is serialized as JSON using Pydantic v2 models.

### `OperatorAttributedProfile` (top-level output)

```jsonc
{
  "schema_version": "1.0",
  "capture_metadata": {
    "model_name": "MyModel",
    "torch_version": "2.3.0",
    "compile_mode": "inductor",
    "device_name": "A100 SXM4 80GB",
    "capture_timestamp_utc": "2026-04-04T12:00:00+00:00"
  },
  "operators": [
    {
      "operator_id": "aten::mm_0",
      "operator_name": "aten::mm",
      "call_index": 0,
      "is_fused": false,
      "kernels": [...],
      "aggregated": {
        "total_duration_ns": 14321000,
        "kernel_count": 3,
        "total_dram_bytes_read": 1073741824,
        "mean_achieved_occupancy": 0.87,
        "bottleneck_classification": "memory_bound"
      }
    }
  ],
  "unattributed_kernels": [],
  "warnings": []
}
```

### `RewritePlan`

```jsonc
{
  "plan_version": "1.0",
  "source_profile_id": "1.0/aten::mm_0",
  "description": "Fuse elementwise ops after mm to reduce memory round-trips",
  "ops": [
    {
      "op": "fuse",
      "id": "f0",
      "nodes": ["add_1", "relu_1"],
      "strategy": "inductor_fuse"
    }
  ]
}
```

---

## Testing

Unit tests run without any GPU hardware or NVIDIA tools:

```bash
# Run all unit tests
pytest

# Run with verbose output
pytest -v

# Run integration tests (requires CUDA GPU + nsys/ncu on PATH)
pytest -m integration
```

Test layout:

```
tests/
├── unit/
│   ├── test_interval_tree.py
│   ├── test_schema_roundtrip.py
│   ├── test_metric_aggregator.py
│   └── test_manifest_builder.py
├── integration/
│   ├── test_nvtx_capture.py
│   └── test_range_replay.py
└── fixtures/
    ├── sample_manifest.json
    └── sample_profile.json
```

Integration tests are excluded from the default `pytest` run via `addopts = "-m 'not integration'"` in `pyproject.toml`.

---

## Environment Variables

| Variable | Stage | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Planner | Anthropic API key for `ThetaPlanner` |
| `INDUCTOR_PROVENANCE` | Capture | Set to `1` to enable Inductor provenance sidecar |
| `INDUCTOR_COMPILE_THREADS` | Capture | Set to `1` to serialize compilation (required for deterministic provenance) |
| `INDUCTOR_PROVENANCE_OUTPUT` | Capture | Path to the provenance JSONL file |

---

## Version

`0.1.0` — see `operator_profiler/__init__.py`.
