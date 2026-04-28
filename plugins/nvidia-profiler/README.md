# Operator Profiler — Claude Code Plugin

End-to-end GPU kernel optimization for PyTorch models, driven entirely from your Claude Code session. Pass a `workload.py` file and get back a profiled, analyzed, optimized, and validated PyTorch backend with a measured speedup report.

---

## Quick Start

```
/optimize examples/transformer_block/transformer_block.py
```

That single command runs all 8 pipeline stages — nsys+ncu capture, bottleneck analysis, FX optimization proposals, backend code generation, 5-step validation, re-profiling, comparison, and a final report — and writes all artifacts to the working directory.

---

## Installation

### Prerequisites

| Requirement | Notes |
|---|---|
| [Claude Code](https://claude.ai/code) | Any recent version |
| Python ≥ 3.10 | |
| PyTorch | Install separately with correct CUDA variant from [pytorch.org](https://pytorch.org) |
| NVIDIA Nsight Systems (`nsys`) | For CUDA kernel capture |
| NVIDIA Nsight Compute (`ncu`) | For hardware counter collection; **requires sudo on most Linux systems** |
| CUDA GPU | Ampere, Hopper, or Blackwell (A100, H100, H200, RTX 4090, B100, B200, etc.) |

### Add the marketplace and install

In any Claude Code session:

```
/plugin marketplace add yottalabsai/Profiler
/plugin install nvidia-profiler@profiler-plugins
```

This registers the plugin globally. No need to clone the repository.

---

## Workload Interface

Every workload script must expose a single function:

```python
def get_model_and_input() -> tuple[torch.nn.Module, torch.Tensor]:
    """Return (uncompiled model on CUDA, input tensor on CUDA)."""
    model = YourModel().to("cuda").eval()
    x = torch.randn(batch, seq, dim, device="cuda")
    return model, x
```

Rules:
- Model must be on CUDA, in `.eval()` mode, and **uncompiled** — the pipeline handles `torch.compile` internally
- Input tensor must be on CUDA
- The pipeline does not read any other function from the file

Six ready-to-run example workloads are in `examples/`: `transformer_block`, `conv_block`, `mlp_activations`, `sdpa_attention`, `depthwise_separable_conv`, `embedding_projection`.

---

## Skills

### `/optimize` — End-to-End Pipeline

The primary entry point. Orchestrates all 8 stages and writes every artifact.

```
/optimize workload.py
/optimize workload.py --compile-backend=inductor --ncu-sudo=true
/optimize workload.py --resume --from=backend    # resume from a specific stage
/optimize workload.py --stage=capture            # run one stage only
/optimize workload_a.py workload_b.py            # batch multiple workloads
```

**The 8 stages:**

| Stage | Agent | Output | Skip condition |
|---|---|---|---|
| 0. Capture baseline | capture-agent | `profile.json` | exists + `--resume` |
| 1. Analyze bottlenecks | profile-analyzer | `triage.json` | exists + `--resume` |
| 2. Propose optimizations | optimization-strategist | `optimizations.json` | exists + `--resume` |
| 3. Generate backend | backend-engineer | `{workload}_optimized.py`, `test_*.py`, `OPTIMIZED_WORKLOAD.md` | exists + `--resume` |
| 4. Validate backend | validation-agent | `validation_report.json` | — (always runs; blocks Stage 5 if fails) |
| 5. Re-capture optimized | capture-agent | `profile_optimized.json` | exists + `--resume` |
| 6. Compare results | comparison-agent | `comparison.md` | — |
| 7. Report | — | `report.md` | — |

**Configuration options:**

| Option | Default | Description |
|---|---|---|
| `--compile-backend` | `inductor` | `inductor`, `none` (eager), `cudagraphs`, or any registered backend |
| `--warmup-iters` | `5` | Must match between Stage 0 and Stage 5 |
| `--measure-iters` | `10` | Must match between Stage 0 and Stage 5 |
| `--ncu-sudo` | `auto` | `auto` detects, `true` forces, `false` skips |
| `--ncu-path` / `--nsys-path` | `auto` | Explicit path to profiler executables |
| `--confidence-threshold` | `medium` | Only implement optimizations at or above this level |
| `--max-optimizations` | `10` | Cap number of proposals from Stage 2 |
| `--resume` | `false` | Skip stages whose output artifact already exists |
| `--from` | `capture` | Stage to resume from (with `--resume`) |
| `--audience` | `team` | Report audience: `team` or `executive` |

**Edge case handling:**

| Condition | Action |
|---|---|
| `unattributed_kernels > 10%` | Warn at Stage 1; lower confidence on all Stage 2 proposals |
| `compile_mode == "eager"` | Skip FX pass generation; propose `torch.compile` migration instead |
| Stage 4 fails | Block Stage 5 — do not waste ncu time on broken code |
| `ERR_NVGPUCTRPERM` | Halt and print exact fix: `--ncu-sudo=true` |

---

### `/capture` — GPU Profiling Pipeline

Runs the complete nsys + ncu pipeline on a workload and produces `profile.json` with per-operator hardware metrics.

```
/capture workload.py
/capture workload.py --compile-backend=none          # eager mode
/capture workload.py --ncu-sudo=true                 # force sudo for ncu
/capture workload.py --profile-name=optimized        # → profile_optimized.json
/capture workload.py --warmup-iters=10 --measure-iters=20
```

**Internal pipeline stages:**

1. **Correlation pass** — runs `torch.profiler` as a plain Python invocation (not inside nsys — both use CUPTI and cannot run simultaneously). Produces a `.corr.json` sidecar for HIGH-confidence kernel→operator attribution.
2. **nsys capture** — runs the workload under `nsys profile --trace=cuda,nvtx`. Produces `.nsys-rep`.
3. **SQLite export** — `nsys export --type=sqlite` for programmatic parsing.
4. **Manifest build** — joins CUDA kernel launches to NVTX operator ranges; applies attribution tiers.
5. **ncu kernel replay** — `--replay-mode application`: replays the full workload once per counter group (4–8 passes), collecting 20 hardware counters for all kernels simultaneously. Produces `profile.json`.

**Attribution tiers (ranked by confidence):**

| Tier | Confidence | Method |
|---|---|---|
| torch.profiler correlation | `high` | CUPTI correlation IDs via `--correlation-pass` |
| NVTX enclosure | `medium` | Kernel falls within an `aten::` NVTX range from `emit_nvtx` |
| Unattributed | — | Stored in `unattributed_kernels[]`; never silently dropped |

Expected unattributed rate: 20–40% for Inductor-compiled models without `--correlation-pass`. Always use the correlation pass to minimize this.

**System auto-detection:**

The capture agent detects your system configuration automatically — nsys/ncu executables (with fallback path scans), sudo requirement (reads `/proc/driver/nvidia/params`), and PYTHONPATH.

**Warmup/measure matching rule:**

`--warmup-iters` and `--measure-iters` in the nsys capture **must** be identical to those passed during ncu replay. Mismatching causes invocation count divergence and metric mismatch.

**Permission errors:**

| Error | Solution |
|---|---|
| `ERR_NVGPUCTRPERM` on Linux | Run with `--ncu-sudo=true` or as root |
| `ERR_NVGPUCTRPERM` on Windows | Restart terminal as Administrator; enable DevMode GPU profiling |

---

### `/analyze` — Bottleneck Triage

Classifies every operator by its primary performance bottleneck and outputs `triage.json`.

```
/analyze profile.json
/analyze profile.json --verbose    # include raw metric values in output
```

**Bottleneck classification** (applied in priority order, first match wins):

| Class | Condition | Meaning | Fix |
|---|---|---|---|
| `tensor_core_idle` | `tensor_core_active_pct == 0.0` on a GEMM op | Tensor Cores idle — FP32 SIMT path | Cast to BF16/FP16 (2–16× gain) |
| `compute_bound` | `sm_throughput > 70%` and `memory_throughput < 40%` | SM compute saturated | Tile size, `max-autotune`, algorithm selection |
| `memory_bound` | `dram_throughput > 60%` and `sm_throughput < 30%` | HBM bandwidth is bottleneck | Operator fusion, layout optimization |
| `wave_starvation` | `achieved_occupancy < 20%` | Not enough work to keep all SMs busy | Increase batch size, batch padding, QKV fusion |
| `latency_bound` | `warp_cycles_per_instruction > 20` | Warps stalling on memory latency | Increase occupancy, reduce register pressure |
| `register_pressure` | `registers_per_thread > 128` or `local_memory_spills > 0` | Register file pressure limits occupancy | `reduce-overhead` mode, kernel rewrite |
| `layout_overhead` | `convertTensor_kernel` present | cuDNN coercing NCHW↔NHWC per call | `model.to(memory_format=torch.channels_last)` |

> **Note on `tensor_core_active_pct == null`:** This is NOT a bottleneck. It means the counter is unavailable for this kernel type (elementwise ops) or was removed on Blackwell. Only `== 0.0` on a GEMM kernel is a problem.

**Architecture notes:**
- **Blackwell:** `warp_cycles_per_instruction` was removed — use `eligible_cycles_pct < 20` as the latency-bound indicator instead
- **Hopper:** `sm90_xmma_gemm_bf16bf16_*` in kernel name = Tensor Cores active; `gemmSN_TN_*` = FP32 SIMT
- **Ampere:** `allow_tf32 = True` enables Tensor Cores at lower precision cost than full BF16

---

### `/propose` — Optimization Proposals

Generates ranked, evidence-backed FX transformation proposals and writes `optimizations.json`.

```
/propose profile.json
/propose triage.json                        # from /analyze output
/propose profile.json --max-opts=5
/propose profile.json --min-confidence=high
```

Each proposal includes: the specific operators affected, exact metric evidence from the profile, concrete `fx_steps[]` (actionable Python-level instructions), estimated impact, confidence level, and dependency ordering.

**Transformation types:**

| Type | Where | Confidence |
|---|---|---|
| `dtype_promotion` (BF16/FP16 cast) | `get_model_and_input()` | high |
| `memory_layout` (channels_last) | `get_model_and_input()` | high |
| `qkv_fusion` (3 GEMM → 1) | FX pass | high/medium |
| `sdpa_replacement` (FlashAttention) | FX pass | medium |
| `bn_fold` (BatchNorm into Conv2d) | FX pass | high |
| `pretranspose_weights` (weight layout) | FX pass | high |
| `activation_substitution` (tanh → gelu) | FX pass | medium |
| `batch_padding` (pad to warp tile) | `get_model_and_input()` | medium |
| `algorithm_selection` | Config change | high |
| `stub_detection` | FX stub | low |

**Confidence levels:**

| Level | Meaning | Backend behavior |
|---|---|---|
| `high` | Single counter directly indicates fix; gain theoretically guaranteed | Full FX pass implementation |
| `medium` | Pattern matching may not generalize; gain depends on graph structure | FX pass with defensive fallback |
| `low` | Requires custom Triton kernel or shape assumptions not in profile | Detection stub with warning |

The agent uses `context7` to fetch live PyTorch FX API docs and `sequential-thinking` for multi-operator dependency analysis when more than 5 operators exceed the 5% time-budget threshold.

---

### `/backend` — FX Backend Generation

Generates a production-ready `workload_optimized.py` implementing the proposed optimizations as a custom `torch.compile()` backend.

```
/backend workload.py optimizations.json
```

**Outputs:**
- `{workload}_optimized.py` — custom backend with FX graph passes; imports baseline `get_model_and_input()` and wraps it
- `test_{workload}_optimized.py` — pytest suite with 4 required tests
- `OPTIMIZED_WORKLOAD.md` — per-optimization documentation with before/after kernel analysis

**Generated backend structure:**

```python
# Non-graph optimizations (dtype, layout, padding) applied in get_model_and_input()
def get_model_and_input():
    model, x = _get_baseline_model_and_input()   # imports baseline
    if next(model.parameters()).dtype != torch.bfloat16:
        model = model.to(torch.bfloat16)
        x = x.to(torch.bfloat16)
    return model, x

# FX graph passes registered as a torch.compile() backend
@register_backend
def my_backend(gm, example_inputs):
    gm = pass_fuse_qkv(gm)
    gm = pass_replace_sdpa(gm)
    gm = pass_pretranspose_weights(gm)
    return compile_fx(gm, example_inputs)
```

Each FX pass is implemented defensively: it inspects the graph for the expected pattern before mutating, logs `INFO` on success or `WARNING` if the pattern is not found (graceful degradation).

---

### `/validate` — Backend Validation

Runs a fixed 5-step validation sequence and reports which FX passes applied vs. degraded gracefully.

```
/validate workload_optimized.py
/validate workload_optimized.py --backend-name=my_backend
```

**Validation steps (always run in this order):**

| Step | Command | Pass condition |
|---|---|---|
| 1. Syntax | `python -m py_compile {file}` | Exit code 0 |
| 2. Import | `python -c "import {module}"` | No traceback |
| 3. Registration | `assert '{backend}' in torch._dynamo.list_backends()` | Backend found |
| 4. Test suite | `pytest {test_file} -v --tb=short` | All 4 tests pass |
| 5. Smoke test | `run_workload.py --compile-backend {backend} --warmup-iters 1 --measure-iters 1` | Exit code 0 |

After Step 5, the agent parses logger output to classify each FX pass:

```
INFO  [pass_fuse_qkv]        Applied: 3 mm nodes → 1    → APPLIED
INFO  [pass_pretranspose]     Applied                     → APPLIED
WARNING [pass_replace_sdpa]  Pattern not found           → NOT_APPLIED (graceful)
WARNING [pass_bn_fold]        Failed: AttributeError     → FAILED
```

If any step fails, the agent reports `BLOCKED — Fix step N before profiling` and does not proceed.

---

### `/compare` — Performance Comparison

Compares baseline and optimized profiles, attributes speedups to specific transformations, and identifies residual opportunities.

```
/compare profile.json profile_optimized.json
```

Normalizes for batch-size differences before computing speedups. Matches operators across profiles by name. Attributes measured gains to specific hardware counter changes (e.g., "tensor_core_active_pct: 0% → 87% — confirms BF16 cast took effect").

---

### `/report` — Human-Readable Summary

Generates `report.md` summarizing the complete optimization lifecycle.

```
/report
/report --audience=executive    # executive summary without raw metrics
/report --output=my_report.md
```

Covers: hardware context, bottleneck classification with evidence, transformations applied and which patterns matched, measured speedup with kernel count before/after, reproduction commands, and known caveats (ncu replay timing, unattributed kernels).

---

## Agents

The plugin ships 6 specialized agents. The `/optimize` skill orchestrates them in sequence; individual skills invoke their corresponding agent directly.

### `capture-agent`

Orchestrates the full nsys + ncu profiling pipeline (Stages 0a-pre through 0d). Handles executable auto-detection, sudo permission detection, PYTHONPATH propagation under `sudo env KEY=VAL`, and `--script-args` ordering enforcement.

**Tools:** `Bash`, `Read`, `Glob`

### `profile-analyzer`

Parses `profile.json`, applies the bottleneck decision tree, computes per-operator wall-time percentages, looks up GPU hardware limits (SM count, ridge point, peak bandwidth), and flags the 8 attribution edge cases.

**Tools:** `Read`, `Bash`

### `optimization-strategist`

Maps bottleneck classifications to concrete FX graph transformations. Produces `optimizations.json` (Schema B) with ranked proposals, exact metric citations, `fx_steps[]`, dependency ordering, and confidence ratings. Fetches live PyTorch API docs via `context7` and uses `sequential-thinking` for multi-operator dependency analysis.

**Tools:** `Read`, `mcp__sequential_thinking`, `mcp__context7`, `mcp__exa__search`, `mcp__memory`

### `backend-engineer`

Generates `{workload}_optimized.py` from `optimizations.json`. Implements each FX graph pass defensively (pattern detection before mutation, `gm.graph.lint()` after each mutation, graceful fallback on pattern miss). Also generates the test script and `OPTIMIZED_WORKLOAD.md`.

**Tools:** `Read`, `Write`, `Edit`, `Bash`, `mcp__context7`

### `validation-agent`

Runs the fixed 5-step validation sequence in order, interprets logger `INFO`/`WARNING` output to classify pass application, and reports `READY FOR PROFILING` or `BLOCKED — Fix step N`. Does not suggest improvements — only reports pass/fail.

**Tools:** `Bash`, `Read`

### `comparison-agent`

Normalizes baseline and optimized profiles for batch-size differences. Matches operators by name across profiles. Attributes speedups to specific transformations by cross-referencing hardware counter changes with the transformations listed in `optimizations.json`. Identifies residual opportunities.

**Tools:** `Read`, `mcp__memory`

---

## Hooks

The plugin registers four automatic hooks that fire during your Claude Code session.

### Session start announcement

**Trigger:** Session opens  
**Action:** Prints available commands to the terminal:
```
[profiler-plugin] Commands: /capture /analyze /propose /backend /validate /compare /report /optimize
```

### Post-write: profile.json → suggest `/analyze`

**Trigger:** A `Write` tool call produces a file named `profile.json`  
**Action:** Prompts the next step — running `/analyze profile.json`

This fires when `/capture` completes, keeping you in the flow without needing to remember what comes next.

### Post-write: optimizations.json → suggest `/backend`

**Trigger:** A `Write` tool call produces `optimizations.json`  
**Action:** Prompts running `/backend workload.py optimizations.json`

### Post-write: `*_optimized.py` → suggest `/validate`

**Trigger:** A `Write` tool call produces a file matching `*_optimized.py`  
**Action:** Prompts running `/validate {file}`

### Pre-Bash: `--script-args` order check

**Trigger:** Any `Bash` tool call containing `operator-profiler map`  
**Action:** Warns if `--script-args` appears before other `map` flags. `--script-args` must be the last flag — anything after it is passed to the workload script, not to `operator-profiler map`.

```bash
# CORRECT — all map flags before --script-args
operator-profiler map manifest.json \
    --ncu-sudo true --ncu-env PYTHONPATH=/repo \
    --script-args --workload workload.py --compile-backend inductor

# WRONG — --ncu-env after --script-args is silently ignored as a map flag
operator-profiler map manifest.json \
    --script-args --workload workload.py \
    --ncu-env PYTHONPATH=/repo
```

---

## Knowledge Base

The plugin bundles a reference knowledge base used by the analyzer, strategist, and backend agents during reasoning.

### `knowledge/hardware-limits.md`

GPU database covering 12 architectures (A100, H100, H200, B100, B200, RTX 4090, A10G, and more). Per-GPU data includes: SM count, peak TFLOPS for BF16/FP16/FP32, HBM bandwidth, ridge point (FLOP/byte), and Tensor Core tile requirements. Used by `/analyze` for Waves/SM calculation and bottleneck calibration.

### `knowledge/fx-patterns.md`

Canonical FX graph pass implementations for every supported transformation type: QKV fusion, SDPA replacement, BatchNorm fold, pre-transposed weights, activation substitution, and more. Includes weight node detection patterns for Inductor-traced graphs and graph mutation discipline (lint, recompile). The `backend-engineer` agent references this for correct pass structure.

### `knowledge/edge-cases.md`

The 8 attribution and metric edge cases that affect how `profile.json` data should be interpreted:

| Edge case | Summary |
|---|---|
| Clock domain mismatch | `start_ns` may be implausible; use `duration_ns` only |
| CUDA graph replay | Kernel counts are replay counts; divide by `measure_iters` |
| Multi-stream overlap | `total_duration_ns` overestimates wall time when streams overlap |
| JIT warm-up inflation | Compilation kernels inflate `kernel_count`; ManifestBuilder excludes outliers |
| Async kernel launch | GPU-side `duration_ns` is authoritative; ignore host-side timing |
| Dynamic shapes | Wide variance across same operator's `call_index`; avoid batch padding |
| Fused kernel multi-NVTX | `is_fused` kernels are shared; `aggregated` fields already account for this |
| ncu replay timing | All `duration_ns` values are 2–5× real execution time — use for relative comparison only |

### `knowledge/schema-versions.md`

`profile.json` schema versioning and backward compatibility notes. Documents the `schema_version` field, deprecated pre-v1.0 fields, and migration guidance.

### `rules/gpu-optimization.md`

Bottleneck classification rules, fix strategies, and architecture-specific notes used as a reasoning reference by the profile-analyzer and optimization-strategist agents.

---

## Output Artifacts

| File | Produced by | Contents |
|---|---|---|
| `profile.json` | `/capture` | Per-operator hardware metrics (20 counters per kernel) |
| `triage.json` | `/analyze` | Bottleneck classifications, time budget, edge case flags |
| `optimizations.json` | `/propose` | Ranked FX transformation proposals with evidence and `fx_steps[]` |
| `{workload}_optimized.py` | `/backend` | Custom `torch.compile()` backend + `get_model_and_input()` wrapper |
| `test_{workload}_optimized.py` | `/backend` | Pytest suite for the generated backend |
| `OPTIMIZED_WORKLOAD.md` | `/backend` | Per-optimization before/after kernel analysis |
| `validation_report.json` | `/validate` | 5-step pass/fail results + FX pass application summary |
| `profile_optimized.json` | `/capture --profile-name=optimized` | Hardware metrics for the optimized backend |
| `comparison.md` | `/compare` | Per-operator speedup table with hardware counter evidence |
| `report.md` | `/report` | Human-readable optimization lifecycle summary |

---

## Troubleshooting

**`ERR_NVGPUCTRPERM` during capture**

ncu requires elevated access to GPU performance counters. On Linux:
```bash
# Option 1: pass --ncu-sudo=true to /capture or /optimize
/capture workload.py --ncu-sudo=true

# Option 2: temporarily allow all users
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia.conf
sudo update-initramfs -u && sudo reboot
```

**High unattributed kernel rate (> 40%)**

The correlation pass provides HIGH-confidence attribution for most kernels. If you skipped it:
```
/capture workload.py    # runs correlation pass automatically
```
If you're seeing high unattributed rates even with the correlation pass, the model may be using `cudagraphs` mode — switch to `inductor` for profiling.

**Validation Step 2 fails: `ModuleNotFoundError`**

The generated backend imports the profiler library. Make sure the repo is on `PYTHONPATH`:
```bash
export PYTHONPATH=/path/to/Profiler:$PYTHONPATH
```

**Validation Step 4 fails: shape mismatch in `test_get_model_and_input`**

The optimized `get_model_and_input()` may be returning a different output shape if batch padding was applied. The test checks against the baseline shape — update the expected shape in the test to account for the padded batch dimension.

**`operator-profiler map` flags silently ignored**

This is the `--script-args` ordering problem. All `map` flags must come before `--script-args`. The pre-Bash hook will warn you when it detects this pattern.

**ncu replay produces invocation count mismatch**

`--warmup-iters` and `--measure-iters` must be identical between the nsys capture pass and the ncu replay pass. If you re-ran `/capture` with different iteration counts after generating a manifest, re-run the full `/capture` pipeline from scratch.
