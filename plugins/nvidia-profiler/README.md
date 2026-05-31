# Operator Profiler — Claude Code Plugin

End-to-end GPU kernel optimization for PyTorch models, driven entirely from your Claude Code session. Pass a `workload.py` file and get back a profiled, analyzed, optimized, and validated PyTorch backend with a measured speedup report.

---

## Quick Start

```
/optimize examples/conv_block/conv_block.py
```

That single command runs all 6 pipeline stages — nsys+ncu capture, FX optimization proposals, backend code generation, 5-step validation, re-profiling, and a final report — and writes all artifacts to the working directory.

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

Seven ready-to-run example workloads are in `examples/`: `conv_block`, `mlp_activations`, `sdpa_attention`, `depthwise_separable_conv`, `embedding_projection`, `gpt2`, `lstm_sequence_encoder`.

---

## Skills

### `/optimize` — End-to-End Pipeline

The primary entry point. Orchestrates all 6 stages and writes every artifact.

```
/optimize workload.py
/optimize workload.py --compile-backend=inductor --ncu-sudo=true
/optimize workload.py --resume --from=backend    # resume from a specific stage
/optimize workload.py --stage=capture            # run one stage only
/optimize workload_a.py workload_b.py            # batch multiple workloads
```

**The 7 stages:**

| Stage | Agent | Output | Skip condition |
|---|---|---|---|
| 0. Capture baseline | capture-agent | `profile.json` | `--from` set to later stage |
| 1. Propose optimizations | optimization-strategist | `optimizations.json` | `--from` set to later stage |
| 2. Generate backend | backend-engineer | `{workload}_optimized.py`, `test_*.py`, `profiler_output/implementation_notes.md` | `--from` set to later stage |
| 3. Validate backend | validation-agent | `profiler_output/validation_report.json` | — (always runs; blocks Stage 4 if fails) |
| 4. Re-capture optimized | capture-agent | `profile_optimized.json` | `--from` set to later stage |
| 4.5. Cleanup | — | *(removes `profiler_output/*_inductor_debug/`)* | — |
| 5. Report | — | `report.md` | — |

**Configuration options:**

| Option | Default | Description |
|---|---|---|
| `--from` | `capture` | Start pipeline from this stage; all earlier stages are skipped (`capture`, `propose`, `backend`, `validate`, `report`) |
| `--compile-backend` | *(none)* | Named `@register_backend` backend for optimized workloads. Required for Stage 4 when the optimized workload uses complex FX passes. |
| `--ncu-sudo` | `auto` | `auto` detects, `true` forces, `false` skips sudo for ncu |
| `--min-confidence` | *(none)* | Only include proposals at or above this confidence level (`high`, `medium`, `low`) |
| `--max-opts` | *(unlimited)* | Cap number of proposals from Stage 1 |

**Edge case handling:**

| Condition | Action |
|---|---|
| `unattributed_kernels > 10%` | Warn at Stage 1; lower confidence on all proposals |
| `compile_mode == "eager"` | Skip FX pass generation; propose `torch.compile` migration instead |
| Stage 3 fails | Block Stage 4 — do not waste ncu time on broken code |
| `ERR_NVGPUCTRPERM` | Halt and print exact fix: `--ncu-sudo=true` |

---

### `/capture` — GPU Profiling Pipeline

Runs the complete nsys + ncu pipeline on a workload and produces `profile.json` with per-operator hardware metrics.

```
/capture workload.py
/capture workload.py --ncu-sudo=true                               # force sudo for ncu
/capture workload.py --profile-name=optimized                      # → profile_optimized.json
/capture workload_optimized.py --compile-backend=my_model_opt      # profile optimized workload
```

**Internal pipeline stages:**

1. **Correlation pass** — runs `run_workload.py --correlation-pass` as a plain Python invocation (not inside nsys). nsys and torch.profiler both use CUPTI and cannot run simultaneously. Produces `.corr.json` for HIGH-confidence kernel→operator attribution, and `.part.json` (partition equivalence map) when using the built-in dedup backend.
2. **nsys capture** — runs the workload under `nsys profile --trace=cuda,nvtx` (without `--correlation-pass`). Reuses Inductor cache from step 1. Produces `.nsys-rep`.
3. **SQLite export** — `nsys export --type=sqlite` for programmatic parsing.
4. **Manifest build** — joins CUDA kernel launches to NVTX operator ranges; applies attribution tiers.
5. **ncu kernel replay** — `--replay-mode application`: replays the full workload once per counter group (4–8 passes), collecting 20 hardware counters for all kernels simultaneously. Produces `profile.json`.

**Attribution tiers (ranked by confidence):**

| Tier | Confidence | Method |
|---|---|---|
| torch.profiler correlation | `high` | External-id link in Chrome trace; written to `.corr.json` |
| NVTX enclosure | `medium` | Kernel falls within an `aten::` NVTX range from `emit_nvtx` |
| Unattributed | — | Stored in `unattributed_kernels[]`; never silently dropped |

Expected unattributed rate: near 0% when both `.corr.json` (from the standalone correlation pass) and the Inductor fusion map are active. If the correlation pass was accidentally run inside nsys (CUPTI conflict), `.corr.json` will have 0 entries and unattributed rates rise to 20–40%.

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

### `/propose` — Optimization Proposals

Reads `profile.json` directly, derives time budget, edge case flags, and architecture context, then writes `optimizations.json` with ranked, evidence-backed FX transformation proposals.

```
/propose profile.json
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
| `pretranspose_weights` (weight layout) | Inductor config (`freezing`) | high |
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
- `profiler_output/implementation_notes.md` — backend architecture and design rationale (ingested by `/report`)

**Generated backend structure:**

```python
# Non-graph optimizations (dtype, layout, padding) applied in get_model_and_input()
def get_model_and_input():
    model, x = _get_baseline_model_and_input()   # imports baseline
    if next(model.parameters()).dtype != torch.bfloat16:
        model = model.to(torch.bfloat16)
        x = x.to(torch.bfloat16)
    return model, x

# LEVEL 1 — functional passes on the Dynamo graph, before compile_fx/AOTAutograd.
# Fusion and SDPA run here: the shared activation is one node; decomposition shatters it.
def _run_functional_passes(gm):
    gm = _fpass_fuse_qkv(gm)        # QKV fusion at functional level
    gm = _fpass_replace_sdpa(gm)    # SDPA formation at functional level
    return gm

# LEVEL 2 — aten passes inside _aten_inner_compile, post-AOTAutograd decomposition.
# example_inputs may be FakeTensors; use real_inputs for weight-value-reading passes.
def _aten_inner_compile(gm, example_inputs, *, real_inputs=None, **kwargs):
    weight_source = real_inputs if real_inputs is not None else example_inputs
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    ph_to_tensor = {ph: t for ph, t in zip(placeholders, weight_source)}
    gm = _apass_bf16_promotion(gm)          # dtype cast — aten level
    gm = _apass_fold_bn(gm, ph_to_tensor)  # BN fold — aten level
    return compile_fx_inner(gm, example_inputs, **kwargs)

@register_backend
def my_backend(gm, example_inputs):
    gm = _run_functional_passes(gm)
    inner = functools.partial(_aten_inner_compile, real_inputs=list(example_inputs))
    # LEVEL 3 — inductor_config: scoped config_patches, no graph surgery
    return compile_fx(gm, example_inputs, inner_compile=inner,
                      config_patches={"freezing": True})
```

Passes are routed across three IR levels. **`functional`** passes (fusion, SDPA formation) run before `compile_fx` on the Dynamo graph, where the shared activation is still a single node — decomposition shatters it into per-consumer `aten.view`s, so fusion must happen here. **`aten`** passes (dtype casts, BN fold, channels-last) run inside `_aten_inner_compile` targeting decomposed `torch.ops.aten.*` nodes. **`inductor_config`** passes are scoped `config_patches` dicts on `compile_fx` (e.g. `{"freezing": True}` for constant-weight layout). Each pass degrades gracefully — logs `WARNING` if the pattern is not found and returns the graph unchanged.

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

### `/report` — Human-Readable Summary

Generates `report.md` summarizing the complete optimization lifecycle.

```
/report
```

Covers: hardware context, bottleneck classification with evidence, transformations applied and which patterns matched, measured speedup with kernel count before/after, reproduction commands, and known caveats (ncu replay timing, unattributed kernels).

---

## Agents

The plugin ships 4 specialized agents. The `/optimize` skill orchestrates them in sequence; individual skills invoke their corresponding agent directly.

### `capture-agent`

Orchestrates the full nsys + ncu profiling pipeline (Stages 0a-pre through 0d). Handles executable auto-detection, sudo permission detection, PYTHONPATH propagation under `sudo env KEY=VAL`, and `--script-args` ordering enforcement.

**Tools:** `Bash`, `Read`, `Glob`, `Skill`

### `optimization-strategist`

Reads `profile.json` directly. Derives time budget, edge case flags, and architecture context, then reasons open-endedly from raw hardware counters to produce `optimizations.json` with ranked proposals, exact metric citations, `fx_steps[]`, and dependency ordering. Fetches live PyTorch API docs via `context7` and uses `sequential-thinking` for multi-operator dependency analysis.

**Tools:** `Read`, `Write`, `mcp__sequential_thinking`, `mcp__context7`, `mcp__exa__search`, `mcp__memory`

### `backend-engineer`

Generates `{workload}_optimized.py` from `optimizations.json`. Implements each FX graph pass defensively (pattern detection before mutation, `gm.graph.lint()` after each mutation, graceful fallback on pattern miss). Also generates the test script and `profiler_output/implementation_notes.md`.

**Tools:** `Read`, `Write`, `Edit`, `Bash`, `mcp__context7`

### `validation-agent`

Runs the fixed 5-step validation sequence in order, interprets logger `INFO`/`WARNING` output to classify pass application, and reports `READY FOR PROFILING` or `BLOCKED — Fix step N`. Does not suggest improvements — only reports pass/fail.

**Tools:** `Bash`, `Read`

---

## Hooks

The plugin registers four automatic hooks that fire during your Claude Code session.

### Session start announcement

**Trigger:** Session opens  
**Action:** Prints available commands to the terminal:
```
[profiler-plugin] Commands: /preflight /capture /propose /backend /validate /report /optimize
```

### Session start preflight

**Trigger:** Session opens  
**Action:** Runs `nvidia/scripts/preflight.py` automatically and prints the result. Validates packages, CUDA, nsys, and ncu before any work begins. If any check fails, the output tells you what to fix before capturing.

### Post-write: profile.json → suggest `/propose`

**Trigger:** A `Write` tool call produces a file named `profile.json`  
**Action:** Prompts the next step — running `/propose profile.json`

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

The plugin bundles a reference knowledge base used by the optimization-strategist and backend-engineer agents during reasoning.

### `knowledge/hardware-limits.md`

GPU database covering 11 architectures (A100, H100, H200, B100, B200, RTX 4090, A10G, and more). Per-GPU data includes: SM count, peak TFLOPS for BF16/FP16/FP32, HBM bandwidth, ridge point (FLOP/byte), and Tensor Core tile requirements. Used by `/propose` for Waves/SM calculation and bottleneck calibration.

### `knowledge/fx-patterns.md`

Canonical FX graph pass implementations organized by IR level. **`functional`-level** patterns (QKV fusion, SDPA formation) run before `compile_fx` on the Dynamo graph where the shared activation is still one node. **`aten`-level** patterns (dtype casts, BN fold, channels-last annotation, activation substitution) run inside `_aten_inner_compile` targeting `torch.ops.aten.*` nodes. **`inductor_config`** patterns are scoped `config_patches` dicts (e.g. `{"freezing": True}`). Includes the Aten IR op mapping table, `real_inputs` threading for weight-value-reading passes (because `inner_compile`'s `example_inputs` may be FakeTensors), and the standard three-stage `_compile_unit` funnel. The `backend-engineer` agent uses this as the authoritative implementation reference.

### `knowledge/edge-cases.md`

The 11 attribution and metric edge cases that affect how `profile.json` data should be interpreted:

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
| torch.compile graph breaks | Eager-mode kernels appear unattributed; FX passes have no effect on broken subgraphs |
| Host-device synchronization stalls | `.item()`/`.cpu()` calls hide GPU idle time from `profile.json` |
| SM occupancy misreporting | High shared-memory kernels show low occupancy as a hardware constraint, not wave starvation |

### `knowledge/schema-versions.md`

`profile.json` schema versioning and backward compatibility notes. Documents the `schema_version` field, deprecated pre-v1.0 fields, and migration guidance.

### `knowledge/capture-errors.md`

Error lookup table for the nsys+ncu pipeline. Maps stderr patterns to their cause and exact remediation command. Referenced by the `capture-agent` when any pipeline stage produces unexpected output or exits non-zero.

### `knowledge/validation-failures.md`

Diagnosis guide for non-obvious validation failures in the 5-step validation sequence. Covers `TypeError: 'module' object is not callable`, dynamo cache hits masking backend execution, NaN outputs from BF16 overflow, and common FX graph mutation errors. Referenced by the `validation-agent`.

### `rules/gpu-optimization.md`

Bottleneck classification rules, fix strategies, and architecture-specific notes used as a reasoning reference by the optimization-strategist agent.

---

## Output Artifacts

| File | Produced by | Contents |
|---|---|---|
| `profiler_output/{stem}.corr.json` | `/capture` | HIGH-confidence kernel→operator attribution map (from torch.profiler correlation pass) |
| `profiler_output/{stem}.part.json` | `/capture` | Layer partition equivalence map for deduplication (built-in backend only) |
| `profile.json` | `/capture` | Per-operator hardware metrics (20 counters per kernel) |
| `optimizations.json` | `/propose` | Ranked FX transformation proposals with evidence, time budget, edge case flags, and `fx_steps[]` |
| `{workload}_optimized.py` | `/backend` | Custom `torch.compile()` backend + `get_model_and_input()` wrapper |
| `test_{workload}_optimized.py` | `/backend` | Pytest suite for the generated backend |
| `profiler_output/implementation_notes.md` | `/backend` | Backend architecture and design rationale (ingested by `/report`) |
| `profiler_output/validation_report.json` | `/validate` | 5-step pass/fail results + FX pass application summary |
| `profile_optimized.json` | `/capture --profile-name=optimized` | Hardware metrics for the optimized backend |
| `report.md` | `/report` | Human-readable optimization lifecycle summary including per-operator speedup table |

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
