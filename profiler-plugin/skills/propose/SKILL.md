---
name: propose
description: Generate a structured optimizations.json from profile.json or triage.json. Produces ranked, evidence-backed FX graph transformation proposals with confidence ratings, exact metric citations, actionable fx_steps[], and prerequisite ordering. Uses the existing optimization_proposal_prompt.md as its core reasoning template.
---

# /propose — Optimization Proposal Generation

Analyzes the profiling data and generates `optimizations.json` with ranked, concrete optimization proposals. Each proposal maps directly to an FX graph pass or a non-graph transformation in `get_model_and_input()`.

## Usage

```
/propose profile.json                    # from profile directly
/propose triage.json                     # from /analyze output
/propose profile.json --max-opts=5       # limit number of proposals
/propose profile.json --min-confidence=high  # only high-confidence proposals
```

## What It Produces

`optimizations.json` following Schema B — the structured format with evidence, `fx_steps[]`, and dependency ordering. This is consumed by `/backend` to generate the actual FX passes.

## Core Reasoning Template

The optimization-strategist agent follows `prompts/optimization_proposal_prompt.md` as its primary reasoning guide. This skill adds:
1. Schema B output enforcement (vs. the simpler table format in the prompt)
2. `prerequisite_for[]` dependency DAG construction
3. Architecture-specific calibration from `knowledge/hardware-limits.md`

## Transformation Types

Each optimization proposal has a `transformation.type` from this taxonomy:

| Type | Description | FX or Non-Graph | Confidence |
|---|---|---|---|
| `dtype_promotion` | Cast model/inputs to BF16/FP16 | Non-graph (`get_model_and_input`) | high |
| `memory_layout` | channels_last memory format | Non-graph | high |
| `qkv_fusion` | Fuse 3 projection matrices into 1 GEMM | FX pass | high/medium |
| `sdpa_replacement` | Replace manual attention with FlashAttention | FX pass | medium |
| `bn_fold` | Fold BatchNorm into Conv2d weights | FX pass | high |
| `pretranspose_weights` | Pre-store transposed weight buffers | FX pass | high |
| `activation_substitution` | Replace tanh → gelu(approximate='tanh') | FX pass | medium |
| `batch_padding` | Pad batch dim to warp tile multiple | Non-graph | medium |
| `algorithm_selection` | torch.compile mode or cudnn.benchmark | Config change | high |
| `stub_detection` | Detection only — requires custom Triton | FX stub | low |

## Confidence Calibration

| Confidence | Meaning | Backend behavior |
|---|---|---|
| `high` | Single well-understood counter directly indicates the fix; gain is theoretically guaranteed | Full FX pass implementation |
| `medium` | Pattern matching may not generalize across all Inductor-traced graphs; gain depends on graph structure | FX pass with defensive fallback |
| `low` | Requires custom Triton kernel or depends on shape assumptions not in the profile | Detection stub with warning |

Every `low` confidence optimization MUST include a `notes` field explaining what infrastructure is needed.

## Dependency Ordering Rules

The `prerequisite_for[]` array encodes which optimizations must run first:

1. **Dtype promotion first**: All dtype-sensitive passes (QKV fusion, SDPA, pre-transpose) should run after dtype promotion because the fused weight tensors must match the runtime dtype
2. **Layout before algorithm**: Apply `channels_last` before `cudnn.benchmark` so cuDNN benchmarks the optimal-layout kernels
3. **High confidence before medium**: In the backend registration, high-confidence passes run first; they cannot be invalidated by medium-confidence passes

If you detect a cycle in the dependency graph: remove the lower-confidence optimization from the cycle and add a note.

## Output Schema (Schema B)

```json
{
  "analysis": {
    "model": "ModelName",
    "device": "GPU model string",
    "compile_mode": "inductor",
    "dtype": "FP32",
    "total_profiled_wall_time_ms": 2.12,
    "time_budget": {
      "aten::operator_name": {
        "pct": 42.3,
        "duration_ns": 5234567,
        "kernel_count": 30
      }
    }
  },
  "optimizations": [
    {
      "id": "OPT-1",
      "priority": 1,
      "operators": ["aten::linear (call_index 0, 1, 2)"],
      "bottleneck": {
        "description": "Exact description of the hardware bottleneck",
        "evidence": {
          "metric_name": "exact_value_from_profile",
          "kernel": "dominant_kernel_name",
          "total_launches": 30,
          "total_duration_ns": 5234567
        }
      },
      "transformation": {
        "type": "dtype_promotion",
        "description": "What the transformation does",
        "location": "get_model_and_input() | FX pass",
        "fx_steps": [
          "Step 1: actionable Python statement or pseudocode",
          "Step 2: ...",
          "Step 3: gm.graph.lint(); gm.recompile()"
        ],
        "code_hint": "Optional: exact code snippet"
      },
      "estimated_impact": {
        "latency_reduction_ns": 222176,
        "latency_reduction_pct_of_total": 10.5,
        "kernel_launches_eliminated": 60,
        "memory_traffic_reduction": "~2x"
      },
      "confidence": "high",
      "prerequisite_for": ["OPT-3"],
      "notes": "Only for high/medium; for low: 'Requires custom Triton kernel X'"
    }
  ],
  "global_notes": [
    "All duration values from ncu replay — use for relative comparison only",
    "Architecture-specific notes"
  ]
}
```

## Multi-Profile Batch Mode

When called with multiple profiles (e.g. baseline and a variant):
```
/propose profile_v1.json profile_v2.json
```

Merges time budgets and flags operators that changed bottleneck class between profiles — these are either regressions from a prior optimization attempt or improvements that warrant deeper analysis.

## Common Mistakes to Avoid

- Do NOT cite generic advice ("use BF16") without mapping to specific operators and exact metric values from the profile
- Do NOT propose SDPA replacement if the attention pattern is not visible in the FX graph (compile_mode == "eager" — no FX passes run)
- Do NOT propose both QKV fusion and pre-transposed weights for the same mm nodes (they are mutually exclusive — QKV fusion replaces the individual weight nodes)
- Do NOT include `warp_cycles_per_instruction` evidence on Blackwell (counter was removed; use `eligible_cycles_pct` instead)
- All `fx_steps[]` entries must be actionable Python-level instructions, not prose descriptions
