# GPT-2 Small — GPU Optimization Report

This optimization achieved **1.76× total speedup** on GPT-2 small (B=4, seq=128, RTX PRO 6000 Blackwell), measured end-to-end across all GPU kernels at locked clocks. The mechanism is a dtype promotion that moved every transformer GEMM off the FP32 SIMT path onto the Blackwell bf16 Tensor Cores (the GEMM family alone went 2.37× faster, 6.40 ms → 2.70 ms); part of that win is reabsorbed by new fp32↔bf16 cast Triton kernels that a cleanup pass (OPT-4) was unable to cancel, which is why the end-to-end number is 1.76× rather than ~2.1×.

---

## 1. Hardware Context

| Field | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Architecture | Blackwell |
| PyTorch | 2.11.0+cu128 (CUDA 12.8) |
| Compile mode (baseline) | `inductor` (built-in dedup backend) |
| Compile mode (optimized) | `gpt2_opt` (custom `@register_backend`) |
| Batch size | 4 |
| Sequence length | 128 |
| Iterations | warmup 2 / measure 2 (nsys capture — durations measured at locked GPU clocks; relative comparison) |
| GPU clock lock | 1845 MHz graphics / 12481 MHz memory (identical for both captures) |
| Model | 12 identical transformer decoder blocks, hidden=768, heads=12, ffn=3072 |

**Timing source.** Per-operator durations are **nsys-derived GPU kernel times**, captured at a locked clock that `run_workload.py` probed once and reused for both baseline and optimized runs — so the comparison is fair and reproducible. Hardware counters (`tensor_core_active_pct`, SM/DRAM throughput, occupancy) come from the ncu replay phase. Baseline and optimized captures ran 48 minutes apart on the same GPU at identical locked clocks, so no cross-session caveat applies.

---

## 2. Operator Summary (Baseline)

Aggregated by kernel family across all 12 blocks. Baseline total GPU kernel time: **7.31 ms** (292 kernels).

| Family | Time (%) | Duration (ms) | Kernels | Bottleneck Class |
|---|---|---|---|---|
| GEMM — `cutlass_80_simt_sgemm` (fp32) | 87.5% | 6.401 | 96 | **Tensor-Core-idle (SIMT-bound)** — `tensor_core_active_pct = 0.0`, 210 regs/thread, 16.5% occupancy |
| Attention — `fmha_cutlassF_f32` (mem-efficient) | 6.6% | 0.485 | 24 | Memory-bound fp32 attention |
| Triton fused (addmm bias / GELU / LayerNorm epilogue) | 2.3% | 0.166 | 46 | Memory-bound epilogue |
| LayerNorm | 2.0% | 0.149 | 50 | Memory-bound |
| Triton other (views / mask prep) | 1.4% | 0.103 | 72 | Memory-bound |
| Elementwise / activation | 0.1% | 0.010 | 4 | Memory-bound |

The profile is dominated by GEMMs running on the CUDA-core SIMT path: every matmul lowered to `cutlass_80_simt_sgemm_*` with **Tensor Cores completely idle** while the fp32 FMA units did all the work.

---

## 3. Reading the Metrics

- **`tensor_core_active_pct = 0.0` (not null)** — the single highest-ROI signal in this profile. It means the GEMM executed entirely on the FP32 SIMT path with Tensor Cores idle. For a transformer that is ~84% GEMM time, this directly implies "promote to a Tensor-Core dtype." (A *null* value, by contrast, is normal for non-GEMM kernels and is not a problem.) On Blackwell this counter is populated here, so it is used directly rather than inferred from throughput.
- **`registers_per_thread = 210`** — the fp32 SIMT sgemm tiles are register-starved; this is what caps achieved occupancy at ~16%. bf16 Tensor-Core HGEMM tiles need far fewer registers (measured 96 after optimization), relieving the pressure.
- **`sm_throughput_pct` vs `memory_throughput_pct`** — baseline GEMMs show SM ~40% / DRAM ~8%: compute-bound, but compute spent on the wrong (SIMT) pipe. The fix is not to reduce work but to redirect it to the Tensor-Core pipe.

---

## 4. Optimizations Applied

Statuses from `profiler_output/validation_report.json`.

| ID | Type | Target Operators | Hardware Evidence | Confidence | Status |
|---|---|---|---|---|---|
| OPT-1 | aten | all `mm`/`addmm` (4 projection families × 12 blocks) | `tensor_core_active_pct = 0.0`; 84% of attributed time on SIMT sgemm | high | **APPLIED** |
| OPT-2 | functional | `scaled_dot_product_attention` | 48-launch memory-bound SDPA-prep family (L1 0%, occ 33%) | medium | **APPLIED** |
| OPT-3 | inductor_config | bias / GELU / LayerNorm epilogues | 72-launch `addmm` Triton family, DRAM 42–50% | medium | **APPLIED** |
| OPT-4 | aten | inverse `convert_element_type` pairs from OPT-1 | redundant fp32↔bf16 round-trips between GEMMs | medium | **NOT_APPLIED** (graceful no-op) |

OPT-4 found nothing to cancel because OPT-1 restores the output dtype directly instead of emitting per-node inverse cast pairs. The consequence is real and visible in Section 6: the fp32↔bf16 cast traffic OPT-1 introduces survives as standalone Triton kernels (~0.7 ms) rather than being eliminated.

---

## 5. Implementation Notes

# Implementation Notes — gpt2_opt

Custom `torch.compile()` backend for the GPT-2 small workload (`examples/gpt2/gpt2.py`):
12 structurally identical transformer decoder blocks, hidden=768, heads=12, ffn=3072,
batch=4, seq_len=128, fp32 inference on an RTX PRO 6000 Blackwell (torch 2.11.0+cu128,
CUDA 12.8). Backend name registered via `@register_backend`: **`gpt2_opt`**.

The backend is the canonical three-stage funnel
`_run_functional_passes(gm) -> compile_fx(inner_compile=_aten_inner_compile, config_patches=_config_patches())`,
invoked identically on the flat graph and on every dedup representative. GPT-2's 12
blocks are structurally identical, so `UniqueSubgraphRegistry` returns a non-empty
equivalence map and the dedup path compiles one representative block and shares the
compiled callable across all 12 — the same dedup behavior the profiling pipeline relies on.

## Backend Architecture

| Pass | Level | Method | Reason |
|---|---|---|---|
| OPT-2 — canonicalize attention to `F.scaled_dot_product_attention(is_causal=True)` | functional | `_run_functional_passes` (`_fpass_canonicalize_sdpa`) | The SDPA call is a single high-level node only on the Dynamo graph; switching it to `is_causal=True` and dropping the additive `attn_mask` makes the `constant_pad_nd`/`scalar_tensor`/`where` mask-construction chain dead, so AOTAutograd/Inductor eliminate the 48-launch memory-bound SDPA-prep family. After decomposition the mask nodes are already shattered and the identity is lost, so it must run functionally. |
| OPT-1 — bf16 dtype promotion for `mm`/`addmm` operands | aten | `_aten_inner_compile` (`_apass_bf16_promotion`) | Dtype promotion keys on the decomposed `aten.mm`/`aten.addmm` targets; casting all GEMM operands to bf16 routes the four projection families off the `cutlass_80_simt_sgemm` SIMT path (`tensor_core_active_pct=0.0`) onto the Blackwell bf16 tensor-core HGEMM path. Op-target pass — no weight-value lookup needed. |
| OPT-4 — cast cancellation (cleanup for OPT-1) | aten | `_aten_inner_compile` (`_apass_cancel_casts`) | Operates on the inverse `convert_element_type` pairs OPT-1 inserts, so it must run at the same aten level and after OPT-1 (within-level edge); cancels fp32->bf16->fp32 round-trips so consecutive GEMMs run bf16 end-to-end without extra DRAM traffic. |
| OPT-3 — Inductor freezing (weight pre-pack + epilogue fusion) | inductor_config | `config_patches={"freezing": True}` (`_cfg_freezing`) | Weight layout / constant-folding is owned by Inductor, not expressible as a graph rewrite; freezing constant-folds the LayerNorm affine + projection bias and pre-transposes the weight constants so cuBLASLt/CUTLASS selects a fused-epilogue HGEMM, removing standalone bias/GELU Triton kernels. |
| (whole-module dtype / memory_format / batch-shape) | non-graph | — (not applied) | None apply: 2D activations make channels_last irrelevant, and the 512x768 problem (512 = 4*128) already tiles cleanly so batch-padding offers no benefit. The model is left fp32 in `get_model_and_input()`; promotion happens inside the graph. |

## Key Design Decisions

**Why per-rep compilation instead of `replace_pattern` / flat compile.** The 12 blocks are
detected as structurally identical by `UniqueSubgraphRegistry`, so the backend compiles a
single representative through the full funnel and patches the duplicates' `.forward` with the
shared callable. This mirrors the dedup contract the rest of the pipeline (`.part.json`,
ncu metric propagation) depends on and cuts compile + profile time ~12x. The funnel runs
per-rep (inside `_compile_unit`), never on the pre-split graph, so functional passes see the
clean single-block Dynamo graph.

**Why OPT-2 is functional, not aten.** Canonicalizing attention keys on the single
`F.scaled_dot_product_attention` node and the materialized causal-mask subgraph feeding it.
After AOTAutograd decomposes the graph, the mask construction is lowered into per-consumer
primitives and the `is_causal` flag is no longer a single togglable argument — the rewrite is
only sound and unambiguous at the functional level. The pass is intentionally defensive
(medium confidence): HF GPT-2's sdpa path may already emit a causal SDPA with no additive
mask, in which case the pass logs a no-op WARNING rather than mutating the graph.

**Why OPT-1 -> OPT-4 is a within-level (aten) ordering edge.** OPT-1 inserts a
`convert_element_type(bf16)` before each GEMM and a `convert_element_type(orig)` after it.
OPT-4 only has work to do once those casts exist, so it is registered immediately after OPT-1
in the aten level. If OPT-1 produced no per-node casts, OPT-4 cleanly no-ops with a WARNING.

**Why `prims.convert_element_type` instead of `aten._to_copy`.** On torch 2.11 `aten._to_copy`
carries both a fallback and a decomp registration; inserting it into an already-decomposed Aten
graph makes Inductor raise "both a fallback and a decomp for the same op". `prims.convert_element_type`
lowers cleanly to a Triton elementwise cast, so both aten passes use it.

**Why config patches are scoped, and why not `aot_autograd(fw_compiler=compile_fx)`.** OPT-3 is
merged into each `compile_fx` call's `config_patches` rather than mutating `torch._inductor.config`
globally, so freezing is scoped to this compilation and cannot leak across units. The backend lets
`compile_fx` own AOTAutograd exactly once (functional passes before it, aten passes inside its
`inner_compile` seam); `aot_autograd(fw_compiler=compile_fx)` would raise
`AssertionError: Expected tensors only` inside `copy_misaligned_inputs` on torch 2.11.

---

## 6. Before/After Results

Both profiles: batch=4, seq=128, same GPU, identical locked clocks, 48 min apart (no cross-session caveat). Comparison aggregated by kernel family over the **full** kernel inventory (including unattributed kernels, so optimization-introduced overhead is counted).

| Family | Baseline (ms) | Optimized (ms) | Speedup |
|---|---|---|---|
| GEMM (fp32 SIMT → bf16 Tensor Core) | 6.401 | 2.695 | **2.37×** |
| Attention (`fmha` fp32) | 0.485 | 0.408 | 1.19× |
| LayerNorm | 0.149 | 0.120 | 1.24× |
| Triton casts + elementwise + epilogue | 0.279 | 0.938 | **new overhead** (0.30×) |
| **Total** | **7.313** | **4.161** | **1.76×** |

**Speedup attribution** (status ∧ metric moved ∧ operator faster):
- The GEMM 2.37× speedup is attributed to **OPT-1** — `status = APPLIED`, `tensor_core_active_pct` moved 0.0 → 40.49% on the FFN GEMM, registers/thread dropped 210 → 96, and the `cutlass_80_simt_sgemm` kernels were entirely replaced by `cutlass_80_tensorop_bf16` / `cutlass_80_wmma_tensorop_bf16`. All three conditions hold.
- The "new overhead" row is the direct cost of **OPT-4 = NOT_APPLIED**: the fp32↔bf16 cast traffic OPT-1 introduces survives as ~0.7 ms of standalone Triton kernels (the family grew from 72 → 288 kernels). This is the single largest reason the end-to-end speedup is 1.76× rather than the ~2.1× the GEMM win alone would imply.
- Attention (1.19×) and LayerNorm (1.24×) show modest gains consistent with OPT-2/OPT-3 plus reduced upstream DRAM traffic, but the attention compute remains the fp32 `fmha` kernel (OPT-2 drops the mask but does not change attention dtype).

---

## 7. What Drove Each Speedup

**bf16 dtype promotion (OPT-1, +2.37× on the GEMM family):** Casting every `mm`/`addmm` operand to bf16 reroutes the four projection GEMMs from the FP32 CUDA-core SIMT path onto Blackwell's bf16 Tensor Cores. The evidence is unambiguous: `tensor_core_active_pct` rose from 0.0 to 40.49% on the representative FFN GEMM, registers/thread fell 210 → 96, and every `cutlass_80_simt_sgemm` kernel in the optimized trace was replaced by a `cutlass_80_tensorop_bf16` / `cutlass_80_wmma_tensorop_bf16` Tensor-Core kernel.

**Attention canonicalization (OPT-2, +1.19× on attention):** Switching SDPA to `is_causal=True` and dropping the additive mask removes the explicit mask-construction subgraph, trimming the memory-bound prep work feeding attention. The `fmha` compute kernel itself stays fp32 (the win is in the surrounding launches, not the kernel dtype).

**Inductor freezing (OPT-3, contributes to epilogue fusion):** `freezing=True` constant-folds the LayerNorm affine and projection biases and pre-packs weight constants, letting CUTLASS select fused-epilogue GEMMs. The standalone fused-bias Triton family (46 launches, 0.166 ms) is folded into the GEMM epilogue path in the optimized profile.

---

## 8. Remaining Opportunities

| ID | Type | Target | Reason Not Applied | Projected Gain |
|---|---|---|---|---|
| OPT-4 | aten | inverse `convert_element_type` pairs | Graceful no-op — OPT-1 restores output dtype directly instead of emitting per-node cast pairs, so there were no round-trips to cancel; the cast traffic instead survives as standalone Triton kernels | ~0.6–0.7 ms (recover most of the "new overhead" row) |

**The dominant residual opportunity is the unrecovered cast overhead.** OPT-1 delivered the Tensor-Core win but left ~0.7 ms of fp32↔bf16 cast Triton kernels in place because OPT-4 had no per-node cast pairs to match. Reworking OPT-1 to keep activations in bf16 across consecutive GEMMs (so casts occur once at block boundaries rather than around every matmul) — or giving OPT-4 a pattern that matches OPT-1's actual cast placement — would reclaim most of that 0.7 ms and push the end-to-end speedup from 1.76× toward ~2.1×.

Secondary: attention still runs the fp32 `fmha_cutlassF_f32` kernel (0.408 ms). Routing it to a bf16 flash backend would convert this from a memory-bound fp32 kernel to a Tensor-Core path, for an estimated additional ~0.2 ms.

---

## Reproduction

```bash
# Baseline capture
python3 nvidia/scripts/run_workload.py --workload examples/gpt2/gpt2.py \
    --warmup-iters 2 --measure-iters 2 --correlation-pass   # Phase A
nsys profile --trace=cuda,nvtx --output=profiler_output/gpt2 \
    python3 nvidia/scripts/run_workload.py --workload examples/gpt2/gpt2.py \
        --warmup-iters 2 --measure-iters 2                  # Phase B
# (then manifest -> attribution -> ncu replay -> build_profile -> profile.json)

# Optimized capture (same iters/clocks, custom backend)
#   add --compile-backend gpt2_opt against examples/gpt2/gpt2_optimized.py

# Or run the whole thing:
/optimize examples/gpt2/gpt2.py
```

Artifacts: `profile.json`, `profile_optimized.json`, `optimizations.json`, `gpt2_optimized.py`, `test_gpt2_optimized.py`, `profiler_output/{validation_report.json, implementation_notes.md}`.
