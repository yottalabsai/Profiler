# Implementation Notes: ConvBlock Optimized Backend

Backend name: `conv_block_opt`
Workload: `conv_block_optimized.py`
Model: VGG-style three-stage CNN (`ConvBlock`)
Compile mode: `inductor`
Device: NVIDIA RTX PRO 6000 Blackwell

---

## Backend Architecture

| Pass | ID | Method | Confidence | Reason |
|---|---|---|---|---|
| channels_last layout | OPT-1 | `get_model_and_input()` | high | `memory_format` is a tensor property, not visible in FX IR. Must be set before `torch.compile` traces the graph. |
| BF16 dtype promotion | OPT-2 | `get_model_and_input()` + FX pass | medium | `model.bfloat16()` sets weight dtypes before compilation. FX pass `_pass_insert_bf16_cast` inserts a `.to(bfloat16)` node after the first placeholder so runtime FP32 inputs are cast inside the compiled graph. Both halves are needed: the non-graph step fixes parameter dtypes; the FX pass handles activation dtype at graph execution time. |
| Inductor max-autotune | OPT-3 | Inductor config directives in backend (stub) | medium | Wave starvation on 64→128 and 128→256 convolutions (`sm__warps_active` = 8.3%). The root cause is large per-CTA tile selection by cuDNN heuristics. The appropriate lever is Inductor's Triton conv autotuner (`max_autotune_conv`), which searches smaller-tile variants. This is a stub: it sets `inductor_config.max_autotune = True` and `max_autotune_conv = True` before delegating to `compile_fx`, but does not rewrite any FX nodes. |
| 3-channel conv padding | OPT-4 | FX pass `_pass_pad_shallow_conv` | medium | The 3→64 channel convolution dispatches to `sm80_xmma_fprop_implicit_gemm_indexed_wo_smem` (15% TC utilisation) because K=27 < WMMA alignment minimum. Padding input and weight from 3 to 4 channels (K=36) satisfies alignment and enables the shared-memory-staging GEMM path. The extra zero-padded channel contributes exactly 0 to the output. |

---

## Key Design Decisions

### OPT-1 — non-graph placement
`memory_format=torch.channels_last` is a tensor-level property. The FX graph sees abstract `aten.convolution.default` nodes regardless of the memory layout of their operands — layout is metadata, not a node type. Setting channels_last before `torch.compile` means Dynamo traces with NHWC-shaped fake tensors, so Inductor's shape inference is accurate and cuDNN receives NHWC data without any format-conversion kernel overhead.

The guard `if not first_param.is_contiguous(memory_format=torch.channels_last)` prevents double-conversion if the caller re-invokes `get_model_and_input()` on an already-converted model.

### OPT-2 — two-part implementation
Parameter dtype (weights, BN scale/shift) is set before compilation so Dynamo traces with BF16 tensor shapes. The FX pass `_pass_insert_bf16_cast` is a separate concern: it handles the activation input at graph execution time. Without the FX pass, a caller that passes a FP32 input tensor would trigger a dtype mismatch between the first placeholder (FP32) and the BF16 weight parameters during the first cuDNN convolution. The pass avoids this by inserting a cast node immediately after the first placeholder, so the graph is self-contained with respect to dtype regardless of the caller's input dtype.

The FX cast pass targets only the first placeholder because all subsequent placeholders are weight/bias parameters already in BF16. Casting them again would be a no-op but would add dead nodes to the graph.

BN `running_mean` and `running_var` are kept as FP32 buffers by PyTorch's `BatchNorm2d` regardless of model dtype, because `_native_batch_norm_legit_no_training` internally accumulates in FP32. No special handling is required.

### OPT-3 — stub classification
The wave-starvation bottleneck (OPT-3) manifests as a cuDNN tile-selection choice at dispatch time, not as a structural graph pattern that can be rewritten at the FX level. The two actionable levers are: (a) increase batch size to enlarge the CTA grid, or (b) route convolutions through Inductor's Triton autotuner instead of cuDNN. Option (b) is implemented here as Inductor config directives. The pass is classified as a stub because it does not modify any FX nodes — it only sets config flags before handing off to `compile_fx`. The actual autotuning happens inside Inductor/Triton at kernel compile time. Correctness is guaranteed; speedup is hardware and Triton version dependent.

### OPT-4 — FX pass at pre-Inductor level
The 3-channel padding pass targets `F.conv2d` nodes (pre-Inductor form), not `aten.convolution.default`. At the `@register_backend` level, before Inductor lowers the graph, convolutions appear as Python-level `F.conv2d` calls. The pass inspects `node.meta["val"].shape[1]` (the fake tensor shape set by Dynamo's shape propagation) to identify 3-channel inputs without needing to resolve actual weight tensors from `partition_inputs`. The padding amounts are expressed as `torch.ops.aten.constant_pad_nd.default` nodes inserted before each matching conv node. The kernel receives the padded input and weight, and cuDNN's heuristic now sees K=36 (aligned) instead of K=27 (sub-threshold), allowing algorithm selection of the smem-staging GEMM path.

The pass also handles `torch.ops.aten.convolution.default` and `torch.ops.aten.cudnn_convolution.default` targets for robustness, in case Dynamo lowers to Aten before this backend is invoked.

### Dedup path
`UniqueSubgraphRegistry` splits the FX graph and checks for structurally identical partitions. `ConvBlock` has no repeated transformer blocks, so `build_partition_equivalence_map()` returns an empty dict and the flat compile path is taken. The dedup path is included for completeness (e.g., if the model is wrapped in a loop or extended with repeated blocks). In the dedup path, each FX pass is applied once per unique representative and propagated to duplicates by sharing the compiled callable.

### Pass ordering
OPT-1 before OPT-2 (per `prerequisite_for` in `optimizations.json`): channels_last is applied first so BF16 parameters are stored in NHWC layout from the start. Within the FX backend, OPT-2 (BF16 cast) is applied before OPT-4 (3-channel padding) so the padding nodes see the already-cast activation dtype.

---

## Estimated Impact (from optimizations.json)

| OPT | Estimated saving (ns) | Pct of total | Notes |
|---|---|---|---|
| OPT-1 | 245,000 | 22.6% | Eliminates 18 convertTensor + nhwcToNchw kernel launches |
| OPT-2 | 38,000 | 3.5% | Halves BN-ReLU pointwise DRAM traffic |
| OPT-3 | 60,000 | 5.5% | Wave utilisation 5→10+ waves on 64→128 and 128→256 convs (hardware-dependent) |
| OPT-4 | 20,000 | 1.85% | TC utilisation 15% → 40–50% on 3→64ch convolutions |

All `duration_ns` values are ncu-inflated 2–5× and represent relative, not absolute, latency.

---

## Known Limitations and Caveats

- **BF16 numerical accuracy**: BF16 has a narrower mantissa than FP32 (7 vs 23 bits). Validate output accuracy before deploying in production. The test suite checks for NaN/Inf but does not compare against a FP32 reference.
- **sm80 kernel on Blackwell**: The profile shows `sm80_xmma_fprop` kernels on Blackwell hardware, suggesting cuDNN in torch 2.11+cu128 uses Ampere-generation heuristics. Upgrading to a PyTorch build with native Blackwell/sm100 cuDNN support may resolve OPT-3 and OPT-4 bottlenecks without any code changes.
- **OPT-3 max_autotune first-call latency**: Enabling max_autotune triggers Triton kernel autotuning on the first compilation, which adds significant latency (minutes) to the first forward pass. This is acceptable for offline inference or batch workloads but not for interactive/low-latency use cases. Set `TORCHINDUCTOR_CACHE_DIR` to persist the autotuning cache across runs.
- **OPT-4 weight mismatch**: The 3→4 channel padding inserts zero rows in the padded weight columns. If the model is saved and reloaded with `torch.save`/`torch.load`, the padded weight will not match the original `state_dict`. This pass is intended for inference-time use only and should not be applied to training graphs.
