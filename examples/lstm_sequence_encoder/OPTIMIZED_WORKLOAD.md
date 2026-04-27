# LSTMSequenceEncoder — Optimized Workload

Custom `torch.compile()` backend for the stacked LSTM sequence encoder, derived from ncu hardware profiling.

## Overview

The baseline profile reveals that `aten::_cudnn_rnn` accounts for **98.9% of attributed time** on a H100 PCIe. Under PyTorch 2.11 + `torch.compile(backend="inductor")`, `nn.LSTM` stays as a single cuDNN operator (cuDNN handles the full LSTM forward; Inductor does NOT decompose it into per-timestep Triton kernels on this configuration).

This has two important consequences for optimization:
1. **FX graph passes cannot reach inside the LSTM** — only the mean-pooling and linear classifier head are visible in the FX graph
2. **The highest-ROI optimization is a dtype change** — BF16 causes cuDNN to switch from the legacy `Kernel2` (FP32 SIMT) path to `sm90_xmma_gemm_bf16bf16_*` (WGMMA Tensor Core path), the H100's highest-throughput compute path for this workload

## Quick Start

```bash
# Validate before profiling (saves ncu replay time on broken code)
cd /home/ubuntu/Profiler
python -m pytest examples/lstm_sequence_encoder/test_lstm_sequence_encoder_optimized.py -v

# Compiled smoke test (1 warmup, 1 measure)
PYTHONPATH=. python3 nvidia/scripts/run_workload.py \
    --workload examples/lstm_sequence_encoder/lstm_sequence_encoder_optimized.py \
    --compile-backend lstm_sequence_encoder_opt \
    --warmup-iters 1 --measure-iters 1

# Full optimized profile (requires CUDA + nsys + ncu)
PYTHONPATH=. nsys profile --trace=cuda,nvtx \
    --output examples/lstm_sequence_encoder/profiler_output/lstm_sequence_encoder_opt \
    --force-overwrite=true \
    python3 nvidia/scripts/run_workload.py \
        --workload examples/lstm_sequence_encoder/lstm_sequence_encoder_optimized.py \
        --compile-backend lstm_sequence_encoder_opt \
        --warmup-iters 5 --measure-iters 10
```

## Optimizations

| ID | Optimization | Target Op | Location | Expected Impact | Confidence |
|----|---|---|---|---|---|
| OPT-1 | BF16 dtype promotion | `aten::_cudnn_rnn` (98.9%) | `get_model_and_input()` | 2–4× LSTM speedup (SIMT → WGMMA) | High |
| OPT-2 | `cudnn.benchmark` + `allow_tf32` | `aten::_cudnn_rnn`, `aten::addmm` | `get_model_and_input()` | 5–20% additional on LSTM | High |
| OPT-3 | Pre-transposed classifier weight | `aten::addmm` (0.7%) | FX pass | Negligible (<0.1ms) | Medium |

## Architecture

### Non-Graph Optimizations (applied in `get_model_and_input`)

**OPT-1 — BF16 dtype promotion:**
```python
torch.backends.cudnn.benchmark = True        # OPT-2: benchmark before construction
torch.backends.cuda.matmul.allow_tf32 = True # OPT-2: TF32 for addmm
model = LSTMSequenceEncoder().to(DEVICE).eval()
x = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE, device=DEVICE)
if next(model.parameters()).dtype != torch.bfloat16:
    model = model.to(torch.bfloat16)         # OPT-1: routes cuDNN to WGMMA
    x = x.to(torch.bfloat16)
```

**Why BF16 works for LSTM:** cuDNN dispatches to different kernel implementations based on input dtype. For FP32 inputs, cuDNN selects `Kernel2` (legacy SIMT matrix multiply). For BF16 inputs on H100, cuDNN selects `sm90_xmma_gemm_bf16bf16_*` — the WGMMA instruction path which utilizes H100 Tensor Cores at full throughput.

**Why `cudnn.benchmark` matters:** The first call to `model(x)` with a new input configuration triggers cuDNN's algorithm selection. Without `benchmark=True`, cuDNN uses a fast heuristic. With `benchmark=True`, cuDNN times several algorithm candidates and caches the fastest. This is amortized across the warmup iterations.

### FX Graph Optimization (applied in backend)

**OPT-3 — Pre-transposed classifier weight (`pass_pretranspose_classifier`):**

The Inductor FX graph represents `nn.Linear` as:
```
weight_T = aten.t(get_attr('classifier.weight'))  # on-the-fly transpose
out      = aten.addmm(bias, pooled, weight_T)
```

The pass pre-computes `weight.T.contiguous()` and registers it as a buffer, replacing the per-call `aten.t()` with a direct buffer lookup. Impact is negligible for this 512×10 weight, but demonstrates the FX pass pipeline is active.

### Backend Registration

```python
@register_backend
def lstm_sequence_encoder_opt(gm: fx.GraphModule, example_inputs) -> Callable:
    gm = pass_pretranspose_classifier(gm)        # OPT-3
    return compile_fx(gm, example_inputs)        # delegates to Inductor
```

## Key Design Decisions

1. **BF16 in `get_model_and_input()`, not FX pass** — cuDNN responds to the dtype of input tensors at runtime, not to FX graph annotations. The BF16 cast must happen before `torch.compile()` sees the model.

2. **No LSTM FX surgery** — cuDNN LSTM is opaque. Attempting to traverse the FX graph looking for LSTM gate patterns would fail because `aten::_cudnn_rnn` is a single node with no decomposition into GEMMs visible in the pre-Inductor FX IR.

3. **cudnn.benchmark before model construction** — Setting `benchmark=True` before constructing the model ensures the benchmark flag is active for the very first forward pass during warmup. This prevents caching a non-benchmarked algorithm.

4. **BF16 vs FP16 for LSTM** — BF16 is preferred for LSTM over FP16. LSTM sigmoid and tanh activations on hidden states can produce large intermediate values (e.g., before the sigmoid saturates). BF16's 8-bit exponent range (same as FP32) avoids overflow; FP16's 5-bit exponent can overflow on large hidden states.

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `TypeError: 'module' object is not callable` | Wrong `compile_fx` import | `from torch._inductor.compile_fx import compile_fx` (already correct in this file) |
| `[pass_pretranspose_classifier] Pattern not found` | Inductor graph structure differs | Normal — pass degrades gracefully; does not affect correctness |
| `AssertionError: Expected BF16` in tests | BF16 not applied | Check that `torch.cuda.is_available()` passes; BF16 requires CUDA |
| Compiled output identical to baseline | Backend not registered | Run `torch._dynamo.reset()` before recompiling; verify `lstm_sequence_encoder_opt` in `torch._dynamo.list_backends()` |
| cuDNN still shows Kernel2 after BF16 | cudnn.benchmark cached FP32 plan | Clear cuDNN plan cache: `torch.backends.cudnn.benchmark_limit = 0` then reset |

## Future Work

| Optimization | What It Needs | Expected Impact |
|---|---|---|
| Custom BF16 sigmoid+tanh Triton kernel | Custom CUDA/Triton kernel fusing all 4 gate activations per timestep | ~10-20% on activation ops (currently inside cuDNN, so gains depend on inductor decomposition) |
| `torch.compile(mode='max-autotune')` | PyTorch 2.1+ max-autotune mode | May select faster cuDNN algorithm beyond what `benchmark=True` tries |
| Per-layer BF16 with FP32 accumulation | Custom LSTM cell that accumulates in FP32 but computes in BF16 | Maintains numerical stability with near-BF16 throughput for 20+ layer models |
| Sequence length padding to power of 2 | Pad `seq_len` from 128 to 128 (already a power of 2) | No benefit for this workload |
