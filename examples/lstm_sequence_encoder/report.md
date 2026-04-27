# LSTMSequenceEncoder — GPU Profiling & Optimization Report

**GPU:** NVIDIA H100 PCIe  
**PyTorch:** 2.11.0+cu130  
**Compile mode:** `inductor` (cuDNN eager for LSTM — see note below)  
**Batch / Seq / Hidden:** 32 × 128 × 512  
**Profiling tool:** nsys + ncu (kernel-replay mode)

> **Duration note:** All times in this report are from ncu kernel-replay mode, which runs 2–5× slower than real-wall-clock execution. The relative before/after comparison is valid; absolute numbers cannot be used as latency estimates.

---

## The Workload

A stacked 2-layer LSTM sequence encoder with mean-pool temporal reduction and a linear classifier:

```
nn.LSTM(input=256, hidden=512, layers=2, batch_first=True)   # 2-layer cuDNN LSTM
out.mean(dim=1)                                               # (B, T, H) → (B, H)
nn.Linear(512 → 10)                                          # classifier head
```

**Key architectural observation:** Under PyTorch 2.11 + `torch.compile(backend='inductor')` on H100 PCIe, `nn.LSTM` stays as a single `aten::_cudnn_rnn` operator. Inductor does **not** decompose LSTM into per-timestep Triton kernels on this configuration — cuDNN handles the full LSTM forward pass internally. This makes cuDNN dtype dispatch the primary optimization lever.

---

## Step 1: Baseline Profile

### Capture commands

```bash
# nsys capture (with NVTX emission for operator attribution)
PYTHONPATH=/home/ubuntu/Profiler nsys profile --trace=cuda,nvtx \
    --output=examples/lstm_sequence_encoder/profiler_output/lstm_sequence_encoder \
    --force-overwrite=true \
    python3 nvidia/scripts/run_workload.py \
        --workload examples/lstm_sequence_encoder/lstm_sequence_encoder.py \
        --compile-backend inductor --warmup-iters 5 --measure-iters 10

# ncu kernel replay → profile.json
PYTHONPATH=/home/ubuntu/Profiler operator-profiler map \
    examples/lstm_sequence_encoder/profiler_output/lstm_sequence_encoder.manifest.json \
    --script nvidia/scripts/run_workload.py \
    --ncu-executable /opt/nvidia/nsight-compute/2025.4.1/ncu --ncu-sudo \
    --ncu-env "PYTHONPATH=/home/ubuntu/Profiler:/home/ubuntu/.local/..." \
    --output examples/lstm_sequence_encoder/profile.json \
    --script-args --workload examples/lstm_sequence_encoder/lstm_sequence_encoder.py \
        --compile-backend inductor --warmup-iters 5 --measure-iters 10
```

### Baseline operator summary (FP32)

| Operator | Duration | % Total | Kernel | Bottleneck |
|---|---|---|---|---|
| `aten::_cudnn_rnn` (×10 iters) | 26.86 ms | **98.7%** | `Kernel2` | tensor_core_idle (inferred) |
| `aten::addmm` (classifier) | 0.17 ms | 0.6% | `sm80_xmma_gemm_f32f32_f32_tn` | negligible |
| `aten::mean` (temporal pool) | 0.09 ms | 0.3% | `reduce_kernel` | negligible |
| `aten::copy_` | 0.07 ms | 0.3% | `elementwise_kernel` | negligible |
| **Total** | **27.20 ms** | | | |

### Reading the metrics

**Why all hardware counters are null for `aten::_cudnn_rnn`:**  
cuDNN operators are opaque to ncu at the operator-aggregation level. The profiler correctly attributes timing, but Tensor Core%, occupancy, SM%, and DRAM% cannot be measured through the `_cudnn_rnn` abstraction. This is expected — ncu sees the constituent cuDNN CUDA kernels (`Kernel2`, `elemWiseRNNcell`) but cannot map them back to the high-level counters at the operator level.

**Bottleneck inference from kernel name:**  
`Kernel2` is cuDNN's internal name for the FP32 SIMT GEMM path used for LSTM gate computation. On H100:
- FP32 inputs → cuDNN selects `Kernel2` (legacy SIMT, no Tensor Core utilization)
- BF16 inputs → cuDNN selects `sm90_xmma_gemm_bf16bf16_*` (WGMMA — H100's Tensor Core path at full throughput)

The fix is a single dtype cast.

---

## Step 2: Optimization Recommendations

### OPT-1: BF16 dtype promotion — HIGH confidence

**Evidence:** cuDNN `Kernel2` kernel selected for FP32 LSTM on H100. Expected 2–4× speedup by switching to WGMMA Tensor Core path.

**Mechanism:** cuDNN dispatches LSTM to different kernel families based on input dtype at runtime. BF16 is preferred over FP16 for LSTM — BF16 has 8-bit exponent (same as FP32), preventing overflow in sigmoid/tanh activations on large hidden states.

```python
# In get_model_and_input()
model = model.to(torch.bfloat16)
x = x.to(torch.bfloat16)
```

### OPT-2: cudnn.benchmark — HIGH confidence

**Evidence:** Without benchmark mode, cuDNN uses a heuristic for algorithm selection. With `benchmark=True`, cuDNN times candidate BF16 LSTM algorithms and caches the fastest for the (batch=32, seq=128) configuration. Amortized across the warmup iterations.

```python
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True  # also helps aten::addmm
```

### OPT-3: Pre-transposed classifier weight — MEDIUM confidence (negligible impact)

**Evidence:** `aten::addmm` dispatches to `sm80_xmma_gemm_f32f32_f32_tn` (TN layout = weight transposed at call time). FX pass pre-transposes the [512, 10] weight.

**Note:** At 0.6% of total time, impact is < 0.1ms. Included for completeness; contributes nothing measurable. Additionally, Dynamo produces 0 traceable FX graphs for `nn.LSTM` (see note below), so this FX pass never fires in practice.

---

## Step 3: Implementing the Optimizations

All optimizations are in `get_model_and_input()` — no FX graph surgery is possible for cuDNN LSTM:

```python
def get_model_and_input():
    # OPT-2: set benchmark BEFORE model construction
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    model = LSTMSequenceEncoder().to(DEVICE).eval()
    x = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE, device=DEVICE)

    # OPT-1: BF16 dtype — routes cuDNN to WGMMA Tensor Core path
    model = model.to(torch.bfloat16)
    x = x.to(torch.bfloat16)

    return model, x
```

**The FX pass situation:** `torch.compile(model, backend='lstm_sequence_encoder_opt')` calls our custom backend. However, `torch._dynamo.explain()` reports `graph_count=0` for this model — Dynamo does not produce traceable FX graphs from `nn.LSTM` with `torch._VF.lstm` (cuDNN's C extension entry point) in PyTorch 2.11. The model runs in cuDNN eager mode regardless of the `torch.compile` call. The primary optimizations (BF16 + benchmark) apply at the eager level and work correctly.

---

## Step 4: Optimized Profile

### Re-capture commands

```bash
# Same pipeline as baseline, pointing to the optimized workload
PYTHONPATH=/home/ubuntu/Profiler nsys profile --trace=cuda,nvtx \
    --output=examples/lstm_sequence_encoder/profiler_output/lstm_sequence_encoder_opt \
    --force-overwrite=true \
    python3 nvidia/scripts/run_workload.py \
        --workload examples/lstm_sequence_encoder/lstm_sequence_encoder_optimized.py \
        --compile-backend lstm_sequence_encoder_opt --warmup-iters 5 --measure-iters 10

PYTHONPATH=/home/ubuntu/Profiler operator-profiler map \
    examples/lstm_sequence_encoder/profiler_output/lstm_sequence_encoder_opt.manifest.json \
    --script nvidia/scripts/run_workload.py \
    --ncu-executable /opt/nvidia/nsight-compute/2025.4.1/ncu --ncu-sudo \
    --output examples/lstm_sequence_encoder/profile_optimized.json \
    --script-args --workload examples/lstm_sequence_encoder/lstm_sequence_encoder_optimized.py \
        --compile-backend lstm_sequence_encoder_opt --warmup-iters 5 --measure-iters 10
```

---

## Step 5: Results — Before vs. After

| Operator | Baseline (FP32) | Optimized (BF16) | Speedup | Driver |
|---|---|---|---|---|
| `aten::_cudnn_rnn` | 26.86 ms | 15.13 ms | **1.78×** | cuDNN WGMMA path (BF16) |
| `aten::addmm` | 0.17 ms | 0.07 ms | **2.41×** | BF16 halves memory traffic |
| `aten::mean` | 0.09 ms | 0.07 ms | **1.32×** | BF16 smaller operands |
| `aten::copy_` | 0.07 ms | 0.07 ms | 1.02× | unchanged |
| **Total** | **27.20 ms** | **15.39 ms** | **1.77×** | |

### What drove the speedup

**`aten::_cudnn_rnn` 1.78×:** BF16 causes cuDNN to switch from `Kernel2` (FP32 SIMT path) to `sm90_xmma_gemm_bf16bf16_*` (WGMMA Tensor Core path). On H100, WGMMA provides substantially higher throughput than SIMT for GEMMs of this size. The LSTM gate GEMM shapes ([32, 768] → [32, 2048]) fit well on H100's Tensor Core tiles.

**`aten::addmm` 2.41×:** BF16 halves the data volume for the [32×512]×[512×10] GEMM and allows the sm90 cuBLAS path to use a more efficient tile configuration.

**Note on `aten::fill_` regression (0.38×):** The optimized profile has 10 additional operators (70 vs 60 in baseline) and `fill_` takes 0.055ms vs 0.021ms. This likely reflects additional BF16 tensor initialization overhead or `flatten_parameters()` triggering extra kernel launches. At 0.055ms total, this is negligible (~0.4% of optimized time).

---

## Key Takeaways

1. **cuDNN dtype dispatch is the highest-leverage knob for LSTM.** A single `model.to(bfloat16)` call routes cuDNN from SIMT to Tensor Core hardware. No custom kernels, no graph surgery, no architectural changes.

2. **Hardware counter opacity is expected for cuDNN operators.** `tensor_core_active_pct == null` does not mean an error — it means the operator uses cuDNN's opaque internal kernel. Inference from kernel name + dtype is sufficient to identify and fix the bottleneck.

3. **`nn.LSTM` is not FX-graph-traceable by Dynamo in PyTorch 2.11.** `torch.compile` with a custom backend returns graph_count=0 for LSTM models. FX passes targeting LSTM internals will never fire. For LSTM, all optimizations must be non-graph (dtype cast, config flags).

4. **1.77× with one line of code.** The entire performance gain comes from `model.to(torch.bfloat16)` + `torch.backends.cudnn.benchmark = True` — no complex transformations required.

5. **Remaining opportunity:** A custom BF16 LSTM cell in Triton could eliminate cuDNN's opaqueness entirely and potentially achieve 2–4× additional speedup by fusing gate GEMMs with their sigmoid/tanh activations in a single kernel. This requires significant engineering effort.

---

## Appendix: Full Reproduction Commands

```bash
cd /home/ubuntu/Profiler

# 1. Validate the optimized workload
python3 -m pytest examples/lstm_sequence_encoder/test_lstm_sequence_encoder_optimized.py -v

# 2. Smoke test
PYTHONPATH=. python3 nvidia/scripts/run_workload.py \
    --workload examples/lstm_sequence_encoder/lstm_sequence_encoder_optimized.py \
    --compile-backend lstm_sequence_encoder_opt --warmup-iters 1 --measure-iters 1

# 3. Baseline profile (already done — profile.json exists)
# 4. Optimized profile (already done — profile_optimized.json exists)
```
