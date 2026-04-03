"""
check_hardware.py — Environment and GPU readiness check.

Verifies:
  - PyTorch version and CUDA availability
  - GPU name, VRAM, and a basic tensor smoke test
  - torch.profiler: runs a small Linear model and prints the top-10 events by CPU time

Run this at the start of a session to confirm the environment is set up correctly.
"""
import sys
import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile

print(f"Python:  {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("No CUDA GPU found — profiling tools require a CUDA-capable GPU.")
    sys.exit(1)

print(f"Device:  {torch.cuda.get_device_name(0)}")
print(f"CUDA:    {torch.version.cuda}")
print(f"VRAM:    {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

t = torch.randn(1024, 1024, device="cuda")
assert t.shape == (1024, 1024) and t.device.type == "cuda"
print("Tensor smoke test: OK")

print()
print("--- torch.profiler smoke test ---")
model = nn.Linear(256, 512).cuda().eval()
x = torch.randn(32, 256, device="cuda")

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with torch.no_grad():
        for _ in range(5):
            model(x)
torch.cuda.synchronize()

evts = prof.key_averages()
print(f"Profiler events captured: {len(evts)}")
print(f"{'Event':<35} {'CPU (µs)':>10}  {'CUDA (µs)':>10}")
print("-" * 60)
for evt in sorted(evts, key=lambda e: e.cpu_time_total, reverse=True)[:10]:
    cuda_us = getattr(evt, "cuda_time_total", 0)
    print(f"  {evt.key:<33} {evt.cpu_time_total:>10.0f}  {cuda_us:>10.0f}")

print()
print("Environment check passed.")
