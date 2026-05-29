"""
neuron_capture — run a workload under NeuronConfig and collect all profiling artifacts.

This is the single-phase capture step for Trainium.  Unlike the NVIDIA pipeline
(which requires two separate runs — torch.profiler then nsys), NRT captures both
the operator timeline AND hardware metrics in one execution:

  nrt_inspect_begin_with_options()   ← called inside NeuronConfig.__enter__
  [workload runs]
  nrt_inspect_stop()                 ← called inside NeuronConfig.__exit__
  → writes ntrace.pb, .ntff, trace_info.pb to profile_output_dir

The Chrome trace (trace.json) is then exported by NeuronProfiler.export_trace()
and contains both cpu_op events (aten:: ops with Kineto External IDs) and
privateuse1_driver events (NeuronCore execution windows with the same IDs).

Workload interface
------------------
The workload script must expose:

    def get_model_and_input() -> tuple[nn.Module, torch.Tensor]:
        ...

The returned model should be on XLA (Neuron) device.  Compilation and warm-up
are handled here via --warmup-iters.

Usage (called by profile_cmd.py or run directly):
    result = run_capture(
        workload_fn=lambda: model(inputs),
        profile_output_dir=Path("/tmp/traces"),
        model_name="gpt2",
        warmup_iters=2,
        measure_iters=1,
    )
    trace_json  = result.trace_json_path
    session_dir = result.nrt_session_dir
"""
from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)


@dataclass
class CaptureResult:
    trace_json_path: Path       # Chrome trace exported by NeuronProfiler.export_trace()
    nrt_session_dir: Path       # NRT output directory containing ntrace.pb, .ntff, etc.
    neuron_sdk_version: str | None
    device_name: str | None


def run_capture(
    workload_fn: Callable[[], None],
    profile_output_dir: Path,
    model_name: str = "model",
    warmup_iters: int = 2,
    measure_iters: int = 1,
    modes: list[str] | None = None,
    neuroncore_indices: list[int] | None = None,
) -> CaptureResult:
    """
    Run *workload_fn* under NeuronConfig and collect profiling artifacts.

    Parameters
    ----------
    workload_fn:
        Zero-argument callable that runs one forward pass of the model.
    profile_output_dir:
        Directory where NRT writes trace files.  Created if absent.
    model_name:
        Used for logging only.
    warmup_iters:
        Number of warm-up iterations before profiling begins.
    measure_iters:
        Number of measured iterations under the profiler.
    modes:
        ProfileMode names to enable.  Defaults to ["DEVICE", "RUNTIME"].
    neuroncore_indices:
        Which NeuronCores to profile.  None = all.
    """
    try:
        import torch_neuronx
        from torch_neuronx.profiling import NeuronConfig, NeuronProfiler, ProfileMode
    except ImportError as e:
        raise RuntimeError(
            "torch_neuronx is required for Trainium profiling. "
            "Install it with: pip install torch-neuronx"
        ) from e

    profile_output_dir.mkdir(parents=True, exist_ok=True)

    if modes is None:
        modes = ["DEVICE", "RUNTIME"]

    mode_values = [getattr(ProfileMode, m) for m in modes]

    # Build NeuronConfig
    config_kwargs: dict = dict(
        modes=mode_values,
        profile_output_dir=str(profile_output_dir),
    )
    if neuroncore_indices is not None:
        config_kwargs["capture_enabled_for_nc"] = neuroncore_indices

    neuron_config = NeuronConfig(**config_kwargs)

    log.info("Warming up %s (%d iters) ...", model_name, warmup_iters)
    for _ in range(warmup_iters):
        workload_fn()

    log.info("Profiling %s (%d iter(s)) ...", model_name, measure_iters)

    profiler = NeuronProfiler(neuron_config)
    with profiler.profile() as prof:
        for _ in range(measure_iters):
            workload_fn()

    # Export Chrome trace to the session directory
    session_dir = Path(neuron_config.get_session_dir())
    trace_json_path = session_dir / "trace.json"
    profiler.export_trace(prof, str(trace_json_path))
    log.info("Chrome trace → %s", trace_json_path)

    # Collect version info
    neuron_sdk_version: str | None = None
    try:
        import neuronx_cc  # type: ignore
        neuron_sdk_version = getattr(neuronx_cc, "__version__", None)
    except ImportError:
        try:
            neuron_sdk_version = torch_neuronx.__version__
        except AttributeError:
            pass

    device_name: str | None = _detect_device_name()

    log.info("Capture complete. NRT session dir: %s", session_dir)
    return CaptureResult(
        trace_json_path=trace_json_path,
        nrt_session_dir=session_dir,
        neuron_sdk_version=neuron_sdk_version,
        device_name=device_name,
    )


def _detect_device_name() -> str | None:
    """Return Trainium instance type if detectable, else None."""
    try:
        import subprocess
        result = subprocess.run(
            ["curl", "-sf", "--max-time", "1",
             "http://169.254.169.254/latest/meta-data/instance-type"],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return None
