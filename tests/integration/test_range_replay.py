"""
Integration test — ncu range replay + metric extraction.

REQUIRES: CUDA GPU + ncu (Nsight Compute) installed and on PATH.
Skip automatically if CUDA / ncu are unavailable.

Verifies that replaying an aten::linear range produces non-zero
dram__bytes_read.sum in the output profile.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def _ncu_available() -> bool:
    try:
        result = subprocess.run(["ncu", "--version"], capture_output=True, timeout=10)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.fixture(scope="module")
def replay_profile(tmp_path_factory):
    if not _cuda_available():
        pytest.skip("CUDA not available")
    if not _ncu_available():
        pytest.skip("ncu not available on PATH")

    tmp = tmp_path_factory.mktemp("range_replay")

    script = tmp / "linear_replay.py"
    script.write_text(
        """\
import torch
import torch.autograd.profiler

model = torch.nn.Linear(512, 512).cuda()
x = torch.randn(1, 512, device="cuda")

for _ in range(2):
    _ = model(x)
torch.cuda.synchronize()

with torch.autograd.profiler.emit_nvtx():
    out = model(x)
    torch.cuda.synchronize()
"""
    )

    # First, run nsys to build the manifest
    nsys_prefix = str(tmp / "profile")
    subprocess.run(
        [
            "nsys", "profile",
            "--trace=cuda,nvtx",
            f"--output={nsys_prefix}",
            "--force-overwrite=true",
            sys.executable, str(script),
        ],
        check=True, timeout=120,
    )
    nsys_rep = tmp / "profile.nsys-rep"

    from operator_profiler.mapper.manifest_builder import ManifestBuilder
    from operator_profiler.schema.manifest import CaptureManifestMetadata

    metadata = CaptureManifestMetadata(
        model_name="ReplayTest",
        torch_version=__import__("torch").__version__,
        compile_mode="eager",
        nsys_report_path=str(nsys_rep),
        capture_timestamp_utc="2026-03-19T00:00:00+00:00",
    )
    manifest = ManifestBuilder(nsys_rep_path=nsys_rep, metadata=metadata).build()

    # Run attribution
    from operator_profiler.mapper.attribution_engine import AttributionEngine
    engine = AttributionEngine(manifest)
    operator_records, unattributed = engine.run()

    # Run range replay
    from operator_profiler.mapper.range_replay import RangeReplayConfig, RangeReplayOrchestrator
    replay_config = RangeReplayConfig(
        replay_script=script,
        output_dir=str(tmp),
    )
    orch = RangeReplayOrchestrator(manifest, operator_records, replay_config)
    orch.run()

    # Build profile
    from operator_profiler.aggregator.profile_builder import build_profile
    import torch
    profile = build_profile(
        manifest=manifest,
        operator_records=operator_records,
        unattributed_kernels=unattributed,
        model_name="ReplayTest",
        torch_version=torch.__version__,
    )
    return profile


class TestRangeReplay:
    def test_profile_has_operators(self, replay_profile):
        assert len(replay_profile.operators) > 0

    def test_dram_bytes_nonzero(self, replay_profile):
        """
        After ncu range replay, dram_bytes_read must be > 0 for at least
        one kernel attributed to aten::linear (or aten::mm).
        """
        target_ops = {"aten::linear", "aten::mm", "aten::addmm"}
        for op in replay_profile.operators:
            if op.operator_name in target_ops:
                for kernel in op.kernels:
                    if kernel.metrics.dram_bytes_read is not None:
                        assert kernel.metrics.dram_bytes_read > 0
                        return
        pytest.fail(
            "No kernel with non-zero dram_bytes_read found for linear/mm operators"
        )

    def test_schema_version_preserved(self, replay_profile):
        from operator_profiler.schema.profile import SCHEMA_VERSION
        assert replay_profile.schema_version == SCHEMA_VERSION
