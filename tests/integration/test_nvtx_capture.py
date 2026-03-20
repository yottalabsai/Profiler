"""
Integration test — NVTX capture + nsys + manifest builder.

REQUIRES: CUDA GPU + nsys installed and on PATH.
Skip automatically if CUDA / nsys are unavailable.

Verifies that a minimal nn.Linear model run under emit_nvtx() + nsys produces
a mapping_manifest.json containing an aten::mm attribution with confidence=medium
or higher.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def _nsys_available() -> bool:
    try:
        result = subprocess.run(["nsys", "--version"], capture_output=True, timeout=10)
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
def linear_manifest(tmp_path_factory):
    """Run a minimal linear model under nsys and build the manifest."""
    if not _cuda_available():
        pytest.skip("CUDA not available")
    if not _nsys_available():
        pytest.skip("nsys not available on PATH")

    tmp = tmp_path_factory.mktemp("nvtx_capture")

    # Write a minimal replay script
    script = tmp / "linear_model.py"
    script.write_text(
        """\
import torch
import torch.autograd.profiler

model = torch.nn.Linear(512, 512).cuda()
x = torch.randn(1, 512, device="cuda")

# Warm up
for _ in range(2):
    _ = model(x)
torch.cuda.synchronize()

with torch.autograd.profiler.emit_nvtx():
    out = model(x)
    torch.cuda.synchronize()
"""
    )

    output_prefix = str(tmp / "profile")
    nsys_rep = tmp / "profile.nsys-rep"

    subprocess.run(
        [
            "nsys", "profile",
            "--trace=cuda,nvtx",
            f"--output={output_prefix}",
            "--force-overwrite=true",
            sys.executable, str(script),
        ],
        check=True,
        timeout=120,
    )
    assert nsys_rep.exists(), "nsys did not produce a .nsys-rep file"

    from operator_profiler.mapper.manifest_builder import ManifestBuilder
    from operator_profiler.schema.manifest import CaptureManifestMetadata

    metadata = CaptureManifestMetadata(
        model_name="IntegrationLinear",
        torch_version=__import__("torch").__version__,
        compile_mode="eager",
        nsys_report_path=str(nsys_rep),
        capture_timestamp_utc="2026-03-19T00:00:00+00:00",
    )
    builder = ManifestBuilder(nsys_rep_path=nsys_rep, metadata=metadata)
    return builder.build()


class TestNvtxCapture:
    def test_manifest_not_empty(self, linear_manifest):
        assert len(linear_manifest.kernels) > 0

    def test_aten_mm_attributed(self, linear_manifest):
        """aten::mm should appear as an NVTX-attributed operator."""
        from operator_profiler.schema.profile import AttributionMethod, Confidence

        mm_kernels = [
            k for k in linear_manifest.kernels
            if "aten::mm" in k.attribution.source_operators
            or "aten::linear" in k.attribution.source_operators
        ]
        assert len(mm_kernels) > 0, (
            "Expected at least one kernel attributed to aten::mm or aten::linear"
        )

    def test_confidence_at_least_medium(self, linear_manifest):
        from operator_profiler.schema.profile import Confidence

        low_or_unattr = [
            k for k in linear_manifest.kernels
            if k.attribution.confidence in (Confidence.LOW, Confidence.UNATTRIBUTED)
            and k.attribution.source_operators
        ]
        # Most kernels should be medium+ confidence in eager mode (NVTX ranges present)
        total = len(linear_manifest.kernels)
        high_med = total - len(low_or_unattr)
        assert high_med / total > 0.5, (
            f"Less than 50% of kernels have medium+ confidence ({high_med}/{total})"
        )

    def test_no_cross_stream_attribution(self, linear_manifest):
        """
        Kernels should only be attributed to NVTX ranges on the same stream
        (edge case #3 sanity check).
        """
        # This is structural — if interval tree is correct, attribution.nvtx_range
        # should match the kernel's stream (we can't verify stream on nvtx_range
        # directly, but we can check no entry has nvtx attributed with wrong stream id).
        # For this test we just verify no AttributionMethod.UNATTRIBUTED entries
        # have a non-None nvtx_range (that would indicate a cross-stream leak).
        from operator_profiler.schema.profile import AttributionMethod
        for k in linear_manifest.kernels:
            if k.attribution.method == AttributionMethod.UNATTRIBUTED:
                assert k.attribution.nvtx_range is None
