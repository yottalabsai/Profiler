"""
Integration test — full preprocessing pipeline end-to-end.

REQUIRES: CUDA GPU + nsys + ncu (Nsight Compute) + PyTorch installed.
Skip automatically if any dependency is unavailable.

Covers all pipeline stages:
  1. nsys capture    — workload under nsys profile --trace=cuda,nvtx
  2. nsys export     — .nsys-rep → .sqlite
  3. Manifest build  — ManifestBuilder (query_kernels + query_nvtx_events + attribution)
  4. Attribution     — AttributionEngine.run() → operator_records
  5. ncu replay      — RangeReplayOrchestrator.run() → kernel metrics populated
  6. Aggregation     — build_profile() → AggregatedMetrics per operator
  7. Schema roundtrip — model_dump_json → model_validate_json
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Availability helpers
# ---------------------------------------------------------------------------

def _nsys_available() -> bool:
    try:
        result = subprocess.run(["nsys", "--version"], capture_output=True, timeout=10)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


_NCU_FALLBACK_PATH = "/opt/nvidia/nsight-compute/2025.4.1/ncu"


def _ncu_executable() -> str | None:
    """Return the ncu executable path, checking PATH first then the fallback."""
    import shutil
    found = shutil.which("ncu")
    if found:
        return found
    if Path(_NCU_FALLBACK_PATH).exists():
        return _NCU_FALLBACK_PATH
    return None


def _ncu_available() -> bool:
    exe = _ncu_executable()
    if exe is None:
        return False
    try:
        result = subprocess.run([exe, "--version"], capture_output=True, timeout=10)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Workload script — written to a temp file and used by both nsys and ncu
# ---------------------------------------------------------------------------

_WORKLOAD_SCRIPT = """\
import torch
import torch.autograd.profiler as P

model = torch.nn.Linear(512, 512).cuda()
x = torch.randn(1, 512, device="cuda")

# Warm-up (pre-NVTX: these should be detected as initialization kernels)
for _ in range(2):
    _ = model(x)
torch.cuda.synchronize()

# Measurement window under emit_nvtx (NVTX attribution fires here)
with P.emit_nvtx():
    for _ in range(3):
        _ = model(x)
torch.cuda.synchronize()
"""

_COMPLEX_WORKLOAD_SCRIPT = """\
import torch
import torch.nn as nn
import torch.autograd.profiler as P

class MLPBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(512)

    def forward(self, x):
        return self.norm(x + self.fc2(self.act(self.fc1(x))))

model = MLPBlock().cuda()
x = torch.randn(8, 512, device="cuda")

# Warm-up
for _ in range(3):
    _ = model(x)
torch.cuda.synchronize()

# Measurement window
with P.emit_nvtx():
    for _ in range(3):
        _ = model(x)
torch.cuda.synchronize()
"""


# ---------------------------------------------------------------------------
# Module-scoped fixture: runs the full pipeline once
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def full_pipeline_profile(tmp_path_factory):
    """
    Run the complete 6-stage preprocessing pipeline on real hardware.

    Returns (profile, nsys_rep, manifest, operator_records) so individual
    tests can assert at different pipeline depths.
    """
    if not _cuda_available():
        pytest.skip("CUDA not available (torch not installed or no GPU)")
    if not _nsys_available():
        pytest.skip("nsys not available on PATH")
    ncu_exe = _ncu_executable()
    if ncu_exe is None:
        pytest.skip("ncu not available on PATH or fallback path")

    tmp = tmp_path_factory.mktemp("pipeline_e2e")

    # Write the workload script
    script = tmp / "workload.py"
    script.write_text(_WORKLOAD_SCRIPT)

    # ------------------------------------------------------------------
    # Stage 1: nsys capture
    # ------------------------------------------------------------------
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
        capture_output=True,
    )

    # nsys may not append .nsys-rep in all versions — find it
    if not nsys_rep.exists():
        candidates = list(tmp.glob("*.nsys-rep"))
        if not candidates:
            pytest.fail(f"nsys produced no .nsys-rep file in {tmp}")
        nsys_rep = candidates[0]

    # ------------------------------------------------------------------
    # Stage 2: nsys export → SQLite
    # ------------------------------------------------------------------
    sqlite_path = tmp / "profile.sqlite"

    subprocess.run(
        [
            "nsys", "export",
            "--type=sqlite",
            f"--output={sqlite_path}",
            "--force-overwrite=true",
            str(nsys_rep),
        ],
        check=True,
        timeout=60,
        capture_output=True,
    )

    if not sqlite_path.exists():
        candidates = list(tmp.glob("*.sqlite"))
        if not candidates:
            pytest.fail(f"nsys export produced no .sqlite file in {tmp}")
        sqlite_path = candidates[0]

    # ------------------------------------------------------------------
    # Stage 3: Manifest build (real nsys I/O — no mocks)
    # ------------------------------------------------------------------
    import torch
    from operator_profiler.mapper.manifest_builder import ManifestBuilder
    from operator_profiler.schema.manifest import CaptureManifestMetadata

    metadata = CaptureManifestMetadata(
        model_name="E2ETest",
        torch_version=torch.__version__,
        compile_mode="eager",
        nsys_report_path=str(nsys_rep),
        capture_timestamp_utc="2026-04-01T00:00:00+00:00",
    )
    manifest = ManifestBuilder(nsys_rep_path=nsys_rep, metadata=metadata).build()

    # ------------------------------------------------------------------
    # Stage 4: Attribution engine
    # ------------------------------------------------------------------
    from operator_profiler.mapper.attribution_engine import AttributionEngine

    engine = AttributionEngine(manifest)
    operator_records, unattributed = engine.run()

    # ------------------------------------------------------------------
    # Stage 5: ncu kernel profiling — populates kernel metrics in-place
    # ------------------------------------------------------------------
    import site
    from operator_profiler.mapper.range_replay import RangeReplayConfig, RangeReplayOrchestrator

    # ncu needs GPU counter access (sudo) and the user's Python packages
    # (PYTHONPATH) when running under a different user context.
    user_site = next(
        (p for p in site.getsitepackages()
         if "dist-packages" in p or "site-packages" in p),
        "",
    )
    local_site = site.getusersitepackages()
    pythonpath = ":".join(filter(None, [local_site, user_site]))

    replay_config = RangeReplayConfig(
        replay_script=script,
        output_dir=str(tmp),
        ncu_executable=ncu_exe,
        ncu_sudo=True,
        ncu_extra_env={"PYTHONPATH": pythonpath},
    )
    orch = RangeReplayOrchestrator(manifest, operator_records, replay_config)
    ncu_output_dir = orch.run()

    # ------------------------------------------------------------------
    # Stage 6: build_profile — calls build_aggregated_metrics internally
    # ------------------------------------------------------------------
    from operator_profiler.aggregator.profile_builder import build_profile

    profile = build_profile(
        manifest=manifest,
        operator_records=operator_records,
        unattributed_kernels=unattributed,
        model_name="E2ETest",
        torch_version=torch.__version__,
        ncu_report_path=str(ncu_output_dir),
    )

    profile_json_path = tmp / "profile.json"
    profile_json_path.write_text(profile.model_dump_json(indent=2))
    print(f"\nProfile JSON written to: {profile_json_path}")

    return profile, nsys_rep, manifest, operator_records


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestPreprocessingPipelineIntegration:

    def test_nsys_rep_exists(self, full_pipeline_profile):
        """Stage 1: nsys produced a non-empty .nsys-rep capture file."""
        _, nsys_rep, _, _ = full_pipeline_profile
        assert nsys_rep.exists(), f"Expected .nsys-rep at {nsys_rep}"
        assert nsys_rep.stat().st_size > 0, ".nsys-rep file is empty"

    def test_manifest_has_attributed_kernels(self, full_pipeline_profile):
        """
        Stage 3: ManifestBuilder produced a manifest with attributed kernels.

        Verifies:
        - Total kernel count > 0
        - NVTX attribution rate > 50% in the non-warmup capture window
        - At least one aten::mm / aten::linear / aten::addmm attribution
        """
        from operator_profiler.schema.profile import AttributionMethod

        _, _, manifest, _ = full_pipeline_profile
        assert len(manifest.kernels) > 0, "Manifest has no kernel entries"

        # Split warm-up vs capture kernels using manifest warnings
        warmup_ids: set[str] = set()
        for w in manifest.warnings:
            if "initialization kernel" in w or "warm-up outlier" in w:
                warmup_ids.add(w.split(" ")[0])

        capture_kernels = [k for k in manifest.kernels if k.kernel_id not in warmup_ids]
        assert len(capture_kernels) > 0, (
            "All kernels were classified as warm-up — no capture window kernels found"
        )

        nvtx_count = sum(
            1 for k in capture_kernels
            if k.attribution.method == AttributionMethod.NVTX
        )
        nvtx_rate = nvtx_count / len(capture_kernels)
        assert nvtx_rate > 0.5, (
            f"NVTX attribution rate too low: {nvtx_rate:.1%} "
            f"({nvtx_count}/{len(capture_kernels)} capture kernels). "
            "Check that emit_nvtx() NVTX ranges overlap with kernel timestamps."
        )

        linear_ops = {"aten::mm", "aten::linear", "aten::addmm"}
        # emit_nvtx(record_shapes=True) appends metadata after the op name:
        # "aten::addmm, seq = 5, op_id = 5, sizes = ..."
        # Strip to the base op name for matching.
        attributed_base_ops = {
            op.split(",")[0].strip()
            for k in manifest.kernels
            for op in k.attribution.source_operators
        }
        assert attributed_base_ops & linear_ops, (
            f"No linear/mm attribution found. Base ops: {sorted(attributed_base_ops)[:20]}"
        )

    def test_attribution_engine_produces_operator_records(self, full_pipeline_profile):
        """
        Stage 4: AttributionEngine grouped kernels into OperatorRecords.

        Verifies:
        - At least one operator produced
        - A linear-family operator is present
        - operator_id format is valid (no spaces, has call index suffix)
        - All call_index values are non-negative
        """
        _, _, _, operator_records = full_pipeline_profile
        assert len(operator_records) > 0, "AttributionEngine returned no operator records"

        linear_ops = {"aten::mm", "aten::linear", "aten::addmm"}
        # Strip emit_nvtx metadata suffix to get the base op name
        base_op_names = {op.operator_name.split(",")[0].strip() for op in operator_records}
        assert base_op_names & linear_ops, (
            f"No linear/mm operator found. Operators: {sorted(base_op_names)}"
        )

        for op in operator_records:
            # operator_id is "{operator_name}_{call_index}"; the name itself may
            # contain spaces when emit_nvtx appends shape metadata — that is fine.
            # Just verify it ends with the call_index suffix.
            assert op.operator_id.endswith(f"_{op.call_index}"), (
                f"operator_id '{op.operator_id}' does not end with '_{op.call_index}'"
            )
            assert op.call_index >= 0, (
                f"Negative call_index {op.call_index} for {op.operator_name}"
            )

    def test_ncu_metrics_populated_for_linear_operators(self, full_pipeline_profile):
        """
        Stage 5 (critical): After ncu kernel profiling, at least one kernel
        attributed to a linear/mm operator must have a valid achieved_occupancy
        value (> 0).

        Note: dram__bytes_read.sum returns "n/a" on Blackwell (CC 12.0), so we
        assert on achieved_occupancy which is available on all supported GPUs.

        This is the primary assertion that ncu kernel profiling worked end-to-end.
        """
        profile, _, _, _ = full_pipeline_profile

        linear_ops = {"aten::mm", "aten::linear", "aten::addmm"}
        found_metrics = False

        from operator_profiler.schema.metrics import get_raw_value

        for op in profile.operators:
            if op.operator_name.split(",")[0].strip() not in linear_ops:
                continue
            for kernel in op.kernels:
                occ = get_raw_value(kernel.metrics.raw, "achieved_occupancy")
                if occ is not None:
                    assert occ > 0, (
                        f"achieved_occupancy is 0 for kernel '{kernel.kernel_name}' "
                        f"in operator '{op.operator_name}'"
                    )
                    found_metrics = True
                    break
            if found_metrics:
                break

        assert found_metrics, (
            "No kernel with non-None achieved_occupancy found for any of "
            f"{linear_ops}. ncu kernel profiling may have failed silently."
        )

    def test_aggregated_metrics_correct_per_operator(self, full_pipeline_profile):
        """
        Stage 6: build_profile() called build_aggregated_metrics() correctly.

        Verifies:
        - Every operator has aggregated metrics set
        - kernel_count matches the number of kernels in the record
        - total_duration_ns > 0 for all operators
        - dominant_kernel_id refers to a real kernel in the operator
        - linear/mm operators with ncu metrics have mean_achieved_occupancy > 0
        """
        profile, _, _, _ = full_pipeline_profile

        assert len(profile.operators) > 0, "Profile has no operators"

        linear_ops = {"aten::mm", "aten::linear", "aten::addmm"}

        for op in profile.operators:
            assert op.aggregated is not None, (
                f"Operator '{op.operator_name}' has no aggregated metrics"
            )
            assert op.aggregated.kernel_count == len(op.kernels), (
                f"kernel_count mismatch for '{op.operator_name}': "
                f"aggregated={op.aggregated.kernel_count}, actual={len(op.kernels)}"
            )
            assert op.aggregated.total_duration_ns > 0, (
                f"total_duration_ns is 0 for '{op.operator_name}'"
            )
            kernel_ids = {k.kernel_id for k in op.kernels}
            assert op.aggregated.dominant_kernel_id in kernel_ids, (
                f"dominant_kernel_id '{op.aggregated.dominant_kernel_id}' not found "
                f"in kernels for '{op.operator_name}'"
            )

            if op.operator_name.split(",")[0].strip() in linear_ops:
                from operator_profiler.schema.metrics import get_raw_value
                has_ncu = any(
                    get_raw_value(k.metrics.raw, "achieved_occupancy") is not None
                    for k in op.kernels
                )
                if has_ncu:
                    assert op.aggregated.mean_achieved_occupancy is not None, (
                        f"mean_achieved_occupancy is None for '{op.operator_name}' "
                        "despite kernels having achieved_occupancy set"
                    )
                    assert op.aggregated.mean_achieved_occupancy > 0, (
                        f"mean_achieved_occupancy is 0 for '{op.operator_name}'"
                    )

    def test_unattributed_kernels_not_in_operator_records(self, full_pipeline_profile):
        """
        Every kernel_id appears in exactly one of: operator kernels or unattributed_kernels.
        No kernel should be silently dropped and none should appear in both lists.
        """
        profile, _, manifest, _ = full_pipeline_profile

        op_kernel_ids: set[str] = {
            k.kernel_id
            for op in profile.operators
            for k in op.kernels
        }
        unattr_ids: set[str] = {k.kernel_id for k in profile.unattributed_kernels}

        overlap = op_kernel_ids & unattr_ids
        assert not overlap, (
            f"Kernel IDs appear in both operator records and unattributed_kernels: "
            f"{sorted(overlap)[:5]}"
        )

        # Also verify all non-warmup manifest kernels are accounted for
        warmup_ids: set[str] = set()
        for w in manifest.warnings:
            if "initialization kernel" in w or "warm-up outlier" in w:
                warmup_ids.add(w.split(" ")[0])

        accounted = op_kernel_ids | unattr_ids | warmup_ids
        all_manifest_ids = {k.kernel_id for k in manifest.kernels}
        missing = all_manifest_ids - accounted
        assert not missing, (
            f"{len(missing)} kernel(s) from the manifest are unaccounted for "
            f"(not in operators, unattributed, or warmup): {sorted(missing)[:5]}"
        )

    def test_profile_schema_version_and_metadata(self, full_pipeline_profile):
        """Stage 6: CaptureMetadata and schema_version are propagated correctly."""
        from operator_profiler.schema.profile import SCHEMA_VERSION

        profile, _, _, _ = full_pipeline_profile

        assert profile.schema_version == SCHEMA_VERSION, (
            f"schema_version mismatch: got '{profile.schema_version}', "
            f"expected '{SCHEMA_VERSION}'"
        )
        assert profile.capture_metadata.model_name == "E2ETest"
        assert profile.capture_metadata.torch_version != ""
        assert profile.capture_metadata.compile_mode == "eager"

    def test_profile_json_roundtrip(self, full_pipeline_profile):
        """
        OperatorAttributedProfile survives a JSON serialize → deserialize roundtrip
        with all key fields preserved — including ncu metrics in nested KernelRecords.
        """
        from operator_profiler.schema.profile import OperatorAttributedProfile

        profile, _, _, _ = full_pipeline_profile

        raw = profile.model_dump_json()
        restored = OperatorAttributedProfile.model_validate_json(raw)

        assert restored.schema_version == profile.schema_version
        assert len(restored.operators) == len(profile.operators)
        assert len(restored.unattributed_kernels) == len(profile.unattributed_kernels)

        for orig_op, rest_op in zip(profile.operators, restored.operators):
            assert rest_op.operator_name == orig_op.operator_name
            assert rest_op.operator_id == orig_op.operator_id
            assert rest_op.aggregated is not None
            assert rest_op.aggregated.kernel_count == orig_op.aggregated.kernel_count
            assert rest_op.aggregated.total_duration_ns == orig_op.aggregated.total_duration_ns

            # Verify ncu metrics survive the roundtrip
            for orig_k, rest_k in zip(orig_op.kernels, rest_op.kernels):
                assert rest_k.kernel_id == orig_k.kernel_id
                assert rest_k.metrics.raw == orig_k.metrics.raw
                assert rest_k.attribution_method == orig_k.attribution_method


# ---------------------------------------------------------------------------
# Complex workload fixture + smoke test
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def complex_pipeline_profile(tmp_path_factory):
    """
    Full pipeline on a multi-operator MLP block:
      fc1 (512→2048) → GELU → fc2 (2048→512) → LayerNorm + residual

    Exercises:
      - Multiple operator types (addmm, gelu, layer_norm)
      - GEMM kernels (batch=8, large matrices) rather than GEMV
      - Multiple unique kernel names → multiple ncu profile runs
      - 1:many operator→kernel potential (layer_norm dispatches several kernels)
    """
    if not _cuda_available():
        pytest.skip("CUDA not available")
    if not _nsys_available():
        pytest.skip("nsys not available")
    ncu_exe = _ncu_executable()
    if ncu_exe is None:
        pytest.skip("ncu not available")

    import site
    import torch
    from operator_profiler.mapper.manifest_builder import ManifestBuilder
    from operator_profiler.schema.manifest import CaptureManifestMetadata
    from operator_profiler.mapper.attribution_engine import AttributionEngine
    from operator_profiler.mapper.range_replay import RangeReplayConfig, RangeReplayOrchestrator
    from operator_profiler.aggregator.profile_builder import build_profile

    tmp = tmp_path_factory.mktemp("pipeline_complex")
    script = tmp / "workload.py"
    script.write_text(_COMPLEX_WORKLOAD_SCRIPT)

    output_prefix = str(tmp / "profile")
    nsys_rep = tmp / "profile.nsys-rep"

    subprocess.run(
        ["nsys", "profile", "--trace=cuda,nvtx",
         f"--output={output_prefix}", "--force-overwrite=true",
         sys.executable, str(script)],
        check=True, timeout=120, capture_output=True,
    )
    if not nsys_rep.exists():
        candidates = list(tmp.glob("*.nsys-rep"))
        nsys_rep = candidates[0] if candidates else pytest.fail("no .nsys-rep produced")

    sqlite_path = tmp / "profile.sqlite"
    subprocess.run(
        ["nsys", "export", "--type=sqlite",
         f"--output={sqlite_path}", "--force-overwrite=true", str(nsys_rep)],
        check=True, timeout=60, capture_output=True,
    )
    if not sqlite_path.exists():
        candidates = list(tmp.glob("*.sqlite"))
        sqlite_path = candidates[0] if candidates else pytest.fail("no .sqlite produced")

    metadata = CaptureManifestMetadata(
        model_name="MLPBlock",
        torch_version=torch.__version__,
        compile_mode="eager",
        nsys_report_path=str(nsys_rep),
        capture_timestamp_utc="2026-04-01T00:00:00+00:00",
    )
    manifest = ManifestBuilder(nsys_rep_path=nsys_rep, metadata=metadata).build()
    engine = AttributionEngine(manifest)
    operator_records, unattributed = engine.run()

    user_site = next(
        (p for p in site.getsitepackages() if "dist-packages" in p or "site-packages" in p), ""
    )
    pythonpath = ":".join(filter(None, [site.getusersitepackages(), user_site]))
    replay_config = RangeReplayConfig(
        replay_script=script,
        output_dir=str(tmp),
        ncu_executable=ncu_exe,
        ncu_sudo=True,
        ncu_extra_env={"PYTHONPATH": pythonpath},
    )
    orch = RangeReplayOrchestrator(manifest, operator_records, replay_config)
    ncu_output_dir = orch.run()

    profile = build_profile(
        manifest=manifest,
        operator_records=operator_records,
        unattributed_kernels=unattributed,
        model_name="MLPBlock",
        torch_version=torch.__version__,
        ncu_report_path=str(ncu_output_dir),
    )

    profile_json_path = tmp / "profile.json"
    profile_json_path.write_text(profile.model_dump_json(indent=2))
    print(f"\nComplex profile JSON written to: {profile_json_path}")

    return profile, nsys_rep, manifest, operator_records


class TestComplexWorkloadPipeline:

    def test_complex_profile_produced(self, complex_pipeline_profile):
        """Smoke test: profile has multiple operator types and all kernels attributed."""
        profile, _, _, _ = complex_pipeline_profile

        op_base_names = {op.operator_name.split(",")[0].strip() for op in profile.operators}
        assert len(profile.operators) > 3, (
            f"Expected more than 3 operators, got {len(profile.operators)}: {sorted(op_base_names)}"
        )
        assert not profile.unattributed_kernels, (
            f"{len(profile.unattributed_kernels)} kernels went unattributed"
        )
        for op in profile.operators:
            assert op.aggregated is not None
            assert op.aggregated.dominant_kernel_id is not None
            assert op.aggregated.total_duration_ns > 0
