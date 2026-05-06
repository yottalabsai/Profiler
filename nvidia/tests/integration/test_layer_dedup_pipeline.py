"""
Integration test — full end-to-end pipeline with --layer-deduplicate.

REQUIRES: CUDA GPU + nsys + ncu (Nsight Compute) + PyTorch installed.
Skip automatically if any dependency is unavailable.

Pipeline under test:
  1. nsys capture    — run_workload.py --layer-deduplicate on a 3-stage model
  2. nsys export     — .nsys-rep → .sqlite
  3. Manifest build  — ManifestBuilder tags kernels with layer_partition + is_unique_partition
  4. Attribution     — AttributionEngine.run() → operator_records
  5. ncu replay      — KernelProfileOrchestrator with partition_equivalence_map from .part.json
  6. Aggregation     — build_profile() → full metric coverage for unique + duplicate partitions

New features exercised:
  - run_workload.py --layer-deduplicate flag
  - .part.json equivalence-map sidecar
  - ManifestBuilder._tag_layer_partitions()
  - KernelProfileOrchestrator._propagate_partition_metrics()
  - KernelProfileConfig.partition_equivalence_map
"""
from __future__ import annotations

import json
import site
import subprocess
import sys
from pathlib import Path

import pytest

NVIDIA_ROOT = Path(__file__).resolve().parent.parent.parent   # /root/Profiler/nvidia
RUN_WORKLOAD = NVIDIA_ROOT / "scripts" / "run_workload.py"

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Availability helpers (copied verbatim from test_preprocessing_pipeline.py)
# ---------------------------------------------------------------------------

def _nsys_available() -> bool:
    try:
        result = subprocess.run(["nsys", "--version"], capture_output=True, timeout=10)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


_NSYS_FALLBACK_PATHS = [
    "/usr/local/cuda-12.8/bin/nsys",
    "/usr/local/cuda/bin/nsys",
    "/opt/nvidia/nsight-systems/2024.6.2/target-linux-x64/nsys",
]

_NCU_FALLBACK_PATHS = [
    "/opt/nvidia/nsight-compute/2025.1.1/ncu",
    "/usr/local/cuda-12.8/bin/ncu",
    "/usr/local/cuda/bin/ncu",
    "/opt/nvidia/nsight-compute/2025.4.1/ncu",
]


def _nsys_executable() -> str | None:
    import shutil
    found = shutil.which("nsys")
    if found:
        return found
    for p in _NSYS_FALLBACK_PATHS:
        if Path(p).exists():
            return p
    return None


def _ncu_executable() -> str | None:
    import shutil
    found = shutil.which("ncu")
    if found:
        return found
    for p in _NCU_FALLBACK_PATHS:
        if Path(p).exists():
            return p
    return None


def _nsys_available() -> bool:
    exe = _nsys_executable()
    if exe is None:
        return False
    try:
        result = subprocess.run([exe, "--version"], capture_output=True, timeout=10)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


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
# Workload — 3 structurally identical stages so the registry produces exactly
# 1 unique partition + 2 duplicates.
# ---------------------------------------------------------------------------

_DEDUP_WORKLOAD = """\
import torch
import torch.nn as nn
import torch.nn.functional as F

class Stage(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 64)

    def forward(self, x):
        return F.relu(self.fc(x))

class ThreeStageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.stages = nn.ModuleList([Stage() for _ in range(3)])

    def forward(self, x):
        for s in self.stages:
            x = s(x)
        return x

def get_model_and_input():
    model = ThreeStageModel().cuda()
    x = torch.randn(4, 64, device="cuda")
    return model, x
"""

# ---------------------------------------------------------------------------
# Module-scoped fixture: runs the full pipeline once
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dedup_pipeline(tmp_path_factory):
    """
    Run the complete layer-deduplication pipeline on real hardware.

    Returns a dict with all intermediate artifacts so individual tests can
    assert at different pipeline depths.
    """
    if not _cuda_available():
        pytest.skip("CUDA not available")
    nsys_exe = _nsys_executable()
    if nsys_exe is None:
        pytest.skip("nsys not available on PATH or known install locations")
    ncu_exe = _ncu_executable()
    if ncu_exe is None:
        pytest.skip("ncu not available on PATH or known install locations")

    tmp = tmp_path_factory.mktemp("dedup_e2e")

    # Write workload exposing get_model_and_input()
    workload_script = tmp / "dedup_workload.py"
    workload_script.write_text(_DEDUP_WORKLOAD)

    output_prefix = str(tmp / "profile")
    nsys_rep = tmp / "profile.nsys-rep"
    part_json_path = tmp / "profile.part.json"

    # ------------------------------------------------------------------
    # Stage 1: nsys capture via run_workload.py --layer-deduplicate
    # ------------------------------------------------------------------
    subprocess.run(
        [
            nsys_exe, "profile",
            "--trace=cuda,nvtx",
            f"--output={output_prefix}",
            "--force-overwrite=true",
            sys.executable,
            str(RUN_WORKLOAD),
            "--workload", str(workload_script),
            "--layer-deduplicate",
            "--output-prefix", output_prefix,
            "--warmup-iters", "2",
            "--measure-iters", "3",
        ],
        check=True,
        timeout=300,
        capture_output=True,
    )

    if not nsys_rep.exists():
        candidates = list(tmp.glob("*.nsys-rep"))
        if not candidates:
            pytest.fail(f"nsys produced no .nsys-rep file in {tmp}")
        nsys_rep = candidates[0]

    # ------------------------------------------------------------------
    # Stage 2 / 3: Manifest build — exports .nsys-rep → .sqlite internally,
    # then tags kernels with layer_partition + is_unique_partition.
    # ------------------------------------------------------------------
    import torch
    from nvidia.operator_profiler.mapper.manifest_builder import ManifestBuilder
    from nvidia.operator_profiler.schema.manifest import CaptureManifestMetadata

    metadata = CaptureManifestMetadata(
        model_name="DedupE2ETest",
        torch_version=torch.__version__,
        compile_mode="inductor",
        nsys_report_path=str(nsys_rep),
        capture_timestamp_utc="2026-01-01T00:00:00+00:00",
    )
    manifest = ManifestBuilder(nsys_rep_path=nsys_rep, metadata=metadata, nsys_executable=nsys_exe).build()

    # ------------------------------------------------------------------
    # Stage 4: Attribution engine
    # ------------------------------------------------------------------
    from nvidia.operator_profiler.mapper.attribution_engine import AttributionEngine

    engine = AttributionEngine(manifest)
    operator_records, unattributed = engine.run()

    # ------------------------------------------------------------------
    # Stage 5: ncu replay with partition_equivalence_map from .part.json
    # ------------------------------------------------------------------
    if not part_json_path.exists():
        # .part.json may be named differently — search for it
        candidates = list(tmp.glob("*.part.json"))
        if not candidates:
            pytest.fail(
                f".part.json sidecar not found in {tmp}. "
                "run_workload.py --layer-deduplicate should write it."
            )
        part_json_path = candidates[0]

    equiv_map: dict[str, str] = json.loads(part_json_path.read_text())

    user_site = next(
        (p for p in site.getsitepackages()
         if "dist-packages" in p or "site-packages" in p),
        "",
    )
    local_site = site.getusersitepackages()
    pythonpath = ":".join(filter(None, [local_site, user_site]))

    from nvidia.operator_profiler.mapper.kernel_profiler import (
        KernelProfileConfig,
        KernelProfileOrchestrator,
    )

    # ncu re-runs the SAME run_workload.py --layer-deduplicate script so that
    # the kernel launch sequence during ncu replay matches the nsys-captured
    # sequence — required for correct invocation-order matching in _merge_metrics.
    ncu_replay_prefix = str(tmp / "ncu_replay")
    replay_config = KernelProfileConfig(
        replay_script=RUN_WORKLOAD,
        replay_script_args=[
            "--workload", str(workload_script),
            "--layer-deduplicate",
            "--output-prefix", ncu_replay_prefix,
            "--warmup-iters", "2",
            "--measure-iters", "3",
        ],
        output_dir=str(tmp),
        ncu_executable=ncu_exe,
        ncu_sudo=True,
        ncu_extra_env={"PYTHONPATH": pythonpath},
        partition_equivalence_map=equiv_map,
    )
    orch = KernelProfileOrchestrator(manifest, operator_records, replay_config)
    ncu_output_dir = orch.run()

    # ------------------------------------------------------------------
    # Stage 6: build_profile
    # ------------------------------------------------------------------
    from nvidia.operator_profiler.aggregator.profile_builder import build_profile

    profile = build_profile(
        manifest=manifest,
        operator_records=operator_records,
        unattributed_kernels=unattributed,
        model_name="DedupE2ETest",
        torch_version=torch.__version__,
        ncu_report_path=str(ncu_output_dir),
    )

    profile_json_path = tmp / "profile.json"
    profile_json_path.write_text(profile.model_dump_json(indent=2))
    print(f"\nProfile JSON written to: {profile_json_path}")

    return {
        "profile": profile,
        "nsys_rep": nsys_rep,
        "manifest": manifest,
        "operator_records": operator_records,
        "unattributed": unattributed,
        "part_json_path": part_json_path,
        "equiv_map": equiv_map,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLayerDedupPipeline:

    def test_nsys_rep_and_part_json_written(self, dedup_pipeline):
        """Stage 1: nsys + run_workload.py --layer-deduplicate produced both sidecars."""
        assert dedup_pipeline["nsys_rep"].exists(), "Missing .nsys-rep"
        assert dedup_pipeline["nsys_rep"].stat().st_size > 0, ".nsys-rep is empty"
        assert dedup_pipeline["part_json_path"].exists(), "Missing .part.json sidecar"

    def test_part_json_structure(self, dedup_pipeline):
        """
        Stage 1: .part.json has the expected structure.

        For a 3-stage model with 3 identical layers the registry produces:
          - 1 unique representative (e.g. "modules_0")
          - 2 duplicates pointing at it (e.g. "modules_1" → "modules_0")
        """
        equiv_map = dedup_pipeline["equiv_map"]
        assert isinstance(equiv_map, dict), f"equiv_map is not a dict: {type(equiv_map)}"
        assert len(equiv_map) == 2, (
            f"Expected 2 duplicate entries in .part.json, got {len(equiv_map)}: {equiv_map}"
        )
        # All values must point to the same unique representative
        unique_labels = set(equiv_map.values())
        assert len(unique_labels) == 1, (
            f"Expected all duplicates to share one unique rep, got: {unique_labels}"
        )

    def test_manifest_partition_tagging(self, dedup_pipeline):
        """
        Stage 3: ManifestBuilder._tag_layer_partitions() correctly tags kernels.

        Unique-partition kernels: is_unique_partition=True, layer_partition set.
        Duplicate-partition kernels: is_unique_partition=False, layer_partition set.
        The duplicate count should be approximately 2× the unique count.
        """
        manifest = dedup_pipeline["manifest"]

        unique_kernels = [
            k for k in manifest.kernels
            if k.is_unique_partition and k.layer_partition is not None
        ]
        dup_kernels = [
            k for k in manifest.kernels
            if not k.is_unique_partition and k.layer_partition is not None
        ]

        assert len(unique_kernels) > 0, (
            "No kernels tagged as is_unique_partition=True. "
            "Check that 'layer::unique::' NVTX ranges were emitted in the nsys trace."
        )
        assert len(dup_kernels) > 0, (
            "No kernels tagged as is_unique_partition=False with a partition label. "
            "Check that 'layer::duplicate::' NVTX ranges were emitted."
        )
        # 2 duplicates per unique for a 3-stage model → duplicate count ≈ 2× unique
        ratio = len(dup_kernels) / len(unique_kernels)
        assert 1.5 <= ratio <= 2.5, (
            f"Expected duplicate:unique kernel ratio ≈ 2.0, got {ratio:.2f} "
            f"({len(dup_kernels)} dup, {len(unique_kernels)} unique)"
        )

    def test_unique_partition_kernels_have_metrics(self, dedup_pipeline):
        """
        Stage 5: After ncu replay, unique-partition kernels have metrics populated.

        Builds a lookup {kernel_id → KernelRecord} from operator_records and checks
        that kernels whose manifest entry has is_unique_partition=True have non-None
        metrics (meaning ncu found and assigned hardware counters for them).
        """
        manifest = dedup_pipeline["manifest"]
        operator_records = dedup_pipeline["operator_records"]

        # Build kernel_id → KernelRecord map
        kid_to_kernel = {
            k.kernel_id: k
            for op in operator_records
            for k in op.kernels
        }

        unique_kids_with_metrics = 0
        unique_kids_total = 0

        for entry in manifest.kernels:
            if entry.is_unique_partition and entry.layer_partition is not None:
                unique_kids_total += 1
                kr = kid_to_kernel.get(entry.kernel_id)
                if kr is not None and kr.metrics is not None:
                    unique_kids_with_metrics += 1

        assert unique_kids_total > 0, "No unique-partition kernel entries found in manifest"
        assert unique_kids_with_metrics > 0, (
            f"No unique-partition kernels have ncu metrics "
            f"({unique_kids_total} unique-partition kernel entries in manifest). "
            "Check ncu ran and returned metrics."
        )

    def test_duplicate_partition_metrics_propagated(self, dedup_pipeline):
        """
        Stage 5: _propagate_partition_metrics() copies metrics to duplicate partitions.

        After orch.run(), every kernel in a duplicate partition should have metrics
        (propagated from the structural unique representative).
        """
        manifest = dedup_pipeline["manifest"]
        operator_records = dedup_pipeline["operator_records"]

        kid_to_kernel = {
            k.kernel_id: k
            for op in operator_records
            for k in op.kernels
        }

        dup_kids_with_metrics = 0
        dup_kids_total = 0

        for entry in manifest.kernels:
            if not entry.is_unique_partition and entry.layer_partition is not None:
                dup_kids_total += 1
                kr = kid_to_kernel.get(entry.kernel_id)
                if kr is not None and kr.metrics is not None:
                    dup_kids_with_metrics += 1

        assert dup_kids_total > 0, "No duplicate-partition kernel entries in manifest"
        # At least some duplicate-partition kernels should have propagated metrics
        assert dup_kids_with_metrics > 0, (
            f"No duplicate-partition kernels have metrics after propagation "
            f"({dup_kids_total} duplicate-partition entries). "
            "Check _propagate_partition_metrics() ran."
        )

    def test_full_profile_has_metric_coverage(self, dedup_pipeline):
        """
        Stage 6: build_profile() produces a valid profile with aggregated metrics.

        Verifies:
        - At least one OperatorAttributedProfile entry
        - At least one operator has mean_achieved_occupancy > 0 (real ncu metric)
        - JSON round-trip is clean (no ValidationError)
        """
        profile = dedup_pipeline["profile"]
        from nvidia.operator_profiler.schema.profile import OperatorAttributedProfile

        assert len(profile.operators) > 0, "Profile has no operator entries"

        # At least one operator should have real ncu occupancy data
        occupancy_values = [
            op.aggregated.achieved_occupancy
            for op in profile.operators
            if op.aggregated is not None and op.aggregated.achieved_occupancy is not None
        ]
        assert len(occupancy_values) > 0, (
            "No operator has achieved_occupancy — "
            "check that ncu metrics were populated"
        )
        assert any(v > 0 for v in occupancy_values), (
            f"All occupancy values are zero: {occupancy_values[:5]}"
        )

        # JSON round-trip
        json_str = profile.model_dump_json()
        recovered = OperatorAttributedProfile.model_validate_json(json_str)
        assert len(recovered.operators) == len(profile.operators)

    def test_profile_schema_metadata(self, dedup_pipeline):
        """Stage 6: Profile metadata fields are populated correctly."""
        profile = dedup_pipeline["profile"]
        assert profile.capture_metadata.model_name == "DedupE2ETest"
        assert profile.capture_metadata.torch_version != ""
