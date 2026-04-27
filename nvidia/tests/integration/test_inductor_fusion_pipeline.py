"""
Integration test — inductor fusion attribution pipeline end-to-end.

REQUIRES: CUDA GPU + nsys + PyTorch with Inductor support installed.
Skips automatically if any dependency is unavailable. ncu is NOT required;
the test covers stages 1–4 only (nsys capture → ManifestBuilder → AttributionEngine).

Verifies that:
  - TORCH_COMPILE_DEBUG=1 causes Inductor to write output_code.py trace artifacts
  - parse_inductor_debug_dir() extracts a non-empty, correctly normalized fusion map
  - ManifestBuilder wires the fusion map into the attribution pipeline
  - INDUCTOR_FUSION kernels appear in the manifest with fused_ops populated
  - The INDUCTOR_FUSION pass does not override higher-confidence tiers
  - AttributionEngine correctly propagates fused_ops into OperatorRecord.fused_with
  - All manifest kernels are accounted for (operators + unattributed + warmup)
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Availability helpers (shared pattern with test_preprocessing_pipeline.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Workload script — MLP with activations compiled via torch.compile(inductor).
#
# nn.LSTM routes through cuDNN in PyTorch 2.x and produces no Triton kernels.
# An MLP with ReLU/LayerNorm forces Inductor to emit Triton fused kernels
# (e.g. triton_poi_fused_addmm_relu_0) that appear in output_code.py.
#
# TORCH_COMPILE_DEBUG is injected into the subprocess environment by the
# fixture; Inductor's trace module writes output_code.py to trace.debug_dir.
# ---------------------------------------------------------------------------

_INDUCTOR_WORKLOAD_SCRIPT = """\
import os
import torch
import torch.nn as nn
import torch.autograd.profiler as P

# TORCH_COMPILE_DEBUG=1 is set in the subprocess env by the test fixture.
# Inductor reads it at import time and enables trace output including
# output_code.py.  trace.debug_dir is also set in the env so we redirect
# the trace files to a known location.
_trace_dir = os.environ.get("INDUCTOR_TRACE_DIR", "")
if _trace_dir:
    import torch._inductor.config as _ind_cfg
    _ind_cfg.trace.enabled = True
    _ind_cfg.trace.debug_dir = _trace_dir

class _MLPBlock(nn.Module):
    \"\"\"Two-layer MLP with ReLU and LayerNorm — produces Triton fused kernels.\"\"\"
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, 128)
        self.norm = nn.LayerNorm(128)

    def forward(self, x):
        return self.norm(x + self.fc2(torch.relu(self.fc1(x))))

model = _MLPBlock().cuda().eval()
x = torch.randn(16, 64, 128, device="cuda")

model = torch.compile(model, backend="inductor")

# Warm-up (pre-NVTX): JIT compilation happens here; these kernels should be
# detected as initialization outliers by ManifestBuilder.
with torch.no_grad():
    for _ in range(3):
        _ = model(x)
torch.cuda.synchronize()

# Measurement window under emit_nvtx — NVTX ranges are pushed for aten ops.
with torch.no_grad():
    with P.emit_nvtx():
        for _ in range(3):
            _ = model(x)
torch.cuda.synchronize()
"""


# ---------------------------------------------------------------------------
# Module-scoped fixture: runs stages 1–4 once for all tests in this module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def inductor_pipeline_fixture(tmp_path_factory):
    """
    Run stages 1–4 of the attribution pipeline on a torch.compile workload
    with Inductor debug enabled.

    Returns (manifest, operator_records, unattributed, fusion_map, debug_dir).
    """
    if not _cuda_available():
        pytest.skip("CUDA not available (torch not installed or no GPU)")
    if not _nsys_available():
        pytest.skip("nsys not available on PATH")

    tmp = tmp_path_factory.mktemp("inductor_fusion")

    # Write workload script to a temp file
    script = tmp / "workload.py"
    script.write_text(_INDUCTOR_WORKLOAD_SCRIPT)

    # Inductor trace output (including output_code.py) lands here.
    # TORCH_COMPILE_DEBUG=1 activates trace.enabled; INDUCTOR_TRACE_DIR is
    # read by the workload script to set trace.debug_dir before torch.compile.
    inductor_debug_dir = tmp / "inductor_debug"
    inductor_debug_dir.mkdir()

    # ------------------------------------------------------------------
    # Stage 1: nsys capture — inject TORCH_COMPILE_DEBUG and INDUCTOR_TRACE_DIR
    # so Inductor writes output_code.py trace artifacts to a known location.
    # ------------------------------------------------------------------
    output_prefix = str(tmp / "profile")
    nsys_rep = tmp / "profile.nsys-rep"

    env = os.environ.copy()
    env["TORCH_COMPILE_DEBUG"] = "1"
    env["INDUCTOR_TRACE_DIR"] = str(inductor_debug_dir)

    subprocess.run(
        [
            "nsys", "profile",
            "--trace=cuda,nvtx",
            f"--output={output_prefix}",
            "--force-overwrite=true",
            sys.executable, str(script),
        ],
        check=True,
        timeout=300,
        capture_output=True,
        env=env,
    )

    if not nsys_rep.exists():
        candidates = list(tmp.glob("*.nsys-rep"))
        if not candidates:
            pytest.fail(f"nsys produced no .nsys-rep file in {tmp}")
        nsys_rep = candidates[0]

    # ------------------------------------------------------------------
    # Stage 2: Parse Inductor debug artifacts → fusion map
    # ------------------------------------------------------------------
    import torch
    from nvidia.operator_profiler.capture.inductor_fusion_extractor import (
        parse_inductor_debug_dir,
    )
    from nvidia.operator_profiler.mapper.manifest_builder import ManifestBuilder
    from nvidia.operator_profiler.schema.manifest import CaptureManifestMetadata

    fusion_map = parse_inductor_debug_dir(inductor_debug_dir)

    # ------------------------------------------------------------------
    # Stage 3: Manifest build — passes the real fusion map so INDUCTOR_FUSION
    # attribution is applied in the post-attribution enrichment pass.
    # ------------------------------------------------------------------
    metadata = CaptureManifestMetadata(
        model_name="InductorFusionTest",
        torch_version=torch.__version__,
        compile_mode="inductor",
        nsys_report_path=str(nsys_rep),
        capture_timestamp_utc="2026-04-26T00:00:00+00:00",
    )
    manifest = ManifestBuilder(
        nsys_rep_path=nsys_rep,
        metadata=metadata,
        inductor_fusion_map=fusion_map,
    ).build()

    # ------------------------------------------------------------------
    # Stage 4: Attribution engine
    # ------------------------------------------------------------------
    from nvidia.operator_profiler.mapper.attribution_engine import AttributionEngine

    engine = AttributionEngine(manifest)
    operator_records, unattributed = engine.run()

    print(
        f"\n[inductor_pipeline_fixture] fusion_map entries: {len(fusion_map)}, "
        f"manifest kernels: {len(manifest.kernels)}, "
        f"operator_records: {len(operator_records)}, "
        f"unattributed: {len(unattributed)}"
    )

    return manifest, operator_records, unattributed, fusion_map, inductor_debug_dir


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestInductorFusionPipeline:

    def test_inductor_debug_dir_populated(self, inductor_pipeline_fixture):
        """Stage 1/2: Inductor wrote at least one output_code.py during compilation."""
        _, _, _, _, debug_dir = inductor_pipeline_fixture
        code_files = list(debug_dir.rglob("output_code.py"))
        assert code_files, (
            f"No output_code.py files found under {debug_dir}. "
            "Either TORCHINDUCTOR_CACHE_DIR was not propagated to the subprocess "
            "or torch._inductor.config.debug was not set before torch.compile()."
        )

    def test_fusion_map_nonempty(self, inductor_pipeline_fixture):
        """Stage 2: parse_inductor_debug_dir() extracted at least one kernel entry."""
        _, _, _, fusion_map, _ = inductor_pipeline_fixture
        assert fusion_map, (
            "fusion_map is empty — parse_inductor_debug_dir() found no kernel→ops "
            "pairs in output_code.py. Check that Inductor is writing .run() call "
            "patterns adjacent to the Original ATen comment (no intervening "
            "stream-setup lines that would reset pending_ops)."
        )

    def test_fusion_map_ops_are_aten_namespaced(self, inductor_pipeline_fixture):
        """Stage 2: Every op string in fusion_map is correctly normalized to aten::*."""
        _, _, _, fusion_map, _ = inductor_pipeline_fixture
        aten_re = re.compile(r"^aten::\w+$")
        for kernel_name, ops in fusion_map.items():
            assert ops, f"Empty ops list for kernel '{kernel_name}'"
            for op in ops:
                assert aten_re.match(op), (
                    f"Op '{op}' for kernel '{kernel_name}' is not in aten:: form. "
                    "_normalize_op() may have failed to strip overload suffixes or "
                    "filter non-aten namespaces."
                )

    def test_inductor_fusion_attribution_present(self, inductor_pipeline_fixture):
        """
        Stage 3: At least one manifest kernel was attributed via INDUCTOR_FUSION.

        INDUCTOR_FUSION attribution is applied to unattributed kernels whose names
        appear in the fusion map. If this fails, check that kernel names in the
        manifest match those in fusion_map (short names, no argument suffix).
        """
        from nvidia.operator_profiler.schema.profile import AttributionMethod

        manifest, _, _, _, _ = inductor_pipeline_fixture
        inductor_kernels = [
            k for k in manifest.kernels
            if k.attribution.method == AttributionMethod.INDUCTOR_FUSION
        ]
        assert inductor_kernels, (
            "No INDUCTOR_FUSION kernels found in the manifest. "
            "Either all kernels were already attributed via NVTX (which is fine "
            "but unexpected for an inductor-compiled model with emit_nvtx()) "
            "or the fusion map entries don't match kernel names in the manifest. "
            f"fusion_map keys (first 5): {list(inductor_pipeline_fixture[3])[:5]}"
        )

    def test_inductor_fusion_kernels_have_fused_ops(self, inductor_pipeline_fixture):
        """
        Stage 3: Every INDUCTOR_FUSION kernel has non-empty fused_ops metadata.

        _apply_inductor_fusion() must populate fused_ops when upgrading an
        UNATTRIBUTED kernel to INDUCTOR_FUSION.
        """
        from nvidia.operator_profiler.schema.profile import AttributionMethod

        manifest, _, _, _, _ = inductor_pipeline_fixture
        for k in manifest.kernels:
            if k.attribution.method == AttributionMethod.INDUCTOR_FUSION:
                assert k.attribution.fused_ops, (
                    f"Kernel '{k.kernel_name}' (id={k.kernel_id}) has "
                    "method=INDUCTOR_FUSION but empty fused_ops. "
                    "_apply_inductor_fusion() must set fused_ops when upgrading."
                )

    def test_nvtx_attribution_not_overridden_to_inductor(self, inductor_pipeline_fixture):
        """
        Stage 3: The INDUCTOR_FUSION enrichment pass does not override kernels
        that already have a valid (non-UNATTRIBUTED) attribution.

        A kernel must not simultaneously have method=INDUCTOR_FUSION and
        confidence=HIGH — HIGH confidence comes from the torch.profiler tier
        which the fusion pass must never override.
        """
        from nvidia.operator_profiler.schema.profile import AttributionMethod, Confidence

        manifest, _, _, _, _ = inductor_pipeline_fixture
        violations = [
            k for k in manifest.kernels
            if k.attribution.method == AttributionMethod.INDUCTOR_FUSION
            and k.attribution.confidence == Confidence.HIGH
        ]
        assert not violations, (
            f"{len(violations)} kernel(s) have method=INDUCTOR_FUSION and "
            "confidence=HIGH — the fusion pass incorrectly overrode a HIGH-confidence "
            "attribution. Only UNATTRIBUTED kernels should be upgraded to INDUCTOR_FUSION."
        )

    def test_operator_records_fused_with_nonempty(self, inductor_pipeline_fixture):
        """
        Stage 4: At least one OperatorRecord has is_fused=True and a non-empty
        fused_with list, showing that AttributionEngine._build_operator_records()
        correctly reads fused_ops from the manifest entry.

        For a multi-op fused Triton kernel (e.g. fusing sigmoid + tanh + mul),
        the primary op becomes the operator_name and the remaining ops appear
        in fused_with.
        """
        _, operator_records, _, _, _ = inductor_pipeline_fixture
        fused_ops_records = [
            op for op in operator_records
            if op.is_fused and op.fused_with
        ]
        assert fused_ops_records, (
            "No OperatorRecord has both is_fused=True and non-empty fused_with. "
            "Either no multi-op fused kernels were produced by Inductor, or "
            "_build_operator_records() is not reading fused_ops from manifest entries. "
            f"Total operator records: {len(operator_records)}, "
            f"is_fused=True count: {sum(1 for op in operator_records if op.is_fused)}"
        )

    def test_all_kernels_accounted_for(self, inductor_pipeline_fixture):
        """
        Stage 4: Every non-warmup manifest kernel appears in exactly one of
        operator_records or unattributed_kernels — no kernel is silently dropped
        and none appears in both lists.
        """
        manifest, operator_records, unattributed, _, _ = inductor_pipeline_fixture

        op_kernel_ids: set[str] = {
            k.kernel_id
            for op in operator_records
            for k in op.kernels
        }
        unattr_ids: set[str] = {k.kernel_id for k in unattributed}

        overlap = op_kernel_ids & unattr_ids
        assert not overlap, (
            f"{len(overlap)} kernel_id(s) appear in both operator_records and "
            f"unattributed_kernels: {sorted(overlap)[:5]}"
        )

        warmup_ids: set[str] = set()
        for w in manifest.warnings:
            if "initialization kernel" in w or "warm-up outlier" in w:
                warmup_ids.add(w.split(" ")[0])

        accounted = op_kernel_ids | unattr_ids | warmup_ids
        all_manifest_ids = {k.kernel_id for k in manifest.kernels}
        missing = all_manifest_ids - accounted
        assert not missing, (
            f"{len(missing)} kernel(s) in the manifest are unaccounted for — "
            "not in operator_records, unattributed_kernels, or warmup set: "
            f"{sorted(missing)[:5]}"
        )
