"""
Integration tests for the full Trainium profiling pipeline.

These tests require a real Trainium instance (trn1/trn2) with torch_neuronx
installed and NRT available.

Run with:
    pytest -m integration trainium/tests/integration/

Excluded by default (no GPU/Trainium required):
    pytest trainium/tests/unit/
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


@pytest.mark.integration
class TestFullCapturePipeline:
    @pytest.fixture
    def simple_workload(self):
        """A minimal linear model on Neuron XLA device."""
        try:
            import torch
            import torch_neuronx
        except ImportError:
            pytest.skip("torch_neuronx not available")

        import torch.nn as nn

        class SimpleModel(nn.Module):
            def forward(self, x):
                return x @ x.T

        device = "xla"
        model = SimpleModel().to(device)
        inputs = torch.randn(4, 4).to(device)
        return model, inputs

    def test_end_to_end_produces_valid_profile(self, simple_workload, tmp_path):
        from trainium.operator_profiler.capture.neuron_capture import run_capture
        from trainium.operator_profiler.mapper.attribution_engine import AttributionEngine
        from trainium.operator_profiler.mapper.manifest_builder import ManifestBuilder
        from trainium.operator_profiler.aggregator.profile_builder import build_profile
        from trainium.operator_profiler.schema.manifest import CaptureManifestMetadata
        from trainium.operator_profiler.schema.profile import OperatorAttributedProfile
        from datetime import datetime, timezone

        model, inputs = simple_workload

        def workload_fn():
            model(inputs)

        capture_result = run_capture(
            workload_fn=workload_fn,
            profile_output_dir=tmp_path / "traces",
            model_name="test_model",
            warmup_iters=1,
            measure_iters=1,
        )

        assert capture_result.trace_json_path.exists()
        assert capture_result.nrt_session_dir.is_dir()

        import torch
        metadata = CaptureManifestMetadata(
            model_name="test_model",
            torch_version=torch.__version__,
            compile_mode="neuron",
            nrt_session_dir=str(capture_result.nrt_session_dir),
            capture_timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )

        builder = ManifestBuilder(
            trace_json_path=capture_result.trace_json_path,
            nrt_session_dir=capture_result.nrt_session_dir,
            metadata=metadata,
        )
        manifest = builder.build()
        assert len(manifest.kernels) > 0

        engine = AttributionEngine(manifest)
        operator_records, unattributed = engine.run()

        profile = build_profile(
            manifest=manifest,
            operator_records=operator_records,
            unattributed_kernels=unattributed,
            model_name="test_model",
            torch_version=torch.__version__,
        )

        # Validate schema
        assert isinstance(profile, OperatorAttributedProfile)
        assert profile.schema_version == "1.0"
        assert len(profile.operators) + len(profile.unattributed_kernels) > 0

        # Serialise / deserialise round-trip
        json_str = profile.model_dump_json(indent=2)
        profile2 = OperatorAttributedProfile.model_validate_json(json_str)
        assert profile == profile2

    def test_attribution_rate_above_threshold(self, simple_workload, tmp_path):
        """At least 50% of device events should be attributed on a simple model."""
        from trainium.operator_profiler.capture.neuron_capture import run_capture
        from trainium.operator_profiler.mapper.attribution_engine import AttributionEngine
        from trainium.operator_profiler.mapper.manifest_builder import ManifestBuilder
        from trainium.operator_profiler.schema.manifest import CaptureManifestMetadata
        from datetime import datetime, timezone
        import torch

        model, inputs = simple_workload

        capture_result = run_capture(
            workload_fn=lambda: model(inputs),
            profile_output_dir=tmp_path / "traces",
            warmup_iters=1,
            measure_iters=1,
        )

        metadata = CaptureManifestMetadata(
            model_name="test",
            torch_version=torch.__version__,
            compile_mode="neuron",
            capture_timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )
        builder = ManifestBuilder(
            trace_json_path=capture_result.trace_json_path,
            nrt_session_dir=capture_result.nrt_session_dir,
            metadata=metadata,
        )
        manifest = builder.build()

        from trainium.operator_profiler.schema.profile import AttributionMethod
        attributed = sum(
            1 for k in manifest.kernels
            if k.attribution.method == AttributionMethod.NRT_CORRELATION
        )
        total = len(manifest.kernels)
        assert total > 0
        assert attributed / total >= 0.5, (
            f"Attribution rate too low: {attributed}/{total} = {attributed/total:.0%}"
        )
