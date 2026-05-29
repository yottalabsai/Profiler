"""
Schema round-trip tests — verify that OperatorAttributedProfile and MappingManifest
serialise to JSON and deserialise back to identical objects.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent.parent / "fixtures"


class TestManifestRoundtrip:
    def test_sample_manifest_parses(self):
        from trainium.operator_profiler.schema.manifest import MappingManifest

        raw = (FIXTURES / "sample_manifest.json").read_text()
        manifest = MappingManifest.model_validate_json(raw)
        assert manifest.schema_version == "1.0"
        assert len(manifest.kernels) == 3

    def test_manifest_roundtrip(self):
        from trainium.operator_profiler.schema.manifest import MappingManifest

        raw = (FIXTURES / "sample_manifest.json").read_text()
        manifest = MappingManifest.model_validate_json(raw)
        serialised = manifest.model_dump_json(indent=2)
        manifest2 = MappingManifest.model_validate_json(serialised)
        assert manifest == manifest2

    def test_manifest_attribution_methods(self):
        from trainium.operator_profiler.schema.manifest import MappingManifest
        from trainium.operator_profiler.schema.profile import AttributionMethod

        raw = (FIXTURES / "sample_manifest.json").read_text()
        manifest = MappingManifest.model_validate_json(raw)

        methods = {k.attribution.method for k in manifest.kernels}
        assert AttributionMethod.NRT_CORRELATION in methods
        assert AttributionMethod.UNATTRIBUTED in methods


class TestProfileRoundtrip:
    def test_sample_profile_parses(self):
        from trainium.operator_profiler.schema.profile import OperatorAttributedProfile

        raw = (FIXTURES / "sample_profile.json").read_text()
        profile = OperatorAttributedProfile.model_validate_json(raw)
        assert profile.schema_version == "1.0"
        assert len(profile.operators) == 1
        assert len(profile.unattributed_kernels) == 1

    def test_profile_roundtrip(self):
        from trainium.operator_profiler.schema.profile import OperatorAttributedProfile

        raw = (FIXTURES / "sample_profile.json").read_text()
        profile = OperatorAttributedProfile.model_validate_json(raw)
        serialised = profile.model_dump_json(indent=2)
        profile2 = OperatorAttributedProfile.model_validate_json(serialised)
        assert profile == profile2

    def test_profile_aggregated_metrics_present(self):
        from trainium.operator_profiler.schema.profile import OperatorAttributedProfile

        raw = (FIXTURES / "sample_profile.json").read_text()
        profile = OperatorAttributedProfile.model_validate_json(raw)
        op = profile.operators[0]
        assert op.aggregated is not None
        assert op.aggregated.total_duration_ns == 5000
        assert op.aggregated.kernel_count == 1

    def test_capture_metadata_neuron_fields(self):
        from trainium.operator_profiler.schema.profile import OperatorAttributedProfile

        raw = (FIXTURES / "sample_profile.json").read_text()
        profile = OperatorAttributedProfile.model_validate_json(raw)
        meta = profile.capture_metadata
        assert meta.neuron_sdk_version == "2.18.0"
        assert meta.nrt_session_dir is not None
        assert meta.neuroncore_count == 2

    def test_nvidia_compatible_fields_present(self):
        """Fields that NVIDIA downstream tools read should be present on Trainium profiles."""
        from trainium.operator_profiler.schema.profile import OperatorAttributedProfile

        raw = (FIXTURES / "sample_profile.json").read_text()
        profile = OperatorAttributedProfile.model_validate_json(raw)
        op = profile.operators[0]
        agg = op.aggregated

        # These may be None (no hardware counters yet) but the fields must exist
        assert hasattr(agg, "total_dram_bytes_read")
        assert hasattr(agg, "total_dram_bytes_written")
        assert hasattr(agg, "sm_throughput_pct")
        assert hasattr(agg, "achieved_occupancy")
        assert hasattr(agg, "l1_hit_rate")
        assert hasattr(agg, "l2_hit_rate")
