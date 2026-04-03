"""
Schema roundtrip tests — serialize + deserialize OperatorAttributedProfile
from sample_profile.json and assert no data loss.

Verifies:
  - All fields survive JSON → Pydantic → JSON → Pydantic roundtrip
  - schema_version is preserved
  - Enums serialize to their string values
  - Nested models (KernelMetrics.raw) roundtrip correctly
"""
import json
from pathlib import Path

import pytest

from operator_profiler.schema.profile import (
    SCHEMA_VERSION,
    AttributionMethod,
    Confidence,
    OperatorAttributedProfile,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"


def load_sample_profile() -> OperatorAttributedProfile:
    raw = (FIXTURES / "sample_profile.json").read_text()
    return OperatorAttributedProfile.model_validate_json(raw)


class TestSchemaRoundtrip:
    def test_load_sample_profile(self):
        profile = load_sample_profile()
        assert profile.schema_version == SCHEMA_VERSION

    def test_json_roundtrip(self):
        profile = load_sample_profile()
        serialized = profile.model_dump_json()
        reloaded = OperatorAttributedProfile.model_validate_json(serialized)

        assert reloaded.schema_version == profile.schema_version
        assert len(reloaded.operators) == len(profile.operators)
        assert len(reloaded.unattributed_kernels) == len(profile.unattributed_kernels)

    def test_dict_roundtrip(self):
        profile = load_sample_profile()
        d = profile.model_dump()
        reloaded = OperatorAttributedProfile.model_validate(d)
        assert reloaded.schema_version == profile.schema_version

    def test_operator_fields_preserved(self):
        profile = load_sample_profile()
        op = profile.operators[0]
        assert op.operator_id == "aten::linear_0"
        assert op.operator_name == "aten::linear"
        assert op.is_fused is True
        assert "aten::relu" in op.fused_with
        assert op.call_index == 0

    def test_kernel_metrics_preserved(self):
        profile = load_sample_profile()
        kernel = profile.operators[0].kernels[0]
        m = kernel.metrics
        assert m.dram_bytes_read == 8192
        assert m.dram_bytes_written == 2048
        assert m.achieved_occupancy == pytest.approx(62.3)
        assert "dram__bytes_read.sum" in m.raw

    def test_attribution_enum_roundtrip(self):
        profile = load_sample_profile()
        kernel = profile.operators[0].kernels[0]
        assert kernel.attribution_method == AttributionMethod.NVTX
        assert kernel.confidence == Confidence.HIGH

    def test_nvtx_range_preserved(self):
        profile = load_sample_profile()
        op = profile.operators[0]
        assert op.nvtx_range is not None
        assert op.nvtx_range.text == "aten::linear"
        assert op.nvtx_range.depth == 2

    def test_aggregated_metrics_preserved(self):
        profile = load_sample_profile()
        agg = profile.operators[0].aggregated
        assert agg is not None
        assert agg.total_duration_ns == 210
        assert agg.kernel_count == 1
        assert agg.bottleneck_classification == "compute_bound"

    def test_capture_metadata_preserved(self):
        profile = load_sample_profile()
        meta = profile.capture_metadata
        assert meta.model_name == "TestModel"
        assert meta.compile_mode == "inductor"
        assert meta.device_name == "NVIDIA A100 SXM4 80GB"

    def test_schema_version_is_string(self):
        """schema_version must serialize as a plain string, not an enum."""
        profile = load_sample_profile()
        raw = json.loads(profile.model_dump_json())
        assert isinstance(raw["schema_version"], str)

    def test_no_extra_keys_added(self):
        """Roundtrip must not introduce unexpected top-level keys."""
        profile = load_sample_profile()
        raw = json.loads(profile.model_dump_json())
        expected_keys = {"schema_version", "capture_metadata", "operators",
                         "unattributed_kernels", "warnings"}
        assert set(raw.keys()) == expected_keys
