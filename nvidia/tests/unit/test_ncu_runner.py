"""
Unit tests for ncu_runner.run_kernel_profile().

Verifies ncu command-line construction for application replay mode without
invoking a real GPU. The subprocess is mocked so tests run anywhere.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nvidia.operator_profiler.mapper.ncu_runner import NcuKernelProfileConfig, run_kernel_profile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_capture(config: NcuKernelProfileConfig) -> list[str]:
    """Run run_kernel_profile() with a mocked subprocess; return the captured cmd."""
    captured: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        captured.append(list(cmd))
        # Create the output file so run_kernel_profile doesn't raise on missing path
        Path(config.output_path).touch()

    with patch("nvidia.operator_profiler.mapper.ncu_runner.run_subprocess", side_effect=fake_run):
        run_kernel_profile(config)

    assert len(captured) == 1, "Expected exactly one subprocess call"
    return captured[0]


def _base_config(**overrides) -> NcuKernelProfileConfig:
    defaults: dict = {"script": "/tmp/replay.py"}
    defaults.update(overrides)
    return NcuKernelProfileConfig(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_application_mode_always_present(tmp_path):
    config = _base_config(output_path=str(tmp_path / "out.ncu-rep"))
    cmd = _run_capture(config)
    assert "--replay-mode" in cmd
    idx = cmd.index("--replay-mode")
    assert cmd[idx + 1] == "application", "Must use application replay, not kernel"


def test_no_kernel_name_flag_when_filter_is_none(tmp_path):
    config = _base_config(
        kernel_name_filter=None,
        output_path=str(tmp_path / "out.ncu-rep"),
    )
    cmd = _run_capture(config)
    assert "--kernel-name" not in cmd


def test_kernel_name_flag_present_when_filter_set(tmp_path):
    """_profile_one() path: targeted single-kernel filter should still work."""
    config = _base_config(
        kernel_name_filter="my_fused_kernel",
        output_path=str(tmp_path / "out.ncu-rep"),
    )
    cmd = _run_capture(config)
    assert "--kernel-name" in cmd
    idx = cmd.index("--kernel-name")
    assert cmd[idx + 1] == "my_fused_kernel"


def test_sudo_prefix_and_env_injection(tmp_path):
    config = _base_config(
        use_sudo=True,
        extra_env={"PYTHONPATH": "/home/ubuntu/Profiler"},
        output_path=str(tmp_path / "out.ncu-rep"),
    )
    cmd = _run_capture(config)
    assert cmd[0] == "sudo"
    assert cmd[1] == "env"
    assert "PYTHONPATH=/home/ubuntu/Profiler" in cmd


def test_no_sudo_prefix_when_disabled(tmp_path):
    config = _base_config(
        use_sudo=False,
        output_path=str(tmp_path / "out.ncu-rep"),
    )
    cmd = _run_capture(config)
    assert cmd[0] != "sudo"


def test_metric_set_overrides_metrics_list(tmp_path):
    config = _base_config(
        ncu_metric_set="full",
        metrics=["metric_a", "metric_b"],
        output_path=str(tmp_path / "out.ncu-rep"),
    )
    cmd = _run_capture(config)
    assert "--set" in cmd
    idx = cmd.index("--set")
    assert cmd[idx + 1] == "full"
    assert "--metrics" not in cmd


def test_metrics_list_used_when_no_metric_set(tmp_path):
    config = _base_config(
        ncu_metric_set="",
        metrics=["sm__cycles_active.sum", "dram__bytes_read.sum"],
        output_path=str(tmp_path / "out.ncu-rep"),
    )
    cmd = _run_capture(config)
    assert "--metrics" in cmd
    idx = cmd.index("--metrics")
    assert "sm__cycles_active.sum" in cmd[idx + 1]
    assert "--set" not in cmd


def test_output_path_and_force_overwrite(tmp_path):
    out = str(tmp_path / "report.ncu-rep")
    config = _base_config(output_path=out)
    cmd = _run_capture(config)
    assert "--export" in cmd
    assert out in cmd
    assert "--force-overwrite" in cmd


def test_python_script_invoked_with_sys_executable(tmp_path):
    config = _base_config(
        script="/some/replay.py",
        script_args=["--iters", "10"],
        output_path=str(tmp_path / "out.ncu-rep"),
    )
    cmd = _run_capture(config)
    assert sys.executable in cmd
    assert "/some/replay.py" in cmd
    assert "--iters" in cmd
    assert "10" in cmd
