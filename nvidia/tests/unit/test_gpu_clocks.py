"""
Unit tests for utils.gpu_clocks.

Verifies clock-target resolution, the lock/reset command sequence, and the
guaranteed-reset / graceful-degradation behavior of gpu_clocks_locked() — all
with nvidia-smi mocked so the tests run on any machine.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from nvidia.operator_profiler.utils import gpu_clocks
from nvidia.operator_profiler.utils.subprocess_utils import SubprocessError

# Representative `--query-supported-clocks=gr,mem --format=csv,noheader,nounits` output,
# with mid-range graphics steps so a probed sustainable clock snaps to a realistic value.
SUPPORTED_CSV = ("2430, 12481\n2422, 12481\n2100, 12481\n1500, 12481\n"
                 "187, 405\n180, 405\n")


def _ok(stdout: str = ""):
    return SimpleNamespace(stdout=stdout, stderr="", returncode=0)


# ---------------------------------------------------------------------------
# resolve_target_clocks
# ---------------------------------------------------------------------------

def test_resolve_max_returns_highest_supported():
    with patch.object(gpu_clocks, "run_subprocess", return_value=_ok(SUPPORTED_CSV)):
        assert gpu_clocks.resolve_target_clocks("max", gpu_index=0) == (2430, 12481)


def test_resolve_explicit_exact_pair():
    with patch.object(gpu_clocks, "run_subprocess", return_value=_ok(SUPPORTED_CSV)):
        assert gpu_clocks.resolve_target_clocks("2422,12481") == (2422, 12481)


def test_resolve_explicit_snaps_to_nearest_supported():
    with patch.object(gpu_clocks, "run_subprocess", return_value=_ok(SUPPORTED_CSV)):
        # 2400 → nearest supported graphics is 2422; 12000 → 12481
        assert gpu_clocks.resolve_target_clocks("2400,12000") == (2422, 12481)


def test_resolve_bad_spec_raises():
    with patch.object(gpu_clocks, "run_subprocess", return_value=_ok(SUPPORTED_CSV)):
        with pytest.raises(ValueError):
            gpu_clocks.resolve_target_clocks("not-a-freq")


def test_resolve_raises_when_query_fails():
    with patch.object(gpu_clocks, "run_subprocess", side_effect=SubprocessError("boom")):
        with pytest.raises(ValueError):
            gpu_clocks.resolve_target_clocks("max")


# ---------------------------------------------------------------------------
# resolve_target_clocks — probe path (default)
# ---------------------------------------------------------------------------

def test_resolve_probe_snaps_down_to_supported():
    """Default (None / 'probe') probes and snaps graphics DOWN to largest supported ≤ probed."""
    with patch.object(gpu_clocks, "run_subprocess", return_value=_ok(SUPPORTED_CSV)), \
         patch.object(gpu_clocks, "probe_sustainable_clock", return_value=2186):
        # 2186 → largest supported ≤ 2186 is 2100; memory stays at max (12481)
        assert gpu_clocks.resolve_target_clocks("probe") == (2100, 12481)
        assert gpu_clocks.resolve_target_clocks(None) == (2100, 12481)


def test_resolve_probe_unavailable_falls_back_to_max():
    with patch.object(gpu_clocks, "run_subprocess", return_value=_ok(SUPPORTED_CSV)), \
         patch.object(gpu_clocks, "probe_sustainable_clock", return_value=None):
        assert gpu_clocks.resolve_target_clocks("probe") == (2430, 12481)


# ---------------------------------------------------------------------------
# probe-result cache — baseline probes once, optimized reuses
# ---------------------------------------------------------------------------

def test_cache_written_then_reused(tmp_path):
    cache_dir = str(tmp_path)
    with patch.object(gpu_clocks, "run_subprocess", return_value=_ok(SUPPORTED_CSV)):
        # First call: probe runs, writes the cache.
        with patch.object(gpu_clocks, "probe_sustainable_clock", return_value=2186) as p1:
            first = gpu_clocks.resolve_target_clocks("probe", cache_dir=cache_dir)
            assert first == (2100, 12481)
            assert p1.called
        assert (tmp_path / ".gpu_clock_lock.json").exists()

        # Second call: cache hit → probe must NOT run.
        with patch.object(gpu_clocks, "probe_sustainable_clock",
                          side_effect=AssertionError("probe should not run on cache hit")):
            second = gpu_clocks.resolve_target_clocks("probe", cache_dir=cache_dir)
            assert second == (2100, 12481)


def test_cache_expired_reprobes(tmp_path):
    cache_dir = str(tmp_path)
    (tmp_path / ".gpu_clock_lock.json").write_text(json.dumps({
        "gpu_name": None, "graphics_mhz": 1500, "memory_mhz": 12481,
        "ts": time.time() - (7 * 3600),  # older than the 6 h TTL
    }))
    with patch.object(gpu_clocks, "run_subprocess", return_value=_ok(SUPPORTED_CSV)), \
         patch.object(gpu_clocks, "probe_sustainable_clock", return_value=2186) as p:
        result = gpu_clocks.resolve_target_clocks("probe", cache_dir=cache_dir)
        assert p.called
        assert result == (2100, 12481)  # re-probed, not the stale 1500


def test_cache_gpu_mismatch_reprobes(tmp_path):
    cache_dir = str(tmp_path)
    (tmp_path / ".gpu_clock_lock.json").write_text(json.dumps({
        "gpu_name": "Some Other GPU", "graphics_mhz": 1500, "memory_mhz": 12481,
        "ts": time.time(),
    }))
    with patch.object(gpu_clocks, "run_subprocess", return_value=_ok(SUPPORTED_CSV)), \
         patch.object(gpu_clocks, "_current_gpu_name", return_value="This GPU"), \
         patch.object(gpu_clocks, "probe_sustainable_clock", return_value=2186) as p:
        result = gpu_clocks.resolve_target_clocks("probe", cache_dir=cache_dir)
        assert p.called
        assert result == (2100, 12481)


# ---------------------------------------------------------------------------
# lock_gpu_clocks
# ---------------------------------------------------------------------------

def test_lock_issues_lgc_and_lmc_with_sudo():
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return _ok()

    with patch.object(gpu_clocks, "run_subprocess", side_effect=fake_run):
        assert gpu_clocks.lock_gpu_clocks(2430, 12481, gpu_index=0, use_sudo=True) is True

    joined = [" ".join(c) for c in calls]
    assert any("-lgc 2430" in j and j.startswith("sudo -n nvidia-smi") for j in joined)
    assert any("-lmc 12481" in j and j.startswith("sudo -n nvidia-smi") for j in joined)


def test_lock_returns_false_on_failure_without_raising():
    def fake_run(cmd, **kwargs):
        if "-lgc" in cmd:
            raise SubprocessError("no permission")
        return _ok()

    with patch.object(gpu_clocks, "run_subprocess", side_effect=fake_run):
        assert gpu_clocks.lock_gpu_clocks(2430, 12481, use_sudo=True) is False


# ---------------------------------------------------------------------------
# gpu_clocks_locked context manager
# ---------------------------------------------------------------------------

def test_context_locks_then_resets():
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return _ok(SUPPORTED_CSV if "--query-supported-clocks=gr,mem" in cmd else "")

    with patch.object(gpu_clocks, "run_subprocess", side_effect=fake_run):
        with gpu_clocks.gpu_clocks_locked("max", gpu_index=0, use_sudo=True) as locked:
            assert locked == (2430, 12481)

    joined = " | ".join(" ".join(c) for c in calls)
    assert "-lgc 2430" in joined and "-lmc 12481" in joined
    assert "-rgc" in joined and "-rmc" in joined


def test_context_resets_even_when_body_raises():
    reset_seen = {"rgc": False, "rmc": False}

    def fake_run(cmd, **kwargs):
        if "-rgc" in cmd:
            reset_seen["rgc"] = True
        if "-rmc" in cmd:
            reset_seen["rmc"] = True
        return _ok(SUPPORTED_CSV if "--query-supported-clocks=gr,mem" in cmd else "")

    with patch.object(gpu_clocks, "run_subprocess", side_effect=fake_run):
        with pytest.raises(RuntimeError):
            with gpu_clocks.gpu_clocks_locked("max"):
                raise RuntimeError("capture blew up")

    assert reset_seen == {"rgc": True, "rmc": True}


def test_context_disabled_does_not_touch_clocks():
    with patch.object(gpu_clocks, "run_subprocess") as m:
        with gpu_clocks.gpu_clocks_locked("max", enabled=False) as locked:
            assert locked is None
        m.assert_not_called()


def test_context_skips_reset_when_lock_failed():
    """If locking fails (e.g. no permission), the block runs unlocked and no reset is issued."""
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if "-lgc" in cmd:
            raise SubprocessError("no permission")
        return _ok(SUPPORTED_CSV if "--query-supported-clocks=gr,mem" in cmd else "")

    with patch.object(gpu_clocks, "run_subprocess", side_effect=fake_run):
        with gpu_clocks.gpu_clocks_locked("max") as locked:
            assert locked is None

    joined = " | ".join(" ".join(c) for c in calls)
    assert "-rgc" not in joined and "-rmc" not in joined
