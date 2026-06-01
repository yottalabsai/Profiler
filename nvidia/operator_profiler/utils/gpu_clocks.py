"""
GPU clock locking for reproducible capture timing.

The nsys capture phase measures kernel durations while the workload runs at the
GPU's *dynamic* boost clock, which floats with temperature and power and differs
between two captures (different code → different power/heat, separate processes,
minutes-to-hours apart).  That clock drift is a confound in any baseline-vs-optimized
duration comparison.

This module pins the SM (graphics) and memory clocks to a fixed frequency for the
duration of the capture and guarantees a reset afterwards.  Everything is best-effort:
clock setting requires root, so calls use ``sudo -n`` (non-interactive) and degrade to
an unlocked run with a WARNING rather than prompting or aborting when permission is
unavailable.

The default target is **probe-and-lock**: a brief synthetic load measures the clock the
GPU actually *sustains* (the power/thermal cap sits below the max boost clock), and the
lock is set at/below that value so it is genuinely held rather than capped.  The probed
result is cached per GPU under the capture's output directory, so the baseline capture
probes once and the optimized capture reuses the *identical* clock — making the two
captures' durations directly comparable.

ncu's replay phase is unaffected — it already self-locks to base clocks and the reported
durations do not come from it.
"""
from __future__ import annotations

import json
import logging
import signal
import statistics
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from nvidia.operator_profiler.utils.subprocess_utils import run_subprocess, SubprocessError

log = logging.getLogger(__name__)

_NVIDIA_SMI = "nvidia-smi"
_CLOCK_CACHE_NAME = ".gpu_clock_lock.json"
_CLOCK_CACHE_TTL_S = 6 * 3600  # 6 h — matches the report skill's cross-session threshold


def _smi_cmd(args: list[str], *, use_sudo: bool) -> list[str]:
    """Build an nvidia-smi command, prefixed with ``sudo -n`` when use_sudo is set."""
    base = [_NVIDIA_SMI, *args]
    return (["sudo", "-n", *base]) if use_sudo else base


def resolve_target_clocks(target: str | None, gpu_index: int = 0,
                          cache_dir: str | None = None) -> tuple[int, int]:
    """
    Resolve a clock target to a concrete ``(graphics_mhz, memory_mhz)`` pair.

    ``target`` may be:
      - ``None`` or ``"probe"`` (default) — measure the clock the GPU sustains under a
        brief synthetic load and lock graphics to the largest supported clock at or below
        it (memory stays at max supported).  When ``cache_dir`` is given, a previously
        probed result for this GPU (< 6 h old) is reused instead of re-probing, so the
        baseline and optimized captures lock to an identical clock.  Falls back to ``max``
        when the probe is unavailable (no torch / CUDA).
      - ``"max"`` — the highest supported graphics and memory clocks.  Note this may sit
        above the sustainable cap, so under load the GPU can run below it.
      - ``"<gr>,<mem>"`` — an explicit pair, snapped to the nearest supported value
        (bypasses the cache).

    Raises ValueError on an unparseable target or when supported clocks cannot be read.
    """
    supported = _query_supported_clocks(gpu_index)
    if not supported:
        raise ValueError(
            "could not read supported clocks from nvidia-smi "
            f"(gpu {gpu_index}); cannot resolve target '{target}'"
        )

    max_gr = max(gr for gr, _ in supported)
    max_mem = max(mem for _, mem in supported)
    norm = (target or "probe").strip().lower()

    if norm == "max":
        return max_gr, max_mem

    if norm == "probe":
        # Reuse a cached probe for this GPU so baseline & optimized lock identically.
        cached = _read_clock_cache(cache_dir, gpu_index) if cache_dir else None
        if cached is not None:
            log.info("Reusing cached sustainable clock %d/%d MHz (no re-probe).", *cached)
            return cached

        probed = probe_sustainable_clock(gpu_index)
        if probed is None:
            log.warning("Clock probe unavailable; falling back to max supported clocks "
                        "(%d/%d MHz).", max_gr, max_mem)
            return max_gr, max_mem

        gr = _largest_supported_at_most([g for g, _ in supported], probed)
        resolved = (gr, max_mem)
        log.info("Probed sustainable graphics clock ~%d MHz; locking to %d MHz "
                 "(graphics) / %d MHz (memory).", probed, gr, max_mem)
        if cache_dir:
            _write_clock_cache(cache_dir, gpu_index, resolved)
        return resolved

    parts = norm.replace(" ", "").split(",")
    if len(parts) != 2 or not all(p.lstrip("-").isdigit() for p in parts):
        raise ValueError(
            f"invalid --lock-clocks-freq '{target}'; "
            "expected 'probe', 'max', or '<gr>,<mem>' in MHz"
        )
    want_gr, want_mem = int(parts[0]), int(parts[1])

    gr = min((g for g, _ in supported), key=lambda g: abs(g - want_gr))
    mem = min((m for _, m in supported), key=lambda m: abs(m - want_mem))
    if gr != want_gr or mem != want_mem:
        log.warning(
            "Requested clocks %d/%d MHz not supported; snapping to nearest %d/%d MHz",
            want_gr, want_mem, gr, mem,
        )
    return gr, mem


def _largest_supported_at_most(values: list[int], ceiling: int) -> int:
    """Largest value <= ceiling; falls back to the minimum when all exceed the ceiling."""
    at_most = [v for v in values if v <= ceiling]
    return max(at_most) if at_most else min(values)


def _query_supported_clocks(gpu_index: int) -> list[tuple[int, int]]:
    """Return the list of supported (graphics_mhz, memory_mhz) pairs, or [] on failure."""
    cmd = [
        _NVIDIA_SMI, "-i", str(gpu_index),
        "--query-supported-clocks=gr,mem", "--format=csv,noheader,nounits",
    ]
    try:
        result = run_subprocess(cmd, description="nvidia-smi query supported clocks",
                                capture_output=True)
    except (SubprocessError, FileNotFoundError) as exc:
        log.warning("Could not query supported GPU clocks: %s", exc)
        return []

    pairs: list[tuple[int, int]] = []
    for line in (result.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        cols = [c.strip() for c in line.split(",")]
        if len(cols) != 2:
            continue
        try:
            pairs.append((int(cols[0]), int(cols[1])))
        except ValueError:
            continue
    return pairs


def _query_current_gr_clock(gpu_index: int) -> int | None:
    """Return the current graphics clock in MHz, or None on failure."""
    cmd = [
        _NVIDIA_SMI, "-i", str(gpu_index),
        "--query-gpu=clocks.gr", "--format=csv,noheader,nounits",
    ]
    try:
        result = run_subprocess(cmd, description="nvidia-smi query current clock",
                                capture_output=True)
    except (SubprocessError, FileNotFoundError):
        return None
    text = (result.stdout or "").strip().splitlines()
    if not text:
        return None
    try:
        return int(text[0].strip())
    except ValueError:
        return None


def probe_sustainable_clock(gpu_index: int = 0, *, warmup_s: float = 1.0,
                            sample_s: float = 1.5, sample_interval_s: float = 0.15) -> int | None:
    """
    Measure the graphics clock the GPU *sustains* under a heavy synthetic load.

    The max boost clock is not holdable under sustained load (the power/thermal cap sits
    below it), so locking to it leaves the clock floating.  This runs a background matmul
    loop to saturate the GPU, samples the graphics clock past an initial warmup window,
    and returns the median steady-state reading — a conservative, holdable frequency.

    Returns the sustained clock in MHz, or None if torch/CUDA is unavailable or the load
    could not be created (caller then falls back to the max supported clock).
    """
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None

    try:
        dev = torch.device(f"cuda:{gpu_index}")
        a = torch.randn(8192, 8192, device=dev)
        b = torch.randn(8192, 8192, device=dev)
    except Exception as exc:  # OOM or device error → caller falls back to max
        log.warning("Clock probe could not allocate load tensors: %s", exc)
        return None

    stop = threading.Event()

    def _load():
        # .item() forces a sync each iteration, keeping continuous pressure on the GPU.
        while not stop.is_set():
            (a @ b).sum().item()

    worker = threading.Thread(target=_load, daemon=True)
    samples: list[int] = []
    try:
        worker.start()
        time.sleep(warmup_s)  # let clocks ramp to steady state before sampling
        deadline = time.monotonic() + sample_s
        while time.monotonic() < deadline:
            c = _query_current_gr_clock(gpu_index)
            if c:
                samples.append(c)
            time.sleep(sample_interval_s)
    finally:
        stop.set()
        worker.join(timeout=5.0)
        del a, b
        torch.cuda.empty_cache()

    if not samples:
        return None
    return int(statistics.median(samples))


def _clock_cache_path(cache_dir: str) -> Path:
    return Path(cache_dir) / _CLOCK_CACHE_NAME


def _current_gpu_name(gpu_index: int) -> str | None:
    try:
        import torch
        return torch.cuda.get_device_name(gpu_index)
    except Exception:
        return None


def _read_clock_cache(cache_dir: str, gpu_index: int) -> tuple[int, int] | None:
    """Return cached (gr, mem) when present, fresh (< TTL), and matching this GPU; else None."""
    path = _clock_cache_path(cache_dir)
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    try:
        gr = int(data["graphics_mhz"])
        mem = int(data["memory_mhz"])
        name = data.get("gpu_name")
        ts = float(data.get("ts", 0))
    except (KeyError, TypeError, ValueError):
        return None

    if time.time() - ts > _CLOCK_CACHE_TTL_S:
        log.debug("Clock cache at %s is stale (> %ds); re-probing.", path, _CLOCK_CACHE_TTL_S)
        return None
    cur_name = _current_gpu_name(gpu_index)
    if cur_name is not None and name is not None and cur_name != name:
        log.debug("Clock cache GPU '%s' != current '%s'; re-probing.", name, cur_name)
        return None
    return gr, mem


def _write_clock_cache(cache_dir: str, gpu_index: int, clocks: tuple[int, int]) -> None:
    """Persist the probed (gr, mem) so sibling captures reuse it.  Failures are non-fatal."""
    path = _clock_cache_path(cache_dir)
    payload = {
        "gpu_name": _current_gpu_name(gpu_index),
        "graphics_mhz": clocks[0],
        "memory_mhz": clocks[1],
        "ts": time.time(),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    except OSError as exc:
        log.debug("Could not write clock cache to %s: %s", path, exc)


def lock_gpu_clocks(gr_mhz: int, mem_mhz: int, *, gpu_index: int = 0,
                    use_sudo: bool = True) -> bool:
    """
    Pin graphics and memory clocks to the given frequencies.

    Best-effort: enables persistence mode, then locks both clock domains.  Returns
    True only if both ``-lgc`` and ``-lmc`` succeeded; returns False (with a WARNING)
    on any failure — never raises.  A False return means the caller should treat the
    run as unlocked and skip the corresponding reset.
    """
    # Persistence is best-effort: locking can still work without it, so a failure here
    # is logged but not fatal to the lock attempt.
    try:
        run_subprocess(_smi_cmd(["-i", str(gpu_index), "-pm", "1"], use_sudo=use_sudo),
                       description="nvidia-smi persistence mode", capture_output=True)
    except (SubprocessError, FileNotFoundError) as exc:
        log.debug("Could not enable persistence mode (continuing): %s", exc)

    try:
        run_subprocess(
            _smi_cmd(["-i", str(gpu_index), "-lgc", str(gr_mhz)], use_sudo=use_sudo),
            description="nvidia-smi lock graphics clock", capture_output=True)
        run_subprocess(
            _smi_cmd(["-i", str(gpu_index), "-lmc", str(mem_mhz)], use_sudo=use_sudo),
            description="nvidia-smi lock memory clock", capture_output=True)
    except (SubprocessError, FileNotFoundError) as exc:
        log.warning(
            "Could not lock GPU clocks to %d/%d MHz (%s); capture will run at "
            "dynamic boost clocks. Durations remain valid but are less reproducible "
            "across captures.", gr_mhz, mem_mhz, exc,
        )
        return False

    log.info("Locked GPU %d clocks to %d MHz (graphics) / %d MHz (memory)",
             gpu_index, gr_mhz, mem_mhz)
    return True


def reset_gpu_clocks(*, gpu_index: int = 0, use_sudo: bool = True) -> None:
    """Restore dynamic (auto-boost) clocks.  Failures are logged, never raised."""
    for args, what in ((["-rgc"], "graphics"), (["-rmc"], "memory")):
        try:
            run_subprocess(_smi_cmd(["-i", str(gpu_index), *args], use_sudo=use_sudo),
                           description=f"nvidia-smi reset {what} clock", capture_output=True)
        except (SubprocessError, FileNotFoundError) as exc:
            log.warning("Could not reset %s clock on GPU %d: %s", what, gpu_index, exc)
    log.info("Reset GPU %d clocks to dynamic", gpu_index)


@contextmanager
def gpu_clocks_locked(target: str | None, *, gpu_index: int = 0,
                      use_sudo: bool = True, enabled: bool = True,
                      cache_dir: str | None = None):
    """
    Context manager that locks GPU clocks for the duration of the block and guarantees
    a reset on exit — including on exception, subprocess timeout, or Ctrl-C.

    Yields the resolved ``(graphics_mhz, memory_mhz)`` tuple when the lock took effect,
    or ``None`` when locking was disabled or failed (the block still runs, unlocked).

    ``cache_dir`` is forwarded to ``resolve_target_clocks`` so a probed clock is reused
    across the baseline and optimized captures of a workload.

    A SIGTERM handler is installed for the lifetime of the block so an external ``kill``
    still triggers a reset before the process dies (SIGKILL cannot be caught — recover
    with ``nvidia-smi -rgc -rmc``).
    """
    locked = False
    resolved: tuple[int, int] | None = None

    if enabled:
        try:
            gr_mhz, mem_mhz = resolve_target_clocks(target, gpu_index, cache_dir=cache_dir)
            locked = lock_gpu_clocks(gr_mhz, mem_mhz, gpu_index=gpu_index, use_sudo=use_sudo)
            if locked:
                resolved = (gr_mhz, mem_mhz)
        except ValueError as exc:
            log.warning("Skipping GPU clock lock: %s", exc)

    # Install a SIGTERM handler that resets clocks before re-raising the default action,
    # so an external kill doesn't leave the GPU pinned.  Restored in finally.
    prev_handler = None
    if locked:
        def _on_sigterm(signum, frame):  # pragma: no cover - signal path
            reset_gpu_clocks(gpu_index=gpu_index, use_sudo=use_sudo)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.raise_signal(signal.SIGTERM)
        try:
            prev_handler = signal.signal(signal.SIGTERM, _on_sigterm)
        except ValueError:
            # signal() only works in the main thread; skip the handler off-main-thread.
            prev_handler = None

    try:
        yield resolved
    finally:
        if locked:
            reset_gpu_clocks(gpu_index=gpu_index, use_sudo=use_sudo)
        if prev_handler is not None:
            try:
                signal.signal(signal.SIGTERM, prev_handler)
            except ValueError:
                pass
