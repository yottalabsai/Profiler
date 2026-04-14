"""
verify_ncu_pipeline.py
======================
End-to-end verification of the ncu range-replay and build_profile stages,
picking up where verify_preprocessing_pipeline.py left off.

Stages verified:
  0. Environment check  — ncu on PATH, CUDA available
  1. nsys capture       — run workload under nsys with emit_nvtx
  2. ManifestBuilder    — build MappingManifest + AttributionEngine
  3. ncu kernel profile   — KernelProfileOrchestrator collects KernelMetrics
  4. build_profile      — assemble OperatorAttributedProfile with metrics
  5. Metrics check      — assert metrics are non-empty, bottleneck classified

Usage:
    python scripts/verify_ncu_pipeline.py
    python scripts/verify_ncu_pipeline.py --sqlite <path> --workload <path>
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SEP = "=" * 72


def _header(stage: int, title: str) -> None:
    print(f"\n{SEP}")
    print(f"  Stage {stage}: {title}")
    print(SEP)


def _ok(msg: str) -> None:   print(f"  [PASS]  {msg}")
def _fail(msg: str) -> None: print(f"  [FAIL]  {msg}")
def _warn(msg: str) -> None: print(f"  [WARN]  {msg}")
def _info(msg: str) -> None: print(f"          {msg}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", metavar="PATH",
                        help="Skip stage 1 and use an existing .sqlite export.")
    parser.add_argument("--workload", metavar="PATH", default=None,
                        help="Workload script (default: scripts/workload.py).")
    parser.add_argument("--ncu-executable", default="ncu",
                        help="Path to ncu executable.")
    args = parser.parse_args(argv)

    overall: list[tuple[str, bool]] = []

    workload_script = Path(args.workload) if args.workload else ROOT / "scripts" / "workload.py"

    # -----------------------------------------------------------------------
    # Stage 0 — environment check
    # -----------------------------------------------------------------------
    _header(0, "Environment check")

    ncu_bin = shutil.which(args.ncu_executable) or shutil.which("ncu")
    if ncu_bin:
        try:
            ver = subprocess.run([ncu_bin, "--version"], capture_output=True, text=True)
            _ok(f"ncu found: {ncu_bin}")
            _info(ver.stdout.strip().splitlines()[0] if ver.stdout else "(version unknown)")
        except Exception as exc:
            _warn(f"ncu found but --version failed: {exc}")
    else:
        _fail("ncu not found on PATH")
        overall.append(("env_check", False))
        return _print_summary(overall)

    nsys_bin = shutil.which("nsys")
    if not nsys_bin and not args.sqlite:
        _fail("nsys not found — pass --sqlite to skip capture")
        overall.append(("env_check", False))
        return _print_summary(overall)
    if nsys_bin:
        _ok(f"nsys found: {nsys_bin}")

    import torch
    _ok(f"torch {torch.__version__}  |  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        _info(f"GPU: {torch.cuda.get_device_name(0)}")
    overall.append(("env_check", True))

    # -----------------------------------------------------------------------
    # Stage 1 — nsys capture (or load existing SQLite)
    # -----------------------------------------------------------------------
    sqlite_path: Path | None = None

    if args.sqlite:
        sqlite_path = Path(args.sqlite)
        _header(1, "nsys capture  (SKIPPED — using existing SQLite)")
        _info(f"SQLite: {sqlite_path}")
        overall.append(("nsys_capture", True))
    else:
        _header(1, f"nsys capture  ({workload_script.name})")
        out_dir = Path(tempfile.mkdtemp(prefix="op_profiler_ncu_"))
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = workload_script.stem
        rep_path = out_dir / f"{stem}.nsys-rep"
        sqlite_path = out_dir / f"{stem}.sqlite"

        cmd = [
            "nsys", "profile",
            "--trace=cuda,nvtx",
            f"--output={out_dir / stem}",
            "--force-overwrite=true",
            sys.executable,
            str(ROOT / "scripts" / "run_workload.py"),
            "--workload", str(workload_script),
            "--compile-backend", "inductor",
            "--warmup-iters", "5",
        ]
        _info(f"cmd: {' '.join(cmd)}")
        t0 = time.perf_counter()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            elapsed = time.perf_counter() - t0
            if result.returncode != 0:
                _fail(f"nsys profile exited {result.returncode}")
                _info(result.stderr[-600:] if result.stderr else "(no stderr)")
                overall.append(("nsys_capture", False))
                return _print_summary(overall)

            # nsys may create the file with or without .nsys-rep extension
            if not rep_path.exists():
                candidates = list(out_dir.glob("*.nsys-rep"))
                rep_path = candidates[0] if candidates else rep_path

            # Export to SQLite
            export_cmd = [
                "nsys", "export", "--type=sqlite",
                f"--output={sqlite_path}", "--force-overwrite=true",
                str(rep_path),
            ]
            subprocess.run(export_cmd, capture_output=True, check=True, timeout=120)

            _ok(f"Capture + export in {elapsed:.1f}s  →  {sqlite_path.name}")
            overall.append(("nsys_capture", True))
        except Exception as exc:
            _fail(f"nsys capture/export failed: {exc}")
            overall.append(("nsys_capture", False))
            return _print_summary(overall)

    # -----------------------------------------------------------------------
    # Stage 2 — ManifestBuilder + AttributionEngine
    # -----------------------------------------------------------------------
    _header(2, "ManifestBuilder + AttributionEngine")

    from operator_profiler.mapper.nsys_export import query_kernels, query_nvtx_events
    from operator_profiler.mapper.manifest_builder import ManifestBuilder
    from operator_profiler.mapper.attribution_engine import AttributionEngine
    from operator_profiler.schema.manifest import (
        CaptureManifestMetadata, KernelManifestEntry, MappingManifest,
    )
    from operator_profiler.schema.profile import NvtxRangeInfo
    from operator_profiler.mapper.interval_tree import NvtxIntervalForest

    try:
        kernel_rows = query_kernels(sqlite_path)
        nvtx_rows = query_nvtx_events(sqlite_path)
        _ok(f"Queried {len(kernel_rows)} kernel rows, {len(nvtx_rows)} NVTX rows")

        meta = CaptureManifestMetadata(
            model_name="TransformerBlock",
            torch_version=torch.__version__,
            compile_mode="inductor",
            nsys_report_path=str(sqlite_path),
            capture_timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        )

        # Build manifest using internal helpers (no second nsys export call)
        builder = ManifestBuilder.__new__(ManifestBuilder)
        builder.nsys_rep_path = sqlite_path
        builder.metadata = meta
        builder.sqlite_cache_dir = None

        forest = builder._build_forest(nvtx_rows)
        outlier_ids = builder._detect_initialization_kernels(kernel_rows, nvtx_rows)

        entries: list[KernelManifestEntry] = []
        warnings: list[str] = []
        for i, kr in enumerate(kernel_rows):
            kid = f"k_{i:05d}"
            attribution = builder._attribute(kr, forest)
            if kid in outlier_ids:
                warnings.append(f"{kid} ({kr.kernel_name}): flagged as initialization kernel")
            entries.append(KernelManifestEntry(
                kernel_id=kid, kernel_name=kr.kernel_name,
                stream_id=kr.stream_id, device_id=kr.device_id,
                start_ns=kr.start_ns, end_ns=kr.end_ns,
                duration_ns=max(0, kr.end_ns - kr.start_ns),
                grid_dim=(kr.grid_x, kr.grid_y, kr.grid_z) if kr.grid_x else None,
                block_dim=(kr.block_x, kr.block_y, kr.block_z) if kr.block_x else None,
                attribution=attribution,
            ))

        manifest = MappingManifest(capture_metadata=meta, kernels=entries, warnings=warnings)

        engine = AttributionEngine(manifest)
        operator_records, unattributed = engine.run()

        # Count kernels with an NVTX range — only those can be replayed
        nvtx_attributed = sum(
            1 for e in entries
            if e.attribution.nvtx_range is not None
            and e.kernel_id not in outlier_ids
        )
        unique_ranges = {
            e.attribution.nvtx_range.text
            for e in entries
            if e.attribution.nvtx_range is not None
            and e.kernel_id not in outlier_ids
        }

        _ok(f"Manifest: {len(entries)} kernels  |  {len(outlier_ids)} init excluded  "
            f"|  {nvtx_attributed} with NVTX range")
        _ok(f"AttributionEngine: {len(operator_records)} operators  "
            f"|  {len(unattributed)} unattributed")
        _info(f"Unique NVTX ranges available for replay: {len(unique_ranges)}")
        for r in sorted(unique_ranges)[:10]:
            _info(f"    {r[:80]}")

        overall.append(("manifest_build", True))
    except Exception as exc:
        import traceback; traceback.print_exc()
        _fail(f"ManifestBuilder/AttributionEngine raised: {exc}")
        overall.append(("manifest_build", False))
        return _print_summary(overall)

    # -----------------------------------------------------------------------
    # Stage 3 — ncu kernel profile
    # -----------------------------------------------------------------------
    _header(3, "ncu kernel profile  (KernelProfileOrchestrator)")

    from operator_profiler.mapper.kernel_profiler import KernelProfileConfig, KernelProfileOrchestrator

    ncu_out_dir = Path(tempfile.mkdtemp(prefix="op_profiler_ncu_out_"))
    replay_config = KernelProfileConfig(
        replay_script=str(workload_script),
        replay_script_args=[],
        output_dir=str(ncu_out_dir),
        ncu_executable=ncu_bin,
    )

    try:
        orch = KernelProfileOrchestrator(manifest, operator_records, replay_config)
        t0 = time.perf_counter()
        orch.run()
        elapsed = time.perf_counter() - t0
        _ok(f"Range replay completed in {elapsed:.1f}s")

        # Count how many kernels now have populated metrics
        metrics_populated = 0
        metrics_empty = 0
        for op in operator_records:
            for k in op.kernels:
                if any([
                    k.metrics.dram_bytes_read,
                    k.metrics.achieved_occupancy,
                    k.metrics.sm_active_cycles,
                ]):
                    metrics_populated += 1
                else:
                    metrics_empty += 1

        _ok(f"KernelMetrics populated: {metrics_populated}  |  empty: {metrics_empty}")
        if metrics_populated == 0:
            _warn("No metrics were populated — ncu may have captured no kernels")

        # Show sample metrics from first non-empty kernel
        for op in operator_records:
            for k in op.kernels:
                m = k.metrics
                if m.dram_bytes_read or m.achieved_occupancy:
                    _info(f"Sample ({op.operator_name} / {k.kernel_name[:50]}):")
                    _info(f"  dram_read={m.dram_bytes_read}  dram_write={m.dram_bytes_written}")
                    _info(f"  occupancy={m.achieved_occupancy}  tensor_core%={m.tensor_core_active_pct}")
                    _info(f"  arith_intensity={m.arithmetic_intensity}  sm_cycles={m.sm_active_cycles}")
                    break
            else:
                continue
            break

        overall.append(("ncu_kernel_profile", metrics_populated > 0))
    except Exception as exc:
        import traceback; traceback.print_exc()
        _fail(f"KernelProfileOrchestrator raised: {exc}")
        overall.append(("ncu_kernel_profile", False))

    # -----------------------------------------------------------------------
    # Stage 4 — build_profile
    # -----------------------------------------------------------------------
    _header(4, "build_profile  (OperatorAttributedProfile assembly)")

    from operator_profiler.aggregator.profile_builder import build_profile

    try:
        profile = build_profile(
            manifest=manifest,
            operator_records=operator_records,
            unattributed_kernels=unattributed,
            model_name="TransformerBlock",
            torch_version=torch.__version__,
            device_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        )

        ops_with_metrics = [o for o in profile.operators if o.aggregated is not None]
        _ok(f"Profile assembled: {len(profile.operators)} operators  "
            f"|  {len(ops_with_metrics)} with AggregatedMetrics")

        overall.append(("build_profile", True))
    except Exception as exc:
        import traceback; traceback.print_exc()
        _fail(f"build_profile raised: {exc}")
        overall.append(("build_profile", False))
        return _print_summary(overall)

    # -----------------------------------------------------------------------
    # Stage 5 — metrics quality check
    # -----------------------------------------------------------------------
    _header(5, "Metrics quality check")

    ops_with_agg = [o for o in profile.operators if o.aggregated is not None]
    total_ns = sum(o.aggregated.total_duration_ns for o in ops_with_agg) or 1

    # Check which fields are actually populated
    occ_count    = sum(1 for o in ops_with_agg if o.aggregated.achieved_occupancy is not None)
    tc_count     = sum(1 for o in ops_with_agg if o.aggregated.tensor_core_active_pct is not None)
    dram_count   = sum(1 for o in ops_with_agg if (o.aggregated.total_dram_bytes_read or 0) > 0)

    _info(f"Operators with occupancy:    {occ_count}/{len(ops_with_agg)}")
    _info(f"Operators with tensor core:  {tc_count}/{len(ops_with_agg)}")
    _info(f"Operators with DRAM reads:   {dram_count}/{len(ops_with_agg)}")

    metrics_ok = occ_count > 0 or dram_count > 0
    if metrics_ok:
        _ok("At least some hardware metrics are populated")
    else:
        _warn("All hardware metrics are empty — check ncu output")

    # Top-10 operators by GPU time
    top10 = sorted(ops_with_agg, key=lambda o: o.aggregated.total_duration_ns, reverse=True)[:10]
    _info("\n  Top-10 operators by GPU time:")
    _info(f"  {'Operator':<45} {'Duration µs':>12} {'%':>6} {'Kernels':>8} "
          f"{'Occupancy':>10} {'TC%':>6}")
    _info("  " + "-" * 95)
    for op in top10:
        agg = op.aggregated
        dur_us = agg.total_duration_ns / 1e3
        pct = 100.0 * agg.total_duration_ns / total_ns
        occ_str = f"{agg.achieved_occupancy:.1f}" if agg.achieved_occupancy is not None else "n/a"
        tc_str  = f"{agg.tensor_core_active_pct:.1f}" if agg.tensor_core_active_pct is not None else "n/a"
        conf = op.kernels[0].confidence.value if op.kernels else "n/a"
        _info(f"  {op.operator_name[:45]:<45} {dur_us:>12.1f} {pct:>6.1f}% "
              f"{agg.kernel_count:>8} {occ_str:>10} {tc_str:>6}  [{conf}]")

    overall.append(("metrics_quality", metrics_ok))

    return _print_summary(overall)


def _print_summary(overall: list[tuple[str, bool]]) -> int:
    print(f"\n{SEP}")
    print("  NCU PIPELINE VERIFICATION SUMMARY")
    print(SEP)
    passed = sum(1 for _, ok in overall if ok)
    failed = len(overall) - passed
    for label, ok in overall:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}]  {label}")
    print(SEP)
    print(f"  {passed}/{len(overall)} stages passed")
    print(SEP)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
