"""
verify_preprocessing_pipeline.py
=================================
End-to-end verification of the data-preprocessing pipeline on a freshly
profiled nsys capture with proper aten:: NVTX annotation.

Stages verified (each reported as PASS / FAIL / WARN):
  0. nsys availability check
  1. nsys profile   — capture nvtx_workload.py under nsys
  2. nsys export    — .nsys-rep → .sqlite
  3. query_kernels  — CUPTI_ACTIVITY_KIND_KERNEL rows
  4. query_nvtx     — NVTX_EVENTS rows  (surfaces schema mismatch bug)
  5. interval forest — per-stream NvtxIntervalForest built from NVTX rows
  6. manifest build  — KernelManifestEntry list (attribution per kernel)
  7. attribution stats — breakdown by method + confidence
  8. metric aggregation — AggregatedMetrics per operator

Usage:
    python scripts/verify_preprocessing_pipeline.py [--sqlite path/to/existing.sqlite]

Pass --sqlite to skip stages 0-2 and run from an existing export.
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

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from operator_profiler.mapper.nsys_export import query_kernels
from operator_profiler.mapper.interval_tree import NvtxIntervalForest
from operator_profiler.schema.manifest import (
    CaptureManifestMetadata,
    KernelAttribution,
    KernelManifestEntry,
    MappingManifest,
)
from operator_profiler.schema.profile import (
    AttributionMethod,
    Confidence,
    NvtxRangeInfo,
)
from operator_profiler.mapper.manifest_builder import ManifestBuilder, _OP_NAME_FRAGMENTS
from operator_profiler.mapper.attribution_engine import AttributionEngine
from operator_profiler.aggregator.metric_aggregator import build_aggregated_metrics

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEP = "=" * 72

def _header(stage: int, title: str) -> None:
    print(f"\n{SEP}")
    print(f"  Stage {stage}: {title}")
    print(SEP)


def _ok(msg: str) -> None:
    print(f"  [PASS]  {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL]  {msg}")


def _warn(msg: str) -> None:
    print(f"  [WARN]  {msg}")


def _info(msg: str) -> None:
    print(f"          {msg}")


# ---------------------------------------------------------------------------
# Stage 4 workaround: direct NVTX query that handles the actual nsys schema
# ---------------------------------------------------------------------------

def _query_nvtx_direct(db_path: Path) -> list:
    """
    Query NVTX_EVENTS directly, tolerating the current nsys 2024+ schema which
    lacks both NVTX_DOMAIN and the nestingLevel column.

    Returns a list of NvtxRow-compatible namedtuples.
    """
    from operator_profiler.mapper.nsys_export import NvtxRow

    rows = []
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row

        # Probe which columns exist
        col_info = conn.execute("PRAGMA table_info(NVTX_EVENTS)").fetchall()
        col_names = {r["name"] for r in col_info}

        has_nesting = "nestingLevel" in col_names
        has_domain_table = bool(
            conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='NVTX_DOMAIN'"
            ).fetchone()
        )

        nesting_expr = "nestingLevel" if has_nesting else "0"
        domain_expr = (
            "COALESCE(d.name, 'default')"
            if has_domain_table
            else "'default'"
        )
        domain_join = (
            "LEFT JOIN NVTX_DOMAIN AS d ON d.id = n.domainId"
            if has_domain_table
            else ""
        )

        sql = f"""
            SELECT
                n.text,
                n.start          AS start_ns,
                n.end            AS end_ns,
                {nesting_expr}   AS nesting_level,
                {domain_expr}    AS domain,
                COALESCE(n.globalTid, 0) AS stream_id,
                0                AS device_id
            FROM NVTX_EVENTS AS n
            {domain_join}
            WHERE n.end IS NOT NULL
            ORDER BY n.start ASC
        """
        for r in conn.execute(sql).fetchall():
            rows.append(
                NvtxRow(
                    text=r["text"] or "",
                    start_ns=r["start_ns"],
                    end_ns=r["end_ns"],
                    nesting_level=r["nesting_level"] or 0,
                    domain=r["domain"],
                    stream_id=r["stream_id"],
                    device_id=r["device_id"],
                )
            )
    return rows, has_nesting, has_domain_table


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sqlite", metavar="PATH",
        help="Skip stages 0-2 and use an existing .sqlite export directly.",
    )
    parser.add_argument(
        "--output-dir", metavar="DIR", default=None,
        help="Directory for nsys output files (default: temp dir).",
    )
    args = parser.parse_args(argv)

    overall: list[tuple[str, bool]] = []   # (label, passed)

    # -----------------------------------------------------------------------
    # Stage 0 — nsys availability
    # -----------------------------------------------------------------------
    sqlite_path: Path | None = None

    if args.sqlite:
        sqlite_path = Path(args.sqlite)
        _header(0, "nsys availability  (SKIPPED — using existing SQLite)")
        _info(f"SQLite: {sqlite_path}")
        overall.append(("nsys availability", True))
        overall.append(("nsys profile", True))
        overall.append(("nsys export", True))
    else:
        _header(0, "nsys availability check")
        nsys_bin = shutil.which("nsys")
        if nsys_bin:
            _ok(f"nsys found: {nsys_bin}")
            overall.append(("nsys availability", True))
        else:
            _fail("nsys not found in PATH — install Nsight Systems or pass --sqlite")
            overall.append(("nsys availability", False))
            return _print_summary(overall)

        # -------------------------------------------------------------------
        # Stage 1 — nsys profile
        # -------------------------------------------------------------------
        _header(1, "nsys profile  (nvtx_workload.py)")
        workload = ROOT / "scripts" / "nvtx_workload.py"
        if not workload.exists():
            _fail(f"Workload script not found: {workload}")
            overall.append(("nsys profile", False))
            return _print_summary(overall)

        out_dir = Path(args.output_dir) if args.output_dir else Path(tempfile.mkdtemp(prefix="op_profiler_"))
        out_dir.mkdir(parents=True, exist_ok=True)
        rep_path = out_dir / "nvtx_workload.nsys-rep"
        sqlite_path = out_dir / "nvtx_workload.sqlite"

        cmd = [
            "nsys", "profile",
            "--trace=cuda,nvtx",
            f"--output={out_dir / 'nvtx_workload'}",
            "--force-overwrite=true",
            sys.executable, str(workload),
        ]
        _info(f"cmd: {' '.join(cmd)}")
        t0 = time.perf_counter()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            elapsed = time.perf_counter() - t0
            if result.returncode != 0:
                _fail(f"nsys profile exited {result.returncode}")
                _info(result.stderr[-800:] if result.stderr else "(no stderr)")
                overall.append(("nsys profile", False))
                return _print_summary(overall)
            if not rep_path.exists():
                # nsys may not append .nsys-rep in all versions
                candidates = list(out_dir.glob("*.nsys-rep"))
                if candidates:
                    rep_path = candidates[0]
                else:
                    _fail(f"No .nsys-rep found in {out_dir}")
                    overall.append(("nsys profile", False))
                    return _print_summary(overall)
            _ok(f"Profile captured in {elapsed:.1f}s → {rep_path.name}")
            overall.append(("nsys profile", True))
        except subprocess.TimeoutExpired:
            _fail("nsys profile timed out after 300s")
            overall.append(("nsys profile", False))
            return _print_summary(overall)
        except Exception as exc:
            _fail(f"nsys profile error: {exc}")
            overall.append(("nsys profile", False))
            return _print_summary(overall)

        # -------------------------------------------------------------------
        # Stage 2 — nsys export → SQLite
        # -------------------------------------------------------------------
        _header(2, "nsys export  (.nsys-rep → .sqlite)")
        cmd = [
            "nsys", "export",
            "--type=sqlite",
            f"--output={sqlite_path}",
            "--force-overwrite=true",
            str(rep_path),
        ]
        _info(f"cmd: {' '.join(cmd)}")
        t0 = time.perf_counter()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            elapsed = time.perf_counter() - t0
            if result.returncode != 0:
                _fail(f"nsys export exited {result.returncode}")
                _info(result.stderr[-800:] if result.stderr else "(no stderr)")
                overall.append(("nsys export", False))
                return _print_summary(overall)
            if not sqlite_path.exists():
                candidates = list(out_dir.glob("*.sqlite"))
                if candidates:
                    sqlite_path = candidates[0]
                else:
                    _fail(f"No .sqlite found in {out_dir}")
                    overall.append(("nsys export", False))
                    return _print_summary(overall)
            size_mb = sqlite_path.stat().st_size / 1e6
            _ok(f"Exported in {elapsed:.1f}s → {sqlite_path.name}  ({size_mb:.1f} MB)")
            overall.append(("nsys export", True))
        except subprocess.TimeoutExpired:
            _fail("nsys export timed out after 120s")
            overall.append(("nsys export", False))
            return _print_summary(overall)
        except Exception as exc:
            _fail(f"nsys export error: {exc}")
            overall.append(("nsys export", False))
            return _print_summary(overall)

    # -----------------------------------------------------------------------
    # Stage 3 — query_kernels
    # -----------------------------------------------------------------------
    _header(3, "query_kernels  (CUPTI_ACTIVITY_KIND_KERNEL)")
    try:
        kernel_rows = query_kernels(sqlite_path)
        if not kernel_rows:
            _warn("Zero kernel rows returned — was the workload captured?")
            overall.append(("query_kernels", False))
        else:
            total_dur_ms = sum(max(0, r.end_ns - r.start_ns) for r in kernel_rows) / 1e6
            unique_names = len({r.kernel_name for r in kernel_rows})
            streams      = len({r.stream_id for r in kernel_rows})
            _ok(f"{len(kernel_rows)} kernel rows  |  {unique_names} unique names  "
                f"|  {streams} stream(s)  |  {total_dur_ms:.1f} ms total GPU time")
            _info(f"First kernel: {kernel_rows[0].kernel_name[:80]}")
            _info(f"Last  kernel: {kernel_rows[-1].kernel_name[:80]}")
            overall.append(("query_kernels", True))
    except Exception as exc:
        _fail(f"query_kernels raised: {exc}")
        overall.append(("query_kernels", False))
        kernel_rows = []

    # -----------------------------------------------------------------------
    # Stage 4 — query_nvtx_events (surface bug + apply workaround)
    # -----------------------------------------------------------------------
    _header(4, "query_nvtx_events  (NVTX_EVENTS schema probe)")

    # First, run the existing pipeline function to surface the known bug
    from operator_profiler.mapper.nsys_export import query_nvtx_events
    builtin_passed = False
    try:
        nvtx_rows_builtin = query_nvtx_events(sqlite_path)
        builtin_passed = True
        _ok(f"query_nvtx_events (built-in): {len(nvtx_rows_builtin)} rows")
    except Exception as exc:
        _warn(f"query_nvtx_events (built-in) FAILED: {exc}")
        _info("This is the known schema bug: nsys 2024+ exports lack nestingLevel")
        _info("and NVTX_DOMAIN table.  Applying schema-aware workaround...")

    # Always run the workaround to get the actual rows
    try:
        nvtx_rows, has_nesting, has_domain_table = _query_nvtx_direct(sqlite_path)
        _ok(f"query_nvtx (workaround): {len(nvtx_rows)} rows")
        _info(f"Schema: nestingLevel col={has_nesting}  NVTX_DOMAIN table={has_domain_table}")
        if nvtx_rows:
            # Show distribution of NVTX event text prefixes
            prefixes = Counter()
            for r in nvtx_rows:
                prefix = r.text.split("[")[0].split(" ")[0][:40]
                prefixes[prefix] += 1
            _info("Top-10 NVTX text prefixes:")
            for txt, cnt in prefixes.most_common(10):
                _info(f"    {txt:<45}  {cnt:>6} events")
        overall.append(("query_nvtx_events", builtin_passed))
    except Exception as exc:
        _fail(f"NVTX workaround also failed: {exc}")
        overall.append(("query_nvtx_events", False))
        nvtx_rows = []

    # -----------------------------------------------------------------------
    # Stage 5 — NvtxIntervalForest
    # -----------------------------------------------------------------------
    _header(5, "NvtxIntervalForest  (per-stream interval tree)")
    try:
        forest = NvtxIntervalForest()
        inserted = 0
        skipped_no_text = 0
        for row in nvtx_rows:
            if not row.text:
                skipped_no_text += 1
                continue
            forest.insert(
                stream_id=row.stream_id,
                device_id=row.device_id,
                range_info=NvtxRangeInfo(
                    text=row.text,
                    depth=row.nesting_level,
                    start_ns=row.start_ns,
                    end_ns=row.end_ns,
                    domain=row.domain,
                ),
            )
            inserted += 1

        stream_keys = forest.stream_keys
        _ok(f"Forest built: {inserted} ranges inserted across {len(stream_keys)} stream(s)")
        if skipped_no_text:
            _info(f"Skipped {skipped_no_text} NVTX rows with empty text")

        # Spot-check: query enclosing for the first kernel
        if kernel_rows:
            k = kernel_rows[0]
            enclosing = forest.query_enclosing(k.stream_id, k.device_id, k.start_ns)
            _info(f"Enclosing ranges for kernel[0] (stream={k.stream_id}): "
                  f"{[r.text[:50] for r in enclosing] or '(none)'}")
        overall.append(("interval_forest", True))
    except Exception as exc:
        _fail(f"NvtxIntervalForest raised: {exc}")
        import traceback; traceback.print_exc()
        overall.append(("interval_forest", False))
        forest = NvtxIntervalForest()

    # -----------------------------------------------------------------------
    # Stage 6 — MappingManifest build (via ManifestBuilder internals)
    # -----------------------------------------------------------------------
    _header(6, "MappingManifest build  (ManifestBuilder attribution loop)")
    try:
        import time as _time
        meta = CaptureManifestMetadata(
            model_name="TransformerBlock",
            torch_version=torch.__version__,
            compile_mode="eager",
            nsys_report_path=str(sqlite_path),
            capture_timestamp_utc=_time.strftime("%Y-%m-%dT%H:%M:%S+00:00", _time.gmtime()),
        )

        # Re-use ManifestBuilder private methods (no nsys export call)
        builder = ManifestBuilder.__new__(ManifestBuilder)
        builder.nsys_rep_path = sqlite_path
        builder.provenance_jsonl_path = None
        builder.metadata = meta
        builder.sqlite_cache_dir = None

        outlier_ids = builder._detect_warmup_outliers(kernel_rows)
        provenance   = builder._load_provenance()

        entries: list[KernelManifestEntry] = []
        warnings: list[str] = []
        for i, kr in enumerate(kernel_rows):
            kid = f"k_{i:05d}"
            attribution = builder._attribute(kr, forest, provenance)
            is_warmup   = kid in outlier_ids
            if is_warmup:
                warnings.append(f"{kid} ({kr.kernel_name}): flagged as warm-up outlier")
            entries.append(
                KernelManifestEntry(
                    kernel_id=kid,
                    kernel_name=kr.kernel_name,
                    stream_id=kr.stream_id,
                    device_id=kr.device_id,
                    start_ns=kr.start_ns,
                    end_ns=kr.end_ns,
                    duration_ns=max(0, kr.end_ns - kr.start_ns),
                    grid_dim=(kr.grid_x, kr.grid_y, kr.grid_z) if kr.grid_x else None,
                    block_dim=(kr.block_x, kr.block_y, kr.block_z) if kr.block_x else None,
                    attribution=attribution,
                )
            )

        manifest = MappingManifest(
            capture_metadata=meta,
            kernels=entries,
            warnings=warnings,
        )
        _ok(f"Manifest: {len(entries)} kernel entries, {len(warnings)} warning(s), "
            f"{len(outlier_ids)} warm-up outlier(s)")
        overall.append(("manifest_build", True))
    except Exception as exc:
        _fail(f"ManifestBuilder raised: {exc}")
        import traceback; traceback.print_exc()
        overall.append(("manifest_build", False))
        manifest = None

    # -----------------------------------------------------------------------
    # Stage 7 — Attribution statistics
    # -----------------------------------------------------------------------
    _header(7, "Attribution statistics")
    if manifest is None:
        _fail("No manifest — skipping attribution stats")
        overall.append(("attribution_stats", False))
    else:
        method_counts: Counter = Counter()
        conf_counts:   Counter = Counter()
        for e in manifest.kernels:
            method_counts[e.attribution.method.value] += 1
            conf_counts[e.attribution.confidence.value] += 1

        total = len(manifest.kernels)
        _ok(f"Total kernels: {total}")
        _info("Attribution method breakdown:")
        for method, cnt in sorted(method_counts.items(), key=lambda x: -x[1]):
            pct = 100 * cnt / total if total else 0
            _info(f"    {method:<20}  {cnt:>7}  ({pct:.1f}%)")
        _info("Confidence breakdown:")
        for conf, cnt in sorted(conf_counts.items(), key=lambda x: -x[1]):
            pct = 100 * cnt / total if total else 0
            _info(f"    {conf:<20}  {cnt:>7}  ({pct:.1f}%)")

        nvtx_pct = 100 * method_counts.get("nvtx", 0) / total if total else 0
        heur_pct = 100 * method_counts.get("name_heuristic", 0) / total if total else 0
        unat_pct = 100 * method_counts.get("unattributed", 0) / total if total else 0

        if nvtx_pct > 0:
            _ok(f"NVTX attribution rate: {nvtx_pct:.1f}%")
        else:
            _warn("Zero NVTX attributions — NVTX ranges may not overlap kernels "
                  "(GPU vs CPU timestamp domain mismatch or empty NVTX data)")
        _info(f"Name heuristic: {heur_pct:.1f}%   Unattributed: {unat_pct:.1f}%")
        overall.append(("attribution_stats", True))

    # -----------------------------------------------------------------------
    # Stage 8 — AttributionEngine + MetricAggregator
    # -----------------------------------------------------------------------
    _header(8, "AttributionEngine + MetricAggregator")
    if manifest is None:
        _fail("No manifest — skipping")
        overall.append(("attribution_engine", False))
    else:
        try:
            engine = AttributionEngine(manifest)
            operator_records, unattributed = engine.run()

            _ok(f"AttributionEngine: {len(operator_records)} operators, "
                f"{len(unattributed)} unattributed kernels")

            # Aggregate metrics for each operator record
            for op in operator_records:
                kernel_metrics = [k.metrics for k in op.kernels]
                total_ns = sum(k.duration_ns for k in op.kernels)
                op.aggregated = build_aggregated_metrics(kernel_metrics, total_ns)

            _ok(f"MetricAggregator: aggregated metrics for {len(operator_records)} operators")

            # Top-5 operators by total duration
            top5 = sorted(
                operator_records,
                key=lambda o: o.aggregated.total_duration_ns,
                reverse=True,
            )[:5]
            _info("Top-5 operators by GPU time:")
            for op in top5:
                dur_us = op.aggregated.total_duration_ns / 1e3
                _info(f"    {op.operator_name:<35}  "
                      f"{dur_us:>10.1f} µs  "
                      f"kernels={op.aggregated.kernel_count}  "
                      f"bottleneck={op.aggregated.bottleneck_classification}")

            overall.append(("attribution_engine", True))
        except Exception as exc:
            _fail(f"AttributionEngine/MetricAggregator raised: {exc}")
            import traceback; traceback.print_exc()
            overall.append(("attribution_engine", False))

    return _print_summary(overall)


def _print_summary(overall: list[tuple[str, bool]]) -> int:
    print(f"\n{SEP}")
    print("  PIPELINE VERIFICATION SUMMARY")
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
