"""
ncu CSV parser — converts `ncu --import <file> --csv` output into KernelMetrics.

ncu CSV format (2024.x):
  "ID","Process ID","Process Name","Host Name","Kernel Name","Kernel Time",
  "Context","Stream","Section Name","Metric Name","Metric Unit","Metric Value"

Each kernel appears as multiple rows (one per metric).  We pivot on
(Kernel Name, ID) to produce one KernelMetrics per kernel row.

Edge case #8: ncu timestamps are intentionally ignored here.  Only metric
values are extracted and stored in KernelMetrics.raw / named fields.
"""
from __future__ import annotations

import csv
import io
import logging
from collections import defaultdict

from operator_profiler.schema.profile import KernelMetrics

log = logging.getLogger(__name__)

# ncu CSV column names (vary slightly between versions; try both spellings)
_COL_ALTERNATIVES = {
    "kernel_name": ["Kernel Name", "KernelName"],
    "metric_name": ["Metric Name", "MetricName"],
    "metric_value": ["Metric Value", "MetricValue"],
    "kernel_id": ["ID", "Id"],
}


def _resolve_header(header: list[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in header:
            return c
    return None


def parse_ncu_csv(csv_text: str) -> dict[str, KernelMetrics]:
    """
    Parse ncu CSV text into a dict mapping kernel_name → KernelMetrics.

    If a kernel name appears multiple times (different IDs), the last entry
    wins.  Callers that need per-invocation granularity should use
    parse_ncu_csv_by_id() instead.
    """
    by_id = parse_ncu_csv_by_id(csv_text)
    # Merge: last ID for each kernel name
    result: dict[str, KernelMetrics] = {}
    for (kernel_name, _kid), metrics in sorted(by_id.items()):
        result[kernel_name] = metrics
    return result


def parse_ncu_csv_by_id(
    csv_text: str,
) -> dict[tuple[str, str], KernelMetrics]:
    """
    Parse ncu CSV text into a dict keyed by (kernel_name, invocation_id).

    Returns a dict suitable for cross-referencing with manifest kernel_ids
    by launch index within a range (edge case #1 — no timestamp join).
    """
    reader = csv.DictReader(io.StringIO(csv_text))
    if reader.fieldnames is None:
        log.warning("ncu CSV has no header row")
        return {}

    header = list(reader.fieldnames)

    # Resolve column names
    col_kernel = _resolve_header(header, _COL_ALTERNATIVES["kernel_name"])
    col_metric = _resolve_header(header, _COL_ALTERNATIVES["metric_name"])
    col_value = _resolve_header(header, _COL_ALTERNATIVES["metric_value"])
    col_id = _resolve_header(header, _COL_ALTERNATIVES["kernel_id"])

    if not all([col_kernel, col_metric, col_value]):
        log.error(
            "ncu CSV missing required columns. Found: %s", header
        )
        return {}

    # Accumulate raw metric rows keyed by (kernel_name, invocation_id)
    raw: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)

    for row in reader:
        kernel_name = row.get(col_kernel, "").strip()
        metric_name = row.get(col_metric, "").strip()
        metric_value = row.get(col_value, "").strip()
        kid = row.get(col_id, "0").strip() if col_id else "0"

        if not kernel_name or not metric_name:
            continue
        raw[(kernel_name, kid)][metric_name] = metric_value

    return {key: _build_metrics(metrics_dict) for key, metrics_dict in raw.items()}


def _build_metrics(raw_dict: dict[str, str]) -> KernelMetrics:
    """Convert a flat {metric_name: value_str} dict → KernelMetrics.

    Non-numeric values (e.g. "n/a") are dropped so that KernelMetrics.raw
    contains only counters that actually fired on this GPU.  This prevents
    architecture-specific fallback names from cluttering profile.json with
    redundant "n/a" entries alongside the counter that produced a real value.
    """
    raw: dict[str, float | int | str] = {}
    for metric_name, value_str in raw_dict.items():
        parsed = _try_parse_numeric(value_str)
        if parsed is not None:
            raw[metric_name] = parsed
    return KernelMetrics(raw=raw)


def _try_parse_numeric(value: str) -> float | int | None:
    """Return int, float, or None for non-numeric strings."""
    value = value.replace(",", "")  # ncu uses comma separators for large numbers
    try:
        i = int(value)
        return i
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return None


