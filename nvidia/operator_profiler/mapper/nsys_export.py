"""
nsys export helpers.

Runs `nsys export --type sqlite` and queries the resulting SQLite database
for CUPTI kernel activity and NVTX events.

Column name conventions follow Nsight Systems 2024.x SQLite schema:
  - CUPTI_ACTIVITY_KIND_KERNEL: (id, start, end, streamId, deviceId,
      correlationId, shortName, gridX/Y/Z, blockX/Y/Z)
  - NVTX_EVENTS: (id, text, start, end, eventType, nestingLevel, domain)

Edge cases handled:
  - #5 (async launch): GPU-domain timestamps from CUPTI are used — not host timestamps.
  - #3 (multi-stream): stream_id and device_id are always returned so the
      caller can build per-stream interval trees.
"""
from __future__ import annotations

import logging
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path

from nvidia.operator_profiler.utils.subprocess_utils import run_subprocess

log = logging.getLogger(__name__)


@dataclass
class KernelRow:
    correlation_id: int
    kernel_name: str
    start_ns: int
    end_ns: int
    stream_id: int
    device_id: int
    grid_x: int
    grid_y: int
    grid_z: int
    block_x: int
    block_y: int
    block_z: int
    # Host thread that launched this kernel (from CUPTI_ACTIVITY_KIND_RUNTIME join).
    # Used to match NVTX ranges, which are keyed by globalTid, not GPU streamId.
    host_tid: int = 0
    # CPU-side timestamp of the cuLaunchKernel() call from RUNTIME table.
    # NVTX ranges use CPU timestamps; GPU start_ns is always slightly later due
    # to async dispatch.  Use cpu_launch_start_ns for NVTX enclosure queries.
    cpu_launch_start_ns: int = 0


@dataclass
class NvtxRow:
    text: str
    start_ns: int
    end_ns: int
    nesting_level: int
    domain: str
    stream_id: int
    device_id: int


def export_to_sqlite(nsys_rep_path: str | Path, output_dir: str | Path | None = None) -> Path:
    """
    Run `nsys export --type sqlite` on *nsys_rep_path* and return the
    path to the resulting .sqlite file.
    """
    nsys_rep_path = Path(nsys_rep_path)
    if output_dir is None:
        output_dir = nsys_rep_path.parent
    output_dir = Path(output_dir)
    output_path = output_dir / nsys_rep_path.with_suffix(".sqlite").name

    cmd = [
        "nsys", "export",
        "--type", "sqlite",
        "--output", str(output_path),
        "--force-overwrite", "true",
        str(nsys_rep_path),
    ]
    log.info("Exporting nsys report to SQLite: %s", " ".join(cmd))
    run_subprocess(cmd, description="nsys export")
    return output_path


def query_kernels(db_path: str | Path) -> list[KernelRow]:
    """
    Query CUPTI_ACTIVITY_KIND_KERNEL rows from the nsys SQLite export.

    Returns GPU-timestamped kernel rows with full launch parameters.
    """
    db_path = Path(db_path)
    rows: list[KernelRow] = []

    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        # Column names vary slightly across nsys versions; we use the most
        # common 2023+ schema.  Adapt if columns are absent.
        #
        # CUPTI_ACTIVITY_KIND_RUNTIME is joined to retrieve globalTid for each
        # kernel.  NVTX_EVENTS rows are keyed by globalTid (host thread), not
        # GPU streamId, so this is the correct key for NVTX enclosure lookups.
        try:
            cursor = conn.execute(
                """
                SELECT
                    k.correlationId,
                    s.value                      AS kernel_name,
                    k.start                      AS start_ns,
                    k.end                        AS end_ns,
                    k.streamId                   AS stream_id,
                    k.deviceId                   AS device_id,
                    k.gridX, k.gridY, k.gridZ,
                    k.blockX, k.blockY, k.blockZ,
                    COALESCE(r.globalTid, 0)     AS host_tid,
                    COALESCE(r.start, 0)         AS cpu_launch_start_ns
                FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
                LEFT JOIN StringIds AS s ON s.id = k.shortName
                LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS r
                       ON r.correlationId = k.correlationId
                ORDER BY k.start ASC
                """
            )
        except sqlite3.OperationalError:
            # Fallback: try without StringIds join (older exports store name inline)
            cursor = conn.execute(
                """
                SELECT
                    k.correlationId,
                    k.shortName                  AS kernel_name,
                    k.start                      AS start_ns,
                    k.end                        AS end_ns,
                    k.streamId                   AS stream_id,
                    k.deviceId                   AS device_id,
                    k.gridX, k.gridY, k.gridZ,
                    k.blockX, k.blockY, k.blockZ,
                    COALESCE(r.globalTid, 0)     AS host_tid,
                    COALESCE(r.start, 0)         AS cpu_launch_start_ns
                FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
                LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS r
                       ON r.correlationId = k.correlationId
                ORDER BY k.start ASC
                """
            )

        for r in cursor.fetchall():
            rows.append(
                KernelRow(
                    correlation_id=r["correlationId"],
                    kernel_name=r["kernel_name"] or "",
                    start_ns=r["start_ns"],
                    end_ns=r["end_ns"],
                    stream_id=r["stream_id"],
                    device_id=r["device_id"],
                    grid_x=r["gridX"] or 0,
                    grid_y=r["gridY"] or 0,
                    grid_z=r["gridZ"] or 0,
                    block_x=r["blockX"] or 0,
                    block_y=r["blockY"] or 0,
                    block_z=r["blockZ"] or 0,
                    host_tid=r["host_tid"] or 0,
                    cpu_launch_start_ns=r["cpu_launch_start_ns"] or 0,
                )
            )

    log.info("Queried %d kernel rows from %s", len(rows), db_path)
    return rows


def query_nvtx_events(db_path: str | Path) -> list[NvtxRow]:
    """
    Query NVTX_EVENTS rows from the nsys SQLite export.

    eventType 59 = NvtxRangeStart/End pairs (the common aten:: ranges).
    We include all range types; the caller filters by nesting_level / text.

    Schema note: nsys 2024+ exports dropped the ``nestingLevel`` column and
    the ``NVTX_DOMAIN`` table.  We probe the schema at runtime and adapt the
    query so the function works across all supported nsys versions.
    """
    db_path = Path(db_path)
    rows: list[NvtxRow] = []

    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row

        # Probe which optional schema elements are present.
        col_names = {
            r["name"]
            for r in conn.execute("PRAGMA table_info(NVTX_EVENTS)").fetchall()
        }
        has_nesting = "nestingLevel" in col_names
        has_domain_table = bool(
            conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='NVTX_DOMAIN'"
            ).fetchone()
        )

        nesting_expr = "n.nestingLevel" if has_nesting else "0"
        domain_expr = (
            "COALESCE(d.name, 'default')" if has_domain_table else "'default'"
        )
        domain_join = (
            "LEFT JOIN NVTX_DOMAIN AS d ON d.id = n.domainId"
            if has_domain_table
            else ""
        )

        sql = f"""
            SELECT
                n.text,
                n.start                      AS start_ns,
                n.end                        AS end_ns,
                {nesting_expr}               AS nesting_level,
                {domain_expr}                AS domain,
                COALESCE(n.globalTid, 0)     AS stream_id,
                0                            AS device_id
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

    log.info("Queried %d NVTX event rows from %s", len(rows), db_path)
    return rows
