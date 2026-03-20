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

from operator_profiler.utils.subprocess_utils import run_subprocess

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
        try:
            cursor = conn.execute(
                """
                SELECT
                    k.correlationId,
                    s.value          AS kernel_name,
                    k.start          AS start_ns,
                    k.end            AS end_ns,
                    k.streamId       AS stream_id,
                    k.deviceId       AS device_id,
                    k.gridX, k.gridY, k.gridZ,
                    k.blockX, k.blockY, k.blockZ
                FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
                LEFT JOIN StringIds AS s ON s.id = k.shortName
                ORDER BY k.start ASC
                """
            )
        except sqlite3.OperationalError:
            # Fallback: try without StringIds join (older exports store name inline)
            cursor = conn.execute(
                """
                SELECT
                    correlationId,
                    shortName        AS kernel_name,
                    start            AS start_ns,
                    end              AS end_ns,
                    streamId         AS stream_id,
                    deviceId         AS device_id,
                    gridX, gridY, gridZ,
                    blockX, blockY, blockZ
                FROM CUPTI_ACTIVITY_KIND_KERNEL
                ORDER BY start ASC
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
                )
            )

    log.info("Queried %d kernel rows from %s", len(rows), db_path)
    return rows


def query_nvtx_events(db_path: str | Path) -> list[NvtxRow]:
    """
    Query NVTX_EVENTS rows from the nsys SQLite export.

    eventType 59 = NvtxRangeStart/End pairs (the common aten:: ranges).
    We include all range types; the caller filters by nesting_level / text.
    """
    db_path = Path(db_path)
    rows: list[NvtxRow] = []

    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                """
                SELECT
                    n.text,
                    n.start          AS start_ns,
                    n.end            AS end_ns,
                    n.nestingLevel   AS nesting_level,
                    COALESCE(d.name, 'default') AS domain,
                    COALESCE(n.globalTid, 0)    AS stream_id,
                    0                           AS device_id
                FROM NVTX_EVENTS AS n
                LEFT JOIN NVTX_DOMAIN AS d ON d.id = n.domainId
                WHERE n.end IS NOT NULL
                ORDER BY n.start ASC
                """
            )
        except sqlite3.OperationalError:
            # Minimal fallback without domain join
            cursor = conn.execute(
                """
                SELECT
                    text,
                    start       AS start_ns,
                    end         AS end_ns,
                    nestingLevel AS nesting_level,
                    'default'   AS domain,
                    0           AS stream_id,
                    0           AS device_id
                FROM NVTX_EVENTS
                WHERE end IS NOT NULL
                ORDER BY start ASC
                """
            )

        for r in cursor.fetchall():
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
