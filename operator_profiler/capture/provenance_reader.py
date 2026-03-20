"""
Inductor provenance JSONL reader.

Parses the sidecar file emitted when INDUCTOR_PROVENANCE=1 and
INDUCTOR_COMPILE_THREADS=1 are set.

Each line has the form:
    {"generated_kernel_name": "triton_per_fused_linear_relu_0",
     "source_ops": ["aten::linear", "aten::relu"],
     "source_locations": [{"file": "model.py", "line": 42, "op": "aten::linear"}]}

INDUCTOR_COMPILE_THREADS=1 serializes compilation so provenance lines are
unambiguous (no interleaved writes from concurrent threads).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class ProvenanceEntry:
    generated_kernel_name: str
    source_ops: list[str] = field(default_factory=list)
    source_locations: list[dict] = field(default_factory=list)


def read_provenance_jsonl(path: str | Path) -> list[ProvenanceEntry]:
    """
    Read and parse a provenance JSONL sidecar.

    Returns a list of ProvenanceEntry objects, one per non-empty line.
    Malformed lines are logged as warnings and skipped.
    """
    path = Path(path)
    if not path.exists():
        log.warning("Provenance file not found: %s", path)
        return []

    entries: list[ProvenanceEntry] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                log.warning("Provenance JSON parse error at line %d: %s", lineno, exc)
                continue

            name = record.get("generated_kernel_name", "")
            if not name:
                log.warning("Provenance line %d missing generated_kernel_name", lineno)
                continue

            entries.append(
                ProvenanceEntry(
                    generated_kernel_name=name,
                    source_ops=record.get("source_ops", []),
                    source_locations=record.get("source_locations", []),
                )
            )

    log.info("Read %d provenance entries from %s", len(entries), path)
    return entries


def provenance_as_dict(entries: list[ProvenanceEntry]) -> dict[str, ProvenanceEntry]:
    """Return a dict keyed by generated_kernel_name for fast lookup."""
    return {e.generated_kernel_name: e for e in entries}
