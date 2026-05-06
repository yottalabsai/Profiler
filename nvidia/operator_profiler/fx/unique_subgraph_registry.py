"""
UniqueSubgraphRegistry — splits an FX graph by layer and groups the resulting
partitions by structural signature for deduplicated profiling and graph editing.
"""
from __future__ import annotations

import logging
from collections import defaultdict

import torch.fx as fx
from torch.fx.passes.split_module import split_module

from nvidia.operator_profiler.capture.layer_graph_splitter import (
    graph_signature,
    make_layer_partitioner,
)

log = logging.getLogger(__name__)


class UniqueSubgraphRegistry:
    """
    Splits an FX GraphModule by layer and groups the resulting partitions by
    structural signature.

    After construction:
    - ``split``         — split GraphModule; functionally identical to ``gm``
    - ``labels``        — partition id → human-readable label ("modules_0", "prologue", …)
    - ``unique_reps``   — one (name, submod) per unique structural signature
    - ``duplicates_of`` — all non-representative submods sharing a signature
    """

    def __init__(self, gm: fx.GraphModule) -> None:
        callback, self.labels = make_layer_partitioner(gm)
        self.split = split_module(gm, gm, callback)

        # sig → ordered list of (submodule_name, submodule)
        # First entry in each list is the representative.
        self._groups: dict[tuple, list[tuple[str, fx.GraphModule]]] = defaultdict(list)
        for name, submod in self.split.named_children():
            if isinstance(submod, fx.GraphModule):
                sig = graph_signature(submod)
                self._groups[sig].append((name, submod))

        n_unique = len(self._groups)
        n_total = sum(len(g) for g in self._groups.values())
        log.info(
            "UniqueSubgraphRegistry: %d total partition(s) → %d unique signature(s)",
            n_total,
            n_unique,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def unique_reps(self) -> list[tuple[str, fx.GraphModule]]:
        """One (name, submodule) per unique structural signature (representatives)."""
        return [group[0] for group in self._groups.values()]

    @property
    def unique_rep_names(self) -> set[str]:
        """Set of submodule names that are unique representatives."""
        return {name for name, _ in self.unique_reps}

    def duplicates_of(self, rep_name: str) -> list[tuple[str, fx.GraphModule]]:
        """All non-representative submodules sharing the same signature as ``rep_name``."""
        for group in self._groups.values():
            if group[0][0] == rep_name:
                return group[1:]
        return []

    def all_in_class(self, rep_name: str) -> list[tuple[str, fx.GraphModule]]:
        """Representative plus all duplicates with the same signature."""
        for group in self._groups.values():
            if group[0][0] == rep_name:
                return list(group)
        return []

    def partition_label(self, submod_name: str) -> str:
        """Human-readable label for a submodule by its split name (e.g. 'modules_0')."""
        try:
            pid = int(submod_name.replace("submod_", ""))
            return self.labels.get(pid, submod_name)
        except ValueError:
            return submod_name

    def is_unique_rep(self, submod_name: str) -> bool:
        """Return True if ``submod_name`` is a unique representative."""
        return submod_name in self.unique_rep_names

    def build_partition_equivalence_map(self) -> dict[str, str]:
        """
        Return a mapping from every duplicate partition label to its unique
        representative label.

        Example for a 12-layer transformer:
            {"modules_1": "modules_0", "modules_2": "modules_0", ...}

        Pass this to ``KernelProfileConfig.partition_equivalence_map`` so the
        profiling pipeline can propagate ncu metrics from unique partitions to
        all structural duplicates.
        """
        equiv: dict[str, str] = {}
        for rep_name, _ in self.unique_reps:
            rep_label = self.partition_label(rep_name)
            for dup_name, _ in self.duplicates_of(rep_name):
                dup_label = self.partition_label(dup_name)
                equiv[dup_label] = rep_label
        return equiv
