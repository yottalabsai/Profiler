"""
FxPassRunner — applies torch.fx graph transformation passes to unique subgraph
representatives and propagates identical edits to all structural duplicates.
"""
from __future__ import annotations

import inspect
import logging
from typing import Callable

import torch.fx as fx
from torch.fx.subgraph_rewriter import replace_pattern

from .unique_subgraph_registry import UniqueSubgraphRegistry

log = logging.getLogger(__name__)


def _normalize_pattern(pattern: Callable) -> fx.GraphModule:
    """
    Trace `pattern` with symbolic_trace and strip default-valued kwargs from
    call_function nodes.

    Dynamo captures ops without their default keyword arguments (e.g. relu is
    stored as `relu(x)` with `kwargs={}`) while `symbolic_trace` includes
    them (e.g. `relu(x, inplace=False)`).  SubgraphMatcher's ``_match_nodes``
    requires an exact kwargs-key match, so the pattern must match the dynamo
    format before being passed to ``replace_pattern``.
    """
    gm = fx.symbolic_trace(pattern)
    for node in gm.graph.nodes:
        if node.op != "call_function" or not node.kwargs:
            continue
        try:
            sig = inspect.signature(node.target)
        except (ValueError, TypeError):
            continue
        stripped = {}
        for k, v in node.kwargs.items():
            param = sig.parameters.get(k)
            if param is None or param.default is inspect.Parameter.empty or param.default != v:
                stripped[k] = v
        if stripped != dict(node.kwargs):
            node.kwargs = stripped
    gm.recompile()
    return gm


class FxPassRunner:
    """
    Applies ``torch.fx.subgraph_rewriter.replace_pattern`` passes only to
    structurally unique subgraph representatives, then re-applies the same pass
    to all structural duplicates.

    Since duplicates have identical graph structure to their representative, the
    same pattern matches at identical positions — no bespoke propagation logic is
    required beyond re-running ``replace_pattern`` on each duplicate.

    Usage::

        registry = UniqueSubgraphRegistry(gm)
        runner   = FxPassRunner(registry)
        n        = runner.apply_pass(pattern_fn, replacement_fn)
        # registry.split now has the transformation applied to every partition.
    """

    def __init__(self, registry: UniqueSubgraphRegistry) -> None:
        self.registry = registry

    def apply_pass(
        self,
        pattern: Callable,
        replacement: Callable,
    ) -> int:
        """
        Apply ``replace_pattern(pattern, replacement)`` to each unique representative
        subgraph, then propagate to all structural duplicates.

        Parameters
        ----------
        pattern:
            Callable describing the subgraph pattern to match (same semantics as
            ``torch.fx.subgraph_rewriter.replace_pattern``).
        replacement:
            Callable describing the replacement subgraph.

        Returns
        -------
        int
            Total number of replacement matches across all partitions (unique +
            all duplicates).
        """
        total = 0
        any_match = False

        # Normalize both pattern and replacement to dynamo's kwargs format.
        # symbolic_trace includes default kwargs (e.g. inplace=False) that
        # dynamo strips; SubgraphMatcher requires an exact kwargs-key match.
        pattern_gm = _normalize_pattern(pattern)
        replacement_gm = _normalize_pattern(replacement)

        for rep_name, rep_mod in self.registry.unique_reps:
            matches = replace_pattern(rep_mod, pattern_gm, replacement_gm)
            if not matches:
                continue

            rep_mod.recompile()
            total += len(matches)
            any_match = True
            log.debug(
                "apply_pass: %d match(es) in representative %r", len(matches), rep_name
            )

            for dup_name, dup_mod in self.registry.duplicates_of(rep_name):
                dup_matches = replace_pattern(dup_mod, pattern_gm, replacement_gm)
                dup_mod.recompile()
                total += len(dup_matches)
                log.debug(
                    "apply_pass: propagated %d match(es) to duplicate %r",
                    len(dup_matches),
                    dup_name,
                )

        if any_match:
            self.registry.split.recompile()
            log.info(
                "apply_pass: %d total replacement(s) across all partitions", total
            )

        return total
