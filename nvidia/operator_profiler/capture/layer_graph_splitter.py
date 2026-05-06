"""
layer_graph_splitter — FX graph partitioning by auto-detected repeated layer index.

When torch._dynamo traces a model, the resulting FX graph contains node names
that embed the originating module path with dots replaced by underscores, e.g.:

    encoder_layers_0_attn_q_proj
    encoder_layers_1_attn_q_proj
    transformer_h_3_mlp_c_proj
    model_stage_2_conv_weight

This module auto-detects which prefix is a repeated layer index by scanning the
graph for ``word_N_`` patterns that appear with more than one distinct N value,
then uses that prefix to split the graph into per-layer submodules via
``torch.fx.passes.split_module.split_module()``.  Structurally identical
partitions can then be detected by comparing ``graph_signature()`` hashes.

Works with any model architecture — HuggingFace, custom PyTorch, vision
transformers, etc. — without requiring a keyword allow-list.

Usage
-----
    from torch.fx.passes.split_module import split_module
    from nvidia.operator_profiler.capture.layer_graph_splitter import (
        graph_signature,
        make_layer_partitioner,
    )

    callback, labels = make_layer_partitioner(gm)
    split = split_module(gm, gm, callback)
    for name, submod in split.named_children():
        sig = graph_signature(submod)
        ...
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Callable

import torch.fx as fx

log = logging.getLogger(__name__)

# Matches word_N_ where the prefix is a simple alphanumeric token (no underscores)
# and must be preceded by a non-alphanumeric character or start-of-string.
# This avoids greedily consuming underscore-joined path prefixes like
# "l_self_modules_stage" when the real token is just "stage".
# Examples: "layer_0_attn" → "layer_0", "h_1_mlp" → "h_1", "stage_3_conv" → "stage_3".
LAYER_RE = re.compile(r"(?<![a-zA-Z0-9])([a-zA-Z][a-zA-Z0-9]*)_(\d+)(?=_)")


def _detect_split_pattern(gm: fx.GraphModule) -> re.Pattern | None:
    """
    Scan ``gm``'s node names and return a compiled pattern for the prefix that
    repeats with the most distinct index values.  Returns None when no prefix
    appears with more than one distinct index (no repeated structure found).
    """
    prefix_indices: dict[str, set[int]] = defaultdict(set)
    for node in gm.graph.nodes:
        for m in LAYER_RE.finditer(node.name):
            prefix_indices[m.group(1)].add(int(m.group(2)))

    repeated = {p: idxs for p, idxs in prefix_indices.items() if len(idxs) > 1}
    if not repeated:
        return None

    best = max(repeated, key=lambda p: len(repeated[p]))
    return re.compile(rf"(?<![a-zA-Z0-9]){re.escape(best)}_(\d+)(?=_|$)")


def graph_signature(gm: fx.GraphModule) -> tuple:
    """
    Return a hashable tuple encoding the computation structure of ``gm``.

    Each element records one node's op kind and functional target.  Module
    instances are identified by type rather than name so that structurally
    identical subgraphs across different layers compare equal.
    """
    sig: list[tuple] = []
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            sig.append(("placeholder",))
        elif node.op == "output":
            sig.append(("output",))
        elif node.op == "get_attr":
            sig.append(("get_attr",))
        elif node.op == "call_function":
            sig.append(("call_function", getattr(node.target, "__name__", str(node.target))))
        elif node.op == "call_method":
            sig.append(("call_method", str(node.target)))
        elif node.op == "call_module":
            try:
                type_name = type(gm.get_submodule(node.target)).__name__
            except AttributeError:
                type_name = node.target
            sig.append(("call_module", type_name))
    return tuple(sig)


def make_layer_partitioner(
    gm: fx.GraphModule,
) -> tuple[Callable[[fx.Node], int], dict[int, str]]:
    """
    Analyse ``gm`` and return a ``(callback, labels)`` pair.

    ``callback(node) -> int`` assigns each node a partition ID.  The splitting
    prefix is auto-detected from the graph: whichever ``word_N`` pattern
    appears with the most distinct values of N is used.  Nodes that carry no
    matching tag are absorbed into the most recent partition (or a
    ``"prologue"`` partition if no layer has been seen yet).

    ``labels`` maps each partition ID to a human-readable name derived from
    the detected layer tag (e.g. ``"layer_0"``, ``"stage_2"``).

    The callback is suitable for direct use with
    ``torch.fx.passes.split_module.split_module(gm, gm, callback)``.

    Works for both complex models (where dynamo encodes the module path into
    ``call_module`` node names) and simple inlined models (where computation
    nodes carry generic names like ``"linear"`` but their parameter
    placeholder inputs carry the full module path).
    """
    pattern = _detect_split_pattern(gm) or LAYER_RE
    log.debug("make_layer_partitioner: splitting by pattern %r", pattern.pattern)

    # Pre-index parameter placeholder names → layer key.  Dynamo embeds the
    # full module path in placeholder names even when computation node names
    # are generic (e.g. "l_self_modules_stage_0_modules_linear_parameters_weight_").
    placeholder_key: dict[str, str] = {}
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            m = pattern.search(node.name)
            if m:
                placeholder_key[node.name] = m.group(0)

    node_pid: dict[str, int] = {}
    pid_to_label: dict[int, str] = {}
    current_pid = 0
    last_layer_key: str | None = None

    for node in gm.graph.nodes:
        if node.op in ("placeholder", "output"):
            continue

        # Primary: match node's own name (works when dynamo uses call_module
        # nodes whose names embed the full module path).
        m = pattern.search(node.name)
        layer_key = m.group(0) if m else None

        # Fallback: any direct parameter-placeholder input carries the path
        # (works when dynamo inlines everything into call_function nodes).
        if layer_key is None:
            for inp in node.all_input_nodes:
                if inp.op == "placeholder":
                    k = placeholder_key.get(inp.name)
                    if k:
                        layer_key = k
                        break

        if layer_key is not None and layer_key != last_layer_key:
            if pid_to_label:
                current_pid += 1
            pid_to_label[current_pid] = layer_key
            last_layer_key = layer_key
        elif current_pid not in pid_to_label:
            pid_to_label[current_pid] = "prologue"

        node_pid[node.name] = current_pid

    log.debug("make_layer_partitioner: %d partition(s) from %d nodes", len(pid_to_label), len(node_pid))

    def _callback(node: fx.Node) -> int:
        return node_pid.get(node.name, 0)

    return _callback, pid_to_label
