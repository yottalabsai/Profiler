"""
dynamo_graph_analysis.py — Empirical analysis of torch._dynamo graph compilation
for HuggingFace transformer models.

Theory under test: transformers have a small number of compiled graphs that are
reused across forward passes — dynamo compiles once per unique graph segment,
then caches and replays it.

Requires:
    pip install transformers   # not in requirements.txt — install manually
    pip install rich           # already in requirements.txt

Usage:
    python examples/transformer_dynamo_analysis/dynamo_graph_analysis.py

    # Verbose graph-break info:
    TORCH_LOGS=graph_breaks python examples/transformer_dynamo_analysis/dynamo_graph_analysis.py
"""
from __future__ import annotations

import sys
from collections import Counter

import torch
import torch.fx as fx
from torch.fx.passes.split_module import split_module

from nvidia.operator_profiler.capture.layer_graph_splitter import (
    LAYER_RE,
    graph_signature,
    make_layer_partitioner,
)

# ---------------------------------------------------------------------------
# Dependency check — transformers is not in requirements.txt
# ---------------------------------------------------------------------------
try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    print(
        "Error: 'transformers' is not installed.\n"
        "Install it with:\n"
        "    pip install transformers\n"
        "Then re-run this script."
    )
    sys.exit(1)

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    _RICH = True
except ImportError:
    _RICH = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODELS = [
    "distilbert-base-uncased",  # 66M  — BERT distilled, encoder
    "bert-base-uncased",        # 110M — classic BERT, encoder
    "roberta-base",             # 125M — BERT variant with BPE tokenizer
    "gpt2",                     # 124M — decoder-only, causal attention
]
MAX_LENGTH = 128
N_RUNS     = 5
DEVICE     = "cpu"   # CPU avoids CUDA driver dependency for this analysis

SENTENCE = (
    "The transformer architecture relies on self-attention mechanisms "
    "to capture long-range dependencies in sequential data."
)

# ---------------------------------------------------------------------------
# Backend factory — returns a fresh (backend_fn, state) pair per model
# so there is no shared global state between runs.
# ---------------------------------------------------------------------------

def make_backend() -> tuple:
    state: dict = {"count": 0, "graphs": [], "example_inputs": None}

    def backend(gm: fx.GraphModule, example_inputs: list) -> callable:
        state["count"] += 1
        state["graphs"].append((state["count"], gm))
        if state["example_inputs"] is None:
            state["example_inputs"] = example_inputs
        return gm.forward   # eager execution — no optimization, pure capture

    return backend, state


# ---------------------------------------------------------------------------
# Per-graph analysis
# ---------------------------------------------------------------------------

def analyze_graph(gm: fx.GraphModule) -> dict:
    op_counts: Counter[str] = Counter()
    cf_targets: Counter[str] = Counter()

    for node in gm.graph.nodes:
        op_counts[node.op] += 1
        if node.op == "call_function":
            target = getattr(node.target, "__name__", str(node.target))
            cf_targets[target] += 1

    total = sum(op_counts.values())
    top_cf = ", ".join(t for t, _ in cf_targets.most_common(5)) or "—"

    return {
        "total": total,
        "placeholders": op_counts["placeholder"],
        "call_function": op_counts["call_function"],
        "call_module": op_counts["call_module"],
        "top_cf": top_cf,
    }


def param_count_m(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def demo_layer_split(gm: fx.GraphModule, model_name: str, example_inputs: list) -> None:
    """Split the captured FX graph by layer, detect unique subgraphs, verify reconstruction."""
    print(f"\n{'─' * 70}")
    print(f"  Layer-split: {model_name}")
    print(f"{'─' * 70}")

    callback, labels = make_layer_partitioner(gm)
    split = split_module(gm, gm, callback)

    # Collect per-partition stats and signatures
    rows: list[dict] = []
    sigs: dict[int, tuple] = {}
    for name, submod in split.named_children():
        if not isinstance(submod, fx.GraphModule):
            continue
        pid   = int(name.replace("submod_", ""))
        stats = analyze_graph(submod)
        sig   = graph_signature(submod)
        sigs[pid] = sig
        stats["label"] = labels.get(pid, name)
        stats["pid"]   = pid
        rows.append(stats)

    # Mark duplicates against previously seen signatures
    seen: dict[tuple, str] = {}
    for r in rows:
        sig = sigs[r["pid"]]
        if sig in seen:
            r["structure"] = f"same as {seen[sig]}"
        else:
            seen[sig] = r["label"]
            r["structure"] = "unique"

    # Print table
    if _RICH:
        console = Console()
        table = Table(
            title=f"Layer Split — {model_name}",
            box=box.SIMPLE_HEAVY,
            show_lines=True,
        )
        table.add_column("Partition",     style="cyan")
        table.add_column("Nodes",         justify="right")
        table.add_column("call_function", justify="right")
        table.add_column("Top ops",       overflow="fold")
        table.add_column("Structure",     style="dim")
        for r in rows:
            table.add_row(
                r["label"],
                str(r["total"]),
                str(r["call_function"]),
                r["top_cf"],
                r["structure"],
            )
        console.print(table)
    else:
        hdr = f"{'Partition':<16}  {'Nodes':>5}  {'call_fn':>7}  {'Top ops':<50}  Structure"
        print(hdr)
        print("-" * len(hdr))
        for r in rows:
            print(f"{r['label']:<16}  {r['total']:>5}  {r['call_function']:>7}  "
                  f"{r['top_cf']:<50}  {r['structure']}")

    # Verify reconstruction: split module must produce identical outputs
    print()
    with torch.no_grad():
        out_orig  = gm(*example_inputs)
        out_split = split(*example_inputs)

    def first_tensor(out):
        return out[0] if isinstance(out, tuple) else out.last_hidden_state

    max_err = (first_tensor(out_orig) - first_tensor(out_split)).abs().max().item()
    verdict = "✓ numerically identical" if max_err < 1e-5 else "✗ mismatch"
    print(f"  Reconstruction max abs error: {max_err:.2e}  ({verdict})")


# ---------------------------------------------------------------------------
# Per-model analysis
# ---------------------------------------------------------------------------

def run_model(model_name: str) -> dict | None:
    """Load, explain, and profile one model. Returns a summary dict or None on error."""
    console = Console() if _RICH else None

    def out(msg: str = "") -> None:
        print(msg)

    out(f"\n{'─' * 70}")
    out(f"  Model: {model_name}")
    out(f"{'─' * 70}")

    # -- Load --
    out("  Loading …")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(DEVICE).eval()
    except Exception as e:
        out(f"  ERROR loading model: {e}")
        return None

    params_m = param_count_m(model)
    out(f"  Parameters: {params_m:.1f}M")

    # GPT-2 tokenizer has no pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    enc = tokenizer(
        SENTENCE,
        return_tensors="pt",
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
    )
    input_ids      = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    # -- explain() --
    out("  Running torch._dynamo.explain() …")
    torch._dynamo.reset()
    try:
        with torch.no_grad():
            explanation = torch._dynamo.explain(model)(
                input_ids, attention_mask=attention_mask
            )
        explain_graphs = explanation.graph_count
        break_reasons  = explanation.break_reasons or []
        explain_breaks = len(break_reasons)
    except Exception as e:
        out(f"  explain() failed: {e}")
        explain_graphs = -1
        explain_breaks = -1
        break_reasons  = []

    out(f"  explain() → {explain_graphs} graph(s), {explain_breaks} break(s)")
    for br in break_reasons:
        reason = getattr(br, "reason", str(br))
        out(f"    • {reason}")

    # -- Reuse capture --
    out(f"  Running {N_RUNS} forward passes with capturing_backend …")
    backend, state = make_backend()
    torch._dynamo.reset()
    compiled = torch.compile(model, backend=backend)
    try:
        with torch.no_grad():
            for _ in range(N_RUNS):
                compiled(input_ids, attention_mask=attention_mask)
    except Exception as e:
        out(f"  Forward pass failed: {e}")
        return None

    n_compiled = state["count"]
    out(f"  Backend called {n_compiled} time(s) across {N_RUNS} runs")

    # -- Per-graph table --
    rows = []
    for idx, gm in state["graphs"]:
        stats = analyze_graph(gm)
        stats["idx"] = idx
        rows.append(stats)

    if _RICH and console is not None:
        table = Table(
            title=f"Graph Details — {model_name}",
            box=box.SIMPLE_HEAVY,
            show_lines=True,
        )
        table.add_column("Graph #",      justify="center", style="cyan")
        table.add_column("Nodes",        justify="right")
        table.add_column("Placeholders", justify="right")
        table.add_column("call_function",justify="right")
        table.add_column("call_module",  justify="right")
        table.add_column("Top call_function targets", overflow="fold")
        for r in rows:
            table.add_row(
                str(r["idx"]),
                str(r["total"]),
                str(r["placeholders"]),
                str(r["call_function"]),
                str(r["call_module"]),
                r["top_cf"],
            )
        console.print(table)
    else:
        hdr = f"{'#':>4}  {'Nodes':>6}  {'Pholds':>6}  {'call_fn':>7}  {'call_mod':>8}  Top ops"
        out(hdr)
        out("-" * len(hdr))
        for r in rows:
            out(f"{r['idx']:>4}  {r['total']:>6}  {r['placeholders']:>6}  "
                f"{r['call_function']:>7}  {r['call_module']:>8}  {r['top_cf']}")

    # -- Reuse verdict --
    if n_compiled < N_RUNS:
        ratio = N_RUNS / n_compiled
        out(f"  ✓ Reuse confirmed — {N_RUNS} runs / {n_compiled} compile(s) = {ratio:.2f}x reuse")
    elif n_compiled == N_RUNS:
        out("  ✗ No reuse — every run triggered a fresh compile")
    else:
        out(f"  ✗ Unexpected: {n_compiled} compilations for {N_RUNS} runs")

    total_nodes = sum(r["total"] for r in rows)
    captured_gm = state["graphs"][0][1] if state["graphs"] else None

    return {
        "model":          model_name,
        "params_m":       params_m,
        "explain_graphs": explain_graphs,
        "explain_breaks": explain_breaks,
        "compiled_graphs":n_compiled,
        "total_nodes":    total_nodes,
        "reuse_ratio":    N_RUNS / n_compiled if n_compiled else 0,
        "reuse":          n_compiled < N_RUNS,
        "gm":             captured_gm,
        "example_inputs": state["example_inputs"],
    }


# ---------------------------------------------------------------------------
# Comparative summary
# ---------------------------------------------------------------------------

def print_comparison(summaries: list[dict]) -> None:
    if not summaries:
        return

    if _RICH:
        console = Console()
        table = Table(
            title="Cross-Model Comparison",
            box=box.ROUNDED,
            show_lines=True,
        )
        table.add_column("Model",          style="bold")
        table.add_column("Params",         justify="right")
        table.add_column("Graphs\n(explain)", justify="center")
        table.add_column("Breaks",         justify="center")
        table.add_column("Compiled\ngraphs", justify="center")
        table.add_column("Total\nnodes",   justify="right")
        table.add_column("Reuse\nratio",   justify="right")
        table.add_column("Reuse?",         justify="center")

        for s in summaries:
            table.add_row(
                s["model"].split("/")[-1],
                f"{s['params_m']:.0f}M",
                str(s["explain_graphs"]),
                str(s["explain_breaks"]),
                str(s["compiled_graphs"]),
                str(s["total_nodes"]),
                f"{s['reuse_ratio']:.2f}x",
                "✓" if s["reuse"] else "✗",
            )
        console.print()
        console.print(table)
    else:
        print()
        print("=== Cross-Model Comparison ===")
        hdr = f"{'Model':<30}  {'Params':>7}  {'Graphs':>6}  {'Breaks':>6}  {'Compiled':>8}  {'Nodes':>6}  {'Reuse':>7}  OK?"
        print(hdr)
        print("-" * len(hdr))
        for s in summaries:
            name = s["model"].split("/")[-1]
            print(
                f"{name:<30}  {s['params_m']:>6.0f}M  {s['explain_graphs']:>6}  "
                f"{s['explain_breaks']:>6}  {s['compiled_graphs']:>8}  {s['total_nodes']:>6}  "
                f"{s['reuse_ratio']:>6.2f}x  {'✓' if s['reuse'] else '✗'}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    summaries = []
    for model_name in MODELS:
        result = run_model(model_name)
        if result is not None:
            summaries.append(result)

    print_comparison(summaries)

    for s in summaries:
        if s.get("gm") is not None and s.get("example_inputs") is not None:
            demo_layer_split(s["gm"], s["model"], s["example_inputs"])

    print()


if __name__ == "__main__":
    main()
