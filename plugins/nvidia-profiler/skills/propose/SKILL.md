---
name: propose
description: Reads profile.json directly, derives time budget and edge case flags, then proposes concrete FX graph transformations with confidence ratings, evidence citations, and dependency ordering. Produces optimizations.json. Supports --max-opts, --min-confidence, --threshold, --focus, and --operator flags.
---

# /propose

## Usage

```
/propose profile.json
/propose profile.json --max-opts=5
/propose profile.json --min-confidence=high
/propose profile.json --threshold=5 --focus=memory
/propose profile.json --operator=aten::linear
/propose profile_v1.json profile_v2.json
```

## Flags

| Flag | Default | Agent instruction |
|---|---|---|
| `--max-opts=N` | unlimited | "Include at most N proposals, selecting by highest priority." |
| `--min-confidence=X` | none | "Only include proposals with confidence 'X' or above." (`high`, `medium`, `low`) |
| `--threshold=PCT` | 1 | "Only analyze operators contributing at least PCT% of total profiled time." |
| `--focus=TYPE` | none | "Restrict proposals to TYPE bottlenecks only." (`compute`, `memory`, `launch_overhead`) |
| `--operator=NAME` | none | "Focus analysis on the operator NAME only; omit all others from the time budget." |

## Execution

Delegates to: optimization-strategist

Translate any flags present into their agent instructions above and include them in the human-turn prompt alongside the profile path.
