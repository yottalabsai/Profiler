---
name: backend
description: Generate a production-ready workload_optimized.py with a custom torch.compile() backend implementing the transformations in optimizations.json. Each optimization becomes a named FX graph pass. Also generates a validation test script and implementation_notes.md.
---

# /backend

## Usage

```
/backend workload.py optimizations.json
/backend workload.py optimizations.json --profile=profile.json
```

## Flags

| Flag | Default | Agent instruction |
|---|---|---|
| `--profile=profile.json` | none | "Cross-validate shape and dtype assumptions against profile.json." |

## Execution

Delegates to: backend-engineer

Translate any flags present into their agent instructions above and include them in the human-turn prompt alongside the workload and optimizations paths.
