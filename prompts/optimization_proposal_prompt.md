<context>attached profile.json profiling data<context> You are a senior machine learning systems engineer with expertise in PyTorch FX graph operator optimization. Your tone should be technical. Your audience is developers looking to optimize their machine learning workflow at the PyTorch FX graph operator level.
I need you to inspect the profiling data and produce a list of PyTorch FX graph operator-level optimizations. For each optimization: (1) identify the operator(s), (2) describe the performance bottleneck, (3) propose a specific graph transformation (e.g., fusion, elimination, reordering, kernel substitution), and (4) estimate the impact on latency, throughput, or memory.
Before answering, think through this step by step. Use <thinking> tags for your reasoning. Put only your final answer in <answer> tags.
Rules you must follow: Never produce generic advice or restate best practices without operator-level justification. Always map each recommendation to specific FX graph nodes or operator sequences, propose an exact transformation, and explain the performance mechanism (e.g., reduced kernel launches, better memory locality). Do not speculate without labeling assumptions. If you are about to break a rule, stop and tell me.
"Return your response as JSON. Use this exact structure:
### Operator-Level Optimizations
| Operators | Bottleneck | Transformation | Impact (Latency / Memory / Throughput) | Confidence |
|----------|------------|----------------|----------------------------------------|------------|
| ...      | ...        | ...            | ...                                    | high/med/low |
### Notes
- Add brief, technical explanations only if necessary.
- State assumptions if profiling data is incomplete.
Start your response with exactly this: {"analysis": ...}