# Canonical FX Graph Pass Patterns

Complete, copy-pasteable implementations for the most common GPU optimization passes. Each pattern includes detection, transformation, lint+recompile, and error handling.

All patterns assume the standard imports:
```python
import logging
import operator
import torch
import torch.fx as fx
import torch.nn.functional as F
from torch._dynamo import register_backend
from torch._inductor.compile_fx import compile_fx  # function, not module

logger = logging.getLogger(__name__)
```

---

## Pre-Inductor vs Post-Inductor IR

`@register_backend` functions receive the graph **before** Inductor lowers it. At this level ops are Python-level callables, not aten primitives:

| Source code | Pre-Inductor node (what your pass sees) |
|---|---|
| `nn.Linear(x)` | `call_function: F.linear` — weight is a `placeholder` node |
| `x @ y` | `call_function: operator.matmul` |
| `x * scale` | `call_function: operator.mul` |
| `x / scale` | `call_function: operator.truediv` |
| `torch.softmax(x, dim)` | `call_function: torch.softmax` (same object as `F.softmax`) |
| `x.tanh()` | `call_method: "tanh"` or `call_function: torch.tanh` |
| `nn.Conv2d(x)` | `call_function: F.conv2d` — weight is a `placeholder` node |
| `nn.BatchNorm2d(x)` | `call_function: F.batch_norm` — all stats are `placeholder` nodes |
| `x.transpose(-2, -1)` | `call_method: "transpose"` |
| Any `nn.Module` parameter | `placeholder` node (Dynamo lifts ALL params to function args) |

Inductor later decomposes these: `F.linear → mm + t`, `torch.softmax → amax + sub + exp + sum + div`, etc. Passes that target `aten.mm.default` or `aten._softmax.default` find nothing when applied to the pre-Inductor graph.

---

## Pass Taxonomy

Before writing any FX pass, classify each optimization into one of three categories:

| Category | Criteria | How to apply |
|---|---|---|
| **`replace_pattern`-compatible** | Pure functional: no tuple outputs, no `register_buffer`, no `call_method` nodes | Define `_pattern_fn` + `_replacement_fn`; call `runner.apply_pass(_pattern_fn, _replacement_fn)` |
| **Manual per-rep** | Requires `register_buffer`, involves `call_method` nodes, or needs actual tensor values from `partition_inputs` | Write `def _pass_name(gm, partition_inputs) -> gm`; call per unique rep, then propagate to duplicates |
| **Non-graph** | dtype, memory_format, batch shape — not visible in FX IR | Stays in `get_model_and_input()`, not in the backend at all |

**Why SDPA replacement must be manual per-rep:** The pattern involves `call_method "transpose"` nodes, which `replace_pattern` cannot match. It also requires extracting the pre-transposed `k` from the transpose node's input.

**Why BN fold must be manual per-rep:** Folding writes new weight tensors via `register_buffer`. `replace_pattern` cannot call `register_buffer` on the module.

**Why QKV fusion must be manual per-rep:** The weight tensors are `placeholder` nodes — their actual values must be retrieved from `partition_inputs` (captured by running the partition once). `replace_pattern` operates purely structurally and cannot access tensor values.

**`replace_pattern`-compatible:** tanh→GELU substitution (Pattern 4). QKV and SDPA are manual per-rep.

---

## Utility: `_capture_partition_inputs`

Required for any pass that needs to read actual weight tensor values. Dynamo lifts all `nn.Module` parameters as `placeholder` nodes, so the values flow in as function arguments — the `partition_inputs` list gives the tensor at each placeholder's position.

```python
def _capture_partition_inputs(
    split_gm: fx.GraphModule,
    example_inputs: list,
) -> dict[str, list]:
    """Capture actual input tensors for each partition by running split_gm once."""
    partition_inputs: dict[str, list] = {}
    hooks = []
    for name, submod in split_gm.named_children():
        if isinstance(submod, fx.GraphModule):
            def _hook(mod, args, _name=name):
                partition_inputs[_name] = list(args)
            hooks.append(submod.register_forward_pre_hook(_hook))
    with torch.no_grad():
        split_gm(*example_inputs)
    for h in hooks:
        h.remove()
    return partition_inputs
```

Usage in the backend — call this BEFORE applying any pass that needs weight values:
```python
partition_inputs = _capture_partition_inputs(registry.split, example_inputs)
for rep_name, rep_mod in registry.unique_reps:
    inputs = partition_inputs.get(rep_name, example_inputs)
    _pass_fuse_qkv(rep_mod, inputs)
    _pass_replace_sdpa(rep_mod)
    compiled = compile_fx(rep_mod, inputs)
```

---

## Pattern 1: QKV Weight Fusion

Fuses 3 independent `F.linear(x, W_q)`, `F.linear(x, W_k)`, `F.linear(x, W_v)` calls sharing the same input `x` into a single `F.linear(x, W_qkv)` + `torch.chunk(3, dim=-1)`.

**Detection signal:** Three `F.linear` nodes whose first argument is the same node.

**Weight access:** weights are `placeholder` nodes — resolve actual tensors from `partition_inputs` by matching placeholder node to its position in the inputs list.

```python
def _pass_fuse_qkv(gm: fx.GraphModule, partition_inputs: list) -> fx.GraphModule:
    try:
        # Map placeholder node → actual tensor value
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        ph_to_tensor = {ph: t for ph, t in zip(placeholders, partition_inputs)}

        # Group F.linear calls by their first argument (shared x input)
        lin_groups: dict[str, list] = {}
        for n in gm.graph.nodes:
            if n.op == "call_function" and n.target is F.linear:
                lin_groups.setdefault(n.args[0].name, []).append(n)

        fused = False
        for x_name, lin_list in lin_groups.items():
            if len(lin_list) < 3:
                continue
            q_lin, k_lin, v_lin = lin_list[0], lin_list[1], lin_list[2]

            W_q = ph_to_tensor.get(q_lin.args[1])
            W_k = ph_to_tensor.get(k_lin.args[1])
            W_v = ph_to_tensor.get(v_lin.args[1])
            if W_q is None or W_k is None or W_v is None:
                logger.warning("[pass_fuse_qkv] Weight tensors not in partition inputs")
                continue

            # Validate shapes are fusible
            if not (W_q.shape[1] == W_k.shape[1] == W_v.shape[1]):
                logger.warning("[pass_fuse_qkv] Weight K dims differ — skipping")
                continue

            W_qkv = torch.cat([W_q, W_k, W_v], dim=0)
            gm.register_buffer("_fused_qkv_weight", W_qkv)

            with gm.graph.inserting_before(q_lin):
                w_buf = gm.graph.get_attr("_fused_qkv_weight")
                fused_lin = gm.graph.call_function(F.linear, (q_lin.args[0], w_buf))
                chunks = gm.graph.call_function(torch.chunk, (fused_lin, 3), {"dim": -1})
                q_out = gm.graph.call_function(operator.getitem, (chunks, 0))
                k_out = gm.graph.call_function(operator.getitem, (chunks, 1))
                v_out = gm.graph.call_function(operator.getitem, (chunks, 2))

            q_lin.replace_all_uses_with(q_out)
            k_lin.replace_all_uses_with(k_out)
            v_lin.replace_all_uses_with(v_out)
            for dead in (q_lin, k_lin, v_lin):
                gm.graph.erase_node(dead)

            gm.graph.lint()
            gm.recompile()
            logger.info("[pass_fuse_qkv] Fused 3 F.linear into 1 (input '%s')", x_name)
            fused = True
            break  # one fusion per unique rep; call again for multiple attention heads

        if not fused:
            logger.warning("[pass_fuse_qkv] Pattern not found — pass not applied")
    except Exception as e:
        logger.warning("[pass_fuse_qkv] Failed: %s", e)
    return gm
```

---

## Pattern 2: SDPA Replacement (FlashAttention)

Replaces `operator.matmul(softmax(operator.matmul(q, k_t) * scale), v)` with `F.scaled_dot_product_attention(q, k, v)`.

**Detection anchor:** The final `operator.matmul` that multiplies attention weights by V. Walk backwards through `softmax → scale → qk_matmul → transpose` to verify the full attention pattern.

**K transpose:** `F.scaled_dot_product_attention` transposes K internally. Extract the pre-transposed K from the `call_method "transpose"` node's first argument.

```python
def _pass_replace_sdpa(gm: fx.GraphModule) -> fx.GraphModule:
    try:
        replaced = 0
        for n in list(gm.graph.nodes):
            # Anchor on the final matmul: out = attn @ v
            if n.op != "call_function" or n.target is not operator.matmul:
                continue

            attn_node, v_node = n.args[0], n.args[1]

            # attn must be softmax output
            if not (attn_node.op == "call_function"
                    and attn_node.target is torch.softmax):
                continue

            # softmax input: scaled scores = qk * scale
            scaled_node = attn_node.args[0]
            if not (scaled_node.op == "call_function"
                    and scaled_node.target is operator.mul):
                continue

            # scores: q @ k_t
            qk_node = scaled_node.args[0]
            if not (qk_node.op == "call_function"
                    and qk_node.target is operator.matmul):
                continue

            q_node, k_t_node = qk_node.args[0], qk_node.args[1]

            # Unwrap k.transpose(-2, -1) to get bare k
            # F.scaled_dot_product_attention transposes k internally
            if k_t_node.op == "call_method" and k_t_node.target == "transpose":
                k_node = k_t_node.args[0]
            else:
                logger.warning("[pass_replace_sdpa] k_t is not call_method transpose — skipping")
                continue

            with gm.graph.inserting_before(n):
                sdpa = gm.graph.call_function(
                    F.scaled_dot_product_attention,
                    (q_node, k_node, v_node),
                )

            n.replace_all_uses_with(sdpa)
            for dead in (n, attn_node, scaled_node, qk_node):
                try:
                    if not dead.users:
                        gm.graph.erase_node(dead)
                except Exception:
                    pass
            replaced += 1

        if replaced:
            gm.graph.lint()
            gm.recompile()
            logger.info("[pass_replace_sdpa] Replaced %d attention block(s) with SDPA", replaced)
        else:
            logger.warning("[pass_replace_sdpa] Attention pattern not found — pass not applied")
    except Exception as e:
        logger.warning("[pass_replace_sdpa] Failed: %s", e)
    return gm
```

---

## Pattern 3: Conv-BN Fold (Inference)

Folds `F.batch_norm(training=False)` into the preceding `F.conv2d` by absorbing the BN scale/shift into conv weights and bias.

**Note:** In pre-Inductor form, `F.batch_norm` with `training=False` returns a single tensor (not a tuple). The post-Inductor `getitem(0)` workaround is not needed here.

**Weight access:** All BN and conv parameters are `placeholder` nodes — resolve via `partition_inputs`.

```python
def _pass_fold_bn(gm: fx.GraphModule, partition_inputs: list) -> fx.GraphModule:
    try:
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        ph_to_tensor = {ph: t for ph, t in zip(placeholders, partition_inputs)}

        for bn_node in list(gm.graph.nodes):
            if bn_node.op != "call_function" or bn_node.target is not F.batch_norm:
                continue

            # F.batch_norm args: (input, weight, bias, running_mean, running_var, training, ...)
            conv_node = bn_node.args[0]
            if conv_node.op != "call_function" or conv_node.target is not F.conv2d:
                continue

            # Resolve BN stats from partition_inputs
            bn_weight = ph_to_tensor.get(bn_node.args[1])
            bn_bias   = ph_to_tensor.get(bn_node.args[2])
            run_mean  = ph_to_tensor.get(bn_node.args[3])
            run_var   = ph_to_tensor.get(bn_node.args[4])
            training  = bn_node.args[5] if len(bn_node.args) > 5 else False
            eps       = bn_node.args[7] if len(bn_node.args) > 7 else 1e-5

            if training or any(t is None for t in (bn_weight, bn_bias, run_mean, run_var)):
                continue

            # F.conv2d args: (input, weight, bias, stride, padding, dilation, groups)
            conv_weight = ph_to_tensor.get(conv_node.args[1])
            conv_bias   = ph_to_tensor.get(conv_node.args[2]) if len(conv_node.args) > 2 else None

            if conv_weight is None:
                continue

            # Fold: scale BN into conv weight/bias
            scale = bn_weight / torch.sqrt(run_var + eps)
            new_weight = conv_weight * scale.view(-1, 1, 1, 1)
            if conv_bias is not None:
                new_bias = (conv_bias - run_mean) * scale + bn_bias
            else:
                new_bias = bn_bias - run_mean * scale

            gm.register_buffer("_folded_conv_weight", new_weight)
            gm.register_buffer("_folded_conv_bias", new_bias)

            with gm.graph.inserting_before(conv_node):
                fw = gm.graph.get_attr("_folded_conv_weight")
                fb = gm.graph.get_attr("_folded_conv_bias")

            # Rebuild conv args with folded weight/bias
            new_conv_args = (conv_node.args[0], fw, fb) + conv_node.args[3:]
            with gm.graph.inserting_before(bn_node):
                new_conv = gm.graph.call_function(F.conv2d, new_conv_args)

            bn_node.replace_all_uses_with(new_conv)
            gm.graph.erase_node(bn_node)
            if not conv_node.users:
                gm.graph.erase_node(conv_node)

            gm.graph.lint()
            gm.recompile()
            logger.info("[pass_fold_bn] Folded BatchNorm into Conv2d")
            break  # fold one BN per call; call again for multiple conv-BN pairs

    except Exception as e:
        logger.warning("[pass_fold_bn] Failed: %s", e)
    return gm
```

---

## Pattern 4: Activation Substitution (tanh → GELU)

Replaces `tanh` with `F.gelu(approximate='tanh')` when tanh appears in an FFN context (linear → tanh → linear). Avoids SFU pipeline serialization.

**Pre-Inductor forms of tanh:** Dynamo may trace `x.tanh()` as `call_method "tanh"` or `call_function torch.tanh` depending on whether it was called as a method or function. Both are matched below.

**`replace_pattern`-compatible** for the `call_function` form only. Use manual traversal to handle both forms.

```python
def _pass_tanh_to_gelu(gm: fx.GraphModule) -> fx.GraphModule:
    try:
        replaced = 0
        for node in list(gm.graph.nodes):
            is_tanh = (
                (node.op == "call_function" and node.target is torch.tanh)
                or (node.op == "call_method" and node.target == "tanh")
            )
            if not is_tanh:
                continue

            # FFN context: producer is F.linear AND at least one consumer is F.linear
            producer = node.args[0]
            is_ffn = (
                producer.op == "call_function" and producer.target is F.linear
                and any(
                    u.op == "call_function" and u.target is F.linear
                    for u in node.users
                )
            )
            if not is_ffn:
                continue

            with gm.graph.inserting_after(node):
                gelu = gm.graph.call_function(
                    F.gelu,
                    (node.args[0],),
                    {"approximate": "tanh"},
                )
            node.replace_all_uses_with(gelu)
            gm.graph.erase_node(node)
            replaced += 1

        if replaced:
            gm.graph.lint()
            gm.recompile()
            logger.info("[pass_tanh_to_gelu] Replaced %d tanh → gelu(tanh)", replaced)
    except Exception as e:
        logger.warning("[pass_tanh_to_gelu] Failed: %s", e)
    return gm
```

---

## Stub Pattern: LayerNorm-Linear Fusion

Detection only. Full implementation requires a custom Triton kernel.

```python
def _pass_fuse_ln_linear_stub(gm: fx.GraphModule) -> fx.GraphModule:
    """Detection stub — full implementation requires a custom Triton kernel."""
    try:
        for ln_node in gm.graph.nodes:
            if ln_node.op != "call_function" or ln_node.target is not F.layer_norm:
                continue
            for linear_node in ln_node.users:
                if linear_node.op == "call_function" and linear_node.target is F.linear:
                    logger.warning(
                        "[pass_fuse_ln_linear] LayerNorm→Linear detected but not applied "
                        "— requires custom Triton kernel (e.g. liger-kernel LN-MM)"
                    )
    except Exception as e:
        logger.warning("[pass_fuse_ln_linear_stub] Failed: %s", e)
    return gm  # gm is ALWAYS returned unchanged
```

---

## Channels-Last Conversion (Non-Graph)

Applied in `get_model_and_input()` — not an FX pass. `memory_format` is a tensor property, not visible in the FX graph.

```python
def apply_channels_last(model: torch.nn.Module) -> torch.nn.Module:
    has_conv = any(isinstance(m, torch.nn.Conv2d) for m in model.modules())
    if not has_conv:
        return model
    return model.to(memory_format=torch.channels_last)
```

```python
# In get_model_and_input():
if not next(model.parameters()).is_contiguous(memory_format=torch.channels_last):
    model = apply_channels_last(model)
    x = x.to(memory_format=torch.channels_last)
```
