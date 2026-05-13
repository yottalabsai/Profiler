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

**`replace_pattern`-compatible:** tanh→GELU substitution only.
**Manual per-rep:** QKV fusion, SDPA, BN fold, SiLU/GEGLU fusion, pre-transposed weights.

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

## Pattern 5: Pre-Transposed Weight Buffer

Eliminates runtime transpose overhead on `F.linear` / `operator.matmul` calls where the weight is transposed on every forward pass. Pre-stores `weight.T.contiguous()` as a buffer so cuBLAS receives a contiguous row-major matrix.

**Detection signal:** `call_method "t"` node whose arg is a `placeholder`, consumed as the weight argument of `F.linear` or `operator.matmul`. Co-occurs with `gemmSN_TN` kernel name in the profile.

**Mutual exclusion:** Do NOT apply if `_pass_fuse_qkv` already ran on the same weight nodes — QKV fusion eliminates the original placeholder nodes, leaving no `t()` node to rewrite.

**Why `matmul` not `F.linear`:** `F.linear(x, w)` computes `x @ w.T` internally. If `w` is already the transposed buffer, `F.linear(x, w_T)` = `x @ w_T.T` = `x @ w_original` — wrong. Use `operator.matmul(x, w_T)` directly.

```python
def _pass_pretranspose_weights(gm: fx.GraphModule, partition_inputs: list) -> fx.GraphModule:
    try:
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        ph_to_tensor = {ph: t for ph, t in zip(placeholders, partition_inputs)}

        replaced = 0
        for node in list(gm.graph.nodes):
            # Find: call_method "t" on a placeholder
            if not (node.op == "call_method" and node.target == "t"):
                continue
            ph_node = node.args[0]
            if ph_node.op != "placeholder":
                continue
            weight = ph_to_tensor.get(ph_node)
            if weight is None or not isinstance(weight, torch.Tensor):
                continue
            # Skip small weights — register_buffer overhead not worth it below 512×512
            if weight.shape[0] < 512:
                continue

            # Find consumers: F.linear or operator.matmul using this t() node as weight
            for user in list(node.users):
                if not (
                    (user.op == "call_function" and user.target is F.linear and user.args[1] is node)
                    or (user.op == "call_function" and user.target is operator.matmul and user.args[1] is node)
                ):
                    continue

                buf_name = f"_pretransposed_weight_{replaced}"
                weight_T = weight.T.contiguous()
                gm.register_buffer(buf_name, weight_T)

                with gm.graph.inserting_before(user):
                    buf_node = gm.graph.get_attr(buf_name)
                    if user.target is F.linear:
                        # Replace F.linear(x, t_node, bias?) with matmul(x, w_T) [+ bias]
                        x_node = user.args[0]
                        bias_node = user.args[2] if len(user.args) > 2 else None
                        new_mm = gm.graph.call_function(operator.matmul, (x_node, buf_node))
                        if bias_node is not None:
                            new_node = gm.graph.call_function(operator.add, (new_mm, bias_node))
                        else:
                            new_node = new_mm
                    else:
                        # matmul(x, t_node) → matmul(x, w_T)
                        new_node = gm.graph.call_function(
                            operator.matmul, (user.args[0], buf_node)
                        )

                user.replace_all_uses_with(new_node)
                gm.graph.erase_node(user)
                replaced += 1

        if replaced:
            # Erase orphaned t() nodes
            for node in list(gm.graph.nodes):
                if node.op == "call_method" and node.target == "t" and not node.users:
                    gm.graph.erase_node(node)
            gm.graph.lint()
            gm.recompile()
            logger.info("[pass_pretranspose_weights] Pre-transposed %d weight(s)", replaced)
        else:
            logger.warning("[pass_pretranspose_weights] Pattern not found — pass not applied")
    except Exception as e:
        logger.warning("[pass_pretranspose_weights] Failed: %s", e)
    return gm
```

---

## Pattern 6: SiLU/GEGLU Gated Activation Fusion

Fuses the LLaMA/Mistral FFN gated activation pattern — two parallel `F.linear` calls from the same input `x`, one fed through `F.silu` then element-wise multiplied with the other — into a single wider `F.linear` + `torch.chunk` + `F.silu` + `operator.mul`.

**Detection signal:** `call_function F.silu` whose input is `F.linear(x, W_gate)`, consumed by an `operator.mul` that also takes `F.linear(x, W_up)` with the same `x` node.

**Model context:** `gate_proj` + `up_proj` in HuggingFace LLaMA/Mistral naming.

**Effect:** 2 separate GEMM launches → 1. Eliminates W_up GEMM launch and its DRAM traffic.

```python
def _pass_fuse_silu_geglu(gm: fx.GraphModule, partition_inputs: list) -> fx.GraphModule:
    try:
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        ph_to_tensor = {ph: t for ph, t in zip(placeholders, partition_inputs)}

        fused = False
        for silu_node in list(gm.graph.nodes):
            if not (silu_node.op == "call_function" and silu_node.target is F.silu):
                continue

            gate_lin = silu_node.args[0]
            if not (gate_lin.op == "call_function" and gate_lin.target is F.linear):
                continue
            x_node = gate_lin.args[0]

            # Find mul node consuming silu output
            mul_node = None
            for user in silu_node.users:
                if user.op == "call_function" and user.target is operator.mul:
                    mul_node = user
                    break
            if mul_node is None:
                continue

            # Find the other operand of mul: F.linear(x, W_up)
            other = mul_node.args[1] if mul_node.args[0] is silu_node else mul_node.args[0]
            if not (other.op == "call_function" and other.target is F.linear):
                continue
            if other.args[0] is not x_node:
                continue

            W_gate = ph_to_tensor.get(gate_lin.args[1])
            W_up   = ph_to_tensor.get(other.args[1])
            if W_gate is None or W_up is None:
                logger.warning("[pass_fuse_silu_geglu] Weight tensors not in partition inputs")
                continue
            if W_gate.shape != W_up.shape:
                logger.warning("[pass_fuse_silu_geglu] W_gate/W_up shapes differ — skipping")
                continue

            W_fused = torch.cat([W_gate, W_up], dim=0)
            gm.register_buffer("_fused_gate_up_weight", W_fused)

            with gm.graph.inserting_before(gate_lin):
                w_buf       = gm.graph.get_attr("_fused_gate_up_weight")
                fused_lin   = gm.graph.call_function(F.linear, (x_node, w_buf))
                chunks      = gm.graph.call_function(torch.chunk, (fused_lin, 2), {"dim": -1})
                gate_out    = gm.graph.call_function(operator.getitem, (chunks, 0))
                up_out      = gm.graph.call_function(operator.getitem, (chunks, 1))
                gated       = gm.graph.call_function(F.silu, (gate_out,))
                fused_out   = gm.graph.call_function(operator.mul, (gated, up_out))

            mul_node.replace_all_uses_with(fused_out)
            for dead in (mul_node, silu_node, gate_lin, other):
                if not dead.users:
                    try:
                        gm.graph.erase_node(dead)
                    except Exception:
                        pass

            gm.graph.lint()
            gm.recompile()
            logger.info("[pass_fuse_silu_geglu] Fused gate_proj + up_proj into single linear")
            fused = True
            break  # one fusion per call; call again for multiple FFN blocks

        if not fused:
            logger.warning("[pass_fuse_silu_geglu] SiLU/GEGLU pattern not found — pass not applied")
    except Exception as e:
        logger.warning("[pass_fuse_silu_geglu] Failed: %s", e)
    return gm
```

---

## Stub Pattern: Grouped Query Attention (GQA) Detection

Detection only. Full implementation requires `F.scaled_dot_product_attention(enable_gqa=True)` (PyTorch ≥ 2.3) and explicit K/V head expansion.

```python
def _pass_detect_gqa_stub(gm: fx.GraphModule, partition_inputs: list) -> fx.GraphModule:
    """Detection stub — reports GQA head ratio; does not transform the graph."""
    try:
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        ph_to_tensor = {ph: t for ph, t in zip(placeholders, partition_inputs)}

        # Group F.linear nodes by input activation
        lin_groups: dict[str, list] = {}
        for n in gm.graph.nodes:
            if n.op == "call_function" and n.target is F.linear:
                lin_groups.setdefault(n.args[0].name, []).append(n)

        for x_name, lins in lin_groups.items():
            if len(lins) < 3:
                continue
            out_dims = []
            for ln in lins:
                w = ph_to_tensor.get(ln.args[1])
                if w is not None:
                    out_dims.append(w.shape[0])
            if len(set(out_dims)) > 1:
                q_dim = max(out_dims)
                kv_dim = min(out_dims)
                ratio = q_dim // kv_dim if kv_dim > 0 else "?"
                logger.warning(
                    "[pass_detect_gqa] GQA detected on input '%s': Q heads=%d, KV heads=%d "
                    "(expansion ratio %s). Use F.scaled_dot_product_attention(enable_gqa=True) "
                    "(requires torch >= 2.3). No transformation applied.",
                    x_name, q_dim, kv_dim, ratio,
                )
    except Exception as e:
        logger.warning("[pass_detect_gqa_stub] Failed: %s", e)
    return gm
```

---

## Stub Pattern: Rotary Position Embedding (RoPE) Detection

Detection only. Fusion requires a custom Triton kernel (e.g., `flash-attn` `rotary_embedding` or `liger-kernel`).

```python
def _pass_detect_rope_stub(gm: fx.GraphModule) -> fx.GraphModule:
    """Detection stub — reports RoPE presence; does not transform the graph."""
    try:
        for node in gm.graph.nodes:
            # Signal A: mul(cos_output, ...) or mul(sin_output, ...)
            if node.op == "call_function" and node.target is operator.mul:
                for arg in node.args[:2]:
                    if (
                        hasattr(arg, "target")
                        and arg.op == "call_function"
                        and arg.target in (torch.cos, torch.sin)
                    ):
                        logger.warning(
                            "[pass_detect_rope] RoPE pattern detected (cos/sin mul) — "
                            "requires custom Triton kernel for fusion "
                            "(e.g. flash-attn rotary_embedding or liger-kernel). "
                            "No transformation applied."
                        )
                        return gm
            # Signal B: rotate_half — two getitem slices + cat
            if node.op == "call_function" and node.target is torch.cat:
                cat_args = node.args[0] if node.args else []
                if (
                    len(cat_args) == 2
                    and all(
                        a.op == "call_function" and a.target is operator.getitem
                        for a in cat_args
                        if hasattr(a, "op")
                    )
                ):
                    logger.warning(
                        "[pass_detect_rope] RoPE pattern detected (rotate_half cat) — "
                        "requires custom Triton kernel for fusion. No transformation applied."
                    )
                    return gm
    except Exception as e:
        logger.warning("[pass_detect_rope_stub] Failed: %s", e)
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

---

## Pass Composition Rules

### Mutual Exclusions

| Pass A | Pass B | Conflict | Resolution |
|---|---|---|---|
| `_pass_fuse_qkv` | `_pass_pretranspose_weights` | Both target the same Q/K/V weight placeholder nodes | Run QKV first; after fusion the original placeholder nodes are gone — pre-transpose finds nothing |
| `_pass_replace_sdpa` | `_pass_fuse_qkv` | SDPA must walk the graph after QKV output nodes exist | Always run `_pass_fuse_qkv` before `_pass_replace_sdpa` |

### Ordering Requirements

1. **Non-graph passes** (`dtype_promotion`, `channels_last`, `batch_padding`) live in `get_model_and_input()` only — never inside the `@register_backend` function. They run before `torch.compile` is called.
2. **Node-count reducing FX passes first:** `_pass_fuse_qkv`, `_pass_fold_bn`, `_pass_fuse_silu_geglu`. These replace multiple nodes with fewer, so downstream passes see the simplified graph.
3. **Attention restructuring next:** `_pass_replace_sdpa`. Must see the QKV output nodes produced in step 2.
4. **Weight layout last:** `_pass_pretranspose_weights`. Runs after all fusion passes have finalized the weight node set.
5. **Activation substitution** (`_pass_tanh_to_gelu`) is independent — convention places it at step 5 but it has no structural ordering requirement.
