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
from torch._functorch.aot_autograd import aot_autograd
from torch._inductor.compile_fx import compile_fx  # function, not module

logger = logging.getLogger(__name__)
```

---

## IR Level: All Passes Run at Aten IR

`@register_backend` receives the graph **before** Inductor lowers it (functional level). All FX passes run inside `_aten_fw_compiler`, which `aot_autograd` calls with the fully decomposed **Aten IR** graph. `compile_fx` inside `_aten_fw_compiler` then only handles the Aten → Triton step.

| Source code | Aten IR node (inside `_aten_fw_compiler`) |
|---|---|
| `nn.Linear(x)` (with bias) | `aten.addmm.default(bias_ph, x, aten.t.default(weight_ph))` |
| `nn.Linear(x)` (no bias) | `aten.mm.default(x, aten.t.default(weight_ph))` |
| `x @ y` | `aten.mm.default` |
| `x * scale` | `aten.mul.Tensor` |
| `x / scale` | `aten.div.Tensor` |
| `torch.softmax(x, dim)` | `aten._softmax.default` |
| `x.tanh()` | `aten.tanh.default` |
| `F.gelu(x)` | `aten.gelu.default` |
| `F.silu(x)` | `aten.silu.default` |
| `nn.Conv2d(x)` | `aten.convolution.default` or `aten.cudnn_convolution.default` |
| `nn.BatchNorm2d(x)` (eval) | `aten._native_batch_norm_legit_no_training.default` — returns 3-tuple `(output, save_mean, save_rstd)` |
| `F.layer_norm(x, ...)` | `aten.native_layer_norm.default` — returns 3-tuple `(output, mean, rstd)` |
| `F.scaled_dot_product_attention(...)` | `aten._scaled_dot_product_efficient_attention.default` or `aten.scaled_dot_product_flash_attention.default` — hardware-selected, returns 4-tuple `(output, log_sumexp, seed, offset)` |
| Any `nn.Module` parameter | `placeholder` node — values in `fw_example_inputs`, positionally matched |

**Weight access at Aten IR:** for `aten.addmm.default(bias, x, t_node)`, the weight placeholder is `t_node.args[0]`. Build the lookup with:
```python
placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
ph_to_tensor = {ph: t for ph, t in zip(placeholders, fw_example_inputs)}
weight = ph_to_tensor[addmm_node.args[2].args[0]]  # unwrap aten.t.default
```

---

## Pass Taxonomy

Two categories:

| Category | Criteria | How to apply |
|---|---|---|
| **Aten IR pass** | Any graph transformation — dtype casts, op replacement, kernel selection, QKV fusion, BN fold, pre-transposed weights | Inside `_aten_fw_compiler`; use `fw_example_inputs` for weight value lookup |
| **Non-graph** | dtype, memory_format, batch shape — not visible in any FX graph | `get_model_and_input()` only; never in the backend |

---

## Tuple-Return Aten Ops

`native_layer_norm`, `_native_batch_norm_legit_no_training`, `_scaled_dot_product_efficient_attention`, and `scaled_dot_product_flash_attention` all return tuples. Do NOT replace the op node directly. Instead:

1. Find downstream `getitem(node, 0)` consumer (the live output)
2. Insert replacement node before the getitem
3. `getitem.replace_all_uses_with(replacement_node)`
4. Erase the getitem node
5. Erase the original op node if it has no remaining users
6. Call `gm.graph.eliminate_dead_code()` — removes dead tuple-element consumers and dead upstream inputs (e.g. mask-construction subgraphs)

---

## Standard Backend Pattern

### `_aten_fw_compiler`

```python
def _aten_fw_compiler(gm: fx.GraphModule, fw_example_inputs) -> Callable:
    """
    Receives the Aten IR graph after AOTAutograd decomposition.
    All graph passes run here. fw_example_inputs contains all placeholder tensors
    (parameters + activations) at their actual runtime values.
    """
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    ph_to_tensor = {ph: t for ph, t in zip(placeholders, fw_example_inputs)}

    gm = _pass_one(gm, ph_to_tensor)   # weight-access pass (QKV fusion, BN fold, etc.)
    gm = _pass_two(gm)                  # op-target pass (BF16 casts, SDPA replacement, etc.)
    return compile_fx(gm, fw_example_inputs)
```

### Dedup-aware backend

```python
@register_backend
def {model}_opt(gm: fx.GraphModule, example_inputs) -> Callable:
    registry = UniqueSubgraphRegistry(gm)
    equiv_map = registry.build_partition_equivalence_map()

    if not equiv_map:
        logger.info("{model}_opt: no repeated layers, flat compile path")
        return aot_autograd(fw_compiler=_aten_fw_compiler)(gm, example_inputs)

    logger.info(f"{model}_opt: {len(equiv_map)} duplicate partition(s), dedup path")
    partition_inputs = _capture_partition_inputs(registry.split, example_inputs)
    for rep_name, rep_mod in registry.unique_reps:
        inputs = partition_inputs.get(rep_name, example_inputs)
        compiled = aot_autograd(fw_compiler=_aten_fw_compiler)(rep_mod, inputs)
        rep_mod.forward = compiled
        for _, dup_mod in registry.duplicates_of(rep_name):
            dup_mod.forward = compiled

    return lambda *args: registry.split(*args)
```

`UniqueSubgraphRegistry` splits at the functional level for structural dedup. Do NOT move the split inside `fw_compiler`.

---

## Utility: `_capture_partition_inputs`

Required to get the actual input tensors for each partition so `aot_autograd` can trace each rep with correct example inputs.

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

---

## Pattern 1: QKV Weight Fusion

Fuses 3 independent projections sharing the same input `x` into a single wider GEMM + chunk.

**Detection signal:** Three `aten.addmm.default` nodes (or `aten.mm.default` for bias-free) whose second arg (input activation) is the same node.

**Weight access:** weight placeholder is at `addmm_node.args[2].args[0]` (inside `aten.t.default`).

```python
def _pass_fuse_qkv(gm: fx.GraphModule, fw_example_inputs: list) -> fx.GraphModule:
    try:
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        ph_to_tensor = {ph: t for ph, t in zip(placeholders, fw_example_inputs)}

        # Group aten.addmm.default calls by their input activation arg (args[1])
        addmm_groups: dict[str, list] = {}
        for n in gm.graph.nodes:
            if n.op == "call_function" and n.target is torch.ops.aten.addmm.default:
                addmm_groups.setdefault(n.args[1].name, []).append(n)

        fused = False
        for x_name, addmm_list in addmm_groups.items():
            if len(addmm_list) < 3:
                continue
            q_n, k_n, v_n = addmm_list[0], addmm_list[1], addmm_list[2]

            # Unwrap aten.t.default to get weight placeholder
            def _get_weight(addmm_node):
                t_node = addmm_node.args[2]
                if t_node.op == "call_function" and t_node.target is torch.ops.aten.t.default:
                    return ph_to_tensor.get(t_node.args[0])
                return None

            W_q, W_k, W_v = _get_weight(q_n), _get_weight(k_n), _get_weight(v_n)
            if W_q is None or W_k is None or W_v is None:
                logger.warning("[pass_fuse_qkv] Weight tensors not in fw_example_inputs")
                continue
            if not (W_q.shape[1] == W_k.shape[1] == W_v.shape[1]):
                logger.warning("[pass_fuse_qkv] Weight K dims differ — skipping")
                continue

            W_qkv = torch.cat([W_q, W_k, W_v], dim=0)
            gm.register_buffer("_fused_qkv_weight", W_qkv)

            with gm.graph.inserting_before(q_n):
                w_buf   = gm.graph.get_attr("_fused_qkv_weight")
                # Use aten.mm since we supply the pre-transposed fused weight directly
                fused_mm = gm.graph.call_function(
                    torch.ops.aten.mm.default, (q_n.args[1], w_buf)
                )
                chunks  = gm.graph.call_function(torch.ops.aten.chunk.default, (fused_mm, 3), {"dim": -1})
                q_out   = gm.graph.call_function(operator.getitem, (chunks, 0))
                k_out   = gm.graph.call_function(operator.getitem, (chunks, 1))
                v_out   = gm.graph.call_function(operator.getitem, (chunks, 2))

            q_n.replace_all_uses_with(q_out)
            k_n.replace_all_uses_with(k_out)
            v_n.replace_all_uses_with(v_out)
            for dead in (q_n, k_n, v_n):
                gm.graph.erase_node(dead)

            gm.graph.lint()
            gm.recompile()
            logger.info("[pass_fuse_qkv] Fused 3 addmm into 1 (input '%s') [Aten IR]", x_name)
            fused = True
            break

        if not fused:
            logger.warning("[pass_fuse_qkv] Pattern not found — pass not applied")
    except Exception as e:
        logger.warning("[pass_fuse_qkv] Failed: %s", e)
    return gm
```

---

## Pattern 2: SDPA Replacement

Replaces `aten._scaled_dot_product_efficient_attention` (or flash variant) with `aten.scaled_dot_product_attention(is_causal=True)` to let the SDPA dispatcher select a hardware-native kernel.

**Tuple return:** efficient_attention returns `(output, log_sumexp, seed, offset)`. Replace `getitem(node, 0)`.

```python
_EFFICIENT_ATTN_TARGETS = frozenset({
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten.scaled_dot_product_flash_attention.default,
})

def _pass_replace_efficient_attn_with_sdpa(gm: fx.GraphModule) -> fx.GraphModule:
    try:
        matched = False
        for node in list(gm.graph.nodes):
            if not (node.op == "call_function" and node.target in _EFFICIENT_ATTN_TARGETS):
                continue
            q, k, v = node.args[0], node.args[1], node.args[2]
            scale = node.kwargs.get("scale", None)

            output_getitem = None
            for user in node.users:
                if (user.op == "call_function"
                        and user.target is operator.getitem
                        and user.args[1] == 0):
                    output_getitem = user
                    break
            if output_getitem is None:
                logger.warning(
                    "[_pass_replace_efficient_attn_with_sdpa] "
                    "No getitem(0) user — skipping %s", node.name
                )
                continue

            with gm.graph.inserting_before(output_getitem):
                new_sdpa = gm.graph.call_function(
                    torch.ops.aten.scaled_dot_product_attention.default,
                    args=(q, k, v, None, 0.0, True),
                    kwargs={"scale": scale},
                )
            output_getitem.replace_all_uses_with(new_sdpa)
            gm.graph.erase_node(output_getitem)
            if not node.users:
                gm.graph.erase_node(node)
            gm.graph.eliminate_dead_code()
            matched = True

        if not matched:
            logger.warning(
                "[_pass_replace_efficient_attn_with_sdpa] "
                "No efficient/flash attention nodes found — pass not applied"
            )
            return gm
        gm.graph.lint()
        gm.recompile()
        logger.info(
            "[_pass_replace_efficient_attn_with_sdpa] "
            "Replaced efficient_attention with SDPA (is_causal=True) [Aten IR]"
        )
    except Exception as exc:
        logger.warning("[_pass_replace_efficient_attn_with_sdpa] Failed: %s", exc)
    return gm
```

---

## Pattern 3: Conv-BN Fold (Inference)

Folds `aten._native_batch_norm_legit_no_training` into the preceding `aten.convolution.default` by absorbing BN scale/shift into conv weights and bias.

**Tuple return:** batch_norm returns `(output, save_mean, save_rstd)`. Replace `getitem(bn_node, 0)` downstream users.

```python
def _pass_fold_bn(gm: fx.GraphModule, fw_example_inputs: list) -> fx.GraphModule:
    try:
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        ph_to_tensor = {ph: t for ph, t in zip(placeholders, fw_example_inputs)}

        _BN_TARGET = torch.ops.aten._native_batch_norm_legit_no_training.default
        _CONV_TARGETS = frozenset({
            torch.ops.aten.convolution.default,
            torch.ops.aten.cudnn_convolution.default,
        })

        for bn_node in list(gm.graph.nodes):
            if not (bn_node.op == "call_function" and bn_node.target is _BN_TARGET):
                continue

            # aten._native_batch_norm_legit_no_training(input, weight, bias, running_mean, running_var, momentum, eps)
            conv_node = bn_node.args[0]
            if not (conv_node.op == "call_function" and conv_node.target in _CONV_TARGETS):
                continue

            bn_weight   = ph_to_tensor.get(bn_node.args[1])
            bn_bias     = ph_to_tensor.get(bn_node.args[2])
            run_mean    = ph_to_tensor.get(bn_node.args[3])
            run_var     = ph_to_tensor.get(bn_node.args[4])
            eps         = bn_node.args[6] if len(bn_node.args) > 6 else 1e-5

            if any(t is None for t in (bn_weight, bn_bias, run_mean, run_var)):
                continue

            # aten.convolution.default args: (input, weight, bias, stride, padding, dilation, transposed, output_padding, groups)
            conv_weight = ph_to_tensor.get(conv_node.args[1])
            conv_bias   = ph_to_tensor.get(conv_node.args[2]) if conv_node.args[2] is not None else None
            if conv_weight is None:
                continue

            scale      = bn_weight / torch.sqrt(run_var + eps)
            new_weight = conv_weight * scale.view(-1, 1, 1, 1)
            new_bias   = (conv_bias - run_mean) * scale + bn_bias if conv_bias is not None else bn_bias - run_mean * scale

            gm.register_buffer("_folded_conv_weight", new_weight)
            gm.register_buffer("_folded_conv_bias",   new_bias)

            with gm.graph.inserting_before(conv_node):
                fw = gm.graph.get_attr("_folded_conv_weight")
                fb = gm.graph.get_attr("_folded_conv_bias")

            new_conv_args = (conv_node.args[0], fw, fb) + conv_node.args[3:]
            with gm.graph.inserting_before(bn_node):
                new_conv = gm.graph.call_function(torch.ops.aten.convolution.default, new_conv_args)

            # bn_node returns a tuple; replace the getitem(bn_node, 0) consumer
            for user in list(bn_node.users):
                if (user.op == "call_function"
                        and user.target is operator.getitem
                        and user.args[1] == 0):
                    user.replace_all_uses_with(new_conv)
                    gm.graph.erase_node(user)
            if not bn_node.users:
                gm.graph.erase_node(bn_node)
            if not conv_node.users:
                gm.graph.erase_node(conv_node)

            gm.graph.lint()
            gm.recompile()
            logger.info("[pass_fold_bn] Folded BatchNorm into Conv2d [Aten IR]")
            break

    except Exception as e:
        logger.warning("[pass_fold_bn] Failed: %s", e)
    return gm
```

---

## Pattern 4: Activation Substitution (tanh → GELU)

Replaces `aten.tanh.default` in an FFN context with `aten.gelu.default(approximate='tanh')`.

```python
def _pass_tanh_to_gelu(gm: fx.GraphModule) -> fx.GraphModule:
    try:
        replaced = 0
        for node in list(gm.graph.nodes):
            if not (node.op == "call_function" and node.target is torch.ops.aten.tanh.default):
                continue

            # FFN context: producer is aten.addmm and at least one consumer is aten.addmm
            producer = node.args[0]
            is_ffn = (
                producer.op == "call_function"
                and producer.target is torch.ops.aten.addmm.default
                and any(
                    u.op == "call_function" and u.target is torch.ops.aten.addmm.default
                    for u in node.users
                )
            )
            if not is_ffn:
                continue

            with gm.graph.inserting_after(node):
                gelu = gm.graph.call_function(
                    torch.ops.aten.gelu.default,
                    (node.args[0],),
                    {"approximate": "tanh"},
                )
            node.replace_all_uses_with(gelu)
            gm.graph.erase_node(node)
            replaced += 1

        if replaced:
            gm.graph.lint()
            gm.recompile()
            logger.info("[pass_tanh_to_gelu] Replaced %d tanh → gelu(tanh) [Aten IR]", replaced)
    except Exception as e:
        logger.warning("[pass_tanh_to_gelu] Failed: %s", e)
    return gm
```

---

## Pattern 5: Pre-Transposed Weight Buffer

Eliminates runtime `aten.t.default` overhead by pre-storing `weight.T.contiguous()` as a buffer.

**Detection signal:** `aten.t.default(weight_ph)` consumed as the weight arg of `aten.mm.default`.

```python
def _pass_pretranspose_weights(gm: fx.GraphModule, fw_example_inputs: list) -> fx.GraphModule:
    try:
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        ph_to_tensor = {ph: t for ph, t in zip(placeholders, fw_example_inputs)}

        replaced = 0
        for node in list(gm.graph.nodes):
            # Find: aten.t.default on a placeholder
            if not (node.op == "call_function" and node.target is torch.ops.aten.t.default):
                continue
            ph_node = node.args[0]
            if ph_node.op != "placeholder":
                continue
            weight = ph_to_tensor.get(ph_node)
            if weight is None or not isinstance(weight, torch.Tensor):
                continue
            if weight.shape[0] < 512:
                continue

            for user in list(node.users):
                # aten.mm.default(x, t_node) — t_node is args[1]
                if not (user.op == "call_function"
                        and user.target is torch.ops.aten.mm.default
                        and user.args[1] is node):
                    continue

                buf_name = f"_pretransposed_weight_{replaced}"
                weight_T = weight.T.contiguous()
                gm.register_buffer(buf_name, weight_T)

                with gm.graph.inserting_before(user):
                    buf_node = gm.graph.get_attr(buf_name)
                    new_mm   = gm.graph.call_function(
                        torch.ops.aten.mm.default, (user.args[0], buf_node)
                    )
                user.replace_all_uses_with(new_mm)
                gm.graph.erase_node(user)
                replaced += 1

        if replaced:
            for node in list(gm.graph.nodes):
                if (node.op == "call_function"
                        and node.target is torch.ops.aten.t.default
                        and not node.users):
                    gm.graph.erase_node(node)
            gm.graph.lint()
            gm.recompile()
            logger.info("[pass_pretranspose_weights] Pre-transposed %d weight(s) [Aten IR]", replaced)
        else:
            logger.warning("[pass_pretranspose_weights] Pattern not found — pass not applied")
    except Exception as e:
        logger.warning("[pass_pretranspose_weights] Failed: %s", e)
    return gm
```

---

## Pattern 6: SiLU/GEGLU Gated Activation Fusion

Fuses gate_proj + up_proj into a single wider GEMM + chunk + silu + mul.

**Detection signal:** `aten.silu.default` consuming `aten.addmm.default(bias, x, W_gate_T)`, multiplied by `aten.addmm.default(bias, x, W_up_T)` via `aten.mul.Tensor`.

```python
def _pass_fuse_silu_geglu(gm: fx.GraphModule, fw_example_inputs: list) -> fx.GraphModule:
    try:
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        ph_to_tensor = {ph: t for ph, t in zip(placeholders, fw_example_inputs)}

        def _get_weight(addmm_node):
            t_node = addmm_node.args[2]
            if t_node.op == "call_function" and t_node.target is torch.ops.aten.t.default:
                return ph_to_tensor.get(t_node.args[0])
            return None

        fused = False
        for silu_node in list(gm.graph.nodes):
            if not (silu_node.op == "call_function"
                    and silu_node.target is torch.ops.aten.silu.default):
                continue

            gate_addmm = silu_node.args[0]
            if not (gate_addmm.op == "call_function"
                    and gate_addmm.target is torch.ops.aten.addmm.default):
                continue
            x_node = gate_addmm.args[1]

            mul_node = None
            for user in silu_node.users:
                if user.op == "call_function" and user.target is torch.ops.aten.mul.Tensor:
                    mul_node = user
                    break
            if mul_node is None:
                continue

            other = mul_node.args[1] if mul_node.args[0] is silu_node else mul_node.args[0]
            if not (other.op == "call_function"
                    and other.target is torch.ops.aten.addmm.default
                    and other.args[1] is x_node):
                continue

            W_gate = _get_weight(gate_addmm)
            W_up   = _get_weight(other)
            if W_gate is None or W_up is None:
                logger.warning("[pass_fuse_silu_geglu] Weight tensors not in fw_example_inputs")
                continue
            if W_gate.shape != W_up.shape:
                logger.warning("[pass_fuse_silu_geglu] W_gate/W_up shapes differ — skipping")
                continue

            W_fused = torch.cat([W_gate, W_up], dim=0)
            gm.register_buffer("_fused_gate_up_weight", W_fused)

            with gm.graph.inserting_before(gate_addmm):
                w_buf      = gm.graph.get_attr("_fused_gate_up_weight")
                fused_mm   = gm.graph.call_function(torch.ops.aten.mm.default, (x_node, w_buf))
                chunks     = gm.graph.call_function(torch.ops.aten.chunk.default, (fused_mm, 2), {"dim": -1})
                gate_out   = gm.graph.call_function(operator.getitem, (chunks, 0))
                up_out     = gm.graph.call_function(operator.getitem, (chunks, 1))
                gated      = gm.graph.call_function(torch.ops.aten.silu.default, (gate_out,))
                fused_out  = gm.graph.call_function(torch.ops.aten.mul.Tensor, (gated, up_out))

            mul_node.replace_all_uses_with(fused_out)
            for dead in (mul_node, silu_node, gate_addmm, other):
                if not dead.users:
                    try:
                        gm.graph.erase_node(dead)
                    except Exception:
                        pass

            gm.graph.lint()
            gm.recompile()
            logger.info("[pass_fuse_silu_geglu] Fused gate_proj + up_proj into single GEMM [Aten IR]")
            fused = True
            break

        if not fused:
            logger.warning("[pass_fuse_silu_geglu] SiLU/GEGLU pattern not found — pass not applied")
    except Exception as e:
        logger.warning("[pass_fuse_silu_geglu] Failed: %s", e)
    return gm
```

---

## Stub Pattern: Grouped Query Attention (GQA) Detection

Detection only — reports GQA head ratio; does not transform the graph.

```python
def _pass_detect_gqa_stub(gm: fx.GraphModule, fw_example_inputs: list) -> fx.GraphModule:
    try:
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        ph_to_tensor = {ph: t for ph, t in zip(placeholders, fw_example_inputs)}

        addmm_groups: dict[str, list] = {}
        for n in gm.graph.nodes:
            if n.op == "call_function" and n.target is torch.ops.aten.addmm.default:
                addmm_groups.setdefault(n.args[1].name, []).append(n)

        for x_name, addmms in addmm_groups.items():
            if len(addmms) < 3:
                continue
            out_dims = []
            for n in addmms:
                t_node = n.args[2]
                if t_node.op == "call_function" and t_node.target is torch.ops.aten.t.default:
                    w = ph_to_tensor.get(t_node.args[0])
                    if w is not None:
                        out_dims.append(w.shape[0])
            if len(set(out_dims)) > 1:
                q_dim  = max(out_dims)
                kv_dim = min(out_dims)
                ratio  = q_dim // kv_dim if kv_dim > 0 else "?"
                logger.warning(
                    "[pass_detect_gqa] GQA on input '%s': Q=%d, KV=%d (ratio %s). "
                    "Use aten.scaled_dot_product_attention(enable_gqa=True) (torch >= 2.3).",
                    x_name, q_dim, kv_dim, ratio,
                )
    except Exception as e:
        logger.warning("[pass_detect_gqa_stub] Failed: %s", e)
    return gm
```

---

## Stub Pattern: Rotary Position Embedding (RoPE) Detection

```python
def _pass_detect_rope_stub(gm: fx.GraphModule) -> fx.GraphModule:
    try:
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target is torch.ops.aten.mul.Tensor:
                for arg in node.args[:2]:
                    if (hasattr(arg, "target")
                            and arg.op == "call_function"
                            and arg.target in (torch.ops.aten.cos.default, torch.ops.aten.sin.default)):
                        logger.warning(
                            "[pass_detect_rope] RoPE pattern (cos/sin mul) detected — "
                            "requires custom Triton kernel for fusion. No transformation applied."
                        )
                        return gm
            if node.op == "call_function" and node.target is torch.ops.aten.cat.default:
                cat_args = node.args[0] if node.args else []
                if (len(cat_args) == 2
                        and all(
                            hasattr(a, "op") and a.op == "call_function"
                            and a.target is operator.getitem
                            for a in cat_args
                        )):
                    logger.warning(
                        "[pass_detect_rope] RoPE pattern (rotate_half cat) detected — "
                        "requires custom Triton kernel. No transformation applied."
                    )
                    return gm
    except Exception as e:
        logger.warning("[pass_detect_rope_stub] Failed: %s", e)
    return gm
```

---

## Stub Pattern: LayerNorm-Linear Fusion

```python
def _pass_fuse_ln_linear_stub(gm: fx.GraphModule) -> fx.GraphModule:
    try:
        _LN_TARGET = torch.ops.aten.native_layer_norm.default
        for ln_node in gm.graph.nodes:
            if not (ln_node.op == "call_function" and ln_node.target is _LN_TARGET):
                continue
            # native_layer_norm returns (output, mean, rstd); index 0 feeds linear
            for gi in ln_node.users:
                if not (gi.op == "call_function"
                        and gi.target is operator.getitem
                        and gi.args[1] == 0):
                    continue
                for linear_node in gi.users:
                    if (linear_node.op == "call_function"
                            and linear_node.target is torch.ops.aten.addmm.default):
                        logger.warning(
                            "[pass_fuse_ln_linear] LayerNorm→Linear detected but not applied "
                            "— requires custom Triton kernel (e.g. liger-kernel LN-MM)"
                        )
    except Exception as e:
        logger.warning("[pass_fuse_ln_linear_stub] Failed: %s", e)
    return gm
```

---

## Channels-Last Conversion (Non-Graph)

Applied in `get_model_and_input()` — not an FX pass.

```python
def apply_channels_last(model: torch.nn.Module) -> torch.nn.Module:
    has_conv = any(isinstance(m, torch.nn.Conv2d) for m in model.modules())
    if not has_conv:
        return model
    return model.to(memory_format=torch.channels_last)
```

---

## Pass Composition Rules

### Ordering Requirements

1. **Non-graph passes** live in `get_model_and_input()` only — never inside `_aten_fw_compiler`.
2. **Node-count reducing passes first:** QKV fusion, BN fold, SiLU/GEGLU. These reduce the node set so downstream passes see a simplified graph.
3. **Attention restructuring next:** SDPA replacement. Must see the correct q/k/v nodes after any upstream fusion.
4. **Weight layout last:** pre-transposed weights. Runs after fusion passes have finalised the weight node set.
5. **Activation substitution:** independent — conventional position is step 5 but has no structural ordering requirement.

### Mutual Exclusions

| Pass A | Pass B | Conflict | Resolution |
|---|---|---|---|
| `_pass_fuse_qkv` | `_pass_pretranspose_weights` | Both target the same weight placeholders | Run QKV first; after fusion the original placeholders are gone |
| `_pass_replace_sdpa` | `_pass_fuse_qkv` | SDPA must walk the graph after QKV output nodes exist | Always run `_pass_fuse_qkv` before SDPA replacement |
