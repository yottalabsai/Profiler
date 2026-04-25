# Canonical FX Graph Pass Patterns

Complete, copy-pasteable implementations for the most common GPU optimization passes. Each pattern includes detection, transformation, lint+recompile, and error handling.

All patterns assume the standard imports:
```python
import logging
import torch
import torch.fx as fx
from torch._dynamo import register_backend
from torch._inductor.compile_fx import compile_fx  # function, not module
from collections import defaultdict

logger = logging.getLogger(__name__)
```

---

## Pattern 1: QKV Weight Fusion

Fuses 3 independent `mm(x, W_q)`, `mm(x, W_k)`, `mm(x, W_v)` calls (same input `x`) into a single batched `mm(x, W_fused)` + `chunk(3, dim=-1)`. Reduces kernel launches from 3 to 1.

**Detection signal:** Three `aten::mm` / `aten::linear` nodes sharing the same input activation node.

**Weight-node detection note:** Inductor wraps parameters as `t(get_attr('weight'))`. Must look through the transpose wrapper.

```python
def pass_fuse_qkv(gm: fx.GraphModule) -> fx.GraphModule:
    try:
        # Build map: input_node → list of mm nodes consuming it
        input_to_mms: dict = defaultdict(list)
        for node in gm.graph.nodes:
            if node.target == torch.ops.aten.mm.default:
                input_node = node.args[0]
                input_to_mms[input_node].append(node)

        for input_node, mm_nodes in input_to_mms.items():
            if len(mm_nodes) < 3:
                continue

            # Extract weight from each mm node, looking through aten.t() wrapper
            weights = []
            weight_names = []
            for mm_node in mm_nodes[:3]:
                w_node = mm_node.args[1]
                if (w_node.target == torch.ops.aten.t.default
                        and w_node.args[0].op == 'get_attr'):
                    param_name = w_node.args[0].target
                    weight = gm.get_parameter(param_name)
                    weights.append(weight)
                    weight_names.append(param_name)
                else:
                    weights = []
                    break

            if len(weights) != 3:
                logger.warning("[pass_fuse_qkv] Could not extract weights — pattern not matched")
                continue

            # Validate shapes are fusable (all weights must have same K dim)
            if not (weights[0].shape[1] == weights[1].shape[1] == weights[2].shape[1]):
                logger.warning("[pass_fuse_qkv] Weight K dims differ — skipping fusion")
                continue

            # Fuse: cat along output dim
            W_fused = torch.cat([w.T for w in weights], dim=1).T.contiguous()
            fused_name = "fused_qkv_weight"
            gm.register_buffer(fused_name, W_fused)

            # Insert fused mm + chunk
            with gm.graph.inserting_after(mm_nodes[2]):
                fused_attr = gm.graph.get_attr(fused_name)
            with gm.graph.inserting_after(fused_attr):
                fused_mm = gm.graph.call_function(
                    torch.ops.aten.mm.default, (input_node, fused_attr)
                )
            total_out = W_fused.shape[0]
            chunk_size = total_out // 3
            with gm.graph.inserting_after(fused_mm):
                chunks = gm.graph.call_function(
                    torch.ops.aten.split.Tensor,
                    (fused_mm, chunk_size),
                    {"dim": -1},
                )
            for i, mm_node in enumerate(mm_nodes[:3]):
                with gm.graph.inserting_after(chunks):
                    chunk_i = gm.graph.call_function(
                        operator.getitem, (chunks, i)
                    )
                mm_node.replace_all_uses_with(chunk_i)
                gm.graph.erase_node(mm_node)

            logger.info(f"[pass_fuse_qkv] Fused 3 mm nodes into 1")

        gm.graph.lint()
        gm.recompile()
    except Exception as e:
        logger.warning(f"[pass_fuse_qkv] Failed: {e}")
    return gm
```

---

## Pattern 2: SDPA Replacement (FlashAttention)

Replaces the `mm(Q, K^T) → div/scale → softmax → mm(attn, V)` chain with `F.scaled_dot_product_attention`, which dispatches to FlashAttention-2 on supported hardware.

**Why `replace_pattern` fails:** Inductor decomposes `softmax` into `exp + sum + div` before FX IR reaches the pass. The decomposed subgraph does not match the high-level pattern function. Use manual graph traversal instead.

```python
def pass_replace_sdpa(gm: fx.GraphModule) -> fx.GraphModule:
    try:
        nodes = list(gm.graph.nodes)
        for node in nodes:
            # Look for softmax → mm chain
            if node.target != torch.ops.aten._softmax.default:
                continue
            softmax_node = node

            # softmax consumer should be mm(attn, V)
            softmax_users = list(softmax_node.users)
            if len(softmax_users) != 1:
                continue
            attn_mm = softmax_users[0]
            if attn_mm.target != torch.ops.aten.mm.default:
                continue
            V_node = attn_mm.args[1]

            # softmax input should be div(scores) or mul(scores, scale)
            scores_node = softmax_node.args[0]
            scale = None
            if scores_node.target in (torch.ops.aten.div.Tensor, torch.ops.aten.mul.Tensor):
                scale_arg = scores_node.args[1]
                if isinstance(scale_arg, (int, float)):
                    scale = float(scale_arg)
                qk_mm = scores_node.args[0]
            else:
                qk_mm = scores_node

            if qk_mm.target != torch.ops.aten.mm.default:
                continue

            Q_node = qk_mm.args[0]
            K_T_node = qk_mm.args[1]

            # K_T should be a transpose of K
            if K_T_node.target != torch.ops.aten.t.default:
                continue
            K_node = K_T_node.args[0]

            # Replace with SDPA
            with gm.graph.inserting_after(attn_mm):
                sdpa_node = gm.graph.call_function(
                    torch.nn.functional.scaled_dot_product_attention,
                    (Q_node, K_node, V_node),
                    {"scale": scale},
                )
            attn_mm.replace_all_uses_with(sdpa_node)
            gm.graph.erase_node(attn_mm)
            # Clean up intermediates if no other users
            for dead in [softmax_node, scores_node, qk_mm, K_T_node]:
                if dead in gm.graph.nodes and len(list(dead.users)) == 0:
                    gm.graph.erase_node(dead)

            logger.info("[pass_replace_sdpa] Replaced attention chain with SDPA")
            break  # Only replace first occurrence; call multiple times for multi-head

        gm.graph.lint()
        gm.recompile()
    except Exception as e:
        logger.warning(f"[pass_replace_sdpa] Failed: {e}")
    return gm
```

---

## Pattern 3: Conv-BN Fold (Inference)

Folds `batch_norm(training=False)` into the preceding `conv2d` by absorbing the BN scale/shift into conv weights and bias. Eliminates the separate BN kernel entirely.

**When to apply:** Only when `training=False` (inference mode). BN running stats must be finalized.

```python
def pass_fold_bn(gm: fx.GraphModule) -> fx.GraphModule:
    try:
        nodes = list(gm.graph.nodes)
        for node in nodes:
            if node.target not in (
                torch.ops.aten._native_batch_norm_legit_no_training.default,
                torch.ops.aten.batch_norm.default,
            ):
                continue
            bn_node = node

            # BN inputs: (input, weight, bias, running_mean, running_var, training, momentum, eps, ...)
            bn_args = bn_node.args
            conv_node = bn_args[0]
            if conv_node.target != torch.ops.aten.convolution.default:
                continue

            # Extract BN parameters
            bn_weight_node = bn_args[1]  # gamma
            bn_bias_node = bn_args[2]    # beta
            running_mean_node = bn_args[3]
            running_var_node = bn_args[4]
            eps = bn_args[7] if len(bn_args) > 7 else 1e-5

            # Only handle get_attr parameters (not computed tensors)
            def get_param(n):
                if n is not None and n.op == 'get_attr':
                    return gm.get_parameter(n.target)
                return None

            gamma = get_param(bn_weight_node)
            beta = get_param(bn_bias_node)
            mean = get_param(running_mean_node)
            var = get_param(running_var_node)
            if any(t is None for t in [gamma, beta, mean, var]):
                continue

            # Get conv weight/bias
            conv_args = conv_node.args
            conv_weight_node = conv_args[1]
            conv_bias_node = conv_args[2]
            conv_weight = get_param(conv_weight_node)
            conv_bias = get_param(conv_bias_node)
            if conv_weight is None:
                continue

            # Fold: new_weight = gamma / sqrt(var + eps) * weight
            scale = gamma / torch.sqrt(var + eps)
            new_weight = conv_weight * scale.view(-1, 1, 1, 1)
            if conv_bias is not None:
                new_bias = (conv_bias - mean) * scale + beta
            else:
                new_bias = beta - mean * scale

            # Register folded parameters
            gm.register_buffer('_folded_conv_weight', new_weight)
            gm.register_buffer('_folded_conv_bias', new_bias)

            # Rewire conv to use folded parameters
            with gm.graph.inserting_before(conv_node):
                fw_node = gm.graph.get_attr('_folded_conv_weight')
                fb_node = gm.graph.get_attr('_folded_conv_bias')
            new_conv_args = (conv_args[0], fw_node, fb_node) + conv_args[3:]
            with gm.graph.inserting_after(conv_node):
                new_conv = gm.graph.call_function(
                    torch.ops.aten.convolution.default, new_conv_args
                )

            # BN output is a tuple (out, mean, rstd); replace the getitem(0) user
            for user in list(bn_node.users):
                if user.target == operator.getitem and user.args[1] == 0:
                    user.replace_all_uses_with(new_conv)
                    gm.graph.erase_node(user)
            conv_node.replace_all_uses_with(new_conv)
            gm.graph.erase_node(conv_node)
            if len(list(bn_node.users)) == 0:
                gm.graph.erase_node(bn_node)

            logger.info("[pass_fold_bn] Folded BatchNorm into Conv2d")

        gm.graph.lint()
        gm.recompile()
    except Exception as e:
        logger.warning(f"[pass_fold_bn] Failed: {e}")
    return gm
```

---

## Pattern 4: Pre-Transposed Weights

Detects `mm(x, aten.t(W))` where W is a large parameter (K ≥ 512) and replaces with `mm(x, W_T_pre)` using a pre-stored transposed buffer. Switches cuBLAS from `gemmSN_TN` (transposed path) to `gemmSN_NN` (non-transposed path).

```python
def pass_pretranspose_weights(gm: fx.GraphModule, min_k: int = 512) -> fx.GraphModule:
    try:
        pretransposed: dict[str, str] = {}  # original param name → buffer name

        for node in list(gm.graph.nodes):
            if node.target != torch.ops.aten.mm.default:
                continue
            w_node = node.args[1]

            # Detect aten.t() wrapping a parameter
            if (w_node.target != torch.ops.aten.t.default
                    or w_node.args[0].op != 'get_attr'):
                continue

            param_name = w_node.args[0].target
            weight = gm.get_parameter(param_name)

            # Only pre-transpose large weights
            if weight.shape[0] < min_k and weight.shape[1] < min_k:
                continue

            # Register pre-transposed buffer (once per unique parameter)
            if param_name not in pretransposed:
                buf_name = f"_pretransposed_{param_name.replace('.', '_')}"
                gm.register_buffer(buf_name, weight.T.contiguous())
                pretransposed[param_name] = buf_name
            else:
                buf_name = pretransposed[param_name]

            # Replace t(get_attr(param)) with get_attr(buf)
            with gm.graph.inserting_before(w_node):
                buf_node = gm.graph.get_attr(buf_name)
            w_node.replace_all_uses_with(buf_node)
            if len(list(w_node.users)) == 0:
                gm.graph.erase_node(w_node)

            logger.info(f"[pass_pretranspose_weights] Pre-transposed {param_name} ({weight.shape})")

        gm.graph.lint()
        gm.recompile()
    except Exception as e:
        logger.warning(f"[pass_pretranspose_weights] Failed: {e}")
    return gm
```

---

## Pattern 5: Activation Substitution (tanh → GELU)

Replaces `aten.tanh.default` with `aten.gelu.default(approximate='tanh')` when the tanh node appears in an FFN context (mm → tanh → mm). Avoids SFU pipeline serialization that makes `tanh` 5–10× slower than GELU on modern hardware.

**When to apply:** Only in FFN-like contexts. Do NOT apply to attention scaling or output-projection activations.

```python
def pass_tanh_to_gelu(gm: fx.GraphModule) -> fx.GraphModule:
    try:
        replaced = 0
        for node in list(gm.graph.nodes):
            if node.target != torch.ops.aten.tanh.default:
                continue
            tanh_node = node

            # Check FFN context: producer is mm AND consumer is mm
            producer = tanh_node.args[0]
            is_ffn = (
                producer.target == torch.ops.aten.mm.default
                and any(u.target == torch.ops.aten.mm.default for u in tanh_node.users)
            )
            if not is_ffn:
                continue

            with gm.graph.inserting_after(tanh_node):
                gelu_node = gm.graph.call_function(
                    torch.ops.aten.gelu.default,
                    (tanh_node.args[0],),
                    {"approximate": "tanh"},
                )
            tanh_node.replace_all_uses_with(gelu_node)
            gm.graph.erase_node(tanh_node)
            replaced += 1

        if replaced:
            logger.info(f"[pass_tanh_to_gelu] Replaced {replaced} tanh → gelu(tanh)")
            gm.graph.lint()
            gm.recompile()
    except Exception as e:
        logger.warning(f"[pass_tanh_to_gelu] Failed: {e}")
    return gm
```

---

## Stub Pattern: LayerNorm-Linear Fusion

Detection only. Full implementation requires a custom Triton kernel that fuses the LN normalization epilogue into the subsequent GEMM prologue.

```python
def pass_fuse_ln_linear_stub(gm: fx.GraphModule) -> fx.GraphModule:
    """
    Detection stub for LayerNorm → Linear fusion.
    Full implementation requires a custom Triton kernel.
    """
    try:
        for node in list(gm.graph.nodes):
            if node.target != torch.ops.aten.native_layer_norm.default:
                continue
            ln_node = node
            for user in ln_node.users:
                # layer_norm output is tuple; getitem(0) is the normalized tensor
                if (user.target == __import__('operator').getitem
                        and user.args[1] == 0):
                    for mm_user in user.users:
                        if mm_user.target == torch.ops.aten.mm.default:
                            logger.warning(
                                "[pass_fuse_ln_linear] LayerNorm→Linear pattern detected "
                                "but not applied — requires custom Triton kernel "
                                "(e.g. triton-flash-attn or liger-kernel LN-MM)"
                            )
    except Exception as e:
        logger.warning(f"[pass_fuse_ln_linear_stub] Failed: {e}")
    return gm  # gm is unchanged
```

---

## Channels-Last Conversion (Non-Graph)

This optimization is applied in `get_model_and_input()`, not as an FX pass, because `memory_format` is a tensor property not visible in the Aten IR graph.

```python
def apply_channels_last(model: torch.nn.Module) -> torch.nn.Module:
    """Convert all Conv2d modules to channels_last memory format.
    
    Eliminates convertTensor_kernel launches from cuDNN NHWC coercion.
    Only effective for models with Conv2d layers.
    """
    has_conv = any(isinstance(m, torch.nn.Conv2d) for m in model.modules())
    if not has_conv:
        return model
    return model.to(memory_format=torch.channels_last)
```

Apply in `get_model_and_input()`:
```python
# Check if already channels_last before applying
if not next(model.parameters()).is_contiguous(memory_format=torch.channels_last):
    model = apply_channels_last(model)
    x = x.to(memory_format=torch.channels_last)
```
