# Canonical FX Graph Pass Patterns

Complete, copy-pasteable implementations for the most common GPU optimization passes. Each pattern includes detection, transformation, lint+recompile, and error handling.

All patterns assume the standard imports:
```python
import logging
import operator
import torch
import torch.fx as fx
import torch.nn.functional as F
import functools
from torch._dynamo import register_backend
from torch._inductor.compile_fx import compile_fx, compile_fx_inner  # functions, not module

logger = logging.getLogger(__name__)
```

---

## IR Levels: Functional, Aten, Inductor-Config

A pass runs at the level where its pattern is cleanly expressed (backend-engineer Rule 10). Three levels:

- **`functional`** — `@register_backend` receives the Dynamo graph **before** `compile_fx`/AOTAutograd. Here `nn.Linear`→`F.linear` is one node, `F.scaled_dot_product_attention` is one node, parameters are clean nodes, and an activation shared by several ops is **one** node. Run fusion / op-substitution here, on `gm` before calling `compile_fx`. (AOTAutograd recomputes meta when it traces the rewritten graph, so no FakeTensorProp is needed at this level.)
- **`aten`** — inside `_aten_inner_compile` (`compile_fx`'s `inner_compile` hook), on the fully decomposed Aten IR graph; `compile_fx_inner` then handles Aten → Triton. Run dtype/primitive rewrites here. The mapping table below is the source→Aten correspondence for these passes.
- **`inductor_config`** — no graph surgery; a scoped `config_patches` dict on the `compile_fx` call (e.g. `{"freezing": True}` for constant-weight layout).

**Aten IR forms (for `aten`-level passes):**

| Source code | Aten IR node (inside `_aten_inner_compile`) |
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

| Category (`ir_level`) | Criteria | How to apply |
|---|---|---|
| **Functional pass** (`functional`) | Operator fusion (QKV / gate-up), SDPA formation, op substitution — keys on a shared high-level op (`F.linear`, `F.scaled_dot_product_attention`) | `_run_functional_passes(gm)` BEFORE `compile_fx`; AOTAutograd recomputes meta (no FakeTensorProp) |
| **Aten IR pass** (`aten`) | dtype casts, BN fold, channels_last annotation, intermediate-tensor contiguity — decomposed primitives | Inside `_aten_inner_compile`; threaded `real_inputs` for weight values; `_repropagate_meta` after structural rewrites |
| **Inductor-config pass** (`inductor_config`) | constant-weight layout / pre-transpose / freezing — owned by Inductor | return a dict merged into `compile_fx(config_patches=...)`; no graph surgery |
| **Non-graph** | dtype, memory_format, batch shape — not visible in any FX graph | `get_model_and_input()` only; never in the backend |

**Why fusion is `functional`, not `aten`:** a QKV fusion keys on three projections sharing one activation. After decomposition each consumes its own `aten.view` of that activation and any dtype-cast pass inserts a separate cast per `aten.mm`, so the shared-node identity is gone and an Aten-level matcher no-ops. Fuse at the functional level; decomposition lowers the single fused `F.linear` to the single wide `aten.mm`.

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

## Standard Backend Pattern (three-stage funnel)

The backend is one funnel, `_compile_unit`, invoked identically on the flat graph and each dedup rep:

```
_run_functional_passes(gm)  ->  compile_fx(inner_compile=_aten_inner_compile, config_patches=...)
   LEVEL 1 (functional)          LEVEL 2 (aten) + LEVEL 3 (inductor_config)
```

`compile_fx` owns AOTAutograd, the decomposition table, the boxed calling convention, and the
fwd/bwd partitioner — functional passes run *before* it, you swap the leaf compiler for the aten
passes, and config-level passes ride on `config_patches`. **Do NOT** use
`aot_autograd(fw_compiler=compile_fx)`: on torch 2.11 it raises `AssertionError: Expected tensors
only, but got list` in `copy_misaligned_inputs` (a boxing mismatch). The seam is scoped per
`compile_fx` call (no process-global state) and forward-compatible via `**kwargs`.

```python
# A pass declares its ir_level; the router groups by level (default "aten" if absent).
PASS_REGISTRY = [ {"id": "OPT-2", "level": "functional", "fn": _fpass_fuse_qkv}, ... ]
def _passes(level): return [p for p in PASS_REGISTRY if p["level"] == level]

def _run_functional_passes(gm):                       # LEVEL 1 — Dynamo graph, pre-compile_fx
    for p in _passes("functional"):
        try: gm = p["fn"](gm)
        except Exception as e: logger.warning("[%s] functional no-op: %s", p["id"], e)
    return gm

def _aten_inner_compile(gm, example_inputs, *, real_inputs=None, **kwargs):   # LEVEL 2 — post-decomp
    weight_source = real_inputs if real_inputs is not None else example_inputs
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    ph_to_tensor = {ph: t for ph, t in zip(placeholders, weight_source)}
    for p in _passes("aten"):
        try:
            gm = p["fn"](gm, ph_to_tensor) if _reads_weight_values(p) else p["fn"](gm)
            _repropagate_meta(gm, example_inputs)     # inserted nodes get meta['val']
        except Exception as e: logger.warning("[%s] aten no-op: %s", p["id"], e)
    return compile_fx_inner(gm, example_inputs, **kwargs)

def _config_patches():                                # LEVEL 3 — scoped Inductor config
    patches = {}
    for p in _passes("inductor_config"):
        try: patches.update(p["fn"]() or {})
        except Exception as e: logger.warning("[%s] config skipped: %s", p["id"], e)
    return patches

def _compile_unit(gm, example_inputs):
    gm = _run_functional_passes(gm)
    inner = functools.partial(_aten_inner_compile, real_inputs=list(example_inputs))
    return compile_fx(gm, example_inputs, inner_compile=inner, config_patches=_config_patches())
```

> **Pass-argument naming:** the weight-reading aten-pass examples below take a parameter named
> `fw_example_inputs` for historical reasons. It receives the threaded **`real_inputs`** list
> (`weight_source`) — not `inner_compile`'s (possibly Fake) `example_inputs`. Bodies unchanged.

### Dedup-aware backend

```python
@register_backend
def {model}_opt(gm: fx.GraphModule, example_inputs) -> Callable:
    registry = UniqueSubgraphRegistry(gm)
    equiv_map = registry.build_partition_equivalence_map()

    if not equiv_map:
        logger.info("{model}_opt: no repeated layers, flat compile path")
        return _compile_unit(gm, example_inputs)

    logger.info(f"{model}_opt: {len(equiv_map)} duplicate partition(s), dedup path")
    partition_inputs = _capture_partition_inputs(registry.split, example_inputs)
    for rep_name, rep_mod in registry.unique_reps:
        inputs = partition_inputs.get(rep_name, example_inputs)
        compiled = _compile_unit(rep_mod, inputs)      # SAME funnel per rep
        rep_mod.forward = compiled
        for _, dup_mod in registry.duplicates_of(rep_name):
            dup_mod.forward = compiled

    return lambda *args: registry.split(*args)
```

`UniqueSubgraphRegistry` splits at the functional level for structural dedup. Functional passes run **per rep** inside `_compile_unit`, never on the pre-split graph. Do NOT move the split inside `_aten_inner_compile`.

---

## Utility: `_capture_partition_inputs`

Required to get the actual input tensors for each partition so `compile_fx` can lower each rep with correct example inputs (and so they are available as `real_inputs` for weight-value-reading passes).

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

## Pattern 1: QKV Weight Fusion — **`ir_level: functional`**

Fuses 3 independent projections sharing the same input `x` into one wider GEMM + slices.

> **Apply this at the `functional` level — NOT at `aten`.** At the functional level the three
> projections are `F.linear(x, W)` nodes that share the **identical** activation node `x` and carry
> clean weight params. After AOTAutograd decomposition the activation is shattered into a
> per-consumer `aten.view` (so the three `aten.mm` no longer share `args[0]`) and bias-free Linear
> lowers to `aten.mm` (not `aten.addmm`) — an Aten-level shared-activation matcher will silently
> find nothing. The Aten variant at the end of this pattern is kept only for reference.

**Detection signal (functional):** ≥3 `F.linear` nodes whose first arg (the activation) is the same node, all bias-free.

```python
_LINEAR_FNS = {torch.nn.functional.linear}
try: _LINEAR_FNS.add(torch._C._nn.linear)
except Exception: pass
def _is_linear(n):
    return (n.op == "call_function"
            and (n.target in _LINEAR_FNS or getattr(n.target, "__name__", "") == "linear"))

def _fpass_fuse_qkv(gm: fx.GraphModule) -> fx.GraphModule:   # runs in _run_functional_passes
    g = gm.graph
    groups = {}
    for n in g.nodes:
        if _is_linear(n) and isinstance(n.args[0], fx.Node):
            groups.setdefault(n.args[0], []).append(n)
    fused = 0
    for act, lins in groups.items():
        if len(lins) < 3:
            continue
        q, k, v = lins[:3]
        wq, wk, wv = q.args[1], k.args[1], v.args[1]
        if any((len(n.args) > 2 and n.args[2] is not None) or n.kwargs.get("bias") is not None
               for n in (q, k, v)):
            continue  # bias fusion not handled — degrade gracefully
        def _out(w):                       # out-features = weight.shape[0]; cat on dim 0
            mv = w.meta.get("example_value", w.meta.get("val"))
            return int(mv.shape[0]) if mv is not None else None
        nq, nk, nv = _out(wq), _out(wk), _out(wv)
        if None in (nq, nk, nv):
            logger.warning("[fuse_qkv] missing weight meta — pass not applied"); continue
        with g.inserting_before(q):
            w_cat = g.call_function(torch.ops.aten.cat.default, ([wq, wk, wv], 0))
            fused_lin = g.call_function(q.target, (act, w_cat))           # one wide F.linear
            def _chunk(lo, hi):
                s = g.call_function(torch.ops.aten.slice.Tensor, (fused_lin, -1, lo, hi))
                # last-dim slice is strided; make contiguous so downstream .view stays valid
                return g.call_function(torch.ops.aten.clone.default, (s,),
                                       {"memory_format": torch.contiguous_format})
            q_out, k_out, v_out = _chunk(0, nq), _chunk(nq, nq+nk), _chunk(nq+nk, nq+nk+nv)
        q.replace_all_uses_with(q_out); k.replace_all_uses_with(k_out); v.replace_all_uses_with(v_out)
        for d in (q, k, v):
            if not d.users: g.erase_node(d)
        fused += 1
        logger.info("[fuse_qkv] Fused 3 projections into 1 linear (N=%d+%d+%d) [functional]", nq, nk, nv)
        break
    if not fused:
        logger.warning("[fuse_qkv] No 3-way shared-activation linear triplet found — pass not applied")
        return gm
    g.eliminate_dead_code(); g.lint(); gm.recompile()
    return gm
```

After OPT-1 (bf16, `aten` level) casts the resulting single wide `aten.mm`, this lowers to one wide bf16 tensor-core GEMM — the intended end state.

<details><summary>Aten-level variant (reference only — usually no-ops, see warning above)</summary>

```python
def _pass_fuse_qkv_aten(gm: fx.GraphModule, fw_example_inputs: list) -> fx.GraphModule:
    # Groups aten.addmm.default by shared activation. Fails for bias-free Linear (which is
    # aten.mm, not addmm) and when AOTAutograd splits the activation into per-consumer views.
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    ph_to_tensor = {ph: t for ph, t in zip(placeholders, fw_example_inputs)}
    addmm_groups = {}
    for n in gm.graph.nodes:
        if n.op == "call_function" and n.target is torch.ops.aten.addmm.default:
            addmm_groups.setdefault(n.args[1].name, []).append(n)
    # ... (cat weights, aten.mm, aten.chunk, replace, erase) — see git history
    return gm
```
</details>

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

1. **Non-graph passes** live in `get_model_and_input()` only — never inside `_aten_inner_compile`.
2. **Node-count reducing passes first:** QKV fusion, BN fold, SiLU/GEGLU. These reduce the node set so downstream passes see a simplified graph.
3. **Attention restructuring next:** SDPA replacement. Must see the correct q/k/v nodes after any upstream fusion.
4. **Weight layout last:** pre-transposed weights. Runs after fusion passes have finalised the weight node set.
5. **Activation substitution:** independent — conventional position is step 5 but has no structural ordering requirement.

### Mutual Exclusions

| Pass A | Pass B | Conflict | Resolution |
|---|---|---|---|
| `_pass_fuse_qkv` | `_pass_pretranspose_weights` | Both target the same weight placeholders | Run QKV first; after fusion the original placeholders are gone |
| `_pass_replace_sdpa` | `_pass_fuse_qkv` | SDPA must walk the graph after QKV output nodes exist | Always run `_pass_fuse_qkv` before SDPA replacement |
