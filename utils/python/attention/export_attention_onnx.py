#!/usr/bin/env python3
"""
Export a Hugging Face model whose attention uses torch SDPA
(torch.nn.functional.scaled_dot_product_attention) to ONNX, so that SDPA
lowers to the native ONNX `Attention` op (ai.onnx, opset >= 23) and is not
decomposed into MatMul/Softmax.

Requirements:
    pip install torch onnx onnxscript transformers onnxruntime

Examples:
    # Decoders with a fixed-size cache updated by ONNX TensorScatter (opset 24)
    python export_attention_onnx.py --model ibm-granite/granite-4.1-3b --static-cache \
        --max-cache-len 4096 --no-optimize

    # Encoders
    python export_attention_onnx.py --model bert-base-uncased
    python export_attention_onnx.py --model ibm-granite/granite-embedding-311m-multilingual-r2 \
        --no-optimize --seq 256

    # Decoders, no cache (single full-sequence forward)
    python export_attention_onnx.py --model gpt2 --causal

    # Decoders with KV cache as graph inputs/outputs (prefill + decode)
    python export_attention_onnx.py --model ibm-granite/granite-4.1-3b --causal --kv-cache --no-optimize

    # Decoders with a fixed-size cache updated by ONNX TensorScatter (opset 24)
    python export_attention_onnx.py --model ibm-granite/granite-4.1-3b --static-cache \
        --max-cache-len 4096 --no-optimize

    # Offline: tiny randomly initialised model (no download)
    python export_attention_onnx.py --random bert
    python export_attention_onnx.py --random granite --kv-cache --no-optimize

    # Compare against the decomposed form
    python export_attention_onnx.py --random bert --opset 18

KV-cache graph signature (--kv-cache):
    inputs : input_ids        [batch, seq]
             attention_mask   [batch, past_seq + seq]
             position_ids     [batch, seq]
             past_key.{i}     [batch, kv_heads, past_seq, head_dim]   i = 0..L-1
             past_value.{i}   [batch, kv_heads, past_seq, head_dim]
    outputs: logits           [batch, seq, vocab]
             present_key.{i}  [batch, kv_heads, past_seq + seq, head_dim]
             present_value.{i}
    Prefill: feed past_seq = 0 (empty tensors). Decode: feed seq = 1 and the
    previous step's present_* as past_*.

Static-cache graph signature (--static-cache, opset 24):
    inputs : input_ids        [batch, seq]
             position_ids     [batch, seq]
             write_indices    [batch]          slot where this step's K/V go
             past_key.{i}     [batch, kv_heads, max_cache_len, head_dim]
             past_value.{i}   [batch, kv_heads, max_cache_len, head_dim]
    outputs: logits           [batch, seq, vocab]
             present_key.{i}  [batch, kv_heads, max_cache_len, head_dim]
             present_value.{i}
    present_key.{i} = TensorScatter(past_key.{i}, new_k, write_indices,
                                    axis=-2, mode="linear")
    nonpad_kv_seqlen = write_indices + seq
    Y = Attention(Q, present_key.{i}, present_value.{i}, "", "", "",
                  nonpad_kv_seqlen, is_causal=1)        # Attention-24, native GQA
    i.e. query q of sample b sees slots j <= write_indices[b] + q (bottom-right
    causal alignment, offset = nonpad_kv_seqlen - seq). --attn-mask keeps the
    exporter's explicit mask instead; --keep-repeat-kv keeps HF's repeat_kv.
    Prefill: zero buffers, write_indices = 0. Decode: write_indices = tokens
    already in the cache (per sample, may differ). Requires
    write_indices[b] + seq <= max_cache_len.
"""
import argparse
import collections

import torch
import onnx


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def tiny_config(arch: str):
    """Small configs so the script can run without hitting the Hub."""
    from transformers import BertConfig, GPT2Config, LlamaConfig

    if arch == "bert":
        return BertConfig(hidden_size=64, num_hidden_layers=2,
                          num_attention_heads=4, intermediate_size=128,
                          vocab_size=1000), False
    if arch == "gpt2":
        return GPT2Config(n_embd=64, n_layer=2, n_head=4, vocab_size=1000), True
    if arch == "llama":
        return LlamaConfig(hidden_size=64, num_hidden_layers=2,
                           num_attention_heads=4, num_key_value_heads=2,
                           intermediate_size=128, vocab_size=1000), True
    if arch == "granite":
        # Same layer type as ibm-granite/granite-4.1-* (GraniteForCausalLM:
        # GQA, RoPE, SwiGLU, RMSNorm, attention/residual multipliers).
        from transformers import GraniteConfig
        return GraniteConfig(hidden_size=64, num_hidden_layers=2,
                             num_attention_heads=4, num_key_value_heads=2,
                             intermediate_size=128, vocab_size=1000,
                             attention_multiplier=0.0625, embedding_multiplier=12.0,
                             residual_multiplier=0.22, logits_scaling=8.0,
                             tie_word_embeddings=True), True
    if arch == "modernbert":
        # Same attention layout as ModernBERT checkpoints such as
        # ibm-granite/granite-embedding-311m-multilingual-r2: RoPE, a global
        # layer every 3rd layer, 128-token sliding window on the others.
        from transformers import ModernBertConfig
        return ModernBertConfig(hidden_size=64, num_hidden_layers=6,
                                num_attention_heads=4, intermediate_size=128,
                                vocab_size=1000, pad_token_id=0, bos_token_id=1,
                                eos_token_id=2, cls_token_id=1, sep_token_id=2,
                                global_attn_every_n_layers=3,
                                local_attention=128), False
    raise ValueError(arch)


def load_model(args):
    from transformers import AutoModel, AutoModelForCausalLM

    if args.random:
        cfg, causal = tiny_config(args.random)
        cfg._attn_implementation = "sdpa"
        cls = AutoModelForCausalLM if causal else AutoModel
        return cls.from_config(cfg, attn_implementation="sdpa"), causal

    causal = args.causal or args.kv_cache
    cls = AutoModelForCausalLM if causal else AutoModel
    model = cls.from_pretrained(args.model, attn_implementation="sdpa",
                                dtype=torch.float32)
    return model, causal


class Wrapper(torch.nn.Module):
    """Fixed signature, plain tensor output, no KV cache."""

    def __init__(self, model, causal, use_mask):
        super().__init__()
        self.model, self.causal, self.use_mask = model, causal, use_mask

    def forward(self, input_ids, attention_mask=None):
        kw = dict(input_ids=input_ids, return_dict=True)
        if self.use_mask:
            kw["attention_mask"] = attention_mask
        if self.causal:
            kw["use_cache"] = False
            return self.model(**kw).logits
        return self.model(**kw).last_hidden_state


def kv_shape_info(config):
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    kv_heads = getattr(config, "num_key_value_heads", None) or n_heads
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // n_heads
    return n_layers, kv_heads, head_dim


class KVWrapper(torch.nn.Module):
    """Flattened KV cache: past_key/value.{i} in, present_key/value.{i} out."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.n_layers = kv_shape_info(model.config)[0]

    def forward(self, input_ids, attention_mask, position_ids, *past):
        from transformers import DynamicCache

        cache = DynamicCache(config=self.model.config)
        for i in range(self.n_layers):
            cache.update(past[2 * i], past[2 * i + 1], i)
        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         position_ids=position_ids, past_key_values=cache,
                         use_cache=True, return_dict=True)
        presents = []
        for layer in out.past_key_values.layers[: self.n_layers]:
            presents += [layer.keys, layer.values]
        return (out.logits, *presents)


# --------------------------------------------------------------------------
# Static (fixed-size) KV cache updated with ONNX TensorScatter (opset 24)
# --------------------------------------------------------------------------
@torch.library.custom_op("onnx_kv::tensor_scatter", mutates_args=())
def tensor_scatter(past: torch.Tensor, update: torch.Tensor,
                   write_indices: torch.Tensor) -> torch.Tensor:
    """Eager reference of ONNX TensorScatter (mode="linear", axis=-2)."""
    out = past.clone()
    s = update.shape[-2]
    for b in range(past.shape[0]):
        i = int(write_indices[b])
        out[b, ..., i:i + s, :] = update[b]
    return out


@tensor_scatter.register_fake
def _tensor_scatter_fake(past, update, write_indices):
    return torch.empty_like(past)


def tensor_scatter_onnx(past, update, write_indices):
    """Translation of onnx_kv::tensor_scatter into ai.onnx::TensorScatter-24."""
    from onnxscript.values import Opset
    return Opset("", 24).TensorScatter(past, update, write_indices,
                                       axis=-2, mode="linear")


def _make_scatter_cache(config, past, write_indices):
    from transformers import DynamicCache

    class ScatterCache(DynamicCache):
        """Cache whose update writes into fixed-size buffers via TensorScatter
        and hands the whole [.., max_cache_len, ..] buffer to attention."""

        def __init__(self):
            super().__init__(config=config)
            self.presents = {}

        def update(self, key_states, value_states, layer_idx, *args, **kwargs):
            k = tensor_scatter(past[2 * layer_idx], key_states, write_indices)
            v = tensor_scatter(past[2 * layer_idx + 1], value_states, write_indices)
            self.presents[layer_idx] = (k, v)
            return k, v

    return ScatterCache()


class StaticKVWrapper(torch.nn.Module):
    """Fixed-size cache: past_key/value.{i} [B, kv_heads, max_cache_len, D] in,
    present_key/value.{i} (same shape) out, written at write_indices[b]."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.n_layers = kv_shape_info(model.config)[0]

    def forward(self, input_ids, position_ids, write_indices, *past):
        seq = input_ids.shape[1]
        max_len = past[0].shape[2]
        q = torch.arange(seq, device=input_ids.device)
        j = torch.arange(max_len, device=input_ids.device)
        # query q of sample b (absolute slot write_indices[b] + q) may attend to
        # slots 0..write_indices[b] + q: causal, and hides unwritten slots.
        limit = write_indices[:, None] + q[None, :]                   # [B, S]
        mask = (j[None, None, :] <= limit[:, :, None])[:, None]      # [B, 1, S, M]
        cache = _make_scatter_cache(self.model.config, past, write_indices)
        out = self.model(input_ids=input_ids, attention_mask=mask,
                         position_ids=position_ids, past_key_values=cache,
                         use_cache=True, return_dict=True)
        presents = []
        for i in range(self.n_layers):
            presents += list(cache.presents[i])
        return (out.logits, *presents)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def report_ops(path):
    m = onnx.load(path, load_external_data=False)
    ops = collections.Counter(f"{n.domain or 'ai.onnx'}::{n.op_type}" for n in m.graph.node)
    for f in m.functions:
        ops.update(f"{n.domain or 'ai.onnx'}::{n.op_type}" for n in f.node)
    print("opset_import:", {o.domain or "ai.onnx": o.version for o in m.opset_import})
    print("inputs :", [i.name for i in m.graph.input][:8],
          "..." if len(m.graph.input) > 8 else "")
    print("outputs:", [o.name for o in m.graph.output][:8],
          "..." if len(m.graph.output) > 8 else "")
    print("op histogram:")
    for k, v in sorted(ops.items(), key=lambda kv: -kv[1]):
        print(f"  {v:4d}  {k}")
    n_attn = ops.get("ai.onnx::Attention", 0)
    if n_attn:
        print(f"OK: {n_attn} native ONNX Attention op(s)")
    else:
        print("NOTE: no ai.onnx::Attention op; SDPA was decomposed "
              "(opset < 23, or exporter version lacks the mapping).")


def ort_session(path):
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed; skipping numeric check")
        return None
    so = ort.SessionOptions()
    so.log_severity_level = 3  # hide per-run "in-place optimization not activated" notes
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


# --------------------------------------------------------------------------
# Export paths
# --------------------------------------------------------------------------
def export_plain(args, model, causal, out):
    use_mask = not args.no_mask
    wrapped = Wrapper(model, causal, use_mask).eval()
    vocab = model.config.vocab_size
    input_ids = torch.randint(0, vocab, (args.batch, args.seq), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    inputs = (input_ids, attention_mask) if use_mask else (input_ids,)
    input_names = ["input_ids", "attention_mask"][: len(inputs)]

    dynamic_shapes = None
    if not args.static:
        B = torch.export.Dim("batch", min=1, max=64)
        S = torch.export.Dim("seq", min=2, max=args.max_seq)
        dynamic_shapes = tuple({0: B, 1: S} for _ in inputs)

    with torch.no_grad():
        prog = torch.onnx.export(wrapped, inputs, dynamo=True,
                                 opset_version=args.opset,
                                 input_names=input_names, output_names=["output"],
                                 dynamic_shapes=dynamic_shapes,
                                 optimize=not args.no_optimize)
    prog.save(out)
    print(f"saved {out}")
    report_ops(out)

    if args.no_check or (sess := ort_session(out)) is None:
        return
    import numpy as np
    b2, s2 = (args.batch, args.seq) if args.static else (args.batch + 1, args.seq + 7)
    ids2 = torch.randint(0, vocab, (b2, s2), dtype=torch.long)
    mask2 = torch.ones_like(ids2)
    if use_mask and b2 > 1:
        mask2[0, -3:] = 0  # right padding on the first sequence
    inp2 = (ids2, mask2) if use_mask else (ids2,)
    with torch.no_grad():
        ref = wrapped(*inp2).numpy()
    got = sess.run(None, {n: t.numpy() for n, t in zip(input_names, inp2)})[0]
    valid = mask2.bool().numpy() if use_mask else np.ones(ref.shape[:2], bool)
    diff = np.abs(ref - got)[valid].max()
    print(f"onnxruntime: shape {got.shape}, max |torch - ort| = {diff:.3e}")


def export_kv(args, model, out):
    cfg = model.config
    n_layers, kv_heads, head_dim = kv_shape_info(cfg)
    print(f"KV cache: {n_layers} layers x [batch, {kv_heads}, past_seq, {head_dim}]")
    wrapped = KVWrapper(model).eval()

    B, S, P = args.batch, args.seq, args.past_seq
    vocab = cfg.vocab_size
    input_ids = torch.randint(0, vocab, (B, S), dtype=torch.long)
    attention_mask = torch.ones(B, P + S, dtype=torch.long)
    position_ids = torch.arange(P, P + S, dtype=torch.long).expand(B, S).contiguous()
    past = []
    for _ in range(n_layers):
        past += [torch.randn(B, kv_heads, P, head_dim), torch.randn(B, kv_heads, P, head_dim)]
    inputs = (input_ids, attention_mask, position_ids, *past)

    input_names = ["input_ids", "attention_mask", "position_ids"]
    output_names = ["logits"]
    for i in range(n_layers):
        input_names += [f"past_key.{i}", f"past_value.{i}"]
        output_names += [f"present_key.{i}", f"present_value.{i}"]

    dbatch = torch.export.Dim("batch", min=1, max=64)
    dseq = torch.export.Dim("seq", min=1, max=args.max_seq)
    dpast = torch.export.Dim("past_seq", min=0, max=args.max_seq)
    dtotal = torch.export.Dim("total_seq", min=1, max=2 * args.max_seq)
    dynamic_shapes = (
        {0: dbatch, 1: dseq},
        {0: dbatch, 1: dtotal},
        {0: dbatch, 1: dseq},
        tuple({0: dbatch, 2: dpast} for _ in past),  # *past is one varargs entry
    )

    with torch.no_grad():
        prog = torch.onnx.export(wrapped, inputs, dynamo=True,
                                 opset_version=args.opset,
                                 input_names=input_names, output_names=output_names,
                                 dynamic_shapes=dynamic_shapes,
                                 optimize=not args.no_optimize)
    prog.save(out)
    print(f"saved {out}")
    report_ops(out)

    if args.no_check or (sess := ort_session(out)) is None:
        return
    import numpy as np

    # Greedy-free check: prefill T0 tokens with empty past, then decode 3
    # tokens one at a time feeding present_* back as past_*. Compare each
    # step's logits against a single uncached torch forward over all tokens.
    b, t0, steps = 2, 7, 3
    ids = torch.randint(0, vocab, (b, t0 + steps), dtype=torch.long)
    with torch.no_grad():
        ref = model(input_ids=ids, use_cache=False).logits.numpy()

    def feed(step_ids, past_len, pkv):
        s = step_ids.shape[1]
        f = {"input_ids": step_ids.numpy(),
             "attention_mask": np.ones((b, past_len + s), np.int64),
             "position_ids": np.broadcast_to(np.arange(past_len, past_len + s),
                                             (b, s)).astype(np.int64).copy()}
        for i in range(n_layers):
            f[f"past_key.{i}"], f[f"past_value.{i}"] = pkv[2 * i], pkv[2 * i + 1]
        return f

    empty = np.zeros((b, kv_heads, 0, head_dim), np.float32)
    outs = sess.run(None, feed(ids[:, :t0], 0, [empty] * (2 * n_layers)))
    d = np.abs(outs[0] - ref[:, :t0]).max()
    print(f"prefill  ({t0} tok, past 0): logits max diff {d:.3e}, "
          f"present_key.0 {outs[1].shape}")
    for k in range(steps):
        pos = t0 + k
        outs = sess.run(None, feed(ids[:, pos:pos + 1], pos, outs[1:]))
        d = np.abs(outs[0][:, 0] - ref[:, pos]).max()
        print(f"decode   (1 tok, past {pos}): logits max diff {d:.3e}, "
              f"present_key.0 {outs[1].shape}")


def rewrite_attention_nonpad(path, native_gqa=True):
    """Post-export pass on a --static-cache graph (opset 24).

    Each Attention node becomes
        Attention(Q, K_cache, V_cache, "", "", "", nonpad_kv_seqlen, is_causal=1)
    with nonpad_kv_seqlen = write_indices + seq computed once in the graph.
    Attention-24 then applies the offset-aware causal rule
        query i attends key j  iff  j <= i + (nonpad_kv_seqlen - q_seq) = i + write_indices
    which is exactly the mask the exporter produced, so attn_mask is dropped.
    With native_gqa, K/V come straight from TensorScatter (kv_heads < q_heads)
    and the exported repeat_kv chain (Unsqueeze/Expand/Reshape) is removed.
    """
    from onnx import helper

    m = onnx.load(path, load_external_data=False)  # weights stay in .onnx.data
    g = m.graph
    prod = {o: n for n in g.node for o in n.output}

    # nonpad_kv_seqlen [batch] = write_indices + seq_len(input_ids)
    shape_n = helper.make_node("Shape", ["input_ids"], ["nonpad_seq_len"],
                               start=1, end=2, name="nonpad_seq_len")
    add_n = helper.make_node("Add", ["write_indices", "nonpad_seq_len"],
                             ["nonpad_kv_seqlen"], name="nonpad_kv_seqlen")

    def cache_source(t):
        """Walk back K/V through repeat_kv to the TensorScatter output."""
        seen = t
        while t in prod and prod[t].op_type in ("Reshape", "Identity", "Expand", "Unsqueeze"):
            t = prod[t].input[0]
        return t if t in prod and prod[t].op_type == "TensorScatter" else seen

    n_attn = n_gqa = 0
    for n in g.node:
        if n.op_type != "Attention" or n.domain not in ("", "ai.onnx"):
            continue
        q, k, v = n.input[0], n.input[1], n.input[2]
        if native_gqa:
            k2, v2 = cache_source(k), cache_source(v)
            n_gqa += (k2 != k) + (v2 != v)
            k, v = k2, v2
        del n.input[:]
        n.input.extend([q, k, v, "", "", "", "nonpad_kv_seqlen"])
        attrs = [a for a in n.attribute if a.name != "is_causal"]
        del n.attribute[:]
        n.attribute.extend(attrs + [helper.make_attribute("is_causal", 1)])
        n_attn += 1

    nodes = [shape_n, add_n] + list(g.node)
    # Dead-code elimination: drops the old mask subgraph and repeat_kv chains
    live = {o.name for o in g.output}
    keep = []
    for n in reversed(nodes):
        if any(o in live for o in n.output):
            keep.append(n)
            live.update(i for i in n.input if i)
    removed = len(nodes) - len(keep)
    del g.node[:]
    g.node.extend(reversed(keep))
    used = {i for n in g.node for i in n.input}
    for vi in [vi for vi in g.value_info if vi.name not in used and vi.name not in live]:
        g.value_info.remove(vi)
    onnx.save(m, path)
    onnx.checker.check_model(path, full_check=True)
    print(f"rewrite: {n_attn} Attention node(s) -> nonpad_kv_seqlen + is_causal=1, "
          f"attn_mask removed; {n_gqa} repeat_kv input(s) bypassed (native GQA); "
          f"{removed} dead node(s) removed")


def export_static_kv(args, model, out):
    cfg = model.config
    n_layers, kv_heads, head_dim = kv_shape_info(cfg)
    M = args.max_cache_len
    print(f"static KV cache: {n_layers} layers x [batch, {kv_heads}, {M}, {head_dim}]")
    wrapped = StaticKVWrapper(model).eval()

    B, S, P = args.batch, args.seq, args.past_seq
    if P + S > M:
        raise SystemExit(f"--past-seq + --seq ({P + S}) must be <= --max-cache-len ({M})")
    vocab = cfg.vocab_size
    input_ids = torch.randint(0, vocab, (B, S), dtype=torch.long)
    position_ids = torch.arange(P, P + S, dtype=torch.long).expand(B, S).contiguous()
    write_indices = torch.full((B,), P, dtype=torch.long)
    past = []
    for _ in range(n_layers):
        past += [torch.randn(B, kv_heads, M, head_dim), torch.randn(B, kv_heads, M, head_dim)]
    inputs = (input_ids, position_ids, write_indices, *past)

    input_names = ["input_ids", "position_ids", "write_indices"]
    output_names = ["logits"]
    for i in range(n_layers):
        input_names += [f"past_key.{i}", f"past_value.{i}"]
        output_names += [f"present_key.{i}", f"present_value.{i}"]

    dbatch = torch.export.Dim("batch", min=1, max=64)
    dseq = torch.export.Dim("seq", min=1, max=M)
    dynamic_shapes = (
        {0: dbatch, 1: dseq},
        {0: dbatch, 1: dseq},
        {0: dbatch},
        tuple({0: dbatch} for _ in past),  # max_cache_len stays static
    )

    with torch.no_grad():
        prog = torch.onnx.export(
            wrapped, inputs, dynamo=True, opset_version=args.opset,
            input_names=input_names, output_names=output_names,
            dynamic_shapes=dynamic_shapes, optimize=not args.no_optimize,
            custom_translation_table={
                torch.ops.onnx_kv.tensor_scatter.default: tensor_scatter_onnx},
        )
    prog.save(out)
    print(f"saved {out}")
    if not args.attn_mask:
        rewrite_attention_nonpad(out, native_gqa=not args.keep_repeat_kv)
    report_ops(out)

    if args.no_check or (sess := ort_session(out)) is None:
        return
    import numpy as np

    # Prefill t0 tokens into an all-zero buffer at slot 0, then decode 3 tokens
    # one at a time at write_indices = current length, feeding present -> past.
    # Sample 1 uses a shorter prompt (t0 - 2) so write_indices differ per batch.
    b, t0, steps = 2, 7, 3
    lens = [t0, t0 - 2]
    ids = torch.randint(0, vocab, (b, t0 + steps), dtype=torch.long)
    refs = []
    with torch.no_grad():
        for k in range(b):
            seq_k = torch.cat([ids[k, :lens[k]], ids[k, t0:]])[None]
            refs.append(model(input_ids=seq_k, use_cache=False).logits[0].numpy())

    def run(step_ids, pos, pkv):
        s = step_ids.shape[1]
        f = {"input_ids": step_ids.numpy(),
             "position_ids": (np.asarray(pos)[:, None] + np.arange(s)).astype(np.int64),
             "write_indices": np.asarray(pos, np.int64)}
        for i in range(n_layers):
            f[f"past_key.{i}"], f[f"past_value.{i}"] = pkv[2 * i], pkv[2 * i + 1]
        return sess.run(None, f)

    zero = np.zeros((b, kv_heads, M, head_dim), np.float32)
    # prefill: right-pad sample 1's prompt; its pad positions get overwritten
    # by the first decode step because write_indices[1] = lens[1].
    outs = run(ids[:, :t0], [0, 0], [zero] * (2 * n_layers))
    d = max(np.abs(outs[0][k, :lens[k]] - refs[k][:lens[k]]).max() for k in range(b))
    print(f"prefill  (prompt lens {lens}, write_indices [0, 0]): logits max diff {d:.3e}, "
          f"present_key.0 {outs[1].shape}")
    pos = list(lens)
    for step in range(steps):
        tok = torch.stack([ids[0, t0 + step], ids[1, t0 + step]])[:, None]
        outs = run(tok, pos, outs[1:])
        d = max(np.abs(outs[0][k, 0] - refs[k][lens[k] + step]).max() for k in range(b))
        print(f"decode   (1 tok, write_indices {pos}): logits max diff {d:.3e}, "
              f"present_key.0 {outs[1].shape}")
        pos = [p + 1 for p in pos]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="bert-base-uncased", help="HF model id or local path")
    p.add_argument("--causal", action="store_true", help="load as AutoModelForCausalLM")
    p.add_argument("--random", choices=["bert", "gpt2", "llama", "granite", "modernbert"],
                   help="tiny random model instead of --model (offline)")
    p.add_argument("--kv-cache", action="store_true",
                   help="decoder only: expose past/present KV cache as graph inputs/outputs")
    p.add_argument("--static-cache", action="store_true",
                   help="with --kv-cache: fixed-size cache updated by TensorScatter (opset 24)")
    p.add_argument("--attn-mask", action="store_true",
                   help="--static-cache: keep the exported attn_mask instead of "
                        "rewriting to nonpad_kv_seqlen + is_causal=1")
    p.add_argument("--keep-repeat-kv", action="store_true",
                   help="--static-cache: keep the exported repeat_kv (no native GQA)")
    p.add_argument("--max-cache-len", type=int, default=256,
                   help="cache buffer length for --static-cache (static dim)")
    p.add_argument("--opset", type=int, default=None,
                   help="default 23 (24 with --static-cache); >=23 for native Attention")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seq", type=int, default=16, help="example sequence length for tracing")
    p.add_argument("--past-seq", type=int, default=8, help="example past length (--kv-cache)")
    p.add_argument("--max-seq", type=int, default=4096, help="upper bound of dynamic seq dims")
    p.add_argument("--no-mask", action="store_true", help="don't pass attention_mask")
    p.add_argument("--static", action="store_true", help="disable dynamic batch/seq")
    p.add_argument("--no-optimize", action="store_true",
                   help="skip onnxscript graph optimization (no RotaryEmbedding/RMSNorm fusion)")
    p.add_argument("--out", default=None)
    p.add_argument("--no-check", action="store_true", help="skip onnxruntime comparison")
    args = p.parse_args()

    if args.static_cache:
        args.kv_cache = True
    if args.opset is None:
        args.opset = 24 if args.static_cache else 23
    if args.static_cache and args.opset < 24:
        raise SystemExit("--static-cache needs --opset >= 24 (TensorScatter)")

    torch.manual_seed(0)
    model, causal = load_model(args)
    model.eval()
    impl = model.config._attn_implementation
    print(f"attn_implementation = {impl}")
    if impl != "sdpa":
        print("WARNING: model is not using SDPA; there will be no Attention op to map.")

    tag = (args.random or args.model).replace("/", "_")
    kind = "_static_kv" if args.static_cache else ("_kv" if args.kv_cache else "")
    out = args.out or f"{tag}{kind}_opset{args.opset}.onnx"

    if args.kv_cache:
        if not causal:
            raise SystemExit("--kv-cache needs a decoder (--causal or --random gpt2/llama/granite)")
        if args.static_cache:
            export_static_kv(args, model, out)
        else:
            export_kv(args, model, out)
    else:
        export_plain(args, model, causal, out)


if __name__ == "__main__":
    main()
