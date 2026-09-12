"""Export our model to GGUF (module 22) — the format llama.cpp loads.

Implements the GGUF v3 container by hand (struct packing) with the `llama`
architecture metadata and a gpt2 byte-BPE tokenizer, so a checkpoint exported
here loads in llama.cpp/ollama/LM Studio.

GGUF layout: magic | version(u32) | n_tensors(u64) | n_kv(u64) | metadata KVs
| tensor infos | tensor data (F32). Types: 0=F32. Metadata values:
UINT32=4, FLOAT32=6, BOOL=7, STRING=8, ARRAY=9.
"""
import struct
import numpy as np
import torch

from .model import Transformer

_MAGIC = b"GGUF"
_VERSION = 3
_T_U32, _T_I32, _T_F32, _T_BOOL, _T_STR, _T_ARR = 4, 5, 6, 7, 8, 9
_T_F32_TENSOR = 0


def _write_str(f, s: bytes):
    f.write(struct.pack("<Q", len(s)))
    f.write(s)


def _kv(f, key: str, vtype: int, value):
    _write_str(f, key.encode())
    f.write(struct.pack("<I", vtype))
    if vtype == _T_U32:
        f.write(struct.pack("<I", value))
    elif vtype == _T_F32:
        f.write(struct.pack("<f", value))
    elif vtype == _T_BOOL:
        f.write(struct.pack("<?", value))
    elif vtype == _T_STR:
        _write_str(f, value if isinstance(value, bytes) else value.encode())
    elif vtype == _T_ARR:
        f.write(struct.pack("<I", value[0]))
        f.write(struct.pack("<Q", len(value[1])))
        elem_t, elems = value
        for e in elems:
            if elem_t == _T_STR:
                _write_str(f, e if isinstance(e, bytes) else e.encode())
            elif elem_t == _T_F32:
                f.write(struct.pack("<f", e))
            elif elem_t == _T_U32:
                f.write(struct.pack("<I", e))
            elif elem_t == _T_I32:
                f.write(struct.pack("<i", e))
    else:
        raise NotImplementedError(vtype)


def _bytes_to_unicode() -> dict[int, str]:
    """The canonical GPT-2 bytes->unicode table (from Radford's encoder.py;
    tiktoken uses the same internally)."""
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("\u00a1"), ord("\u00ac") + 1))
          + list(range(ord("\u00ae"), ord("\u00ff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def gpt2_tokenizer_metadata():
    """Extract the full gpt2 vocab from tiktoken: tokens, types, merges."""
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    byte_decoder = _bytes_to_unicode()

    def b2u(b: bytes) -> str:
        return "".join(byte_decoder[c] for c in b)

    ranks = enc._mergeable_ranks  # token bytes -> rank (bytes = rank for bytes)
    tokens, types, scores = [], [], []
    for i in range(enc.n_vocab):
        if i == enc.eot_token:
            tokens.append("<|endoftext|>")
            types.append(3)  # CONTROL (llama.cpp's own gpt2 export does this)
            scores.append(-1000.0)
        else:
            bs = next(b for b, r in ranks.items() if r == i)
            tokens.append(b2u(bs))
            # NOTE: NORMAL (1) for byte tokens too — llama.cpp's gpt2 export
            # marks only "<0xXX>"-style tokens as BYTE; gpt2 bytes are looked
            # up by their raw char via byte_encode, so typing them BYTE breaks
            # token_to_byte (it substrs "<0xXX>" text).
            types.append(1)
            scores.append(0.0)
    # Recover the merge sequence: each merged token (rank >= 256) came from
    # merging two known pieces at the earliest possible step — the split whose
    # larger piece has MINIMUM rank.
    merges = []
    for r in range(256, enc.n_vocab - 1):
        bs = next(b for b, rank in ranks.items() if rank == r)
        best = None
        for s in range(1, len(bs)):
            if bs[:s] in ranks and bs[s:] in ranks:
                m = max(ranks[bs[:s]], ranks[bs[s:]])
                if best is None or m < best[0]:
                    best = (m, s)
        assert best is not None, f"no valid split for rank {r}"
        s = best[1]
        merges.append(f"{b2u(bs[:s])} {b2u(bs[s:])}")
    return tokens, types, merges, scores


def tensor_map(model: Transformer, with_output: bool = True):
    """Module state_dict keys -> llama.cpp tensor names. The vocab is trimmed
    from the padded 50304 to the real 50257 (padding rows are unused)."""
    sd = model.state_dict()
    out = {}
    n_vocab = 50257
    out["token_embd.weight"] = sd["tok_emb.weight"][:n_vocab]
    for i in range(model.cfg.n_layers):
        b = f"blocks.{i}"
        out[f"blk.{i}.attn_norm.weight"] = sd[f"{b}.norm1"]
        out[f"blk.{i}.ffn_norm.weight"] = sd[f"{b}.norm2"]
        a = f"{b}.attn"
        out[f"blk.{i}.attn_q.weight"] = sd[f"{a}.q.weight"]
        out[f"blk.{i}.attn_k.weight"] = sd[f"{a}.k.weight"]
        out[f"blk.{i}.attn_v.weight"] = sd[f"{a}.v.weight"]
        out[f"blk.{i}.attn_output.weight"] = sd[f"{a}.o.weight"]
        m = f"{b}.mlp"
        out[f"blk.{i}.ffn_gate.weight"] = sd[f"{m}.gate.weight"]
        out[f"blk.{i}.ffn_up.weight"] = sd[f"{m}.up.weight"]
        out[f"blk.{i}.ffn_down.weight"] = sd[f"{m}.down.weight"]
    out["output_norm.weight"] = sd["norm_f"]
    if with_output:
        out["output.weight"] = sd["head.weight"][:n_vocab]
    return out


def export_gguf(model: Transformer, path: str, with_tokenizer: bool = True):
    cfg = model.cfg
    if cfg.n_heads * cfg.head_dim != cfg.hidden:
        raise ValueError(
            f"cannot export: {cfg.n_heads} heads x head_dim {cfg.head_dim} != "
            f"hidden {cfg.hidden}. llama.cpp (and every real arch) requires "
            f"n_heads * head_dim == hidden.")
    tmap = tensor_map(model)
    # llama.cpp requires tensors in sorted order
    names = sorted(tmap)
    tokens, types, merges, scores = gpt2_tokenizer_metadata() \
        if with_tokenizer else (None, None, None, None)

    with open(path, "wb") as f:
        f.write(_MAGIC)
        f.write(struct.pack("<I", _VERSION))
        f.write(struct.pack("<Q", len(tmap)))
        n_kv = 13 + (5 if with_tokenizer else 0)
        f.write(struct.pack("<Q", n_kv))
        _kv(f, "general.architecture", _T_STR, "llama")
        _kv(f, "llama.context_length", _T_U32, cfg.max_seq_len)
        _kv(f, "llama.embedding_length", _T_U32, cfg.hidden)
        _kv(f, "llama.block_count", _T_U32, cfg.n_layers)
        # must equal the actual ffn_gate/ffn_up row count — llama.cpp validates
        # this against the tensor shapes and refuses the file if they disagree.
        _kv(f, "llama.feed_forward_length", _T_U32, cfg.ffn_hidden)
        _kv(f, "llama.attention.head_count", _T_U32, cfg.n_heads)
        _kv(f, "llama.attention.head_count_kv", _T_U32, cfg.n_kv_heads)
        _kv(f, "llama.rope.dimension_count", _T_U32, cfg.head_dim)
        _kv(f, "llama.rope.freq_base", _T_F32, cfg.rope_theta)
        _kv(f, "llama.attention.layer_norm_rms_epsilon", _T_F32, cfg.norm_eps)
        _kv(f, "llama.vocab_size", _T_U32, 50257)
        _kv(f, "general.file_type", _T_U32, 0)  # F32
        _kv(f, "general.name", _T_STR, "lm-course")
        if with_tokenizer:
            _kv(f, "tokenizer.ggml.model", _T_STR, "gpt2")
            _kv(f, "tokenizer.ggml.tokens", _T_ARR, (_T_STR, tokens))
            _kv(f, "tokenizer.ggml.scores", _T_ARR, (_T_F32, scores))
            _kv(f, "tokenizer.ggml.token_type", _T_ARR, (_T_I32, types))
            _kv(f, "tokenizer.ggml.merges", _T_ARR, (_T_STR, merges))

        # tensor infos; offsets are RELATIVE to the (32-byte aligned) data
        # section start — readers add them to that base.
        info_size = sum(8 + len(name.encode()) + 4 + 8 * len(tmap[name].shape) + 4 + 8
                        for name in names)
        data_start = f.tell() + info_size
        align_pad = (-data_start) % 32
        rel_offset = 0
        for name in names:
            t = tmap[name]
            _write_str(f, name.encode())
            f.write(struct.pack("<I", len(t.shape)))
            for d in reversed(t.shape):  # GGUF stores dims reversed (innermost first)
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", _T_F32_TENSOR))
            f.write(struct.pack("<Q", rel_offset))
            rel_offset += t.numel() * 4
        f.write(b"\x00" * align_pad)
        for name in names:
            f.write(tmap[name].detach().float().numpy().tobytes())
    print(f"wrote {path}: {len(tmap)} tensors, "
          f"{sum(t.numel() for t in tmap.values()) / 1e6:.1f}M params")
    return path
