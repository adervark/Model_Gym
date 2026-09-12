"""Config arithmetic, tokenizer, data pipeline."""
import numpy as np
import pytest
import torch

from lm.config import ModelConfig, TrainConfig
from lm.tokenizer import BPETokenizer, get_tokenizer, tokenize_doc
from lm.data import tokenize_to_bin, TokenizedFile


def test_param_count_hand_math():
    """Hand-computed count for a tiny model must match the formula.

    Spell out every matrix — this test exists to catch arithmetic drift in
    ModelConfig.n_params, so it must not reuse n_params' own expressions.
    Two things an earlier version of this test got wrong: SwiGLU is THREE
    matrices (not four), and under GQA the k/v projections are n_kv-sized
    while q/o stay full width."""
    d, V, L, H, n_kv, D = 64, 100, 2, 4, 2, 16
    cfg = ModelConfig(hidden=d, n_layers=L, n_heads=H, n_kv_heads=n_kv,
                      head_dim=D, vocab_size=V, max_seq_len=64)

    emb = V * d                                     # input embedding
    head = V * d                                    # untied readout
    per_attn = d * H * D + d * n_kv * D + d * n_kv * D + H * D * d   # q, k, v, o
    per_mlp = 3 * d * cfg.ffn_hidden                # gate, up, down
    norms = 2 * d * L + d                           # norm1/norm2 per block + norm_f

    assert cfg.ffn_hidden == 192                    # ceil(8*64/3) rounded up to 64
    assert per_attn == 12288
    assert per_mlp == 36864
    assert cfg.n_params == emb + head + norms + (per_attn + per_mlp) * L
    assert cfg.n_params == 111424                   # pinned: catches silent drift


def test_param_count_matches_real_module():
    """The formula must equal an actually-constructed model's parameter count.
    This is the check that makes the hand math above trustworthy."""
    from lm.model import Transformer
    for kw in (dict(hidden=64, n_layers=2, n_heads=4, n_kv_heads=2, head_dim=16,
                    vocab_size=100, max_seq_len=64),
               dict(hidden=128, n_layers=3, n_heads=4, n_kv_heads=1, head_dim=32,
                    vocab_size=256, max_seq_len=64),
               dict(hidden=128, n_layers=2, n_heads=4, n_kv_heads=2, head_dim=32,
                    vocab_size=256, max_seq_len=64, mtp_depth=2)):
        cfg = ModelConfig(**kw)
        real = sum(p.numel() for p in Transformer(cfg).parameters())
        assert cfg.n_params == real, f"{kw}: formula {cfg.n_params} != actual {real}"


def test_moe_param_arithmetic():
    d, V, L, E, k = 96, 50304, 2, 8, 2
    cfg = ModelConfig(hidden=d, n_layers=L, n_heads=4, n_kv_heads=2,
                      max_seq_len=32, n_experts=E, n_experts_active=k)
    # total vs dense: extra (E-1) MLPs per layer, plus the d x E router
    dense = ModelConfig(hidden=d, n_layers=L, n_heads=4, n_kv_heads=2, max_seq_len=32)
    per_mlp = 3 * d * cfg.ffn_hidden
    assert cfg.n_params == dense.n_params + L * (E - 1) * per_mlp + L * d * E
    # active = only k experts run, so it sits between dense and full MoE
    assert dense.n_params < cfg.active_params < cfg.n_params
    assert cfg.active_params == dense.n_params + L * (k - 1) * per_mlp + L * d * E


def test_grad_accum_auto():
    tc = TrainConfig(total_batch_tokens=524288, micro_batch_size=8, seq_len=1024)
    assert tc.grad_accum == 64
    tc2 = TrainConfig(total_batch_tokens=1024, micro_batch_size=8, seq_len=1024)
    assert tc2.grad_accum == 1
    tc3 = TrainConfig(total_batch_tokens=524288, micro_batch_size=8, seq_len=1024,
                      grad_accum_steps=7)
    assert tc3.grad_accum == 7


class TestBPE:
    def test_roundtrip(self):
        text = "abababcdcdabababcd" * 5
        bpe = BPETokenizer(text, vocab_size=32)
        assert bpe.decode(bpe.encode(text)) == text

    def test_compresses_repetition(self):
        text = "abcdefgh" * 50
        bpe = BPETokenizer(text, vocab_size=40)
        ids = bpe.encode(text)
        assert len(ids) < len(text) // 4

    def test_unknown_chars_fail_loudly(self):
        bpe = BPETokenizer("abc", vocab_size=10)
        with pytest.raises(KeyError):
            bpe.encode("xyz")


def test_tiktoken_eot_and_roundtrip():
    enc = get_tokenizer()
    ids = tokenize_doc(enc, "hello world")
    assert ids[-1] == enc.eot_token
    assert enc.decode(ids[:-1]) == "hello world"


def test_bin_roundtrip(tmp_path):
    enc = get_tokenizer()
    text = "the quick brown fox " * 200
    want = enc.encode_ordinary(text) + [enc.eot_token]
    p = str(tmp_path / "t.bin")
    tokenize_to_bin(text, p)
    got = np.fromfile(p, dtype=np.uint16).tolist()
    assert got == want


def test_streaming_dataset_windows(tmp_path):
    enc = get_tokenizer()
    text = " ".join([f"document number {i}" for i in range(300)])
    p = str(tmp_path / "t.bin")
    tokenize_to_bin(text, p)
    ds = TokenizedFile(p, seq_len=16)
    it = iter(ds)
    for _ in range(10):
        x = next(it)
        assert x.shape == (17,)
        assert x.dtype == torch.int64
