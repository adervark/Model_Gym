"""Model internals: shapes, causality, KV cache, RoPE, MoE, checkpointing."""
import math

import pytest
import torch

from lm.config import ModelConfig
from lm.model import (Transformer, rms_norm, apply_rope, precompute_rope_freqs,
                      MoE, MLP)


def test_forward_shapes_and_loss(model, tiny_cfg):
    B, T = 2, 32
    x = torch.randint(0, tiny_cfg.vocab_size, (B, T))
    logits, loss, caches = model(x, x)
    assert logits.shape == (B, T, tiny_cfg.vocab_size)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert isinstance(caches, list) and len(caches) == tiny_cfg.n_layers


def test_causality(model, tiny_cfg, seeded):
    """logits at position t must not depend on tokens after t."""
    B, T = 2, 24
    x = torch.randint(0, tiny_cfg.vocab_size, (B, T))
    logits1, _, _ = model(x)
    x2 = x.clone()
    x2[:, T // 2:] = torch.randint(0, tiny_cfg.vocab_size, (B, T - T // 2))
    logits2, _, _ = model(x2)
    t = T // 2 - 1
    assert torch.allclose(logits1[:, :t + 1], logits2[:, :t + 1], atol=1e-6)


def test_cache_equivalence_full_recompute(model, tiny_cfg, seeded):
    """Incremental decoding with KV cache must reproduce full-sequence logits."""
    B, T = 3, 16
    ids = torch.randint(0, tiny_cfg.vocab_size, (B, T))
    logits_full, _, _ = model(ids)

    logits_0, _, caches = model(ids[:, :1])
    for i in range(1, T):
        li, _, caches = model(ids[:, i:i + 1], cache=caches)
        assert torch.allclose(li[0, -1], logits_full[0, i], atol=1e-5), f"pos {i}"


def test_cache_position_tracks_length(model, tiny_cfg, seeded):
    """The rope start offset must be the cache length (shape[2] bug guard)."""
    B, T = 2, 10
    ids = torch.randint(0, tiny_cfg.vocab_size, (B, T))
    _, _, caches = model(ids)
    # cache k shape: [B, n_kv, T, D] — position = dim 2
    k = caches[0][0]
    assert k.shape[2] == T
    # one more token
    tok = torch.randint(0, tiny_cfg.vocab_size, (B, 1))
    li, _, caches2 = model(tok, cache=caches)
    assert caches2[0][0].shape[2] == T + 1
    # full-recompute equivalence at position T (the new token)
    logits_full, _, _ = model(torch.cat([ids, tok], dim=1))
    assert torch.allclose(li[0, -1], logits_full[0, T], atol=1e-5)


def test_rope_relative_position_property():
    """RoPE dot products must depend only on relative position."""
    d, T = 32, 16
    freqs = precompute_rope_freqs(d, T * 2)
    f = torch.cos(freqs)[None, :, None, :]
    fs = torch.sin(freqs)[None, :, None, :]
    q = torch.randn(2, 2, 1, d)
    k = torch.randn(2, 2, 1, d)
    q_i, k_j = apply_rope(q, f[:, 0:1], fs[:, 0:1]), apply_rope(k, f[:, 3:4], fs[:, 3:4])
    q_ii, k_jj = apply_rope(q, f[:, 2:3], fs[:, 2:3]), apply_rope(k, f[:, 5:6], fs[:, 5:6])
    d1 = (q_i * k_j).sum(-1)
    d2 = (q_ii * k_jj).sum(-1)
    assert torch.allclose(d1, d2, atol=1e-4)


def test_yarn_freqs_differ_and_shape():
    base = precompute_rope_freqs(32, 128, method="none")
    linear = precompute_rope_freqs(32, 128, method="linear", scale=2.0)
    yarn = precompute_rope_freqs(32, 128, method="yarn", scale=2.0)
    assert base.shape == linear.shape == yarn.shape == (128, 16)
    assert not torch.allclose(base, yarn)
    assert not torch.allclose(base, linear)


def test_tied_embeddings():
    cfg = ModelConfig(hidden=64, n_layers=1, n_heads=4, n_kv_heads=2,
                      max_seq_len=32, tie_embeddings=True)
    m = Transformer(cfg)
    assert m.head.weight is m.tok_emb.weight


def test_gradient_checkpointing_matches(model, tiny_cfg, seeded):
    cfg2 = ModelConfig(**tiny_cfg.__dict__)
    m1 = Transformer(cfg2, gradient_checkpointing=False)
    m2 = Transformer(cfg2, gradient_checkpointing=True)
    m2.load_state_dict(m1.state_dict())
    x = torch.randint(0, tiny_cfg.vocab_size, (2, 16))
    _, l1, _ = m1(x, x)
    _, l2, _ = m2(x, x)
    l1.backward()
    l2.backward()
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(p1.grad, p2.grad, atol=1e-5)


def test_moe_aux_loss_and_params():
    cfg = ModelConfig(hidden=96, n_layers=2, n_heads=4, n_kv_heads=2,
                      max_seq_len=32, n_experts=4, n_experts_active=2)
    m = Transformer(cfg)
    assert m.cfg.n_params > m.cfg.active_params
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    _, loss, _ = m(x, x)
    assert torch.isfinite(loss)
    # collapsed routing (zero gate -> topk always picks experts 0..k-1):
    # f = [1]*k, p uniform -> aux == k exactly (validates the formula)
    moe = m.blocks[0].mlp
    flat = torch.randn(2, 32, 96, requires_grad=True)
    with torch.no_grad():
        moe.gate.weight.zero_()
    _, aux = moe(flat)
    assert aux.item() == pytest.approx(cfg.n_experts_active, abs=1e-4)


def test_moe_trains(model=None):
    cfg = ModelConfig(hidden=96, n_layers=2, n_heads=4, n_kv_heads=2,
                      max_seq_len=32, n_experts=4, n_experts_active=2)
    m = Transformer(cfg)
    x = torch.randint(0, cfg.vocab_size, (4, 16))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    losses = []
    for _ in range(40):
        _, loss, _ = m(x, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.98


def test_rms_norm_matches_manual():
    x = torch.randn(4, 8)
    w = torch.ones(8)
    got = rms_norm(x, w)
    want = x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + 1e-5)
    assert torch.allclose(got, want, atol=1e-6)
