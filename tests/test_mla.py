"""MLA (DeepSeek-V2 latent attention): module 17."""
import pytest
import torch

from lm.config import ModelConfig
from lm.model import Transformer


@pytest.fixture
def mla_cfg():
    return ModelConfig(hidden=96, n_layers=2, n_heads=4, n_kv_heads=2,
                       max_seq_len=64, use_mla=True,
                       kv_latent_dim=64, rope_latent_dim=16)


def test_forward_and_loss(mla_cfg):
    m = Transformer(mla_cfg)
    x = torch.randint(0, mla_cfg.vocab_size, (2, 16))
    logits, loss, caches = m(x, x)
    assert logits.shape == (2, 16, mla_cfg.vocab_size)
    assert torch.isfinite(loss)
    assert len(caches) == mla_cfg.n_layers


def test_cache_is_smaller_than_gqa(mla_cfg):
    """The entire point: latent cache < GQA cache at equal head count."""
    gqa = ModelConfig(hidden=96, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64)
    assert mla_cfg.kv_cache_tokens_per_position < gqa.kv_cache_tokens_per_position
    # with 8 kv heads the gap widens
    mla8 = ModelConfig(**{**mla_cfg.__dict__, "n_kv_heads": 8})
    gqa8 = ModelConfig(**{**gqa.__dict__, "n_kv_heads": 8})
    assert mla8.kv_cache_tokens_per_position * 2 < gqa8.kv_cache_tokens_per_position


def test_cache_equivalence(mla_cfg, seeded=True):
    torch.manual_seed(0)
    m = Transformer(mla_cfg)
    B, T = 3, 16
    ids = torch.randint(0, mla_cfg.vocab_size, (B, T))
    logits_full, _, _ = m(ids)
    logits_0, _, caches = m(ids[:, :1])
    for i in range(1, T):
        li, _, caches = m(ids[:, i:i + 1], cache=caches)
        assert torch.allclose(li[0, -1], logits_full[0, i], atol=1e-5), f"pos {i}"


def test_mla_trains(mla_cfg):
    m = Transformer(mla_cfg)
    x = torch.randint(0, mla_cfg.vocab_size, (4, 16))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = None
    for _ in range(120):
        _, loss, _ = m(x, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < first * 0.6


def test_mla_with_moe_combined(mla_cfg):
    """Frontier combo (V2/V3 use both): MLA + MoE must coexist."""
    cfg = ModelConfig(**{**mla_cfg.__dict__, "n_experts": 4, "n_experts_active": 2})
    m = Transformer(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    _, loss, _ = m(x, x)
    assert torch.isfinite(loss)
    loss.backward()
