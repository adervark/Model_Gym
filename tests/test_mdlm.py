"""MDLM (module 18): masked diffusion LMs."""
import pytest
import torch

from lm.config import ModelConfig
from lm.model import Transformer
from lm.mdlm import mdlm_loss, mdlm_generate, MASK_ID


@pytest.fixture
def mdlm_model():
    cfg = ModelConfig(hidden=96, n_layers=2, n_heads=4, n_kv_heads=2,
                      max_seq_len=64, causal=False)  # bidirectional
    return Transformer(cfg), cfg


def test_masking_rate_and_mask_only_loss(mdlm_model):
    m, cfg = mdlm_model
    B, T = 16, 32
    x = torch.randint(0, MASK_ID, (B, T))
    loss, mask = mdlm_loss(m, x)
    assert torch.isfinite(loss)
    assert 0.05 < mask.float().mean() < 0.95  # t ~ U(0,1): masked fraction in range
    # gradient must flow ONLY through masked positions: zeroing unmasked
    # targets' contributions is equivalent (construct two identical inputs,
    # perturb one unmasked position's target -> loss unchanged)
    x2 = x.clone()
    x2[0, 0] = (x2[0, 0] + 1) % MASK_ID
    if not mask[0, 0]:
        loss2, _ = mdlm_loss(m, x2, t_min=0.5, t_max=0.5)
        loss3, _ = mdlm_loss(m, x, t_min=0.5, t_max=0.5)
        assert abs(loss2.item() - loss3.item()) < 1e-6


def test_bidirectional_not_causal(mdlm_model):
    """causal=False: logits at position 0 must depend on FUTURE tokens."""
    m, cfg = mdlm_model
    x = torch.randint(0, MASK_ID, (1, 8))
    l1, _, _ = m(x)
    x2 = x.clone()
    x2[0, 7] = (x2[0, 7] + 1) % MASK_ID
    l2, _, _ = m(x2)
    assert not torch.allclose(l1[0, 0], l2[0, 0])


def test_causal_true_still_masks_future(mdlm_model):
    """Sanity: the AR default still holds (no regression from the causal flag)."""
    cfg2 = ModelConfig(**{**mdlm_model[1].__dict__, "causal": True})
    m2 = Transformer(cfg2)
    x = torch.randint(0, MASK_ID, (1, 8))
    l1, _, _ = m2(x)
    x2 = x.clone()
    x2[0, 7] = (x2[0, 7] + 1) % MASK_ID
    l2, _, _ = m2(x2)
    assert torch.allclose(l1[0, 0], l2[0, 0], atol=1e-6)


def test_training_improves_unmasking():
    """Small vocab (256): the memorized denoising task must become near-perfect."""
    cfg = ModelConfig(hidden=96, n_layers=2, n_heads=4, n_kv_heads=2,
                      max_seq_len=64, causal=False, vocab_size=256)
    m = Transformer(cfg)
    mask_id = 255  # absorbing state inside the small vocab
    # structured sequence (memorizes fast; random tokens are slow by design)
    x = (torch.arange(24) % 200).expand(8, 24)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)

    def accuracy():
        with torch.no_grad():
            t = torch.full((x.shape[0],), 0.5)
            mask = torch.rand_like(x, dtype=torch.float) < t[:, None]
            x_m = torch.where(mask, torch.full_like(x, mask_id), x)
            logits, _, _ = m(x_m)
            preds = logits.argmax(-1)
            return (preds[mask] == x[mask]).float().mean().item()

    before = accuracy()
    for _ in range(250):
        loss, _ = mdlm_loss(m, x, mask_id=mask_id)
        opt.zero_grad()
        loss.backward()
        opt.step()
    after = accuracy()
    assert after > before + 0.3  # memorized 256-way batch: big jump


def test_generate_returns_clean_sequence(mdlm_model):
    m, cfg = mdlm_model
    out = mdlm_generate(m, L=24, steps=24, temperature=1.0)
    assert out.shape == (24,)
    assert (out == MASK_ID).sum() == 0  # fully denoised
    out2 = mdlm_generate(m, L=24, steps=24, temperature=1.0)
    assert out2.shape == (24,)
