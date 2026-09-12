"""Muon optimizer + DAPO upgrades (module 19)."""
import pytest
import torch

from lm.config import ModelConfig
from lm.model import Transformer
from lm.optim import Muon, newton_schulz, split_muon_adam
from lm.grpo import (grpo_loss, group_advantages, filter_degenerate,
                     overlong_shaped_reward)


class TestNewtonSchulz:
    def test_shape_preserved(self):
        """The transpose flag bug returned transposed outputs — must never regress."""
        for shape in [(64, 128), (128, 64), (50304, 96)]:
            G = torch.randn(*shape)
            Q = newton_schulz(G)
            assert Q.shape == G.shape

    def test_reduces_orthogonality_error(self):
        """On momentum-buffer-like matrices (the distribution it's tuned for),
        5 NS steps must move G much closer to orthogonal than it started."""
        torch.manual_seed(0)
        G = torch.randn(64, 128)
        G_n = G / (G.norm() + 1e-7)

        def err(X):
            return (X @ X.T - torch.eye(X.shape[0])).abs().max().item()

        assert err(newton_schulz(G)) < err(G_n) * 0.5

    def test_matches_quintic_math(self):
        """Faithfulness: matches a direct implementation of the published iteration."""
        torch.manual_seed(1)
        G = torch.randn(64, 128)
        a, b, c = 3.4445, -4.7750, 2.0315
        X = G.bfloat16() / (G.norm() + 1e-7)
        for _ in range(5):
            A = X @ X.T
            X = a * X + (b * A + c * A @ A) @ X
        assert torch.allclose(newton_schulz(G), X.float(), atol=0.05)


def test_split_2d_vs_1d():
    cfg = ModelConfig(hidden=96, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=32)
    m = Transformer(cfg)
    muon, adam = split_muon_adam(m)
    assert all(p.ndim >= 2 for p in muon)
    assert all(p.ndim < 2 for p in adam)
    assert len(muon) > 0 and len(adam) > 0  # norms are 1D
    assert sum(p.numel() for p in muon) > sum(p.numel() for p in adam)


def test_muon_trains():
    cfg = ModelConfig(hidden=96, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=32)
    m = Transformer(cfg)
    muon_p, adam_p = split_muon_adam(m)
    opt = Muon(muon_p, adam_p, lr=0.01, adam_lr=3e-4)
    x = torch.randint(0, cfg.vocab_size, (4, 16))
    first = None
    for _ in range(150):
        _, loss, _ = m(x, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < first * 0.5


def test_muon_schedule_bookkeeping():
    cfg = ModelConfig(hidden=96, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=32)
    m = Transformer(cfg)
    muon_p, adam_p = split_muon_adam(m)
    opt = Muon(muon_p, adam_p, lr=0.02, adam_lr=3e-4)
    # the training loop's schedule mechanics must work on Muon groups
    for g in opt.param_groups:
        g["lr"] = g.get("base_lr", 1.0) * 0.5
        if "adam_lr_base" in g:
            g["adam_lr"] = g["adam_lr_base"] * 0.5
    assert opt.param_groups[0]["lr"] == pytest.approx(0.01)
    assert opt.param_groups[1]["adam_lr"] == pytest.approx(1.5e-4)


class TestDAPO:
    def setup_method(self):
        cfg = ModelConfig(hidden=96, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=32)
        self.m = Transformer(cfg)
        self.ref = Transformer(cfg)
        self.prompt = torch.randint(0, cfg.vocab_size, (2, 8))
        self.comp = torch.randint(0, cfg.vocab_size, (2, 16))
        self.mask = torch.ones(2, 16, dtype=torch.bool)
        self.rewards = torch.tensor([1.0, 0.0])
        self.old = torch.full((2, 16), -2.0)

    def test_dapo_and_standard_both_run(self):
        l1, s1 = grpo_loss(self.m, self.ref, self.prompt, self.comp, self.mask,
                           self.rewards, self.old, 2)
        l2, s2 = grpo_loss(self.m, self.ref, self.prompt, self.comp, self.mask,
                           self.rewards, self.old, 2, dapo=True, eps_high=0.28)
        l3, s3 = grpo_loss(self.m, self.ref, self.prompt, self.comp, self.mask,
                           self.rewards, self.old, 2, dapo=True, token_level=True)
        assert torch.isfinite(l1) and torch.isfinite(l2) and torch.isfinite(l3)

    def test_old_logp_must_be_per_token(self):
        """A per-completion mean is not an importance ratio; reject it loudly
        rather than silently training on a meaningless rho."""
        with pytest.raises(ValueError, match="per-token"):
            grpo_loss(self.m, self.ref, self.prompt, self.comp, self.mask,
                      self.rewards, torch.tensor([-2.0, -1.5]), 2)

    def test_matches_reference_surrogate(self):
        """Compare grpo_loss against a from-scratch reference of the published
        objective, for every (dapo, token_level) combination.

        Regression guard: an earlier version left positive-advantage tokens
        completely unclipped under dapo=True (removing the trust region that
        clipping exists to provide) and dropped the pessimistic `min`. Both
        show up here as a mismatch against the reference."""
        from lm.grpo import completion_logps
        eps, eps_high = 0.2, 0.5
        adv = group_advantages(self.rewards, 2)

        with torch.no_grad():
            per_tok = completion_logps(self.m, self.prompt, self.comp, self.mask)
        # Construct old_logp so the ratios are EXACTLY these values, spanning all
        # three regions: below 1-eps, inside the band, and above 1+eps_high.
        # (Letting rho run to ~1e7 instead would make every loss ~1e8, where
        # float32 spacing is ~8 and any tolerance passes trivially.)
        target_rho = torch.tensor([0.5, 1.0, 2.0]).repeat(16)[:16]
        old = per_tok - torch.log(target_rho)[None, :].expand_as(per_tok)

        for dapo in (False, True):
            for token_level in (False, True):
                hi = 1 + (eps_high if dapo else eps)
                rho = torch.exp(per_tok - old)
                surr = torch.min(rho * adv[:, None],
                                 torch.clamp(rho, 1 - eps, hi) * adv[:, None])
                m = self.mask
                want = (-(surr * m).sum() / m.sum() if token_level
                        else -((surr * m).sum(-1) / m.sum(-1)).mean())
                got, _ = grpo_loss(self.m, self.ref, self.prompt, self.comp,
                                   self.mask, self.rewards, old, 2, beta=0.0,
                                   eps=eps, eps_high=eps_high, dapo=dapo,
                                   token_level=token_level)
                assert got.item() == pytest.approx(want.item(), abs=1e-5), (
                    f"dapo={dapo} token_level={token_level}: "
                    f"got {got.item():.5f}, reference {want.item():.5f}")

    def test_token_level_reweights_by_length(self):
        """DAPO's token-level normalization: a long completion must carry more
        weight than a short one. Under GRPO's per-completion mean they are
        equal — that equality is what causes length collapse."""
        mask = torch.ones(2, 16, dtype=torch.bool)
        mask[1, 8:] = False                       # completion 1 is half as long
        old = torch.full((2, 16), -3.0)
        seq, _ = grpo_loss(self.m, self.ref, self.prompt, self.comp, mask,
                           self.rewards, old, 2, beta=0.0, token_level=False)
        tok, _ = grpo_loss(self.m, self.ref, self.prompt, self.comp, mask,
                           self.rewards, old, 2, beta=0.0, token_level=True)
        assert seq.item() != pytest.approx(tok.item(), abs=1e-6)

    def test_dapo_raises_only_the_upper_bound(self):
        """clip-higher widens the band upward; the lower bound stays at 1-eps."""
        # rho << 1 (policy far below rollout): both variants clip at 1-eps,
        # so raising eps_high must make NO difference.
        old = torch.zeros(2, 16)
        a, _ = grpo_loss(self.m, self.ref, self.prompt, self.comp, self.mask,
                         self.rewards, old, 2, beta=0.0, eps=0.2)
        b, _ = grpo_loss(self.m, self.ref, self.prompt, self.comp, self.mask,
                         self.rewards, old, 2, beta=0.0, eps=0.2,
                         dapo=True, eps_high=0.9)
        assert a.item() == pytest.approx(b.item(), abs=1e-6)

    def test_keep_group_filters_degenerate(self):
        keep = filter_degenerate(torch.tensor([1.0, 0.0, 1.0, 0.0]), 4)
        assert keep.tolist() == [True]
        keep = filter_degenerate(torch.tensor([1.0, 1.0, 1.0, 1.0]), 4)
        assert keep.tolist() == [False]
        keep = filter_degenerate(torch.tensor([0.0, 0.0]), 2)
        assert keep.tolist() == [False]
        # mixed: per-group decision
        keep = filter_degenerate(torch.tensor([1.0, 1.0, 0.0, 1.0]), 2)
        assert keep.tolist() == [False, True]

    def test_overlong_shaping(self):
        assert overlong_shaped_reward(1.0, 48, 64, True) == 1.0    # correct: no penalty
        assert overlong_shaped_reward(0.0, 64, 64, False) == -0.5  # wrong + hits limit
        assert overlong_shaped_reward(0.0, 30, 64, False) == 0.0   # wrong, not overlong

    def test_group_advantages_are_standardized(self):
        """Group normalization: mean 0, and +-1/sqrt(2) for a 2-member group
        (torch.std is the unbiased estimator, so std of (1,0) is 0.7071)."""
        adv = group_advantages(torch.tensor([1.0, 0.0, 1.0, 0.0]), 2)
        assert torch.allclose(adv, torch.tensor([0.7071, -0.7071, 0.7071, -0.7071]),
                              atol=1e-3)
        assert adv.view(-1, 2).mean(-1).abs().max().item() < 1e-6
