"""SAE (module 20) and VLM (module 21)."""
import pytest
import torch

from lm.config import ModelConfig
from lm.model import Transformer
from lm.tokenizer import get_tokenizer
from lm.sae import (SparseAutoencoder, collect_activations, feature_top_tokens,
                    dead_features, train_sae)
from lm.vlm import VLM, RandomVisionTower, caption_batch


class TestSAE:
    def test_reconstruction_learns(self):
        torch.manual_seed(0)
        sae = SparseAutoencoder(d=64, f=256)
        x = torch.randn(200, 64)
        opt = torch.optim.AdamW(sae.parameters(), lr=1e-2)
        first_mse = None
        for _ in range(400):
            loss, stats = sae.loss(x, l1_coef=1e-3)
            opt.zero_grad()
            loss.backward()
            opt.step()
            first_mse = first_mse if first_mse is not None else stats["mse"]
        # MSE must drop substantially vs input variance (~1.0)
        assert stats["mse"] < first_mse * 0.5
        assert stats["mse"] < 0.5

    def test_sparsity_and_dead_features(self):
        torch.manual_seed(1)
        sae = SparseAutoencoder(d=64, f=256, k=8)
        x = torch.randn(128, 64)
        opt = torch.optim.AdamW(sae.parameters(), lr=1e-2)
        for _ in range(300):
            loss, _ = sae.loss(x, l1_coef=1e-3)
            opt.zero_grad()
            loss.backward()
            opt.step()
        recon, acts = sae(x)
        # top-k=8: at most 8 active features per sample (zero-valued selections
        # can lower the count slightly; random data keeps most nonzero)
        l0 = acts.count_nonzero(dim=1).float().mean().item()
        assert 5.0 <= l0 <= 8.0  # top-8 selection; L1 pushes some to exact 0
        assert dead_features(sae, acts) < 256 * 0.5  # most features alive

    def test_collect_activations_shape(self, model, tiny_cfg):
        ids = torch.randint(0, tiny_cfg.vocab_size, (2, 16))
        h = collect_activations(model, ids, layer=0)
        assert h.shape == (2 * 16, tiny_cfg.hidden)

    def test_feature_top_tokens(self):
        cfg = ModelConfig(hidden=96, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=32)
        m = Transformer(cfg)
        sae = SparseAutoencoder(d=cfg.hidden, f=64)
        top = feature_top_tokens(sae, m, feature=3, k=5)
        assert len(top) == 5
        assert all(isinstance(t, str) and isinstance(p, float) for t, p in top)

    def test_train_sae_end_to_end(self, model, tiny_cfg):
        ids = torch.randint(0, tiny_cfg.vocab_size, (4, 16))
        h = collect_activations(model, ids, layer=1)
        sae = SparseAutoencoder(d=tiny_cfg.hidden, f=128, k=4)
        batches = iter([h[0:16], h[16:32], h[32:48], h[48:64]] * 100)
        train_sae(sae, batches, steps=200, lr=1e-3, log_every=500)
        recon, acts = sae(h[:16])
        assert ((recon - h[:16]) ** 2).mean().item() < 1.0


class TestVLM:
    def setup_method(self):
        cfg = ModelConfig(hidden=96, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64)
        self.lm = Transformer(cfg)
        self.cfg = cfg

    def test_forward_shapes_and_loss(self):
        vlm = VLM(self.lm, vision_dim=128)
        patches = torch.randn(2, 16, 128)          # [B, P, v_dim]
        x, y = caption_batch(get_tokenizer(), ["a cat", "a dog"], seq_len=16)
        logits, loss = vlm(patches, x, y)
        assert logits.shape[0] == 2
        assert logits.shape[1] == 16 + x.shape[1]  # P + T
        assert torch.isfinite(loss)

    def test_loss_only_on_text(self):
        """Vision prefix tokens must not contribute to the LM loss (-100 targets).
        Verified against a manual cross-entropy over text positions only."""
        vlm = VLM(self.lm, vision_dim=128)
        enc = get_tokenizer()
        patches = torch.randn(2, 16, 128)
        x, y = caption_batch(enc, ["hello world", "hello world"], seq_len=16)
        logits, loss = vlm(patches, x, y)
        # manual: pad targets with -100 for the prefix, CE
        P = 16
        tgt = torch.cat([y.new_full((2, P), -100), y], dim=1)
        manual = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        assert loss.item() == pytest.approx(manual.item(), abs=1e-5)
        # sanity: vision content DOES influence text predictions (attention)
        _, loss2 = vlm(torch.zeros_like(patches), x, y)
        assert loss2.item() != pytest.approx(loss.item(), abs=1e-3)

    def test_training_improves_captioning(self):
        vlm = VLM(self.lm, vision_dim=32)
        tower = RandomVisionTower(vision_dim=32)
        for p in tower.parameters():
            p.requires_grad_(False)  # frozen tower: output must be detached
        enc = get_tokenizer()
        # fixed image, fixed caption — overfit sanity
        img = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            patches = tower(img)
        x, y = caption_batch(enc, ["the quick brown fox"], seq_len=16)
        opt = torch.optim.AdamW(vlm.parameters(), lr=3e-3)
        first = None
        for _ in range(150):
            _, loss = vlm(patches, x, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            first = first if first is not None else loss.item()
        assert loss.item() < first * 0.5

    def test_vision_tower_shapes(self):
        tower = RandomVisionTower(vision_dim=128)
        out = tower(torch.randn(2, 3, 64, 64))
        assert out.shape == (2, 16, 128)

    def test_prefix_embeds_without_targets(self):
        """Generation mode: prefix + tokens, no targets -> no loss, correct shape."""
        vlm = VLM(self.lm, vision_dim=32)
        patches = torch.randn(1, 4, 32)
        x, _ = caption_batch(get_tokenizer(), ["hi"], seq_len=8)
        logits, loss = vlm(patches, x[:, :2])
        assert loss is None
        assert logits.shape[1] == 4 + 2
