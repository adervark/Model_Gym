"""LoRA and quantization."""
import torch

from lm.config import ModelConfig
from lm.model import Transformer
from lm.lora import (apply_lora, lora_trainable_params, merge_lora,
                     count_lora_params, LoraLinear)
from lm.quant import (quantize_int8, quantize_linear_int8, Int8Linear,
                      NF4, gptq_layer, gptq_model, nf4_size_reduction)


class TestLoRA:
    def test_base_frozen_adapters_only(self, tiny_cfg):
        m = Transformer(tiny_cfg)
        apply_lora(m, r=4, alpha=8)
        trainable, frozen = count_lora_params(m)
        assert trainable > 0
        # every trainable param must be an adapter (A or B)
        for name, p in m.named_parameters():
            if p.requires_grad:
                assert name.split(".")[-1] in ("A", "B")
        assert frozen == sum(p.numel() for p in Transformer(tiny_cfg).parameters())
        # adapters are a tiny fraction
        assert trainable / frozen < 0.05

    def test_identity_at_init(self, tiny_cfg, seeded):
        """B=0 init -> logits identical before/after applying LoRA."""
        torch.manual_seed(0)
        base = Transformer(tiny_cfg)
        x = torch.randint(0, tiny_cfg.vocab_size, (2, 16))
        logits_before, _, _ = base(x)
        apply_lora(base, r=4, alpha=8)
        logits_after, _, _ = base(x)
        assert torch.allclose(logits_before, logits_after, atol=1e-6)

    def test_lora_trainable_params_excludes_frozen(self, tiny_cfg):
        m = Transformer(tiny_cfg)
        apply_lora(m, r=4, alpha=8)
        params = lora_trainable_params(m)
        names = set()
        for n, p in m.named_parameters():
            if any(p is q for q in params):
                names.add(n.split(".")[-1])
        assert names <= {"A", "B"}

    def test_merge_equivalence(self, tiny_cfg, seeded):
        """After training, merged model logits == adapter model logits."""
        torch.manual_seed(7)
        m = Transformer(tiny_cfg)
        apply_lora(m, r=4, alpha=8)
        opt = torch.optim.AdamW(lora_trainable_params(m), lr=1e-2)
        x = torch.randint(0, tiny_cfg.vocab_size, (2, 16))
        for _ in range(30):
            _, loss, _ = m(x, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            _, _, _ = m(x)  # ensure caches built
            logits_adapter, _, _ = m(x)
        merge_lora(m)
        assert not any(isinstance(mod, LoraLinear) for mod in m.modules())
        logits_merged, _, _ = m(x)
        assert torch.allclose(logits_adapter, logits_merged, atol=1e-4)


class TestQuant:
    def test_int8_roundtrip_error(self):
        w = torch.randn(512, 512)
        w_q, scale = quantize_int8(w)
        w_dq = w_q.float() * scale
        rel = (w - w_dq).abs().mean() / w.abs().mean()
        # per-TENSOR int8 on N(0,1) weights: ~2-3% mean relative error.
        # (Production int8 is per-channel, hence the small error people quote.)
        assert rel < 0.05

    def test_int8_replace_and_forward(self, tiny_cfg, seeded):
        m = Transformer(tiny_cfg)
        quantize_linear_int8(m, skip=("head",))
        x = torch.randint(0, tiny_cfg.vocab_size, (2, 16))
        logits, _, _ = m(x)
        assert torch.isfinite(logits).all()

    def test_nf4_roundtrip(self):
        w = torch.randn(4096)
        idx, mx = NF4.quantize(w)
        wq = NF4.dequantize(idx, mx, w.shape)
        assert (w - wq).abs().mean() < 0.2 * w.abs().mean()

    def test_nf4_compression_ratio(self):
        w = torch.randn(4096, 4096)
        assert nf4_size_reduction(w) > 3.0

    def test_gptq_beats_naive_rounding(self):
        """GPTQ's error correction must do at least as well as per-column rounding."""
        torch.manual_seed(3)
        W = torch.randn(64, 96)
        X = torch.randn(512, 96)
        # naive: round each column to its own 4-bit grid
        scale = W.abs().max(dim=0).values / 7.5 + 1e-8
        W_naive = (W / scale).round().clamp(-7, 7) * scale
        err_naive = ((W @ X.T) - (W_naive @ X.T)).pow(2).sum().item()
        W_gptq = gptq_layer(W, X)
        err_gptq = ((W @ X.T) - (W_gptq @ X.T)).pow(2).sum().item()
        assert err_gptq <= err_naive * 1.01

    def test_gptq_model_terminates(self, tiny_cfg):
        """Regression: calibration over an infinite dataloader must be bounded
        (this hung forever before the fix)."""
        import time
        from lm.quant import gptq_model
        m = Transformer(tiny_cfg)
        x = torch.randint(0, tiny_cfg.vocab_size, (2, 32))
        loader = iter([x, x, x, x])  # finite iterable passed as calib_loader
        t0 = time.time()
        gptq_model(m, loader)
        assert time.time() - t0 < 30
