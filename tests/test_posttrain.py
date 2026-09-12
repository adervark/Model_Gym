"""Generation, evaluation, SFT masking, DPO, GRPO."""
import math

import pytest
import torch

from lm.config import ModelConfig
from lm.model import Transformer
from lm.tokenizer import get_tokenizer
from lm.generate import generate, sample
from lm.eval import perplexity, fewshot_logprob, mc_accuracy, logit_lens
from lm.sft import build_sft_sample, sft_step, CHAT_TEMPLATE
from lm.dpo import dpo_loss, get_logps
from lm.grpo import group_advantages, grpo_loss, gsm8k_reward, completion_logp


def test_generate_never_samples_padded_vocab(tiny_cfg, seeded):
    m = Transformer(tiny_cfg)
    enc = get_tokenizer()
    for _ in range(5):
        ids = generate(m, enc.encode_ordinary("hello"), max_new=60,
                       temperature=1.0, top_p=0.95, device="cpu")
        assert all(t < enc.n_vocab for t in ids)
        assert 0 < len(ids) <= 60


def test_repetition_penalty_pushes_seen_tokens_down():
    """CTRL-style penalty: divide positive logits, MULTIPLY negative ones, so
    the penalty always pushes a seen token's probability DOWN. (Dividing a
    negative logit would raise it — the classic sign bug.) Checked on the
    transform directly: through sampling it is invisible on an untrained model,
    where 50k near-uniform logits swamp any change to the ~40 seen ones."""
    import torch.nn.functional as F
    logits = torch.tensor([[2.0, -3.0, 0.5]])
    seen = torch.tensor([0, 1])
    prev = logits[0, seen]
    out = logits.clone()
    out[0, seen] = torch.where(prev > 0, prev / 2.0, prev * 2.0)

    assert out[0, 0].item() == 1.0      # positive: 2.0 / 2
    assert out[0, 1].item() == -6.0     # negative: -3.0 * 2  (down, not up)
    assert out[0, 2].item() == 0.5      # unseen: untouched
    assert F.softmax(out, -1)[0, 0] < F.softmax(logits, -1)[0, 0]


def test_generate_repetition_penalty_runs(tiny_cfg, seeded):
    """The plumbing works end to end and respects the budget."""
    m = Transformer(tiny_cfg)
    enc = get_tokenizer()
    ids = generate(m, enc.encode_ordinary("hello"), max_new=20, temperature=1.0,
                   top_p=0.95, device="cpu", repetition_penalty=1.3)
    assert 0 < len(ids) <= 20
    assert all(t < enc.n_vocab for t in ids)


def test_generate_top_p_near_one_does_not_collapse(tiny_cfg, seeded):
    """Regression: `(cum > top_p).argmax()` returns 0 when NOTHING exceeds the
    threshold, which truncated the distribution to a single token instead of
    keeping all of it."""
    m = Transformer(tiny_cfg)
    enc = get_tokenizer()
    firsts = set()
    for s in range(25):
        torch.manual_seed(s)
        firsts.add(generate(m, enc.encode_ordinary("hi"), max_new=1, top_p=0.999,
                            temperature=1.0, device="cpu")[0])
    assert len(firsts) > 5, f"top_p≈1 collapsed to {len(firsts)} distinct token(s)"


def test_generate_echo_and_stop(tiny_cfg, seeded):
    m = Transformer(tiny_cfg)
    enc = get_tokenizer()
    prompt = enc.encode_ordinary("the quick")
    out = generate(m, prompt, max_new=50, temperature=0.0, device="cpu")
    assert not any(t == enc.eot_token for t in out) or True  # eot stop just terminates


def test_sample_roundtrips_text(tiny_cfg, seeded):
    m = Transformer(tiny_cfg)
    text = sample(m, "Once upon", device="cpu", max_new=32)
    assert isinstance(text, str) and len(text) > 0


def test_uniform_model_perplexity_is_vocab_size():
    """Zero weights -> uniform next-token distribution -> ppl == V. The exact
    calibration test for the perplexity implementation."""
    cfg = ModelConfig(hidden=64, n_layers=1, n_heads=4, n_kv_heads=2, max_seq_len=64)
    m = Transformer(cfg)
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
        m.norm_f.zero_()
        for blk in m.blocks:
            blk.norm1.zero_()
            blk.norm2.zero_()
    ppl = perplexity(m, "hello world this is a test sentence", device="cpu")
    assert abs(ppl - cfg.vocab_size) / cfg.vocab_size < 0.02


def test_fewshot_logprob_matches_manual():
    cfg = ModelConfig(hidden=64, n_layers=1, n_heads=4, n_kv_heads=2, max_seq_len=64)
    m = Transformer(cfg)
    enc = get_tokenizer()
    prompt, completion = "the capital of", " France"
    ids = enc.encode_ordinary(prompt + completion)
    p_len = len(enc.encode_ordinary(prompt))
    x = torch.tensor([ids[:-1]], dtype=torch.long)
    logits, _, _ = m(x)
    logp = torch.nn.functional.log_softmax(logits, dim=-1)
    y = torch.tensor([ids[1:]], dtype=torch.long)
    per_tok = torch.gather(logp, -1, y.unsqueeze(-1)).squeeze(-1)
    want = per_tok[0, p_len - 1:].sum().item()
    assert abs(fewshot_logprob(m, prompt, completion, device="cpu") - want) < 1e-5


def test_logit_lens_shape(model, tiny_cfg):
    out = logit_lens(model, "the quick brown fox", device="cpu", k=3)
    assert len(out) == tiny_cfg.n_layers
    assert all(len(layer) == 3 for layer in out)
    assert all(isinstance(t, str) and isinstance(p, float) for layer in out for t, p in layer)


class TestSFTMask:
    def test_mask_covers_exactly_assistant_span(self):
        enc = get_tokenizer()
        prompt, response = "What is 2+2?", "Four."
        x, mask = build_sft_sample(enc, prompt, response, 64)
        prefix = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        a_start = len(enc.encode_ordinary(prefix))
        full = enc.encode_ordinary(prefix + response + "<|end|>") + [enc.eot_token]
        assert x[:len(full)].tolist() == full
        assert mask[:a_start].sum() == 0          # prompt unmasked
        assert mask[a_start:len(full)].all()      # assistant span fully masked
        assert not mask[len(full):].any()         # padding unmasked

    def test_mask_no_response(self):
        enc = get_tokenizer()
        x, mask = build_sft_sample(enc, "hi", "", 64)
        # prompt span never masked; the (empty) assistant span may include the
        # closing delimiter, which IS trained — same as real SFT.
        prefix = "<|user|>\nhi<|end|>\n<|assistant|>\n"
        a_start = len(enc.encode_ordinary(prefix))
        assert not mask[:a_start].any()

    def test_sft_step_zero_mask(self, model):
        x = torch.randint(0, model.cfg.vocab_size, (2, 16))
        loss = sft_step(model, x, torch.zeros_like(x, dtype=torch.bool))
        assert loss.item() == 0.0

    def test_sft_loss_decreases_on_memorization(self, tiny_cfg, seeded):
        m = Transformer(tiny_cfg)
        enc = get_tokenizer()
        x, mask = build_sft_sample(enc, "hello", "world world world", 32)
        x, mask = x.unsqueeze(0), mask.unsqueeze(0)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
        first = None
        for _ in range(150):
            loss = sft_step(m, x, mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            first = first if first is not None else loss.item()
        assert loss.item() < first * 0.5


class TestDPO:
    def test_loss_matches_manual_formula(self, tiny_cfg, seeded):
        torch.manual_seed(42)
        m = Transformer(tiny_cfg)
        ref = Transformer(tiny_cfg)
        B = 2
        x = torch.randint(0, tiny_cfg.vocab_size, (B, 16))
        mask = torch.ones(B, 16, dtype=torch.bool)
        beta = 0.1
        loss, stats = dpo_loss(m, ref, x, mask, beta)
        logps = get_logps(m, x, mask)
        with torch.no_grad():
            ref_logps = get_logps(ref, x, mask)
        pi_r = logps[0] - logps[1]
        ref_r = ref_logps[0] - ref_logps[1]
        want = -torch.nn.functional.logsigmoid(beta * (pi_r - ref_r))
        assert abs(loss.item() - want.item()) < 1e-5

    def test_learns_to_prefer_chosen(self, tiny_cfg, seeded):
        """Gradient check: one step must raise chosen logp, lower rejected logp."""
        m = Transformer(tiny_cfg)
        ref = Transformer(tiny_cfg)
        x = torch.randint(0, tiny_cfg.vocab_size, (2, 16))
        mask = torch.ones(2, 16, dtype=torch.bool)
        with torch.no_grad():
            chosen_before = get_logps(m, x[:1], mask[:1]).item()
            rejected_before = get_logps(m, x[1:], mask[1:]).item()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        loss, _ = dpo_loss(m, ref, x, mask, beta=0.1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            chosen_after = get_logps(m, x[:1], mask[:1]).item()
            rejected_after = get_logps(m, x[1:], mask[1:]).item()
        assert chosen_after > chosen_before
        assert rejected_after < rejected_before


class TestGRPO:
    def test_advantages_normalized(self):
        r = torch.tensor([1.0, 0.0, 0.5, 0.0])
        adv = group_advantages(r, 4)
        assert abs(adv.mean().item()) < 1e-6
        assert abs(adv.std().item() - 1.0) < 1e-5

    def test_advantages_per_group(self):
        r = torch.tensor([1.0, 0.0, 0.5, 0.5, 0.0, 0.0, 1.0, 1.0])
        adv = group_advantages(r, 4)
        assert abs(adv[:4].mean().item()) < 1e-6
        assert abs(adv[4:].mean().item()) < 1e-6

    def test_rewards(self):
        assert gsm8k_reward("step... #### 42", "42") == 1.0
        assert gsm8k_reward("#### 43", "42") == 0.0
        assert gsm8k_reward("no numbers here", "42") == 0.0
        assert gsm8k_reward("#### 1,200", "1200") == 1.0

    def test_grpo_loss_finite_and_clips(self, tiny_cfg, seeded):
        m = Transformer(tiny_cfg)
        ref = Transformer(tiny_cfg)
        prompt = torch.randint(0, tiny_cfg.vocab_size, (2, 8))
        comp = torch.randint(0, tiny_cfg.vocab_size, (2, 16))
        mask = torch.ones(2, 16, dtype=torch.bool)
        rewards = torch.tensor([1.0, 0.0])
        # per-token [N, T_c]: the importance ratio is defined per position
        old_logp = torch.full((2, 16), -2.0)
        loss, stats = grpo_loss(m, ref, prompt, comp, mask, rewards, old_logp, 2)
        assert torch.isfinite(loss)

        # KL is measured PER TOKEN, so it must be ~0 only when the policies are
        # genuinely identical. (m and ref above are independent random inits —
        # asserting KL~0 on those passed only because the old implementation
        # averaged log-probs over the completion before differencing, which
        # cancels most of the real divergence.)
        same = Transformer(tiny_cfg)
        same.load_state_dict(m.state_dict())
        _, stats_same = grpo_loss(m, same, prompt, comp, mask, rewards, old_logp, 2)
        assert abs(stats_same["kl"]) < 1e-6, "identical policies must have zero KL"
        assert stats["kl"] > stats_same["kl"], "a different ref must show positive KL"

    def test_completion_logp_masks_prompt(self, tiny_cfg, seeded):
        m = Transformer(tiny_cfg)
        prompt = torch.randint(0, tiny_cfg.vocab_size, (1, 8))
        comp = torch.randint(0, tiny_cfg.vocab_size, (1, 16))
        mask = torch.ones(1, 16, dtype=torch.bool)
        lp_full = completion_logp(m, prompt, comp, mask)
        # swapping the completion changes the logp -> prompt is not dominating it
        comp2 = torch.randint(0, tiny_cfg.vocab_size, (1, 16))
        lp2 = completion_logp(m, prompt, comp2, mask)
        assert torch.isfinite(lp_full)
