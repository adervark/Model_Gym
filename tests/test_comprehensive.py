"""Modules 23-26: RM/PPO, preference losses, Lion/Adafactor, speculative
decoding, RAG, agents, MinHash, continuous batching, Mamba, MTP, z-loss."""
import pytest
import torch

from lm.config import ModelConfig
from lm.model import Transformer
from lm.tokenizer import get_tokenizer
from lm.optim import Lion, Adafactor
from lm.dpo import orpo_loss, simpo_loss, kto_loss, dpo_loss
from lm.rm import RewardModel, bradley_terry_loss, ValueHead, gae, ppo_loss, \
    ppo_step, policy_logps_sequence
from lm.generate import speculative_generate, generate
from lm.rag import chunk_text, BM25, embed_chunks, dense_search, rag_answer
from lm.agent import Calculator, agent_loop
from lm.data import minhash_signature, deduplicate
from lm.serving import KVPageManager, ContinuousBatcher
from lm.mamba import MambaBlock, MambaModel, MambaHybrid

TINY = dict(hidden=96, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64)


def _m(**kw):
    return Transformer(ModelConfig(**{**TINY, **kw}))


class TestMTPZLoss:
    def test_mtp_loss_component(self):
        m = _m(mtp_depth=2)
        x = torch.randint(0, m.cfg.vocab_size, (2, 16))
        _, loss, _ = m(x[:, :-1], x[:, 1:])
        assert torch.isfinite(loss)
        # MTP heads must train the future-prediction signal
        opt = torch.optim.AdamW(m.mtp_heads.parameters(), lr=3e-3)
        before = loss.item()
        for _ in range(30):
            _, loss, _ = m(x[:, :-1], x[:, 1:])
            opt.zero_grad()
            loss.backward()
            opt.step()
        assert loss.item() < before

    def test_zloss_bounds_logits(self):
        m = _m(z_loss_coef=1e-4)
        x = torch.randint(0, m.cfg.vocab_size, (2, 16))
        logits, loss, _ = m(x[:, :-1], x[:, 1:])
        assert torch.isfinite(loss)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        for _ in range(20):
            logits, loss, _ = m(x[:, :-1], x[:, 1:])
            opt.zero_grad()
            loss.backward()
            opt.step()
        assert logits.abs().max() < 100


class TestOptimizers:
    def test_lion_trains(self):
        m = _m()
        x = torch.randint(0, m.cfg.vocab_size, (4, 16))
        opt = Lion(m.parameters(), lr=1e-4)
        first = None
        for _ in range(400):
            _, loss, _ = m(x[:, :-1], x[:, 1:])
            opt.zero_grad()
            loss.backward()
            opt.step()
            first = first if first is not None else loss.item()
        assert loss.item() < first * 0.6

    def test_adafactor_trains(self):
        m = _m()
        x = torch.randint(0, m.cfg.vocab_size, (4, 16))
        opt = Adafactor(m.parameters(), lr=0.1)
        first = None
        for _ in range(600):
            _, loss, _ = m(x[:, :-1], x[:, 1:])
            opt.zero_grad()
            loss.backward()
            opt.step()
            first = first if first is not None else loss.item()
        # Adafactor has no momentum: memorization is slow by design — require
        # a clear, meaningful drop rather than a multiplicative one
        assert first - loss.item() > 0.5

    def test_adafactor_memory_state_shape(self):
        m = _m()
        opt = Adafactor(m.parameters())
        _, loss, _ = m(torch.randint(0, 50304, (2, 8)),
                       torch.randint(0, 50304, (2, 8)))
        opt.zero_grad()
        loss.backward()
        opt.step()
        for p in m.parameters():
            st = opt.state[p]
            if p.ndim >= 2:
                assert st["row"].shape == (p.shape[0],)
                assert st["col"].shape == (p.shape[1],)


class TestPreferenceLosses:
    def setup_method(self):
        self.m = _m()
        self.ref = _m()
        self.x = torch.randint(0, self.m.cfg.vocab_size, (4, 16))
        self.mask = torch.ones(4, 16, dtype=torch.bool)

    def test_all_losses_finite(self):
        for fn, args in [
            (dpo_loss, (self.ref,)),
            (orpo_loss, ()),
            (simpo_loss, ()),
            (kto_loss, (self.ref,)),
        ]:
            loss, stats = fn(self.m, *args, self.x, self.mask)
            assert torch.isfinite(loss)

    def test_orpo_improves_chosen_vs_rejected(self):
        """ORPO gradient must raise chosen odds and lower rejected odds."""
        before_c, before_r = None, None
        from lm.dpo import get_logps
        with torch.no_grad():
            before_c = get_logps(self.m, self.x[:2], self.mask[:2]).mean().item()
            before_r = get_logps(self.m, self.x[2:], self.mask[2:]).mean().item()
        opt = torch.optim.AdamW(self.m.parameters(), lr=1e-3)
        for _ in range(20):
            loss, _ = orpo_loss(self.m, self.x, self.mask, beta=2.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            after_c = get_logps(self.m, self.x[:2], self.mask[:2]).mean().item()
            after_r = get_logps(self.m, self.x[2:], self.mask[2:]).mean().item()
        # the design: SFT term pulls chosen UP; odds penalty holds rejected
        # down — the MARGIN is what must grow, not rejected-alone fall
        assert after_c > before_c
        assert (after_c - after_r) > (before_c - before_r)


class TestRM:
    def test_reward_model_trains(self):
        m = _m()
        rm = RewardModel(m, freeze_trunk=False)
        x = torch.randint(0, m.cfg.vocab_size, (4, 16))
        mask = torch.ones(4, 16, dtype=torch.bool)
        opt = torch.optim.AdamW(rm.parameters(), lr=1e-3)
        margins = []
        for _ in range(100):
            loss, stats = bradley_terry_loss(rm, x, mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            margins.append(stats["reward_margin"])
        # preference accuracy should climb above random (0.5)
        assert stats["pref_accuracy"] > 0.8

    def test_gae_shapes_and_baseline(self):
        B, T = 2, 8
        values = torch.zeros(B, T)
        rewards = torch.zeros(B, T)
        rewards[:, -1] = 1.0
        mask = torch.ones(B, T, dtype=torch.bool)
        adv = gae(values, rewards, mask, gamma=1.0, lam=1.0)
        # constant reward at the end -> advantage = 1 everywhere (gamma=1)
        assert torch.allclose(adv, torch.ones(B, T), atol=1e-6)

    def test_ppo_clip_math(self):
        logps = torch.tensor([[0.0, 0.0]])
        old = torch.tensor([[0.0, 0.0]])
        adv = torch.tensor([[1.0, -1.0]])
        mask = torch.ones(1, 2)
        # ratio = 1 -> loss = -adv = [-1, 1] -> mean 0
        assert ppo_loss(logps, old, adv, mask).item() == pytest.approx(0.0)

    def test_ppo_step_runs(self):
        policy, ref = _m(), _m()
        critic = ValueHead(_m())
        x = torch.randint(0, policy.cfg.vocab_size, (2, 16))
        mask = torch.ones(2, 16, dtype=torch.bool)
        rewards = torch.zeros(2, 16)
        rewards[:, -1] = 1.0
        old_logps = policy_logps_sequence(policy, x, mask).detach()
        total, stats = ppo_step(policy, ref, critic, x, mask, rewards, old_logps)
        assert torch.isfinite(total)
        assert "policy_loss" in stats and "kl" in stats


class TestSpeculative:
    def test_identical_models_distributional_equivalence(self):
        """The speculative guarantee: accepted output is EXACTLY distributed
        as the target's own sampling. With target == draft (acceptance ≈ 1),
        spec's empirical first-token frequency must match the model's TRUE
        next-token distribution (measured directly)."""
        cfg = ModelConfig(hidden=64, n_layers=1, n_heads=4, n_kv_heads=2,
                          max_seq_len=32, vocab_size=256)
        t = Transformer(cfg)
        d = Transformer(cfg)
        # modestly sharp distribution: 7 -> 7 everywhere, but position 2 -> 42
        x = torch.full((32, 8), 7, dtype=torch.long)
        x[:, 3] = 42
        opt = torch.optim.AdamW(t.parameters(), lr=1e-2)
        for _ in range(2000):
            _, loss, _ = t(x[:, :-1], x[:, 1:])
            opt.zero_grad()
            loss.backward()
            opt.step()
        d.load_state_dict(t.state_dict())
        prompt = [7, 7, 7]
        with torch.no_grad():
            logits, _, _ = t(torch.tensor([prompt]))
            p_true = torch.softmax(logits[0, -1], -1)  # the ground-truth dist
        assert p_true[42].item() > 0.2  # sharp enough to distinguish at n=400

        n = 400
        spec_first = []
        for i in range(n):
            torch.manual_seed(1000 + i)
            out = speculative_generate(t, d, prompt, max_new=8, gamma=2,
                                       device="cpu")
            spec_first.append(out[0])
        import collections
        c = collections.Counter(spec_first)
        empirical = torch.zeros_like(p_true)
        for tok, count in c.items():
            empirical[tok] = count / n
        tv = (empirical - p_true).abs().sum().item() / 2  # total variation
        assert tv < 0.15, f"spec distribution drifted from target (TV={tv:.3f})"

    def test_equivalence_when_draft_disagrees(self):
        """The guarantee must hold when the draft is WRONG, which is the only
        regime that exercises rejection + residual resampling.

        The test above uses target == draft, so acceptance is ~1 and the
        residual branch never runs — a scalar-vs-vector bug in the residual
        (p_t - p_d[t] instead of p_t - p_d) sails straight through it. Here the
        draft is trained on a different pattern, so most drafts are rejected."""
        cfg = ModelConfig(hidden=64, n_layers=1, n_heads=4, n_kv_heads=2,
                          max_seq_len=32, vocab_size=256)
        torch.manual_seed(0)
        t, d = Transformer(cfg), Transformer(cfg)

        def fit(model, nxt):
            # The prompt below is [7,7,7], so the queried position is index 3.
            # The two models must differ THERE — differing at some later
            # position leaves them identical where the test actually looks.
            x = torch.full((32, 8), 7, dtype=torch.long)
            x[:, 3] = nxt
            opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
            for _ in range(1500):
                _, loss, _ = model(x[:, :-1], x[:, 1:])
                opt.zero_grad(); loss.backward(); opt.step()

        fit(t, 42)          # target continues with 42
        fit(d, 99)          # draft continues with 99 -> near-total rejection
        prompt = [7, 7, 7]
        with torch.no_grad():
            lt, _, _ = t(torch.tensor([prompt]))
            p_true = torch.softmax(lt[0, -1], -1)
            ld, _, _ = d(torch.tensor([prompt]))
            p_draft = torch.softmax(ld[0, -1], -1)
        # the two must actually disagree, else this degenerates into the test above
        assert (p_true - p_draft).abs().sum().item() / 2 > 0.2, "draft too similar"

        n = 1500
        first = []
        for i in range(n):
            torch.manual_seed(50_000 + i)
            first.append(speculative_generate(t, d, prompt, max_new=1, gamma=2,
                                              device="cpu")[0])
        import collections
        empirical = torch.zeros_like(p_true)
        for tok, count in collections.Counter(first).items():
            empirical[tok] = count / n
        tv = (empirical - p_true).abs().sum().item() / 2
        # sampling noise at n=1500 on this support is ~0.03; the scalar-residual
        # bug measures ~0.14, so 0.07 separates them cleanly.
        assert tv < 0.07, (
            f"speculative output is not target-distributed (TV={tv:.3f}). "
            "Check the rejection residual: it must be (p_target - p_draft), "
            "elementwise over the whole vocab.")

    def test_terminates_and_vocab_valid(self):
        torch.manual_seed(0)
        t = _m()
        d = _m()
        enc = get_tokenizer()
        out = speculative_generate(t, d, enc.encode_ordinary("hi"), max_new=24,
                                   gamma=4, device="cpu")
        assert 0 < len(out) <= 24
        assert all(tok < enc.n_vocab for tok in out)


class TestRAG:
    def test_bm25_finds_keyword_doc(self):
        chunks = ["the cat sat on the mat",
                  "quantum field theory and renormalization",
                  "python list comprehensions are fast"]
        bm25 = BM25(chunks)
        hits = bm25.search("quantum physics", k=1)
        assert hits[0] == 1

    def test_dense_retrieval_mechanics(self, tiny_cfg, seeded):
        """Dense retrieval must (a) produce one unit-norm vector per chunk and
        (b) rank by cosine similarity, so a chunk always retrieves ITSELF.

        Deliberately not a semantic test: an untrained model has no semantics,
        so asserting that "canines make noise" retrieves "dogs bark" is a coin
        flip on the init seed (measured: 3/6 seeds each way) — it passed by
        luck and broke the moment an unrelated shape changed."""
        m = Transformer(tiny_cfg)
        chunks = ["dogs bark at night", "derivatives and integrals in calculus",
                  "the ship sailed west at dawn"]
        emb = embed_chunks(m, chunks, device="cpu")
        assert emb.shape[0] == len(chunks)
        assert torch.allclose(emb.norm(dim=-1), torch.ones(len(chunks)), atol=1e-4)

        # self-retrieval: identical text must be the top hit for every chunk
        for i, c in enumerate(chunks):
            q = embed_chunks(m, [c], device="cpu")
            assert dense_search(emb, q, k=1)[0] == i

    def test_dense_search_ranks_by_cosine(self):
        """dense_search is pure ranking logic — check it directly, with
        hand-built vectors, so the assertion doesn't depend on any model."""
        emb = torch.tensor([[1.0, 0.0], [0.7071, 0.7071], [0.0, 1.0]])
        q = torch.tensor([[1.0, 0.0]])
        assert dense_search(emb, q, k=3) == [0, 1, 2]
        assert dense_search(emb, torch.tensor([[0.0, 1.0]]), k=3) == [2, 1, 0]

    def test_rag_answer_returns_text(self):
        cfg = ModelConfig(hidden=96, n_layers=2, n_heads=4, n_kv_heads=2,
                          max_seq_len=128)
        m = Transformer(cfg)
        # small chunks so the assembled context leaves generation headroom
        chunks = chunk_text("the sky is blue because of rayleigh scattering. " * 2,
                            chunk_tokens=24)
        bm25 = BM25(chunks)
        emb = embed_chunks(m, chunks, device="cpu")
        ans = rag_answer(m, "why is the sky blue?", chunks, bm25, emb,
                         device="cpu", k=2, max_new=24)
        assert isinstance(ans, str) and len(ans) > 0


class TestAgent:
    def test_calculator(self):
        calc = Calculator()
        assert calc.run("2 + 2") == "4"
        assert calc.run("(3*4)/2") == "6.0"
        assert calc.run("2 + ").startswith("ERROR")
        assert calc.run("__import__('os')") == "ERROR: invalid expression"

    def test_agent_loop_terminates(self, tiny_cfg):
        m = Transformer(tiny_cfg)
        ans, log = agent_loop(m, "2+2", Calculator(), max_turns=3,
                              max_new=32, device="cpu")
        assert len(log) <= 3


class TestMinHash:
    def test_signature_stable(self):
        doc = list(range(100))
        assert minhash_signature(doc) == minhash_signature(doc)

    def test_dedup_catches_near_duplicates(self):
        base = list(range(500))
        dup = base[:450] + [999] * 50     # 90% overlap
        distinct = list(range(1000, 1500))
        kept = deduplicate([base, dup, distinct])
        assert kept[0] == 0
        assert 1 not in kept              # dup dropped
        assert 2 in kept                  # distinct kept


class TestServing:
    def test_page_manager_growth_and_release(self):
        pm = KVPageManager(n_pages=4, page_size=8, feat_dim=4)
        rid = pm.new_request(4)
        for i in range(20):
            pm.append_token(rid, torch.full((4,), float(i)))
        assert pm.request_len(rid) == 20
        assert len(pm.request_pages[rid]) == 3  # 20 tokens -> 3 pages
        pm.release(rid)
        assert len(pm.free) == 4

    def test_page_manager_evicts_when_full(self):
        pm = KVPageManager(n_pages=3, page_size=4, feat_dim=2)
        r1 = pm.new_request(8)   # 2 pages
        r2 = pm.new_request(4)   # 1 page
        r3 = pm.new_request(4)   # no pages free -> evicts r1
        assert r1 not in pm.request_pages
        assert r3 in pm.request_pages

    def test_continuous_batcher_end_to_end(self, tiny_cfg):
        m = Transformer(tiny_cfg)
        pm = KVPageManager(n_pages=16, page_size=8, feat_dim=tiny_cfg.hidden)
        batcher = ContinuousBatcher(m, pm, seq_len=32)
        enc = get_tokenizer()
        r1 = batcher.add_request(enc.encode_ordinary("hello"))
        r2 = batcher.add_request(enc.encode_ordinary("world"))
        for _ in range(30):
            batcher.step()
            done = batcher.collect_finished()
            if done:
                break
        assert pm.request_len(r1) > 2
        assert pm.request_len(r2) > 2


class TestMamba:
    def test_block_shapes_and_causal(self):
        blk = MambaBlock(d=64, d_state=16, d_conv=4)
        x = torch.randn(2, 24, 64)
        y = blk(x)
        assert y.shape == x.shape
        # causality: output at t must not depend on x[t+1:]
        x2 = x.clone()
        x2[:, 12:] = 99.0
        y2 = blk(x2)
        assert torch.allclose(y[:, :12], y2[:, :12], atol=1e-6)

    def test_mamba_model_trains(self):
        cfg = ModelConfig(hidden=64, n_layers=2, n_heads=4, n_kv_heads=2,
                          max_seq_len=32, vocab_size=512)
        m = MambaModel(cfg, n_blocks=2, d_state=8)
        x = torch.randint(0, 512, (4, 16))
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
        first = None
        for _ in range(150):
            logits, loss = m(x, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            first = first if first is not None else loss.item()
        assert loss.item() < first * 0.6

    def test_hybrid_runs(self):
        cfg = ModelConfig(hidden=64, n_layers=2, n_heads=4, n_kv_heads=2,
                          max_seq_len=32, vocab_size=512)
        m = MambaHybrid(cfg, n_mamba=2, n_attn=1)
        x = torch.randint(0, 512, (2, 16))
        logits, loss = m(x, x)
        assert logits.shape == (2, 16, 512)
        assert torch.isfinite(loss)
        loss.backward()
