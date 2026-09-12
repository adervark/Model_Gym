# Module 17 — MLA: Multi-head Latent Attention (DeepSeek-V2/V3/R1)

**Goal:** the attention mechanism inside the models you've been reading about —
DeepSeek-V2, V3, R1 all use it. MLA compresses the KV cache ~4-10x with a
low-rank latent. Implementation: `MLA` in `lm/model.py`, enabled with
`use_mla=True`.

## 1. The memory wall it solves

Inference cost = autoregressive decode = re-reading weights and KV cache per
token (module 12). GQA shrank the cache by sharing heads; MLA attacks the
cache itself. Cache sizes per token per layer (head_dim=64):

| Scheme | Floats/token |
|---|---|
| MHA (8 heads) | 2 · 8 · 64 = 1024 |
| GQA (8 q / 2 kv) | 2 · 2 · 64 = 256 |
| **MLA (d_c=128, d_r=32)** | **128 + 32 = 160** |

The MLA row has **no factor of n_kv in it**, and that is the entire point. You
cache one latent `c_kv` (d_c floats) and one roped key slice (d_r floats) per
position, *shared by every head* — K and V for all heads are rebuilt from them.
Multiplying d_r by n_kv (a natural-looking implementation slip, since the rope
slice is broadcast to the heads at compute time) inflates the cache and throws
away most of the win.

At V3's real scale (d_c=512, d_r=64, 128 heads at head_dim=128):

```
MHA:  2 · 128 · 128 = 32,768 floats/token/layer
MLA:       512 + 64 =    576 floats/token/layer     -> ~57x smaller
```

Long context and big batches live or die on this number.

## 2. The mechanism (three projections replace the whole KV block)

```
c_kv = W_kv(x)              d -> d_c        compressed latent (THE cache)
K_c  = W_uk(c_kv)           d_c -> n_kv·(D - d_r)
V    = W_uv(c_kv)           d_c -> n_kv·D
c_kr = W_kr(x)              d -> d_r        decoupled rope latent
K    = [K_c | rope(c_kr)]                   rope slice carries position
Q    = [Q_c | rope(Q_r)]    same split: compressed + rope slices
```

Three key ideas to internalize:

1. **K and V share one low-rank latent.** The whole KV block is ~rank d_c
   (512 vs 2·128·128 in V2) — there's nothing worth caching beyond it.
2. **RoPE is decoupled.** RoPE must apply per-position; it can't ride inside
   the compressed latent. So a *separate small* rope latent exists only to
   carry position — and it's cached too (post-rotation).
3. **Absorption (the production trick).** At inference, W_uk/W_uv are folded
   into the output projection: W_o_eff = W_o · W_uv (and the query side
   similarly), so recomputing K, V from c_kv costs *zero extra FLOPs* —
   MLA decodes at GQA speed with a fraction of the cache. Our course code
   keeps the explicit up-projections so you can see them; folding is
   exercise 3.

## 3. The cache convention that keeps everything working

All caches in `lm/model.py` are `[B, feat, T]` (features × time) so the shared
position bookkeeping (`cache[0][0].shape[2]`) works for GQA and MLA alike. For
MLA the two cached tensors are `[B, d_c, T]` and `[B, d_r, T]` — note the second
one is d_r, *not* n_kv·d_r; broadcasting to the heads happens at compute time in
`_heads()`, never in storage.
That single invariant is what the module-03 cache-equivalence tests protect —
read the `MLA.forward` cache branch and notice how K/V get *rebuilt from the
latent* at decode instead of being stored.

## 4. Run it

```bash
python -m lm.cli pretrain --out checkpoints/mla-100m --size 100m \
    --use-mla --kv-latent 128 --rope-latent 32 --steps 20000
python -m lm.cli generate --ckpt checkpoints/mla-100m/best.pt
```

Compare `cfg.kv_cache_tokens_per_position` vs the GQA equivalent — the
property `test_cache_is_smaller_than_gqa` asserts. Then push the frontier combo
from module 15:

```bash
python -m lm.cli pretrain --size 100m --use-mla --moe-experts 8 --moe-topk 2 \
    --steps 20000   # V3's architecture family, at course scale
```

## Exercises

1. **Cache byte math.** V3: 61 layers, n_kv=128, d_c=512, d_r=64, bf16.
   Compute per-token cache with GQA vs MLA at 128k context, batch 32.
   How many H100s of HBM (80GB) does each need? (This one calculation
   explains MLA's existence.)
2. **Ablate the rope latent.** Set d_r=0 and re-train briefly. What happens to
   the loss? (Hint: RoPE was the position signal — with d_r=0 the model is
   permutation-blind within the latent; watch the loss floor.)
3. **Implement absorption.** Fold W_uk into W_o (and the query-side
   equivalents) so decode never materializes K, V. Verify cache-equivalence
   tests still pass bit-for-bit. (This is the difference between the paper's
   claim and a naive implementation.)
4. **Cache equivalence under load.** Enable gradient checkpointing + MLA
   together (the V3 training stack) and re-run the test suite's cache tests.

**Papers:**
- DeepSeek-V2 (arXiv:2405.04434) — MLA
- DeepSeek-V3 (arXiv:2412.19437) — MLA at frontier scale, MTP heads
- DeepSeek-R1 (arXiv:2501.12948) — the same architecture reasoning
