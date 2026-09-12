# Module 02 — The transformer, from scratch

**Goal:** implement the GPT-2 architecture from the paper, train it on characters,
and understand why it beats recurrence and bigrams. This is the "everything is
attention" module; module 03 modernizes it.

## 1. The one idea

Language modeling = predict token `t` from all previous tokens. A transformer does
this with a stack of layers, each: **attention** (which previous tokens matter
right now?) + **MLP** (process what you gathered). No recurrence: the whole
sequence is processed in parallel, causality enforced by a mask.

The building blocks (implement each one before reading on):

### 1a. Scaled dot-product attention

```
Attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V
```

Q, K, V are your token vectors projected 3 ways. The QK^T matrix is "how much
does token i look at token j", softmax makes it a distribution, times V = gather.
The `sqrt(d)` keeps softmax out of saturation (dot products of d-dim vectors have
std sqrt(d)).

### 1b. Multi-head

One attention function learns one relation pattern ("verb → subject"). 8-16 heads
in parallel, concatenated: attention with multiple simultaneous hypotheses.
`head_dim` (64) is a hyperparameter that's held constant across model sizes.

### 1c. Causal masking

Token i may only see tokens <= i. Implement by setting the upper triangle of the
attention logits to -inf before softmax.

### 1d. Positional embeddings

Attention is permutation-invariant; tokens need positions. GPT-2: learned lookup
table added to token embeddings. (Dead in 2025 — module 03 replaces with RoPE.)

### 1e. The residual stream

```
x = x + attn(layernorm(x))    # not: x = layernorm(x + attn(x))  (post-norm, pre-2023)
x = x + mlp(layernorm(x))
```

Pre-norm (normalize *before* the sublayer) trains deeper models more stably —
Llama's choice. The residual stream is the model's communication channel: every
layer reads from it and writes back to it. Keep this mental model; it explains
90% of interpretability work (module 13).

## 2. Build it

```python
import math, torch, torch.nn as nn, torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        assert d % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.qkv = nn.Linear(d, 3 * d)          # one matrix, split later
        self.out = nn.Linear(d, d)

    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        att = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)   # [B, H, T, T]
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v                              # [B, H, T, D]
        y = y.transpose(1, 2).reshape(B, T, d)
        return self.out(y)
```

Then: `Block = pre-norm + attn + mlp` (MLP: `Linear(d, 4d) -> GELU -> Linear(4d, d)`),
then `Transformer = tok_emb + pos_emb + [Block] * L + norm + head`.
Use learned positional embeddings. Target: 6 layers / 6 heads / d=384.

Train on characters with the loop from module 01 (replace bigram with this model).
Your code, if right, gets ~1.7 loss on tiny Shakespeare vs 2.5 for bigram. That gap
is attention looking back at context.

## 3. Why this works: the computational argument (no mysticism)

- RNNs process tokens sequentially; gradients flow through T serial steps (vanish).
  Attention: every token attends to all previous in O(1) parallel layers, gradient
  path length 1. This is why transformers scale to trillions of tokens.
- MLP provides the per-token compute; attention provides routing. Neither alone
  works well. The ratio is **2:1** (MLP : attention params) and is remarkably
  stable across scales: attention is q,k,v,o = 4d², the MLP is d→4d→d = 8d².
  (Count it yourself — it is the fastest sanity check on any architecture
  diagram you are handed.)
- O(T^2) memory is the tax; module 03/12 shows how the field pays it (FlashAttention,
  KV cache, sparse attention in frontier models).

## 4. KV cache (write it now — you'll use it forever)

At generation time you recompute everything per token (O(T^2) work). Instead, cache
K, V per layer; a new token only needs Q_new @ [K_cache; K_new]. See `Attention.forward`
in `lm/model.py` — the `cache` argument is exactly this. Speedup: generation goes
from O(T^2) to O(T) per step after prefill.

## Exercises

1. **Ablate attention.** Zero out the attention output (identity residual only) and
   re-evaluate: what loss do you get? (Should be ~bigram.) Then ablate the MLP.
   Which hurts more? Explain with the routing/compute split.
2. **Gradient checkpointing.** Recompute activations in backward instead of storing
   them: memory O(L) instead of O(L * T). Implement for the attention block with
   `torch.utils.checkpoint.checkpoint`. Measure peak memory at L=12, T=1024.
   (Production models all do this or FSDP's equivalent.)
3. **Precision bug hunt.** Run your attention in fp16 without the mask: what goes
   wrong (NaN softmax — why?) What does `-inf` prevent?

**Papers:**
- Vaswani et al., *Attention Is All You Need* (arXiv:1706.03762) — the source; the
  "Transformer" block.
- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2) —
  the decoder-only scale-up, learned positional embeddings, byte-level BPE.
- Kaplan et al., *Scaling Laws for Neural Language Models* (arXiv:2001.08361) —
  preview of module 08; why GPT-2 wasn't trained bigger at the time.
