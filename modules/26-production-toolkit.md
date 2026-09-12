# Module 26 — The production toolkit: everything else a comprehensive course includes

**Goal:** the remaining production-grade techniques that didn't warrant a
module each but appear in every serious training/serving stack: two more
optimizers, two training auxiliaries, dedup, and continuous batching.

## 1. The optimizer zoo is now complete

You've met AdamW (module 06), Muon (module 19). Add:

- **Lion** (`lm/optim.py`): sign-based — `update = sign(b1·m + (1−b1)·g)`
  with decoupled decay. One state instead of AdamW's two → half the
  optimizer memory; slightly worse convergence, often better generalization
  (the "sharpness" argument). lr ~3-10x SMALLER than AdamW.
- **Adafactor** (`lm/optim.py`): PaLM's optimizer — *factored* second
  moments: row/col statistics instead of a full matrix → O(1) extra memory
  per tensor. No momentum (by default), RMS-normalized updates with relative
  step sizing. The answer when 8 bytes/param (AdamW's state) doesn't fit.

Memory per param: AdamW 8B, Lion 4B, Muon 4B, Adafactor ~0B. That table is
the whole decision.

## 2. Two training auxiliaries (both config-gated in `model.py`)

- **z-loss** (`z_loss_coef`): PaLM's stability trick —
  `L_z = mean(logsumexp(logits)²)`. Penalizes the logit *scale* drifting —
  cheap insurance against logit explosion at scale.
- **MTP — Multi-Token Prediction** (`mtp_depth`): DeepSeek-V3's heads that
  predict tokens 2, 3, ... k ahead from the same hidden states. Auxiliary
  loss that densifies the training signal (each position teaches multiple
  predictions); V3 reports better data efficiency + free speculative-decode
  draft heads (module 12's speculative decoding with no separate draft
  model!). The full V3 version adds a small transformer block per depth;
  the course's linear heads teach the mechanism.

## 3. MinHash dedup (`lm/data.py`)

Module 05's dedup exercise, made real: hash each doc's k-shingles, keep the
smallest m as a signature; signature overlap ≈ Jaccard similarity (Broder).
`deduplicate` drops near-duplicates at any threshold you choose. Run it on
your next scrape before training — dedup is the cheapest data-quality win
there is.

## 4. Continuous batching + paged KV (`lm/serving.py`)

Module 12's serving exercise, made real, at the scheduler level:

- **KVPageManager**: the KV cache in fixed-size pages with an allocation
  table per request and LRU eviction — PagedAttention's core idea. The
  arena grows per request *on demand*; no pre-allocation waste (the 60-80%
  fragmentation naive batching suffers).
- **ContinuousBatcher**: every decode step is ONE batched forward over all
  live requests; finished requests leave, new ones join mid-flight. This
  batching is the 10-30x throughput multiplier between naive serving and
  vLLM-class serving.

The honest boundary: this teaches the *scheduling and paging mechanics*; the
fused kernels live in vLLM (module 12 §3). Knowing both sides of that line
is the professional level.

## Exercises

1. **The full optimizer sweep.** One model, one budget: AdamW vs Lion vs
   Adafactor vs Muon — loss vs wall-clock AND peak memory. Write the
   recommendation table a lab would actually use.
2. **MTP = free speculation.** Train with mtp_depth=2. Then use the MTP
   heads as the *draft* in module 12's speculative decoding — no separate
   draft model needed. Measure the speedup. (This is a genuine 2025
   frontier trick: Medusa/V3-style self-speculation.)
3. **z-loss under stress.** Train with fp16-style numerics (large lr, no
   clipping) with and without z-loss. Find the setting where z-loss saves
   the run. When does it not matter?
4. **Serving under load.** Drive the ContinuousBatcher with 64 staggered
   requests (random arrival times). Measure: throughput vs one-at-a-time,
   and page-eviction behavior when the arena is too small. Then tune
   arena size until evictions stop.

**Papers:**
- Chen et al., *Lion* (arXiv:2302.06675) · Shazeer & Stern, *Adafactor*
  (arXiv:1804.04235)
- Chowdhery et al., *PaLM* (arXiv:2204.02311) — z-loss
- Gloeckle et al., *Better & Faster LLMs via Multi-token Prediction*
  (arXiv:2404.19737) · DeepSeek-V3 (arXiv:2412.19437) — MTP
- Kwon et al., *PagedAttention* (arXiv:2309.06180) · Yu et al., *Orca*
  (arXiv:2306.02707) — continuous batching
