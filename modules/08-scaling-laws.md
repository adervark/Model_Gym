# Module 08 — Scaling laws & hyperparameter transfer

**Goal:** the mathematics of "how big, how much data, what lr" — so you can spend
millions of dollars of compute deliberately instead of by vibes. Two tools:
Chinchilla (budget → size/data split) and muP (hyperparameters that survive
scale).

## 1. Chinchilla: the compute budget equation

Empirical law (Hoffmann et al. 2022), fit over 400+ training runs:

```
L(N, D) = E + A/N^alpha + B/D^beta     E≈1.69, A≈406, B≈410, alpha≈0.34, beta≈0.28
```

and the *optimal allocation*: for compute budget C, train with roughly

```
N_opt ≈ 0.6 · C^0.46 · (GPT-3-equivalent constants)     D_opt ≈ 20 tokens per parameter
```

i.e. **~20 tokens/param, params ∝ sqrt(compute)**. Your 100M model: D ≈ 2B tokens.
A 7B model: ~140B tokens (~1 epoch of FineWeb). GPT-3 (175B / 300B tokens) was
wildly undertrained by this law — which is why Llama 3 8B (15T tokens) beats
GPT-3-class models.

**Where the law sits in 2025 (important):** Chinchilla was fit at ≤10B scale. The
frontier now overtrains relative to Chinchilla — Llama 3 8B used ~15T tokens
(~2000 tokens/param), because inference is cheaper than training, so smaller
models trained longer serve more economically. The law still correctly predicts
*the tradeoff curve*; the industry moved the operating point. The usable form:

- Loss is a power law in N and D; marginal gain per token decreases monotonically.
- At fixed budget, you can predict the loss of a model you haven't trained
  (fit alpha/beta on 3 small runs, extrapolate).
- Diminishing returns are brutal: 10x compute → ~2x quality improvement.

## 2. muP: why your hyperparameters don't transfer

Classical init (Glorot/He) makes activation scale grow with width; attention
logits grow as sqrt(d); each width needs its own lr. **muP (maximal update
parameterization)** rescales init and lr per-matrix by width ratio so that the
learning dynamics are *identical* across widths. The grouping is by *how a
matrix's fan-in scales with width*, which is not the same as "input vs output
layer" — getting this grouping wrong is the usual reason a muP sweep fails to
transfer:

| Class | What | Init (rel. to base) | LR (Adam) |
|---|---|---|---|
| **input** | token embeddings | unchanged | **unchanged** — fan-in is vocab, which doesn't grow with width |
| **hidden** | q, k, v, o, gate, up, down, router | × sqrt(base/w) | × base/w |
| **output** | the readout head | × (base/w) | × base/w |
| **vector** | RMSNorm gains (1-D) | unchanged | unchanged |

Two traps worth stating explicitly: the **embedding lr must NOT be scaled**
(it's an O(1) quantity), and **every** hidden matrix must be — scaling only the
output projections (o, down) while leaving q/k/v/gate/up alone produces curves
that look muP-ish and do not transfer. muP also replaces attention's
1/sqrt(head_dim) with 1/head_dim; this course holds head_dim fixed at 64 and
grows width via `n_heads`, so that term is a width-independent constant here and
is deliberately omitted (see `lm/train.py::mup_scale`).

Then: tune lr once on a 10M model at base width 256, reuse it at 1B. This is how
real labs tune (GPT-4-scale recipes are found on small proxies; PaLM-2, Gemini
used muP). The math that makes it work is the same "coordinate check" as residual
init in module 03: keep every tensor's size O(1) as width grows.

Our implementation: `lm/train.py::mup_param_class` (the four-way grouping
above), `mup_scale` (post-init rescale) and `make_optimizer` (per-group lr).
Verify it worked:

```bash
python -m lm.cli pretrain --size tiny --mup --steps 100 --lr 0.02   # tuned on small
# lr 0.02 transfers to:
python -m lm.cli pretrain --size 100m --mup --steps 5000 --lr 0.02
```

Check invariance yourself with the module-03 init logic: run a width sweep and
watch activation/gradient norms stay flat (exercise 1). The shape you're looking
for — residual-stream RMS after a few steps, base width 256, all else equal:

| width | no-muP | muP |
|---|---|---|
| 128 | 13.6 | 161 |
| 256 | 79.0 | 79.0 |
| 512 | 755.6 | 137 |
| 1024 | 6408.9 | 166 |

Without muP the scale explodes ~470x across this range, which is exactly why a
learning rate tuned at one width is wrong at another. With muP it stays inside a
~2x band (it is a coordinate check, not an identity — expect noise from finite
depth and step count). Width 256 matches by construction: it's the base, so
every scale factor is 1. **If your muP curve is not visibly flatter than the
no-muP one, your parameter grouping is wrong** — re-read the table above before
blaming the method.

## 3. The full hyperparameter transfer stack (2025 practice)

| Knob | Rule |
|---|---|
| width, depth | scale width primarily (Llama 3 8B→70B: width-heavy) |
| batch size | tokens/step fixed ~0.5M-4M, grows slowly with scale |
| lr | muP transfer, or keep ~3e-4 across sizes (modern default) |
| wd 0.1, beta 0.9/0.95, clip 1.0 | fixed across ALL sizes — don't touch |
| schedule | WSD (module 06) so runs are extendable |
| data mix | the real frontier hyperparameter — curriculum, ratios |

The unsolved part in 2025: **data scaling laws** — how quality, mix, and
repetition epochs interact. (Read DeepSeek-V3's training section for the current
state of the art in these decisions.)

## 4. Memory-frugal optimizers (when AdamW's 8 bytes/param hurts)

- **Lion** (arXiv:2302.06675): sign-based, 2 states → ~1/2 AdamW memory. Slightly
  worse convergence, used in some vision/LM work.
- **Adafactor**: factorized second moments → O(1) extra memory. Used by PaLM.
- **8-bit Adam / Adam-mini**: quantize optimizer state. Growing adoption in 2025.

## Exercises

1. **The muP sweep.** Train 4 widths (64..512) × {no-muP, muP}, 100 steps each.
   Plot: activation RMS, grad RMS, and final loss vs width. muP curves should be
   flat; no-muP will show width-dependent loss. This single experiment is worth
   more than the rest of this module combined.
2. **Fit Chinchilla yourself.** Train 6 (N, D) points from {tiny, 50m} ×
   {0.5M, 2M, 8M} tokens. Fit alpha, beta by least squares on log-loss. Extrapolate
   the loss of 100m/2B tokens. Train it. How close were you? (This is literally
   the technique of the original paper, at hobby scale.)
3. **Compute-optimal split.** Given your fitted law and a 10x compute budget,
   compute the optimal (N, D). Then verify by training your recommended split vs
   2 alternatives. Where does your optimum land relative to Chinchilla's 20
   tokens/param, and why?
4. **Optimizer memory math.** For a 100M model: AdamW state size. Repeat for
   Lion and Adafactor. At 70B, what's the GPU count difference (80GB A100s)?
   This is the whole pitch for frugal optimizers.

**Papers:**
- Hoffmann et al., *Training Compute-Optimal LLMs (Chinchilla)* (arXiv:2203.15556)
- Kaplan et al., *Scaling Laws* (arXiv:2001.08361) — the originals (params-scale,
  superseded by Chinchilla's data-aware form)
- Yang et al., *Tensor Programs V: Tuning Large NNs via Zero-Shot Hyperparameter
  Transfer (muP)* (arXiv:2203.03466)
- Muennighoff et al., *Scaling Data-Constrained LMs* (arXiv:2305.16264) — data
  epochs, the post-Chinchilla refinement
- DeepSeek-V3 technical report (arXiv:2412.19437) — § training: the frontier's
  actual hyperparameter table
