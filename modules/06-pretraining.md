# Module 06 — Pretraining

**Goal:** the full modern training loop — AdamW, bf16, gradient clipping,
schedules, checkpoints — and a 100M model actually learning language. This is the
module where loss curves stop being noise.

## 1. The optimizer: AdamW (and why the details matter)

```
m_t = beta1 m_{t-1} + (1-beta1) g        # momentum (EMA of grads)
v_t = beta2 v_{t-1} + (1-beta2) g^2      # variance (EMA of squared grads)
update = lr * m_t/(sqrt(v_t)+eps)        # normalized, per-parameter lr
w = w - update - lr * wd * w             # DECOUPLED weight decay (the "W")
```

- **Decoupled weight decay** (Loshchilov & Hutter): decay applied directly to
  weights, not through the Adam normalization. This is what makes wd actually act
  as L2 regularization rather than being renormalized away.
- **beta2 = 0.95** (not 0.999): shorter gradient-variance memory handles the
  distribution shift of training better. GPT-3/Llama choice. beta1 0.9.
- **Gradient clipping** (`clip_grad_norm_`, 1.0): a single pathological batch
  (the internet contains anything) produces a loss spike -> gradient spike -> the
  variance EMA is poisoned for 1/(1-beta2) ~ 20 steps. Clip instead.
- `eps=1e-8` (AdamW default): with bf16's low precision, smaller eps destabilizes.
- **Bias correction** is on by default (m, v divided by 1-beta^t); keeps early
  steps from exploding when m,v start at 0.

**Why not just SGD?** AdamW needs no per-layer lr tuning; its per-parameter
normalization is what lets one lr train embeddings, attention, and head
simultaneously. Every serious LLM run uses AdamW (or a memory-frugal variant:
Lion, Adafactor — see module 08).

## 2. Precision: bf16 is the default for a reason

- **bf16** = fp32's 8-bit exponent + 7-bit mantissa. Same range as fp32 (no
  overflow/underflow), half the memory, double the tensor-core throughput on
  Ampere+. Loss scaling (fp16's GradScaler hack) becomes unnecessary — the range
  is wide enough. This is why bf16 won.
- fp16 still exists on pre-Ampere GPUs: needs GradScaler. If you see NaN loss with
  fp16, it's an overflow — scale the loss.
- Master weights stay fp32 (optimizer state). Memory during training:

  | Tensor | dtype | bytes/param |
  |---|---|---|
  | weights | bf16 | 2 |
  | gradients | bf16 | 2 |
  | AdamW m, v | fp32 | 8 |
  | master weights | fp32 | 4 |
  | **total** | | **16** + activations |

  Memorize 16 bytes/param — module 07 §1 builds the whole distributed-training
  argument on it, and it is the number that tells you at a glance that a 7B
  model (112GB) cannot train on one 80GB card.

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    logits, loss, _ = model(x[:, :-1], x[:, 1:])   # autocast picks per-op precision
```

`autocast` doesn't just downcast: some ops (softmax, norms) stay fp32 internally
where precision matters. Don't hand-write `half()` casts; let autocast choose.

## 3. Batch size & gradient accumulation

The frontier insight (GPT-3): total batch size (tokens per optimizer step) is a
hyperparameter — ~0.5M tokens for 100M-1B models, up to ~4M+ at frontier scale.
Hardware can't fit that: accumulate gradients over micro-batches:

```python
for _ in range(cfg.grad_accum):
    loss = model(batch) / cfg.grad_accum   # normalize BEFORE backward
    loss.backward()                        # grads accumulate
optimizer.step()
```

Gradient accumulation = larger batches at fixed memory. The scale is roughly
linear: doubling batch lets you halve steps (at fixed lr) until the noise floor
(too-small batches are noisier; too-large wastes tokens per update).

## 4. Learning rate schedule: warmup → hold → decay

- **Warmup** (~500 steps): gradient-variance EMA is cold at init; early large
  steps wreck the freshly-initialized weights. Linear ramp fixes it.
- **Cosine decay** to ~10% of peak: standard for compute-limited runs — the final
  decay phase contributes disproportionately to quality (why: it anneals into a
  low-loss basin; a famous Chinchilla-era finding).
- **WSD** (warmup-stable-decay, DeepSeek/Cold-DL): hold lr flat until ~90% of
  budget, decay at the end. Costs nothing, and — critically — lets you *continue*
  training with another schedule cycle (the stable phase is resumable; cosine
  isn't). DeepSeek-V3 was trained with WSD.
- All three are in `lm/train.py::get_lr`. LR for 100M: ~3e-4. For 7B: ~3e-4 (modern
  practice doesn't scale lr down much with size — module 08/muP).

## 5. Logging: wandb (2 minutes, worth it)

Terminal logs stop being readable once runs take hours. One flag:

```bash
pip install wandb
python -m lm.cli pretrain --out checkpoints/base-100m --wandb --wandb-project lm-course
```

`train.py` logs loss, val loss, lr, tok/s per step (`--log-every` controls the
cadence). Once you have curves, you develop the two habits that separate
deliberate training from watching a progress bar:
- **Superpose runs**: every config change (lr, batch, arch) is one line on the
  same loss-vs-token plot. The plot *is* the experiment.
- **Watch val loss vs train loss gap**: divergence = overfitting (data too
  small / too many epochs). With WSD (module 06 §4) you can catch it and stop.

## 6. Run the real thing

```bash
python -m lm.cli pretrain --out checkpoints/base-100m --size 100m \
    --dataset fineweb_edu --steps 20000 --lr 3e-4
```

Then sample:

```bash
python -m lm.cli generate --ckpt checkpoints/base-100m/best.pt --max-new 200
```

## 7. Read `lm/train.py` with this checklist

Open it now. Find each: (1) the 5-line loop from module 01, (2) grad accumulation
with `loss / grad_accum`, (3) autocast context, (4) clip_grad_norm_, (5) AdamW
with betas/wd, (6) the lr schedule function, (7) eval under `no_grad` every N
steps, (8) checkpoint = {model, optimizer, step, configs} — always save the
optimizer state (resume without replaying momentum) and configs (reconstruct the
model without guessing).

## Exercises

1. **The ablation matrix.** Same seed, 500 steps each: (a) remove weight decay,
   (b) no warmup, (c) no clipping, (d) SGD instead of AdamW, (e) constant lr.
   Rank the damage. (You'll find clipping+wd+schedule are each worth ~0.1-0.3 loss
   — boring but load-bearing.)
2. **Loss spike autopsy.** Train with clipping disabled on FineWeb — find the step
   where loss explodes, dump that batch's text, read what caused it. Then restore
   clipping and confirm recovery. This is the actual failure mode clipping exists for.
3. **batch vs steps.** Grid: {32k, 128k, 512k} tokens/step × matched total tokens
   (adjust steps). Plot val loss vs batch at fixed token count. At what batch does
   the curve bend? Explain with gradient noise.
4. **Checkpoint resume.** Kill a run at step 1000, resume from checkpoint, verify
   the loss curve is continuous (identical to uninterrupted run within noise).
   Checkpoint bugs are invisible until the run you can't afford to lose dies.

**Papers:**
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (arXiv:1711.05101)
- Brown et al., *GPT-3* (arXiv:2005.14165) — batch size, lr schedule, the recipe
- Hoffmann et al., *Chinchilla* (arXiv:2203.15556) — the decay-phase finding
- Hu et al., *WSD: Warmup-Stable-Decay* (arXiv:2404.06395)
