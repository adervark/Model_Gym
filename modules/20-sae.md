# Module 20 — Sparse Autoencoders: reading the residual stream

**Goal:** the interpretability frontier's core tool. An SAE decomposes your
model's internal activations into sparse, human-readable *features* — the
machinery behind Anthropic's "Golden Gate Claude" and Google's Gemma Scope.
Implementation: `lm/sae.py`.

## 1. Why "reading" a model is a compression problem

Module 02's mental model: the residual stream is the model's communication
channel. But a single activation vector h ∈ R^d is a *superposition* — one
neuron firing for 50 unrelated concepts at once (the "superposition
hypothesis", Anthropic's Toy Models). You can't read it directly.

The SAE's bet: h is a sparse sum of interpretable feature directions,

```
h ≈ Σᵢ aᵢ·dᵢ,      a = ReLU(W_enc(h − b_pre) + b_enc)     a has ~k ≪ f nonzeros
```

- **f ≫ d features**: an overcomplete dictionary — each feature can mean one
  precise thing ("the token 'Marvel'", "math in progress").
- **Sparsity** is what buys interpretability: `L = MSE(h, Σaᵢdᵢ) + λ·|a|₁`.
- **TopK** (Gao et al., Gemma Scope): instead of λ, hard-select the k largest
  activations — exactly-k sparsity, no λ tuning.
- The **decoder rows dᵢ are unit-norm** (scaling is absorbed into aᵢ).

## 2. The training loop (`lm/sae.py`)

```python
batches = activation_batches(model, ids, layer)   # hook residual stream
sae = SparseAutoencoder(d=cfg.hidden, f=2048, k=32)
train_sae(sae, batches, steps=5000)
```

`collect_activations` runs the model and hooks the output of a chosen layer —
the same trick as the logit lens, collecting instead of decoding. Then it's a
plain autoencoder: 5000 steps on a laptop. The two health metrics that matter:

- **Dead features**: dictionary entries that never activate (`dead_features`).
  >50% dead = too sparse; retune k.
- **Explained variance**: MSE / activation variance. <60% = too few features.

## 3. Reading a feature (`feature_top_tokens`)

A feature's decoder direction dᵢ is a vector in residual space — project it
through the unembedding (the logit lens from module 13) and read its top
tokens:

```bash
python scripts/train_sae.py --ckpt checkpoints/base-100m/best.pt \
    --layer 6 --features 2048 --k 32 --steps 5000
# prints: feature 0: ['Queen', 'Elizabeth', 'II']  ...
```

Coherent token clusters = the feature "means" something. Incoherent ones =
the feature is entangled (or the model's just under-trained — at 100M, expect
a mix; that mix IS the data you're collecting).

## 4. What SAEs are actually for (the frontier map)

1. **Steering**: amplify a feature's activation (add c·dᵢ to the stream) and
   watch behavior change — Anthropic's Golden Gate Claude.
2. **Auditing reasoning models**: activate features *during* <think> traces
   and check whether stated reasoning matches internal features (the frontier
   safety question from module 13 §4.7).
3. **Circuit analysis**: which features cause which (patching experiments on
   SAE activations) — the path from correlational to causal claims.

The honest caveat: SAEs explain a fraction of the variance; the residual is
"dark matter" nobody can read yet. That's the open problem.

## Exercises

1. **The sparsity-interpretability curve.** Train SAEs at k ∈ {4, 16, 64}
   on the same layer. Plot: explained variance, dead features, and the
   coherence of the top-20 features' top tokens (judged by you). Where's the
   knee? (Gemma Scope's sweep, in miniature.)
2. **Feature steering.** Find a feature whose top tokens are proper nouns.
   Generate twice: baseline, and with +3σ added along dᵢ. Does the
   distribution of generated names shift? (You now have a steering vector
   — the RLHF-free alignment mechanism.)
3. **Layer anatomy.** Train one SAE per layer (0, L/2, L−1). Compare top
   features across depth: early = syntactic/format, middle = semantics,
   late = next-token. This depth gradient is a standard mechinterp result —
   reproduce it on your model.
4. **Residual vs MLP features.** Train on the residual stream at layer ℓ and
   on the MLP *output* at layer ℓ. Which is more interpretable? Why?
   (Hint: the residual stream mixes attention + MLP contributions.)

**Papers:**
- Anthropic, *Towards Monosemanticity: Decomposing LMs With Dictionary
  Learning* (transformer-circuits.pub, 2023)
- Bricken et al., *Scaling Monosemanticity* (2024)
- Gao et al., *Scaling and evaluating sparse autoencoders* (arXiv:2406.04093)
  — TopK SAEs, the Gemma Scope method
- Templeton et al., *Scaling Monosemanticity: Golden Gate Claude* (2024)
