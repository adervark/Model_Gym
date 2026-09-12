# Module 19 — The 2025 training-stack upgrades: Muon + DAPO

**Goal:** two upgrades that post-date the main course modules — Muon (the
optimizer that trained Moonlight at half the FLOPs of AdamW) and DAPO (the
ByteDance Seed + Tsinghua AIR fixes that make GRPO actually work well). Both are
drop-ins into code you already have: `lm/optim.py` and `lm/grpo.py`.

## 1. Muon: momentum, orthogonalized

AdamW's per-parameter variance normalization is effective but expensive and —
Muon's argument — *hurts*: dividing by sqrt(v) distorts the update's
eigenspectrum. Muon instead: keep a **momentum buffer** per weight matrix and
**orthogonalize** it every step:

```
update = lr · orthogonalize(momentum_buffer)     momentum = 0.95·momentum + grad
```

Why orthogonalization: in the limit, orthogonalized momentum = the polar
factor of the gradient flow — a pure rotation of the weight basis that
preserves the update's spectral structure instead of shrinking every
coordinate differently. The math:

- **Newton-Schulz iteration** computes the orthogonal factor of a matrix
  without SVDs — just 5 matrix multiplies (`lm/optim.py::newton_schulz`).
- The quintic coefficients (3.4445, -4.7750, 2.0315) maximize the slope at
  zero — tuned for the empirical singular-value distribution of momentum
  buffers. **Read the honest caveat in the docstring**: this is an
  *approximation* — the tests verify shape preservation, error reduction,
  and faithfulness to the reference, not exact orthogonality on arbitrary
  inputs.
- **The recipe**: Muon for 2D weight matrices at lr≈0.02 (an order of
  magnitude above AdamW — the orthogonalization removes the need for tiny
  per-parameter lr), plain AdamW for 1D params (norms, biases) at 3e-4.
- Result: Moonlight (16B) trained with ~50% of AdamW's FLOPs. The lr is
  *transferable across widths* — a muP-like property Muon gets for free.

```bash
python -m lm.cli pretrain --out checkpoints/muon-100m --size 100m \
    --steps 20000 --optim muon --lr 0.02
```

## 2. DAPO: the four GRPO fixes (Yu et al., arXiv:2503.14476)

GRPO (module 10) as published has pathologies; DAPO diagnosed and fixed them.
All four are implemented in `lm/grpo.py`:

| Fix | Problem | Mechanism |
|---|---|---|
| **Clip-higher** | a symmetric band caps how fast a low-probability token can be promoted → kills the exploratory long-CoT gradient | *raise the upper bound only*: clip to [1-eps_low, 1+eps_high] with eps_high≈0.28 > eps_low≈0.2 (`dapo=True`) |
| **Dynamic sampling** | groups where all G rewards are equal contribute adv≈0 — wasted compute, diluted gradients | filter them (`filter_degenerate`) |
| **Token-level loss** | per-completion mean loss gives a 20-token and a 2000-token rollout equal weight → length collapse | normalize by TOTAL tokens in the batch, not per completion (`token_level=True`) |
| **Overlong reward shaping** | models learn to stop *late* (reward=1 until max_len, then silence) | penalize wrong answers that hit the length limit (`overlong_shaped_reward`) |

The deep pattern: **all four are about gradient flow**. GRPO's advantage
signal is fragile — clipping, noise, and normalization each quietly shrink
it, and the model responds by shortening its thinking. DAPO = removing every
place the signal leaks.

Both GRPO and DAPO share one clipped surrogate, evaluated **per token**:

```
rho_t  = exp(logp_t - old_logp_t)
surr_t = min(rho_t * A, clip(rho_t, 1-eps_low, 1+eps_high) * A)
```

and differ only in the normalization (per-completion mean vs total tokens).
Note both branches keep the pessimistic `min` and clip on *both* sides — the
asymmetry widens the trust region, it does not remove it. Dropping the clip for
positive advantages is not "clip-higher", it is "no trust region", and it blows
up the moment a rollout gets a large advantage at low probability.

```python
# old_logp must be PER-TOKEN [N, T_c] — the ratio is per token, so a
# per-completion mean is not a ratio of anything (grpo_loss rejects it).
with torch.no_grad():
    old_logp = completion_logps(model, prompt, comp, mask)
loss, stats = grpo_loss(model, ref, prompt, comp, mask, rewards, old_logp,
                        group_size, dapo=True, eps_high=0.28, token_level=True)
```

## 3. What this tells you about the field

Muon and DAPO are both *post-hoc fixes to the 2022-2023 default stack* —
and both came from small teams shipping real runs (Keller Jordan; the
DeepSeek-AI post-training group). The frontier's optimizers and RL recipes
are not settled science; they're the current best hack, and the people
finding the next one are running exactly the kind of experiments this
course's test suite locks in: gradient math checked against hand-derived
formulas, ablation grids, and overfit sanity checks.

## Exercises

1. **Muon vs AdamW at matched FLOPs.** Same model, same steps: `--optim muon
   --lr 0.02` vs adamw 3e-4. Plot loss vs *wall-clock* (Muon's 5 extra
   matmuls per step cost time). Where does Muon win? Then halve Muon's
   steps (its FLOP claim) and compare final loss.
2. **Watch the length collapse, then kill it.** GRPO with standard loss:
   log mean completion length per step — it shrinks. Enable token-level +
   clip-higher: what happens to length? Then add overlong shaping with
   `--max-new` tight. Report all three curves.
3. **Muon + muP.** Muon's lr transfers across widths — verify: tune lr on
   tiny, transfer to 100m. Does it hold as well as muP's transfer (module 08)?
4. **NS from scratch.** Implement the quadratic NS iteration (X ← 1.5X −
   0.5·X·Xᵀ·X) and compare convergence to the quintic on the momentum-buffer
   distribution. When does each win? (The quadratic is the textbook version;
   the quintic is the field's.)

**Papers:**
- Keller Jordan, *Muon: An optimizer for hidden layers in neural networks*
  (2024) — a blog post (kellerjordan.github.io), not an arXiv paper; the
  quintic coefficients and the recipe come from here
- Liu et al., *Muon is Scalable for LLM Training* / Moonlight
  (arXiv:2502.16982) — Muon at 16B scale, the ~2x-FLOP-efficiency claim
- Yu et al., *DAPO: An Open-Source LLM Reinforcement Learning System at
  Scale* (arXiv:2503.14476) — ByteDance Seed + Tsinghua AIR
- Shao et al., *DeepSeekMath* (arXiv:2402.03300) — GRPO itself, the thing
  DAPO fixes
