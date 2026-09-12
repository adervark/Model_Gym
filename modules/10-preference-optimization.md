# Module 10 — Preference optimization: RLHF → DPO → GRPO/RLVR

**Goal:** train behavior that SFT can't reach — correctness on verifiable tasks,
preference alignment — via the three generations of methods. This is the module
where 2025's frontier (R1, o-series) is made.

## 0. The problem SFT can't solve

SFT maximizes likelihood of demonstrations. But for tasks with *verifiable*
outcomes (math: answer is right/wrong; code: tests pass/fail), demonstrations
are noisy and the *gradient signal* "imitate" is weaker than "the answer was
wrong, search more". RL supplies that signal. R1's headline result: RL alone
turned a base model into a reasoner, without SFT. That's why this module exists.

## 1. RLHF (classic, 2022-2024): reward model + PPO

```
1. Collect (prompt, win, lose) pairs      4. PPO: optimize reward while staying
2. Train reward model r(x, y)                 near the SFT model (KL penalty)
3. r scores generations                   policy = argmax E[r(x,y)] - beta KL(pi||pi_sft)
```

PPO needs: value model (critic), importance sampling, advantage estimation
(GAE), multi-epoch clipping. It works (InstructGPT, ChatGPT) but it's expensive
(2 models + ref + rewards) and finicky. **In 2025, nobody new starts PPO.**

## 2. DPO: collapse RLHF into one loss

Rafailov et al.'s move: for Bradley-Terry preferences
`P(win over lose) = sigma(r_win - r_lose)`, the *closed-form optimal policy* of
the RLHF objective is

```
pi*(y|x) ∝ pi_ref(y|x) · exp(r(x,y)/beta)      ⟺      r(x,y) = beta log(pi*/pi_ref)
```

Substitute r into the preference loss → train the policy directly on pairs:

```
L_DPO = -log sigma( beta [ log pi(w|x)/pi_ref(w|x) - log pi(l|x)/pi_ref(l|x) ] )
```

One model, one loss, no reward model, no PPO. Your implementation:
`lm/dpo.py::dpo_loss`. Read the derivation (paper is short); the mechanism worth
internalizing: DPO is just gradient ascent on preferred completions + descent on
rejected ones, with `pi_ref` as the anchor preventing drift.

**DPO's failure mode** (well-documented by 2024): the preference data only tells
you relative quality *within the data distribution*. Off-policy methods can
plateau where the SFT policy starts; length bias sneaks in; and nothing here
helps on *verifiable* tasks (no reward signal, just comparisons). Hence:

## 3. GRPO / RLVR: the 2025 workhorse (what R1 uses)

**RLVR** (RL with Verifiable Rewards — the name was popularized by Tulu 3,
arXiv:2411.15124, cited below): replace the learned
reward model with a *rule* — math answer equality, unit tests passing, compiler
output. No reward model at all → no reward hacking, no training a critic.

**GRPO** (DeepSeek, arXiv:2402.03300): replace the critic with group statistics.
For each prompt, sample G completions, score them, and normalize *within the
group*:

```
A_i = (r_i - mean(r_1..r_G)) / std(r_1..r_G)        # group-relative advantage

L = -min(rho * A_i, clip(rho, 1-eps, 1+eps) * A_i) + beta * KL(pi || pi_ref)
    rho = pi(y_i|x) / pi_old(y_i|x)                  # importance ratio vs rollout policy
```

The loop (implemented in `lm/grpo.py` + `scripts/run_grpo.py`):

```
for step:
    sample prompts (e.g., GSM8K questions)
    rollout G completions per prompt from the policy (temperature ~1)
    score with rule-based reward (lm/grpo.py::gsm8k_reward)
    group-normalize -> advantages
    gradient step: clipped policy update + KL to frozen ref (no critic!)
```

This is R1's recipe (their R1-Zero: base model + GRPO on math/code → reasoning
emerges). The `beta * KL` term keeps the model from drifting into reward hacking
("#### 42" spam). And group normalization is doing what the critic did — but
with zero extra parameters, batched and stable.

**When to use which (the practical decision):**

| Task type | Use |
|---|---|
| style/preference data, quick iteration | DPO |
| verifiable tasks (math, code, agents) | GRPO + RLVR |
| general helpfulness at frontier scale | RLVR + a *learned* reward model ensemble (frontier labs use both: verifiable rules where they exist, learned RMs elsewhere) |

## 4. Run it

```bash
python scripts/run_dpo.py  --ckpt checkpoints/sft-100m/best.pt --steps 500
python scripts/run_grpo.py --ckpt checkpoints/sft-100m/best.pt --steps 300 --group 4
```

(These use tiny synthetic preference/math datasets so you can run on CPU; the
same code takes UltraFeedback/GSM8K by swapping the data script.)

## Exercises

1. **DPO's entropy collapse.** Train DPO 2000 steps, watch entropy of
   generations vs step. Find the step where the model's outputs become
   repetitive. The fix (adding an SFT loss term back) is the standard trick —
   implement it, confirm the collapse moves later.
2. **Group size matters.** GRPO with G=2 vs 4 vs 8 at fixed total rollouts.
   Plot reward mean + KL vs steps. Explain the tradeoff (advantage variance vs
   rollout cost). R1 uses G=16 — why not more?
3. **Reward hacking, hands-on.** Train GRPO with the GSM8K reward but *without*
   the KL term. Find what the model learns to output. (Spoiler: reward goes to 1,
   answers are garbage — you just reproduced the #1 failure mode of RL.) Then
   restore KL and watch the difference.
4. **DPO vs GRPO head-to-head.** Same base, same budget (steps × batch): DPO on
   preference pairs vs GRPO on verifiable rewards, evaluated on held-out math
   accuracy. Explain the result from the loss functions alone.

**Papers:**
- Christiano et al., *Deep RL from Human Preferences* (arXiv:1706.03741) — RLHF
- Ouyang et al., *InstructGPT* (arXiv:2203.02155) — PPO at scale
- Rafailov et al., *Direct Preference Optimization* (arXiv:2305.18290)
- Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning*
  (arXiv:2402.03300) — GRPO
- Lambert et al., *Tulu 3: Pushing Frontiers in Open Language Model
  Post-Training* (arXiv:2411.15124) — RLVR
- DeepSeek-R1 (arXiv:2501.12948) — RL-first post-training
