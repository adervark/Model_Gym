# Module 23 — Classical RLHF, complete: reward models, PPO, and the preference-loss zoo

**Goal:** the full 2022-era pipeline that every later method (DPO, GRPO, KTO)
is derived from or against — reward model + PPO — plus the four preference
losses you should be able to reach for. Implementation: `lm/rm.py`, `lm/dpo.py`.

## 1. The pipeline (InstructGPT's three stages)

```
1. SFT (module 09)              imitation baseline
2. Reward model:                r(x, y) trained on (prompt, chosen, rejected)
                                with the Bradley-Terry objective:
                                L = -log sigma(r_w - r_l)
3. PPO:                         optimize r while staying near the SFT model:
                                max E[r(x,y)] - beta·KL(pi || pi_ref)
```

Stage 2 (`lm/rm.py::RewardModel`): your LM's trunk + one scalar head;
mean-pool the final hidden over the completion. Stage 3 is where the
machinery lives — and where everything else in module 10 came from:

- **GAE** (`lm/rm.py::gae`): advantages as exponentially-discounted sums of
  TD errors — the credit-assignment workhorse.
- **Clipped surrogate** (`ppo_loss`): the PPO clip you already know from
  GRPO — GRPO *inherited* it from here.
- **Value model** (`ValueHead`): the critic GRPO deleted by using group
  statistics instead. You now understand both the thing and what replaced it.
- **KL to ref**: PPO's anchor — DPO folds this into its loss; GRPO keeps it.

## 2. Why this died and what replaced it (the genealogy)

| Method | Kills | Cost |
|---|---|---|
| PPO | — (the original) | 4 models (policy, ref, critic, RM), GAE, epochs of tuning |
| DPO (module 10) | the RM + PPO entirely | one loss, but off-policy plateau |
| KTO | **pairs** — needs only desirability labels | slightly weaker than DPO |
| GRPO (module 10) | the critic | group-normalized advantages |
| ORPO/SimPO | the **reference model** | weaker guarantees, cheaper |
| RLVR (module 10) | the **reward model** (rules instead) | verifiable tasks only |

Reading it this way, module 10's methods stop being a menu and become a
family tree. That's the point of this module.

## 3. The preference-loss zoo (all in `lm/dpo.py`)

- **ORPO**: `nll(chosen) + beta·relu(log-odds(rej) − log-odds(chosen))` —
  no reference model; the SFT term and the odds penalty share one gradient.
- **SimPO**: length-normalized logps + target margin γ — no reference, and
  the normalization fixes DPO's length bias.
- **KTO**: per-example losses with a KL-referenced threshold — train on
  labeled examples *without pairs* (the cheapest data requirement).

```python
from lm.dpo import PreferenceLosses
loss, stats = PreferenceLosses.LOSSES["orpo"](model, x, mask, beta=0.1)
```

## Exercises

1. **Train the full chain.** SFT (module 09) → RM on synthetic preferences →
   PPO for 200 steps. Plot reward, KL, and response length over training.
   This is InstructGPT's Figure 2, reproduced on your hardware.
2. **The zoo head-to-head.** Same data: DPO vs ORPO vs SimPO vs KTO, matched
   steps. Rank by (final quality, wall-clock, data requirements). Which wins
   for a dataset with pairs? Without? (The answer maps to what each method
   deletes.)
3. **Derive DPO from PPO.** Start from max E[r] − βKL, solve the optimal
   policy in closed form, substitute into Bradley-Terry. Verify your DPO
   loss equals `lm/dpo.py::dpo_loss` numerically. (If you can do this
   derivation, you understand the entire modern alignment stack.)
4. **Reward hacking, the PPO edition.** Optimize the RM with PPO, no KL
   penalty, low temperature. Find the prompt where r is maximal — and read
   what the model says. (The failure mode that motivated RLVR's rules.)

**Papers:**
- Christiano et al., *Deep RL from Human Preferences* (arXiv:1706.03741)
- Ouyang et al., *InstructGPT* (arXiv:2203.02155)
- Schulman et al., *PPO* (arXiv:1707.06347) · *GAE* (arXiv:1506.02438)
- Hong et al., *ORPO* (arXiv:2403.07691) · Meng et al., *SimPO*
  (arXiv:2405.14734) · Ethayarajh et al., *KTO* (arXiv:2402.01306)
