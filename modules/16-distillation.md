# Module 16 — Distillation + small-scale RLVR: the R1 pipeline on your GPU

**Goal:** run the actual frontier recipe (R1) end-to-end at small scale:
distill reasoning traces from a strong open model → SFT your model on them →
GRPO on verifiable rewards. Everything here runs on your 8GB card with
1.5B-4B models, or with your own 100M as the student.

## 1. Why distillation first

R1's key engineering finding: reasoning is a *distribution*, and it transfers.
The R1-Distill models (1.5B-70B) were not trained with RL at all — they're SFT
on 800k traces sampled from R1 itself, and they beat most same-size models on
math. Distillation gets you ~most of the reasoning gain for a fraction of the
compute; RL then pushes past what the teacher demonstrates. Order matters:
**RL-first on a raw base works (R1-Zero) but is far less sample-efficient than
seeding with traces** — exactly what you'll measure in exercise 3.

## 2. The pipeline (three commands)

```bash
# 1. Teacher generates long-CoT traces on GSM8K (R1-Distill's data step)
python scripts/distill.py --model Qwen/Qwen2.5-0.5B-Instruct --n 500
#    (8GB: Qwen2.5-1.5B-Instruct in bf16 fits; 8B with --load-4bit, batch 1)

# 2. SFT your model on the traces — module 09 mechanics, better data
python -m lm.cli sft --ckpt checkpoints/base-100m/best.pt \
    --data data/distill_sft.jsonl --out checkpoints/distill-100m \
    --steps 4000 --lr 1e-5

# 3. RLVR on top — module 10 mechanics, verifiable rewards
python -m lm.cli grpo --ckpt checkpoints/distill-100m/best.pt \
    --out checkpoints/r1-mini --steps 1000 --group 4
```

For a real-model version, the same script mechanics against Qwen2.5-1.5B:
teacher = 7B/14B via API or --load-4bit, student = 1.5B QLoRA SFT (module 14),
then GRPO-style RL with verifiers (TRL's GRPOTrainer is the production path).

## 3. What makes the traces good (the parts that matter)

The teacher's prompt template (`scripts/distill.py::PROMPT_TEMPLATE`) asks for
step-by-step reasoning and a parseable answer line. Two data-quality levers,
both cheap, both large:

- **Format filtering**: keep only traces whose final answer matches the ground
  truth (GSM8K has answers). Wrong traces teach wrong reasoning — R1 filtered
  heavily; you should too. (One line in distill.py's loop.)
- **The prompt is part of the skill**: teachers reason better with explicit
  format constraints; students learn the format as a thinking scaffold. The
  "####" convention is a *verifier interface* — it's how exercise 2 scores you.

## 4. RLVR after distillation (why the last step is RL, not more SFT)

SFT on traces imitates the teacher's error rate; it cannot exceed it. GRPO with
`gsm8k_reward` (module 10) gives gradient signal *only* on the verifiable
outcome — the model self-corrects on problems the teacher got wrong, and learns
to spend more compute (longer thinking) where it pays. This is the R1-Zero
phenomenon: at ~100M scale you'll see the reward rate climb *above* the
teacher's, because the reward signal is sharper than the imitation signal.

## 5. The frontier version of this pipeline (what R1 actually did)

1. Base model + GRPO on math/code (verifiable) → R1-Zero (emergent thinking)
2. Rejection-sample the best traces → cold-start long-CoT SFT
3. Large GRPO run (rule rewards + learned RM for subjective domains)
4. Rejection sampling again → final SFT for helpfulness/safety
5. **Distill to every size** — the step you're running

## Exercises

1. **Teacher size vs student quality.** Distill 500 traces from 0.5B and 1.5B
   teachers (or one 8B via API). SFT identical students. Measure GSM8K accuracy.
   The gap is the "distillation tax" — how much is the teacher's capability in
   the traces?
2. **Trace quality matters more than quantity.** Filter traces to
   answer-correct only, then train on {all, correct-only} × {100, 500} traces.
   Report the 4-way accuracy table. (R1's finding: filtered correct traces win.)
3. **SFT vs SFT+GRPO.** Student A: distillation SFT only. Student B: + GRPO on
   GSM8K rewards. Student C: GRPO from base (no distillation). Compare accuracies
   and *token-length of reasoning traces*. Which student "thinks" longest?
4. **Self-verification probe.** Prompt the GRPO'd model with "double-check your
   answer" and measure the self-correction rate vs the SFT-only model. Self-
   correction emerging without being taught it is the R1-Zero signature.

**Papers:**
- Hinton et al., *Distilling the Knowledge in a Neural Network* (arXiv:1503.02531)
- DeepSeek-R1 (arXiv:2501.12948) — the pipeline above
- Li et al., *Distilling Reasoning Capabilities into Smaller LMs* (arXiv:2212.10604)
