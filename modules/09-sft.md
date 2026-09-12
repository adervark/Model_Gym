# Module 09 — Supervised Fine-Tuning (SFT)

**Goal:** turn the base model (autocomplete engine) into an assistant (follows
instructions, answers, refuses). SFT is where the tokenizer, chat template, and
loss masking all come together — and it's the cheapest quality lever you have.

## 1. What base models know vs what they do

A pretrained model completes text: prompt "Explain gravity" → it continues like
Wikipedia ("...is one of the four fundamental forces, along with..." — an
*essay fragment*, not an answer). Base models also don't know how to say "I don't
know", follow formats, or stop. SFT teaches the *interaction surface* by training
on (instruction, response) pairs. Crucially: SFT is imitation — the ceiling is
the quality of the demonstrations. RL (module 10) is where the model exceeds its
teachers.

The standard recipe (InstructGPT/Llama 3):
1. Collect ~10k-1M (prompt, answer) pairs from humans.
2. Format with a chat template; loss computed on assistant tokens only.
3. Train 1-3 epochs, low lr (~1e-5), decay.

## 2. Chat templates (this is the interface, get it exactly right)

Llama 3 style:

```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Explain gravity.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Gravity is...<|eot_id|>
```

Facts that matter:
- **Special tokens were added at SFT time** (module 04): their embeddings are
  random at init, learned during SFT. The model has *no idea* what `<|assistant|>`
  means until SFT teaches it.
- **Template drift = quality death**: a model evaluated with a slightly different
  template than training drops sharply (the "format tax"). Frontier labs pin
  templates to the byte.
- Your course template is `lm/sft.py::CHAT_TEMPLATE` — deliberately minimal
  (`<|user|>`/`<|assistant|>`/`<|end|>`); real deployments use Llama 3 / ChatML.

## 3. Loss masking: the one implementation detail

Only train on assistant tokens. If you compute loss on user turns, the model
learns to *regenerate prompts* (and evaluations break). Implementation
(`lm/sft.py::sft_step`): compute per-token CE, zero out (mask) non-assistant
positions, mean over the rest.

Why mean over masked tokens (not the full sequence): loss becomes the
assistant's average log-likelihood, comparable across different-length answers.

## 4. Packing SFT samples (the efficiency trick you'll need at scale)

SFT sequences are short (100-500 tokens). Pad = waste ~70%. Pack: concatenate
multiple (prompt, response) pairs into one seq_len window, mask each pair's
assistant span, and **never pack across a sample boundary in a way that mixes
attention contexts** — standard practice: mask attention between packed samples
too, or rely on the fact that most SFT fine-tunes tolerate concatenation (Llama
3's paper §4 says they pack and only mask the loss; cross-sample attention is
accepted noise at scale). Either is fine at 100M.

## 5. Run it

Generate a synthetic SFT dataset (real ones: OpenAssistant, UltraChat, or your
own domain data):

```bash
python scripts/make_sft_data.py          # writes data/sft_train.bin from tiny templates
python scripts/run_sft.py \
    --ckpt checkpoints/base-100m/best.pt \
    --out checkpoints/sft-100m --steps 2000 --lr 1e-5
```

Then compare:

```bash
python -m lm.cli generate --ckpt checkpoints/base-100m/best.pt   # essay fragment
python -m lm.cli generate --ckpt checkpoints/sft-100m/best.pt    # answers the question
```

## 6. The 2025 landscape (what changed since InstructGPT)

- **Synthetic data is the default.** Llama 3's SFT data was largely
  *model-generated* (from bigger models), then human-filtered. Frontier recipe:
  seed with human examples → generate → filter by quality rubric → train.
- **SFT scale shrank.** R1 famously *skipped* the large SFT stage (RL-first).
  Current practice: small high-quality SFT (10k-100k examples) → RL (module 10).
- **SFT alone plateaus.** Imitation learning can't create behaviors not in the
  data. This is the formal reason RL exists: the gradient toward "behave like the
  data" doesn't reach "solve problems the data can't show".
- **Long-CoT SFT**: seeding RL with even a few thousand *reasoned* answers
  (with thinking traces) is worth a lot (module 13).

## Exercises

1. **Masking ablation.** Train 500 steps with loss on all tokens vs assistant-only.
   Evaluate both by prompting. Describe the failure mode of the first. (This bug
   ships in production constantly.)
2. **The format tax.** Take your SFT model, evaluate with (a) exact training
   template, (b) template with different spacing/order. Measure the drop. Then
   fix it by adding format augmentation to SFT and re-measure.
3. **Scale the dataset.** Train with 1k, 4k, 16k samples (same steps). Plot
   eval quality vs data size. Where's the knee? (The frontier answer is "~100k
   for general assistants" — you'll see the same shape.)
4. **Catastrophic forgetting check.** After SFT, run `lm.cli eval` (perplexity)
   on FineWeb text. Compare to base. If ppl exploded, you overtrained; find the
   epoch count where base knowledge starts dying.

**Papers:**
- Ouyang et al., *InstructGPT* (arXiv:2203.02155) — the original SFT+RLHF recipe
- Wei et al., *Finetuned Language Models are Zero-Shot Learners* (FLAN)
  (arXiv:2109.01652)
- Llama 3 paper (arXiv:2407.21783) §4 — SFT data pipeline at scale, packing
- DeepSeek-R1 (arXiv:2501.12948) — SFT-first vs RL-first
