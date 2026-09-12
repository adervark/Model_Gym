# Module 14 — LoRA/QLoRA: fine-tuning big models on small GPUs

**Goal:** fine-tune models you cannot train. On your 8GB laptop this is the
bridge to real models (7-8B) — you'll never full-train one, but you can
*adapt* one. The mechanics in `lm/lora.py` run on your course models; the
production recipe at the end targets Llama/Qwen via bitsandbytes+peft.

## 1. The idea (Hu et al. 2021)

Fine-tuning updates all weights (ΔW per matrix). LoRA's observation: the ΔW you
need lives in a low-rank subspace (intrinsic dimensionality of fine-tuning is
tiny — Aghajanyan et al. 2020). So parameterize the update:

```
W_eff = W + (alpha/r) · B @ A        A: [r, d_in], B: [d_out, r],  r = 8..64
```

- **A** random init (kaiming), **B** zero init → W_eff == W at step 0, no
  instability.
- `alpha/r` scales the delta so you can change r without retuning lr (alpha=16,
  r=8 is the default pair).
- Training memory: optimizer state only for A/B — 0.1-1% of params. Your frozen
  base weights are read-only and can live in 16-bit (or 4-bit — that's QLoRA).
- Serving: merge B@A into W (one matrix add per layer, `merge_lora`) → zero
  inference overhead. Or keep adapters and hot-swap them per task.

Which layers: attention q,k,v,o + MLP gate/up/down (all projections — the
default and best at small scale; some work applies only q,v).

## 2. Read `lm/lora.py`

- `LoraLinear`: frozen base + A/B; forward adds the adapter term.
- `apply_lora`: wraps targeted Linears in place.
- `lora_trainable_params`: the optimizer gets only adapters (+ optional extras).
- `merge_lora`: folds adapters back, returns a plain model.
- `count_lora_params`: the number that sells LoRA — your trainable/total ratio.

## 3. Run it

```bash
python -m lm.cli lora --ckpt checkpoints/base-100m/best.pt \
    --out checkpoints/lora-100m --steps 2000 --lr 1e-4 --r 8 --alpha 16
python -m lm.cli generate --ckpt checkpoints/lora-100m/best.pt   # merged model
```

Writes both artifacts: `adapters.pt` (the LoRA delta, ~1-2MB) and `best.pt`
(merged, so everything downstream works unchanged).

## 4. QLoRA: the 4-bit trick (Dettmers et al. 2023)

LoRA still needs the base model in VRAM — a 7B model in bf16 is 14GB, dead on
8GB. QLoRA stacks the pieces you already built:

1. **NF4 base weights** (module 12: `lm/quant.py::NF4`) — ~4x less VRAM, and
   NF4's quantiles fit NN weights near-optimally.
2. **Dequantize-on-the-fly** — compute in bf16, store in 4-bit.
3. **LoRA adapters on top** — the *only* trainable params.
4. **Paged AdamW** — optimizer states spill to CPU RAM (like OS paging) so the
   optimizer never OOMs.

Result: Llama-3.1-8B QLoRA fine-tune in ~6-7GB VRAM. It's the standard recipe
for consumer GPUs, and the reason fine-tunes are cheap while pretraining isn't.

## 5. The production recipe (your 8GB, real model)

```bash
pip install transformers accelerate peft bitsandbytes
python - <<'EOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

model_name = "Qwen/Qwen2.5-1.5B-Instruct"   # or meta-llama/Llama-3.1-8B-Instruct
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained(model_name)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj"]))
# then: Trainer with your dataset, SFT-style loss masking — identical mechanics
# to lm/sft.py, just on someone else's 1.5B weights.
EOF
```

8GB guidance: 1.5B-4B QLoRA with batch 1-2, seq 512-1024, gradient checkpointing
on → fits. 8B works with batch 1 and short sequences. The trained adapters
(~10-50MB) are the artifact you ship; merge with `model.merge_and_unload()`.

## Exercises

1. **The r sweep.** Same data, r ∈ {1, 2, 8, 64} (alpha = 2r). Plot eval quality
   vs trainable params. Where's the knee? Explain with the intrinsic-rank claim.
2. **Merge equivalence.** Verify `merge_lora` correctness: logits of
   (base + adapters) vs merged model must match to fp32 error. Then verify the
   *frozen* base never changed: hash the base weights before/after training.
3. **Which targets matter.** LoRA on {q,v} only vs {q,k,v,o} vs all projections,
   same r. Rank them. (This is the ablation that justified all-projections.)
4. **QLoRA at course scale.** NF4-quantize your 100M base (module 12 GPTQ-lite),
   then LoRA on top. Compare eval quality vs LoRA-on-fp16 base vs full SFT.
   That's the QLoRA paper's central table, reproduced on your hardware.

**Papers:**
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (arXiv:2106.09685)
- Dettmers et al., *QLoRA* (arXiv:2305.14314)
- Aghajanyan et al., *Intrinsic Dimensionality Explains the Effectiveness of
  LM Fine-Tuning* (arXiv:2012.13255)
