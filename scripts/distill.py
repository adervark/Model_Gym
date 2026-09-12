"""Distillation (module 16): generate chain-of-thought traces from a strong open
model on GSM8K, format them as SFT data, then fine-tune your small model on the
traces — the R1-Distill recipe at course scale.

Requires: pip install transformers datasets accelerate (network to HF hub).
For 7-8B teachers on an 8GB GPU, add --load-4bit (needs bitsandbytes).

Usage:
  python scripts/distill.py --model Qwen/Qwen2.5-0.5B-Instruct --n 500
  python -m lm.cli sft --ckpt checkpoints/base-100m/best.pt \
      --data data/distill_sft.jsonl --out checkpoints/distill-100m --steps 4000
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

PROMPT_TEMPLATE = """Solve the following math problem step by step. Explain your reasoning, then give the final answer on its own line as "#### {{answer}}".

Question: {question}

Solution:"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", default="data/distill_sft.jsonl")
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--load-4bit", action="store_true")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(load_in_4bit=True,
                                   bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=quant, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
        model = model.to(device)
    tok = AutoTokenizer.from_pretrained(args.model)

    ds = load_dataset("openai/gsm8k", "main", split="train")
    questions = [ex["question"] for ex in ds.select(range(args.n))]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n_done = 0
    with open(args.out, "w") as f:
        for i, q in enumerate(questions):
            prompt = PROMPT_TEMPLATE.format(question=q)
            inputs = tok(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.max_new,
                                     temperature=args.temperature, do_sample=True,
                                     pad_token_id=tok.eos_token_id)
            full = tok.decode(out[0], skip_special_tokens=True)
            response = full[len(prompt):] if full.startswith(prompt) else full
            if not response.strip():
                continue
            f.write(json.dumps({"prompt": q, "response": response}) + "\n")
            n_done += 1
            if i % 25 == 0:
                print(f"distilled {i + 1}/{args.n}")
    print(f"wrote {n_done} traces to {args.out}")
    print("next: python -m lm.cli sft --ckpt <base> --data data/distill_sft.jsonl "
          "--out checkpoints/distill --steps 4000 --lr 1e-5")


if __name__ == "__main__":
    main()
