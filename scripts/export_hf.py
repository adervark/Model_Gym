"""Export a course checkpoint to HuggingFace format (module 12: vLLM, harness).

Usage: python scripts/export_hf.py --ckpt checkpoints/base-100m/best.pt --out hf-export
"""
import argparse

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import os

import torch

from lm.config import ModelConfig
from lm.model import Transformer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="hf-export")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["model_cfg"])
    model = Transformer(cfg)
    model.load_state_dict(ckpt["model"])

    os.makedirs(args.out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out, "pytorch_model.bin"))
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({
            "model_type": "gpt2",  # nearest HF architecture; export for eval only
            "n_embd": cfg.hidden, "n_layer": cfg.n_layers, "n_head": cfg.n_heads,
            "vocab_size": cfg.vocab_size,
        }, f, indent=2)
    print(f"exported to {args.out}/ (config maps to gpt2 shape; fine for "
          f"lm-eval-harness with --model hf)")


if __name__ == "__main__":
    main()
