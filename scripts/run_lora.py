"""LoRA SFT entry point (module 14). Logic lives in lm/runners.py.

Usage:
  python scripts/run_lora.py --ckpt checkpoints/base-100m/best.pt --steps 2000
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lm.runners import run_lora


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/sft.jsonl")
    ap.add_argument("--out", default="checkpoints/lora")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    args = ap.parse_args()
    run_lora(args.ckpt, args.data, args.out, args.steps, args.lr,
             args.batch_size, args.seq_len, args.r, args.alpha)


if __name__ == "__main__":
    main()
