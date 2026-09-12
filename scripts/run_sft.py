"""SFT entry point (module 09). Logic lives in lm/runners.py.

Usage:
  python scripts/run_sft.py --ckpt checkpoints/base-100m/best.pt \
      --out checkpoints/sft-100m --steps 2000 --lr 1e-5
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lm.runners import run_sft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/sft.jsonl")
    ap.add_argument("--out", default="checkpoints/sft")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=128)
    args = ap.parse_args()
    run_sft(args.ckpt, args.data, args.out, args.steps, args.lr,
            args.batch_size, args.seq_len)


if __name__ == "__main__":
    main()
