"""DPO entry point (module 10). Logic lives in lm/runners.py.

Usage: python scripts/run_dpo.py --ckpt checkpoints/sft-100m/best.pt --steps 500
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lm.runners import run_dpo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="checkpoints/dpo")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--seq-len", type=int, default=128)
    args = ap.parse_args()
    run_dpo(args.ckpt, args.out, args.steps, args.lr, args.beta, args.seq_len)


if __name__ == "__main__":
    main()
