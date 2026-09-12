"""GRPO/RLVR entry point (module 10). Logic lives in lm/runners.py.

Usage: python scripts/run_grpo.py --ckpt checkpoints/base-100m/best.pt --steps 300 --group 4
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lm.runners import run_grpo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="checkpoints/grpo")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--eps", type=float, default=0.2)
    ap.add_argument("--max-new", type=int, default=48)
    args = ap.parse_args()
    run_grpo(args.ckpt, args.out, args.steps, args.group, args.lr,
             args.beta, args.eps, args.max_new)


if __name__ == "__main__":
    main()
