"""Scaling-law sweep harness (module 22): grid over (N, D), fit the law.

Runs `lm.cli pretrain` for each (size, steps) cell, collects val losses, and
fits L(N, D) = E + A/N^a + B/D^b by alternating least squares. This is the
Chinchilla paper's technique, automated.

Usage:
  python scripts/sweep_scaling.py --sizes tiny,50m --steps 500,2000 \
      --dataset tiny_shakespeare --data-dir data --out sweeps
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def run_cell(size: str, steps: int, out: str, dataset: str, data_dir: str,
             extra: list[str]) -> float:
    """Train one (N, D) cell; return the final val loss (or NaN on failure)."""
    cmd = [sys.executable, "-m", "lm.cli", "pretrain", "--out", out,
           "--size", size, "--steps", str(steps), "--dataset", dataset,
           "--data-dir", data_dir, "--seq-len", "128", "--batch-size", "2",
           "--grad-accum", "2", "--warmup", str(max(10, steps // 10)),
           "--log-every", str(steps // 2), "--eval-every", str(steps),
           "--save-every", str(steps), "--lr", "3e-4", *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return float("nan")
    losses = re.findall(r"val loss ([\d.]+)", r.stdout)
    return float(losses[-1]) if losses else float("nan")


def fit_law(N: np.ndarray, D: np.ndarray, L: np.ndarray, iters: int = 40) -> dict:
    """Fit L = E + A/N^a + B/D^b.

    For fixed (a, b) the model is linear in (E, A, B); the exponents are found
    by alternating 1-D refinement starting from a coarse grid. Robust and
    dependency-free (this is the Chinchilla fit, for course scale)."""
    def linear_fit(a, b):
        X = np.stack([np.ones_like(L), N ** -a, D ** -b], axis=1)
        w, *_ = np.linalg.lstsq(X, L, rcond=None)
        pred = X @ w
        return w, float(np.sqrt(((L - pred) ** 2).mean()))

    a = b = 0.3
    for _ in range(iters):
        w, _ = linear_fit(a, b)
        for i in range(10):  # refine a along its gradient direction (grid + bisect)
            cands = [a + d for d in (-0.1, -0.02, 0.02, 0.1)]
            errs = [(c, linear_fit(c, b)[1]) for c in cands]
            best_c, best_e = min(errs, key=lambda t: t[1])
            if best_e >= linear_fit(a, b)[1] - 1e-9:
                break
            a = max(best_c, 0.05)
        for i in range(10):
            cands = [b + d for d in (-0.1, -0.02, 0.02, 0.1)]
            errs = [(c, linear_fit(a, c)[1]) for c in cands]
            best_c, best_e = min(errs, key=lambda t: t[1])
            if best_e >= linear_fit(a, b)[1] - 1e-9:
                break
            b = max(best_c, 0.05)
    w, rmse = linear_fit(a, b)
    E, A, B = float(w[0]), float(w[1]), float(w[2])
    return {"E": E, "A": A, "B": B, "alpha": a, "beta": b, "rmse": rmse}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="tiny,50m")
    ap.add_argument("--steps", default="500,2000")
    ap.add_argument("--dataset", default="tiny_shakespeare")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="sweeps")
    ap.add_argument("--extra", nargs="*", default=[])
    args = ap.parse_args()

    from lm.config import ModelConfig
    from lm.cli import PRESETS

    sizes = args.sizes.split(",")
    steps_list = [int(s) for s in args.steps.split(",")]
    os.makedirs(args.out, exist_ok=True)

    results = []
    for size in sizes:
        for steps in steps_list:
            tag = f"{size}_{steps}"
            print(f"[sweep] {tag} ...", flush=True)
            val = run_cell(size, steps, os.path.join(args.out, tag),
                           args.dataset, args.data_dir, args.extra)
            N = ModelConfig(**PRESETS[size]).n_params
            D = steps * 128 * 2 * 2  # tokens = steps * seq * batch * grad_accum
            results.append({"size": size, "N": N, "D": D, "val_loss": val})
            print(f"  val loss {val:.4f} (N={N:.1e}, D={D:.1e})")

    rs = [r for r in results if not math.isnan(r["val_loss"])]
    if len(rs) >= 4:
        N = np.array([r["N"] for r in rs], dtype=float)
        D = np.array([r["D"] for r in rs], dtype=float)
        L = np.array([r["val_loss"] for r in rs], dtype=float)
        fit = fit_law(N, D, L)
        print("\nfit: L(N,D) = E + A/N^a + B/D^b")
        for k, v in fit.items():
            print(f"  {k}: {v:.4f}")
        pred = fit["E"] + fit["A"] / N ** fit["alpha"] + fit["B"] / D ** fit["beta"]
        print(f"  in-sample fit RMSE: {np.sqrt(((L - pred) ** 2).mean()):.4f}")
        with open(os.path.join(args.out, "fit.json"), "w") as f:
            json.dump({**fit, "cells": rs}, f, indent=2)
    else:
        print("\nnot enough successful cells to fit (need >= 4)")


if __name__ == "__main__":
    main()
