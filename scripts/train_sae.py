"""Train an SAE on a checkpoint's residual stream (module 20).

Usage:
  python scripts/train_sae.py --ckpt checkpoints/base-100m/best.pt \
      --layer 6 --features 2048 --k 32 --steps 5000 --out checkpoints/sae.pt
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from lm.config import ModelConfig
from lm.model import Transformer
from lm.sae import SparseAutoencoder, collect_activations, train_sae, feature_top_tokens
from lm.data import load_raw_text
from lm.tokenizer import get_tokenizer


def activation_batches(model, ids, layer, batch=256, buffer=64):
    """Endless iterator of activation batches collected from the corpus."""
    import itertools
    enc = get_tokenizer()
    seq_len = model.cfg.max_seq_len
    while True:
        i = 0
        while i + seq_len < len(ids):
            chunk = torch.tensor(ids[i:i + seq_len]).unsqueeze(0)
            h = collect_activations(model, chunk, layer=layer)  # [T, d]
            for j in range(0, h.shape[0], batch):
                yield h[j:j + batch]
            i += seq_len // 2  # stride half the window for coverage
        # reshuffle window ordering implicitly by looping corpus
        yield from ()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--text", default="data/tiny_shakespeare.txt")
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--features", type=int, default=1024)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--l1", type=float, default=1e-3)
    ap.add_argument("--out", default="checkpoints/sae.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["model_cfg"])
    model = Transformer(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    enc = get_tokenizer()
    # load_raw_text takes the dataset NAME (resolved against data_dir)
    name = os.path.splitext(os.path.basename(args.text))[0]
    ids = enc.encode_ordinary(load_raw_text(name))
    batches = activation_batches(model, ids, args.layer)

    sae = SparseAutoencoder(d=cfg.hidden, f=args.features, k=args.k).to(device)
    train_sae(sae, batches, steps=args.steps, lr=args.lr, l1_coef=args.l1)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"sae": sae.state_dict(), "model_cfg": cfg.__dict__,
                "layer": args.layer, "k": args.k}, args.out)
    print(f"saved {args.out}")

    # report the most interpretable features (top decoder-direction tokens)
    for f in range(0, 10):
        print(f"feature {f:>3}:", [t for t, _ in feature_top_tokens(sae, model, f, 3)])


if __name__ == "__main__":
    main()
