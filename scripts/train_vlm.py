"""Train the tiny VLM (module 21).

Offline mode (default): a small random CNN as the vision tower — exercises the
whole pipeline anywhere. Real mode: a frozen SigLIP/CLIP from transformers.

Usage:
  python scripts/train_vlm.py --ckpt checkpoints/base-100m/best.pt --steps 2000
  python scripts/train_vlm.py --ckpt checkpoints/base-100m/best.pt \
      --vision openai/clip-vit-base-patch32 --data coco
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from lm.config import ModelConfig
from lm.model import Transformer
from lm.vlm import VLM, RandomVisionTower, caption_batch
from lm.tokenizer import get_tokenizer


def offline_data(n_samples=256, n_patches=16):
    """Synthetic image-caption pairs: the caption describes the image stats,
    so a real (vision -> text) mapping exists and training makes sense."""
    rng = torch.Generator().manual_seed(0)
    imgs, caps = [], []
    adjectives = ["red", "blue", "green", "bright", "dark"]
    for _ in range(n_samples):
        c = torch.randint(0, len(adjectives), (1,), generator=rng).item()
        base = torch.randn(3, 64, 64, generator=rng) * 0.1
        img = base + 0.5 * c  # color channel offset encodes the adjective
        imgs.append(img)
        caps.append(f"a {adjectives[c]} picture")
    return torch.stack(imgs), caps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="checkpoints/vlm")
    ap.add_argument("--vision", default=None)  # HF model id for a real tower
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["model_cfg"])
    lm = Transformer(cfg).to(device)
    lm.load_state_dict(ckpt["model"])

    if args.vision is None:
        tower = RandomVisionTower(vision_dim=128).to(device)
        for p in tower.parameters():
            p.requires_grad_(False)
        images, captions = offline_data()
        vlm = VLM(lm, vision_dim=128).to(device)
    else:
        from transformers import AutoModel
        tower = AutoModel.from_pretrained(args.vision).to(device)
        for p in tower.parameters():
            p.requires_grad_(False)
        vision_dim = tower.config.hidden_size
        vlm = VLM(lm, vision_dim=vision_dim).to(device)
        raise NotImplementedError("wire your caption dataset here (module 21 "
                                  "exercise); the offline path below is the "
                                  "reference for the loop")

    enc = get_tokenizer()
    opt = torch.optim.AdamW(vlm.parameters(), lr=args.lr)
    for step in range(args.steps):
        idx = torch.randint(0, len(images), (args.batch_size,))
        with torch.no_grad():
            patches = tower(images[idx].to(device))
        x, y = caption_batch(enc, [captions[i] for i in idx], seq_len=24)
        x, y = x.to(device), y.to(device)
        _, loss = vlm(patches, x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"step {step:>5} | loss {loss.item():.4f}")

    os.makedirs(args.out, exist_ok=True)
    torch.save({"lm": lm.state_dict(), "connector": vlm.connector.state_dict(),
                "model_cfg": cfg.__dict__}, os.path.join(args.out, "vlm.pt"))
    print(f"saved {args.out}/vlm.pt")


if __name__ == "__main__":
    main()
