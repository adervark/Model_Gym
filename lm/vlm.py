"""Tiny VLM (module 21): frozen vision tower + connector + our LM.

Architecture (the LLaVA pattern, reduced to the essentials):
    image -> vision_tower (frozen) -> patch features [B, n_patch, v_dim]
          -> connector Linear(v_dim, d) -> prepend to token embeddings
          -> LM attends over [image | text], predicts caption tokens.

Training: vision tower frozen, connector + LM trainable (or LoRA the LM —
module 14). Data: image/caption pairs. The vision tower is the ONLY part you
don't build here; it comes from transformers (SigLIP/CLIP) or a small random
CNN for offline experiments.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Transformer
from .tokenizer import get_tokenizer

IMG_TOKEN_ID = None  # not used; vision patches occupy raw embedding positions
CAPTION_PROMPT = "Caption: "


class VLM(nn.Module):
    def __init__(self, lm: Transformer, vision_dim: int):
        super().__init__()
        self.lm = lm
        self.connector = nn.Linear(vision_dim, lm.cfg.hidden, bias=False)

    def forward(self, patch_feats: torch.Tensor, tokens: torch.Tensor,
                targets: torch.Tensor | None = None):
        """patch_feats: [B, P, vision_dim] (frozen tower output).
        tokens: [B, T] caption ids. Returns (logits [B, P+T, V], loss)."""
        prefix = self.connector(patch_feats)
        logits, loss, _ = self.lm(tokens, targets, prefix_embeds=prefix)
        return logits, loss


class RandomVisionTower(nn.Module):
    """Stand-in for SigLIP: a small CNN over raw images, trainable or frozen.
    Lets the whole VLM pipeline run offline (module 21 exercises)."""

    def __init__(self, vision_dim: int = 128, n_patches: int = 16):
        super().__init__()
        self.vision_dim = vision_dim
        self.n_patches = n_patches
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.proj = nn.Linear(32, vision_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B, 3, 64, 64] -> patch features [B, 16, vision_dim]."""
        x = self.cnn(images)                       # [B, 32, 4, 4]
        x = x.flatten(2).transpose(1, 2)           # [B, 16, 32]
        return self.proj(x)                        # [B, 16, vision_dim]


def caption_batch(enc, captions: list[str], seq_len: int = 32):
    """Tokenize captions with a prompt prefix; targets shift by one."""
    ids = [enc.encode_ordinary(CAPTION_PROMPT + c)[:seq_len] + [enc.eot_token]
           for c in captions]
    L = max(len(i) for i in ids)
    padded = [i + [enc.eot_token] * (L - len(i)) for i in ids]
    x = torch.tensor(padded, dtype=torch.long)
    return x[:, :-1], x[:, 1:]
