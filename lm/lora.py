"""LoRA: Low-Rank Adaptation (module 14).

Freeze the base weights W and train a low-rank delta: W_eff = W + (alpha/r) A @ B,
A in R^{r x in}, B in R^{out x r}, r << d. Training memory = optimizer state for
A/B only (r/in ~ 0.1-1% of params); serving cost zero after merge.

QLoRA (same adapters, NF4 base + paged optimizers) is the 8GB-laptop path to
fine-tuning 7-8B models — see module 14 for the production recipe (bitsandbytes
+ peft); this file implements the mechanics in pure torch against our model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Transformer

DEFAULT_TARGETS = ("q", "k", "v", "o", "gate", "up", "down")


class LoraLinear(nn.Module):
    """Wraps a frozen Linear with trainable A/B adapters. alpha/r scales the
    delta so r can be changed without retuning lr."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        out_f, in_f = base.weight.shape
        self.r, self.alpha = r, alpha
        self.A = nn.Parameter(torch.empty(r, in_f))
        self.B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.A, a=math_sqrt5())  # B=0 keeps W_eff == W at init
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, x):
        delta = (self.alpha / self.r) * (self.dropout(x) @ self.A.T @ self.B.T)
        return F.linear(x, self.base.weight, self.base.bias) + delta


def math_sqrt5():
    import math
    return math.sqrt(5)


def apply_lora(model: nn.Module, r: int = 8, alpha: float = 16.0,
               targets: tuple = DEFAULT_TARGETS, dropout: float = 0.0) -> nn.Module:
    """Replace every targeted Linear with a LoraLinear wrapper and freeze the
    entire base (all non-adapter params). Modifies in place."""
    for p in model.parameters():
        p.requires_grad_(False)
    for name, m in list(model.named_modules()):
        if isinstance(m, nn.Linear) and name.split(".")[-1] in targets \
                and not isinstance(m, LoraLinear):
            parent = model
            path = name.split(".")
            for part in path[:-1]:
                parent = getattr(parent, part)
            setattr(parent, path[-1], LoraLinear(m, r, alpha, dropout))
    return model


def lora_trainable_params(model: nn.Module, extra: tuple = ()):
    """Adapter params + any named extra (e.g. ('head',) when training new
    special-token embeddings). Everything else stays frozen."""
    params = []
    for name, p in model.named_parameters():
        if p.requires_grad or any(k in name for k in extra):
            p.requires_grad_(True)  # thaw extras
            params.append(p)
    return params


def merge_lora(model: nn.Module) -> nn.Module:
    """Fold A@B into the base weights, drop the adapters. Output is a plain
    Transformer — same logits (modulo fp error), full-speed inference."""
    for name, m in list(model.named_modules()):
        if isinstance(m, LoraLinear):
            base = m.base
            with torch.no_grad():
                base.weight.data = base.weight.data + (m.alpha / m.r) * (m.B @ m.A)
            parent = model
            path = name.split(".")
            for part in path[:-1]:
                parent = getattr(parent, part)
            setattr(parent, path[-1], base)
    return model


def count_lora_params(model: nn.Module) -> tuple[int, int]:
    """(trainable adapter params, frozen base params)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen
