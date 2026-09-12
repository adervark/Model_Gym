"""Masked Diffusion Language Models (module 18): the non-autoregressive frontier.

Autoregression generates left-to-right, one token at a time. Diffusion LMs
(Mercury, LLaDA) instead train to denoise: mask a random fraction of tokens,
let a BIDIRECTIONAL model predict them, and generate by iterative unmasking.
Properties: whole-sequence editing, parallel decoding, ~10x faster sampling.

Objective (absorbing-state discrete diffusion, MDLM):

    L = -E_{t~U, m~Bern(t)} [ log p(x | x_masked) on masked positions ]

t = corruption level (mask probability). At t=1 nothing is known; at t=0
nothing is masked. The same model handles every t.

Generation: x = all [MASK]; for i in 1..T: predict p(x_i | x), unmask the
top-k most-confident positions (k = L*(1 - i/T)), repeat until clean.

Mask token: we use the eot id (50256) for the absorbing state. Real MDLM
adds a dedicated [MASK] token to the vocab — exercise 2.
"""
import torch
import torch.nn.functional as F

from .model import Transformer

MASK_ID = 50256  # gpt2 <|endoftext|> as the absorbing state (see module 18)


def mdlm_loss(model: Transformer, x: torch.Tensor, t_min: float = 0.0,
              t_max: float = 1.0, mask_id: int = MASK_ID):
    """One MDLM training step. x: [B, T] clean token ids.
    Returns (loss, mask) — loss is CE on masked positions only."""
    B, T = x.shape
    t = t_min + (t_max - t_min) * torch.rand(B, device=x.device)
    mask = torch.rand(B, T, device=x.device) < t[:, None]
    x_m = torch.where(mask, torch.full_like(x, mask_id), x)
    logits, _, _ = model(x_m)
    logits = logits.reshape(-1, logits.size(-1))
    targets = x.reshape(-1)
    flat_mask = mask.reshape(-1)
    loss = F.cross_entropy(logits[flat_mask], targets[flat_mask])
    return loss, mask


@torch.no_grad()
def mdlm_generate(model: Transformer, L: int, steps: int = 32,
                  temperature: float = 1.0, mask_id: int = MASK_ID,
                  prompt: torch.Tensor | None = None) -> torch.Tensor:
    """Iterative unmasking: start from all [MASK], reveal the most confident
    tokens over `steps` rounds. prompt: [1, P] prefix kept clean throughout."""
    dev = next(model.parameters()).device
    model.eval()
    x = torch.full((1, L), mask_id, dtype=torch.long, device=dev)
    if prompt is not None:
        x[0, : prompt.shape[1]] = prompt[0].to(dev)
    for i in range(1, steps + 1):
        logits, _, _ = model(x)
        probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
        # confidence = max prob at each masked position
        conf, preds = probs.max(dim=-1)                     # [1, L]
        conf = conf.masked_fill(x != mask_id, -1.0)
        n_known = (x != mask_id).sum().item()
        if i == steps:
            n_reveal = (x == mask_id).sum().item()          # final step: reveal all
        else:
            n_reveal = max(1, int(L * (1 - i / steps))) - n_known
        if n_reveal <= 0:
            continue
        top = conf[0].topk(min(n_reveal, (x == mask_id).sum().item()))
        x[0, top.indices] = preds[0, top.indices]
    return x[0]
