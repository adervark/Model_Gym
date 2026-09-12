"""Sparse Autoencoders on the residual stream (module 20).

An SAE decomposes residual-stream activations h ∈ R^d into a sparse sum of
learned feature directions:

    h ≈ Σ_i a_i d_i,   a = ReLU(W_enc (h - b_pre) + b_enc),  a sparse

Trained with reconstruction MSE + L1 sparsity penalty. The decoder rows d_i
are the model's "features"; you read them with the logit lens (module 13).
This is the machinery behind Anthropic's interpretability results and
Gemma Scope.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Transformer, rms_norm


class SparseAutoencoder(nn.Module):
    def __init__(self, d: int, f: int, k: int = 0):
        """d: residual dim. f: feature count (f >> d). k: top-k sparsity
        (0 = plain ReLU + L1)."""
        super().__init__()
        self.d, self.f, self.k = d, f, k
        self.W_enc = nn.Parameter(torch.randn(f, d) * 0.01)
        self.b_enc = nn.Parameter(torch.zeros(f))
        self.W_dec = nn.Parameter(torch.randn(f, d) * 0.01)
        self.b_pre = nn.Parameter(torch.zeros(d))
        with torch.no_grad():  # unit-norm decoder rows (standard practice)
            self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True).clamp(min=1e-8)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = x - self.b_pre
        pre = F.relu(h @ self.W_enc.T + self.b_enc)
        if self.k > 0:
            _, idx = torch.topk(pre, self.k, dim=-1)
            mask = torch.zeros_like(pre).scatter_(1, idx, 1.0)
            pre = pre * mask
        return pre

    def forward(self, x: torch.Tensor):
        acts = self.encode(x)
        recon = acts @ self.W_dec + self.b_pre
        return recon, acts

    def loss(self, x: torch.Tensor, l1_coef: float = 1e-3):
        recon, acts = self.forward(x)
        mse = ((x - recon) ** 2).mean()
        l1 = acts.abs().sum(-1).mean() * l1_coef
        return mse + l1, {"mse": mse.item(), "l1": l1.item()}


@torch.no_grad()
def collect_activations(model: Transformer, ids: torch.Tensor,
                        layer: int) -> torch.Tensor:
    """Residual-stream activations after block `layer`, from token ids [B, T]."""
    model.eval()
    B, T = ids.shape
    f = model.freqs[:T]
    freqs_cos = torch.cos(f)[None, :, None, :]
    freqs_sin = torch.sin(f)[None, :, None, :]
    h = model.tok_emb(ids)
    for i, block in enumerate(model.blocks):
        h, _, _ = block(h, freqs_cos, freqs_sin)
        if i == layer:
            return h.detach().flatten(0, 1)
    raise ValueError(f"layer {layer} out of range")


@torch.no_grad()
def feature_top_tokens(sae: SparseAutoencoder, model: Transformer,
                       feature: int, k: int = 5) -> list[tuple[str, float]]:
    """What does feature i represent? Project its decoder direction through the
    unembedding (logit lens, module 13)."""
    from .tokenizer import get_tokenizer
    enc = get_tokenizer()
    logits = model.head(sae.W_dec[feature])       # [V]
    probs = F.softmax(logits, dim=-1)
    top = torch.topk(probs, k)
    return [(enc.decode([t.item()]), float(p)) for p, t in zip(top.values, top.indices)]


@torch.no_grad()
def dead_features(sae: SparseAutoencoder, acts: torch.Tensor) -> int:
    """Features that never activate on a batch — the standard health metric."""
    return int((acts.sum(0) == 0).sum())


def train_sae(sae: SparseAutoencoder, batches, steps: int = 1000,
              lr: float = 1e-3, l1_coef: float = 1e-3, log_every: int = 200):
    """batches: iterator yielding activation tensors [B, d]."""
    opt = torch.optim.AdamW(sae.parameters(), lr=lr)
    it = iter(batches)
    for step in range(steps):
        x = next(it)
        loss, stats = sae.loss(x, l1_coef)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % log_every == 0:
            print(f"step {step:>5} | mse {stats['mse']:.4f} | l1 {stats['l1']:.4f}")
    return sae
