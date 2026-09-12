"""Mamba: selective state-space models (module 25).

The other major architecture family. Where attention is O(T^2) in memory and
O(T) sequential steps in decode, a state-space model is O(T) in both: a
recurrent state h_t = A_t h_{t-1} + B_t x_t summarizes the whole past, and
y_t = C_t h_t reads out. The "selective" part (Mamba/S6): A, B, C depend on
the INPUT — the model chooses what to remember, per token. That selectivity
is what makes Mamba competitive with attention while staying linear.

This is a READABLE implementation (sequential scan, no CUDA kernels) —
correct math, course speed. Production Mamba (linear-attention hybrids,
Zamba/Jamba/Griffin/Gemma-3) runs fused parallel scans.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class MambaBlock(nn.Module):
    """One selective-SSM layer (Gu & Dao 2023, simplified to the essentials).

    x -> conv1d (local context) -> SiLU -> selective SSM:
        dt  = softplus(x W_dt + b_dt)          per-token time step
        B_t = x W_B,  C_t = x W_C              input-dependent in/out maps
        A   = -exp(A_log)                      learned decay (fixed, input-independent)
        h_t = exp(dt·A)·h_{t-1} + dt·B_t·x_t   discretized recurrence
        y_t = C_t·h_t
    -> + gated SiLU(x) -> output projection
    """

    def __init__(self, d: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2):
        super().__init__()
        self.d, self.d_state = d, d_state
        self.in_proj = nn.Linear(d, expand * d, bias=False)
        self.conv1d = nn.Conv1d(expand * d, expand * d, d_conv,
                                groups=expand * d, padding=d_conv - 1)
        self.x_proj = nn.Linear(expand * d, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(expand * d, expand * d, bias=True)
        nn.init.constant_(self.dt_proj.bias, 0.01)  # positive initial dt
        # A: d_state decay rates, per inner-dim block of the expanded hidden
        A = torch.arange(1, d_state + 1).float().unsqueeze(0)
        self.A_log = nn.Parameter(torch.log(A).repeat(expand * d, 1))
        self.out_proj = nn.Linear(expand * d, d, bias=False)
        self.norm = nn.LayerNorm(d)

    def _ssm(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, d_exp]. Sequential selective scan (readable; production
        uses a parallel associative scan)."""
        B, L, D = x.shape
        dt = F.softplus(self.dt_proj(x))                     # [B, L, D]
        proj = self.x_proj(x)                                # [B, L, 2·N + 1]
        Bm = proj[..., :self.d_state]                        # [B, L, N]
        C = proj[..., self.d_state:2 * self.d_state]
        A = -torch.exp(self.A_log)                           # [D, N]
        # broadcast dt over the N state dims
        dA = torch.exp(dt.unsqueeze(-1) * A)                 # [B, L, D, N]
        dB = (dt.unsqueeze(-1) * Bm.unsqueeze(2))            # [B, L, D, N]
        h = torch.zeros(B, D, self.d_state, device=x.device)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
            ys.append((C[:, t].unsqueeze(1) * h).sum(-1))    # [B, D]
        return torch.stack(ys, dim=1)                        # [B, L, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.in_proj(x)                                   # [B, L, 2d] gate
        z_conv = self.conv1d(z.transpose(1, 2))[..., :x.shape[1]].transpose(1, 2)
        y = self._ssm(z_conv)                                 # SSM on the raw proj
        out = self.out_proj(y * F.silu(z))                    # gate with z
        return x + out


class MambaModel(nn.Module):
    """A pure-Mamba language model: embeddings + blocks + norm + head.
    Trains with the SAME loop as the transformer (module 01's loop) — swap
    the architecture, keep everything else."""

    def __init__(self, cfg: ModelConfig, n_blocks: int = 4, d_state: int = 16):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden)
        self.blocks = nn.ModuleList([MambaBlock(cfg.hidden, d_state=d_state)
                                     for _ in range(n_blocks)])
        self.norm_f = nn.LayerNorm(cfg.hidden)
        self.head = nn.Linear(cfg.hidden, cfg.vocab_size, bias=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            std = self.cfg.init_std / math.sqrt(2 * len(self.blocks))
            nn.init.normal_(m.weight, std=std)
        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=self.cfg.init_std)

    def forward(self, idx, targets=None):
        """idx: [B, T]. No attention mask needed — causality is structural
        (the recurrence only sees the past)."""
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size),
                                   targets.reshape(-1))
        return logits, loss


class MambaHybrid(nn.Module):
    """The pattern frontier models actually use (Zamba/Jamba/Griffin/Gemma-3):
    interleave Mamba and attention blocks — attention at the boundaries,
    linear recurrence in the middle where long-range memory is cheap."""

    def __init__(self, cfg: ModelConfig, n_mamba: int = 3, n_attn: int = 2):
        super().__init__()
        from .model import Block as AttnBlock
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden)
        self.blocks = nn.ModuleList()
        for i in range(n_mamba + n_attn):
            if i % 2 == 0:
                self.blocks.append(MambaBlock(cfg.hidden))
            else:
                self.blocks.append(AttnBlock(cfg))
        self.norm_f = nn.LayerNorm(cfg.hidden)
        self.head = nn.Linear(cfg.hidden, cfg.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        f = torch.cos(self._freqs()[:T])[None, :, None, :]
        fs = torch.sin(self._freqs()[:T])[None, :, None, :]
        for block in self.blocks:
            if isinstance(block, MambaBlock):
                x = block(x)
            else:
                x, _, _ = block(x, f, fs)
        x = self.norm_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size),
                                   targets.reshape(-1))
        return logits, loss

    def _freqs(self):
        from .model import precompute_rope_freqs
        return precompute_rope_freqs(self.cfg.head_dim, self.cfg.max_seq_len,
                                     self.cfg.rope_theta)
