"""Llama-style decoder: RMSNorm, RoPE, GQA, SwiGLU, FlashAttention (via SDPA).

Written to be read top-to-bottom; this is the assembled version of modules 02-03.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """RMSNorm: normalize by root-mean-square, then apply a learned per-dim gain.
    Cheaper than LayerNorm (no mean-subtraction, no bias) and empirically equal or better."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10_000.0,
                          method: str = "none", scale: float = 1.0) -> torch.Tensor:
    """Rotary angles theta^(-2i/d). With method/scale: context-extension
    (module 03, YaRN). method: none | linear | yarn; scale = target/trained len."""
    if method == "linear":  # naive: slower frequencies by scale (quality drops)
        theta = theta * scale
    elif method == "yarn":  # NTK-aware + ramp blend (Peng et al.)
        theta = theta * scale ** (head_dim / (head_dim - 2))
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    if method == "yarn":
        i = torch.arange(0, head_dim, 2).float() / head_dim
        ramp = torch.clamp((scale * i - 1.0) / (32.0 - 1.0), 0.0, 1.0)  # beta_fast=32
        inv = (1 - ramp) * inv / scale + ramp * inv  # interp low dims, extrap high dims
    pos = torch.arange(max_seq_len).float()
    return torch.outer(pos, inv)  # [T, d/2]


def apply_rope(x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
    """Complex rotation: treat (x0, x1) pairs as real/imag, multiply by e^{i theta}.
    x: [B, H, T, D]; freqs: [1, 1, T, D/2]."""
    d = x.shape[-1]
    x_pair = x.reshape(*x.shape[:-1], d // 2, 2)
    x0, x1 = x_pair[..., 0], x_pair[..., 1]
    return torch.stack([x0 * freqs_cos - x1 * freqs_sin,
                        x0 * freqs_sin + x1 * freqs_cos], dim=-1).reshape_as(x)


def causal_args(causal: bool, q_len: int, kv_len: int, device) -> dict:
    """Arguments for SDPA that enforce causality for ANY (q_len, kv_len) pair.

    `is_causal=True` only means "causal" when q_len == kv_len; with a populated
    KV cache the query block sits at the END of the keys, so the flag would mask
    the wrong triangle. Single-token decode needs no mask at all (one query may
    see every cached key). A multi-token chunk against a non-empty cache — a
    chunked prefill — needs an explicit rectangular mask, which is the case that
    silently produces a model that peeks at its own future if you skip it."""
    if not causal or q_len == 1:
        return {}
    if q_len == kv_len:
        return {"is_causal": True}
    q_pos = torch.arange(kv_len - q_len, kv_len, device=device)[:, None]
    k_pos = torch.arange(kv_len, device=device)[None, :]
    return {"attn_mask": q_pos >= k_pos}


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads, self.n_kv_heads = cfg.n_heads, cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.causal = cfg.causal
        d = cfg.hidden
        self.q = nn.Linear(d, self.n_heads * self.head_dim, bias=False)
        self.k = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.o = nn.Linear(self.n_heads * self.head_dim, d, bias=False)
        # GQA: fewer kv heads than query heads; each kv head serves n_heads/n_kv_heads queries.

    def forward(self, x, freqs_cos, freqs_sin, cache=None):
        B, T, d = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v(x).view(B, T, self.n_kv_heads, self.head_dim)
        q, k = apply_rope(q, freqs_cos, freqs_sin), apply_rope(k, freqs_cos, freqs_sin)

        q = q.transpose(1, 2)                 # [B, H, T, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if cache is not None:                 # inference: append to cache, attend over all
            k_cache, v_cache = cache
            k = torch.cat([k_cache, k], dim=2)   # cache lives in kv-head space (GQA savings)
            v = torch.cat([v_cache, v], dim=2)
        new_cache = (k, v)  # always emit; discarded during training

        if self.n_kv_heads < self.n_heads:    # GQA: broadcast kv across query heads
            reps = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(reps, dim=1)
            v = v.repeat_interleave(reps, dim=1)

        # SDPA dispatches to FlashAttention kernels on Ampere+ when possible.
        y = F.scaled_dot_product_attention(q, k, v, **causal_args(
            self.causal, T, k.shape[2], x.device))
        y = y.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        return self.o(y), new_cache


class MLA(nn.Module):
    """Multi-head Latent Attention (DeepSeek-V2, module 17).

    Instead of caching K/V per head, cache ONE low-rank latent c_kv = W_kv(x);
    K = W_uk(c), V = W_uv(c) are recomputed from it each step. RoPE is
    decoupled: a small per-token rope latent c_kr carries the position signal.
    Cache per token: d_c + d_r instead of 2 * n_kv * head_dim.

    (V2's real trick — folding W_uk/W_uv into the output projection so the
    up-projections cost nothing at inference — is explained in the module.)"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.hidden
        self.n_heads, self.n_kv_heads = cfg.n_heads, cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.causal = cfg.causal
        self.d_c = cfg.kv_latent_dim
        self.d_r = cfg.rope_latent_dim
        assert self.d_r <= self.head_dim // 2 and self.d_r % 2 == 0
        self.wq = nn.Linear(d, self.n_heads * (self.head_dim - self.d_r), bias=False)
        self.wq_r = nn.Linear(d, self.n_heads * self.d_r, bias=False)
        self.wkv = nn.Linear(d, self.d_c, bias=False)
        self.wkr = nn.Linear(d, self.d_r, bias=False)
        # K's compressed part (rope part comes from the decoupled latent);
        # V is full-dim from the shared latent.
        self.wuk = nn.Linear(self.d_c, self.n_kv_heads * (self.head_dim - self.d_r),
                             bias=False)
        self.wuv = nn.Linear(self.d_c, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, d, bias=False)

    def forward(self, x, freqs_cos, freqs_sin, cache=None):
        B, T, d = x.shape
        c_kv = self.wkv(x)                                # [B, T, d_c]
        q_c = self.wq(x).view(B, T, self.n_heads, self.head_dim - self.d_r)
        q_r = self.wq_r(x).view(B, T, self.n_heads, self.d_r)

        # RoPE applies only to the decoupled rope slices
        f_c = freqs_cos[..., : self.d_r // 2]
        f_s = freqs_sin[..., : self.d_r // 2]
        q_r = apply_rope(q_r, f_c, f_s)
        # ONE shared rope slice per position, [B, T, 1, d_r]. It carries position,
        # which is head-independent, so it is cached once and broadcast to the
        # kv-heads at compute time — caching n_kv copies would multiply the very
        # number MLA exists to shrink.
        k_r = apply_rope(self.wkr(x)[:, :, None, :], f_c, f_s)

        def _heads(latent, kr):
            """Rebuild [B, H_kv, T, D] K and V from the cached latent + rope slice."""
            t = latent.shape[1]
            k_c = self.wuk(latent).view(B, t, self.n_kv_heads, self.head_dim - self.d_r)
            k_full = torch.cat([k_c, kr.expand(B, t, self.n_kv_heads, self.d_r)], dim=-1)
            v_full = self.wuv(latent).view(B, t, self.n_kv_heads, self.head_dim)
            return k_full.transpose(1, 2), v_full.transpose(1, 2)

        # [B, T, 1, d_r] -> [B, d_r, T] for storage. squeeze-then-transpose, NOT
        # transpose-then-reshape: reshaping [B, 1, T, d_r] into [B, d_r, T] keeps
        # the elements in the wrong order (it interleaves time and feature).
        kr_store = k_r.squeeze(2).transpose(1, 2)    # [B, d_r, T]

        if cache is not None:
            ckv_c, kr_c = cache                      # [B, d_c, T], [B, d_r, T]
            c_kv = torch.cat([ckv_c, c_kv.transpose(1, 2)], dim=2)
            kr_all = torch.cat([kr_c, kr_store], dim=2)
            # rebuild k/v from the FULL cached latent (kv-head space)
            k, v = _heads(c_kv.transpose(1, 2), kr_all.transpose(1, 2)[:, :, None, :])
            new_cache = (c_kv, kr_all)               # both [B, feat, T_all]
        else:
            k, v = _heads(c_kv, k_r)
            # store cache as [B, feat, T] (shared position logic reads shape[2])
            new_cache = (c_kv.transpose(1, 2), kr_store)
        q = torch.cat([q_c, q_r], dim=-1).transpose(1, 2)  # [B, H, T, D]

        if self.n_kv_heads < self.n_heads:
            reps = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(reps, dim=1)
            v = v.repeat_interleave(reps, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, **causal_args(
            self.causal, T, k.shape[2], x.device))
        y = y.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        return self.wo(y), new_cache


class MLP(nn.Module):
    """SwiGLU: down(silu(gate(x)) * up(x)).

    Width: SwiGLU needs THREE d x h matrices where a GELU MLP needs two, so
    naively reusing h = 4d makes the block 1.5x bigger (12d^2 vs 8d^2). Llama
    keeps the parameter count matched by shrinking the hidden dim to 8d/3
    (= 2/3 of 4d), rounded up to a hardware-friendly multiple — that is what
    "same loss at ~2/3 the params" in module 03 refers to. cfg.ffn_hidden
    implements exactly that rule."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d, h = cfg.hidden, cfg.ffn_hidden
        self.gate = nn.Linear(d, h, bias=False)
        self.up = nn.Linear(d, h, bias=False)
        self.down = nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoE(nn.Module):
    """Mixture-of-Experts (module 15): each token routes to top-k of E expert
    MLPs. Active FLOPs = dense * k/E; load-balancing aux loss keeps usage uniform.

    Router: softmax(topk(gate(x))). Aux loss (Switch/GShard): E * sum(f_i * p_i),
    f_i = routed-token fraction, p_i = mean gate prob. Computed in fp32."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.hidden
        self.top_k = cfg.n_experts_active
        self.n_experts = cfg.n_experts
        self.gate = nn.Linear(d, self.n_experts, bias=False)
        self.experts = nn.ModuleList([MLP(cfg) for _ in range(self.n_experts)])

    def forward(self, x):
        B, T, d = x.shape
        flat = x.view(-1, d)                       # [N, d]
        logits = self.gate(flat)                   # [N, E]
        topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(topk_vals, dim=-1)
        # load-balancing aux loss, fp32 for stability
        f = torch.zeros_like(logits, dtype=torch.float32).scatter_add(
            -1, topk_idx, torch.ones_like(weights, dtype=torch.float32)) / flat.shape[0]
        p = logits.float().softmax(-1).mean(0)
        aux = self.n_experts * (f * p).sum()

        out = torch.zeros_like(flat)
        for e in range(self.n_experts):
            sel = topk_idx == e                      # [N, k]
            sel_mask, _ = sel.max(-1)
            idx = sel_mask.nonzero().squeeze(-1)
            if idx.numel() == 0:
                continue
            y = self.experts[e](flat[idx])          # [n_e, d]
            w = weights[sel][:, None]               # [n_e, 1]
            out.index_add_(0, idx, y * w)
        return out.view(B, T, d), aux


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, gradient_checkpointing: bool = False):
        super().__init__()
        self.norm1 = nn.Parameter(torch.ones(cfg.hidden))
        self.attn = MLA(cfg) if cfg.use_mla else Attention(cfg)
        self.norm2 = nn.Parameter(torch.ones(cfg.hidden))
        self.mlp = MoE(cfg) if cfg.n_experts > 0 else MLP(cfg)
        self.is_moe = cfg.n_experts > 0
        self.gckpt = gradient_checkpointing

    def _forward(self, x, freqs_cos, freqs_sin, cache=None):
        h, new_cache = self.attn(rms_norm(x, self.norm1), freqs_cos, freqs_sin, cache)
        x = x + h
        if self.is_moe:
            out, aux = self.mlp(rms_norm(x, self.norm2))
            x = x + out
        else:
            x = x + self.mlp(rms_norm(x, self.norm2))
            aux = torch.zeros((), device=x.device)  # tensor so checkpoint can graph it
        return x, new_cache, aux

    def forward(self, x, freqs_cos, freqs_sin, cache=None):
        if self.gckpt and self.training:
            # drop the stored activations, recompute in backward: memory O(L) not O(L*T)
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, freqs_cos, freqs_sin, cache, use_reentrant=False)
        return self._forward(x, freqs_cos, freqs_sin, cache)


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig, gradient_checkpointing: bool = False):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden)
        self.blocks = nn.ModuleList([Block(cfg, gradient_checkpointing)
                                     for _ in range(cfg.n_layers)])
        self.norm_f = nn.Parameter(torch.ones(cfg.hidden))
        self.head = nn.Linear(cfg.hidden, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.tok_emb.weight
        self.register_buffer("freqs", precompute_rope_freqs(
            cfg.head_dim, cfg.max_seq_len, cfg.rope_theta,
            cfg.rope_method, cfg.rope_scale), persistent=False)
        # Multi-Token Prediction (DeepSeek-V3, module 15): auxiliary heads that
        # predict tokens 2..mtp_depth+1 ahead from the SAME final hidden state.
        # The full V3 version uses a separate small transformer per depth;
        # this linear variant teaches the mechanism.
        self.mtp_heads = nn.ModuleList(
            [nn.Linear(cfg.hidden, cfg.vocab_size, bias=False)
             for _ in range(cfg.mtp_depth)]) if cfg.mtp_depth > 0 else None
        self.apply(self._init)

    def _init(self, m):
        """Residual-scaled init: block outputs stay small relative to the stream,
        so the residual stream's variance grows ~O(1) rather than ~O(L) with depth.

        Note this applies the 1/sqrt(2L) factor to EVERY Linear, not only to the
        residual output projections (o, down) the way GPT-2/nanoGPT do. It is the
        more conservative choice — uniformly smaller init — and it is what muP's
        width correction in train.py::mup_scale is layered on top of. If you
        compare against a nanoGPT-style baseline, this is the difference."""
        if isinstance(m, nn.Linear):
            std = self.cfg.init_std / math.sqrt(2 * self.cfg.n_layers)
            nn.init.normal_(m.weight, std=std)
        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=self.cfg.init_std)

    def forward(self, idx, targets=None, cache=None, prefix_embeds=None):
        """idx: [B, T] int64. Returns logits, loss (if targets), and per-layer
        KV caches (always, so incremental decoding can resume from them).

        prefix_embeds: [B, P, d] optional non-text embeddings prepended to the
        sequence (vision patches in a VLM, module 21). Targets shift by P and
        are padded with -100 (ignored) over the prefix."""
        B, T = idx.shape
        P = prefix_embeds.shape[1] if prefix_embeds is not None else 0
        # position of the first token in this chunk: cache length when resuming
        start = cache[0][0].shape[2] if cache is not None else 0
        f = self.freqs[start:start + T + P]                    # [T+P, D/2] angles
        freqs_cos = torch.cos(f)[None, :, None, :]             # [1, T+P, 1, D/2]
        freqs_sin = torch.sin(f)[None, :, None, :]

        x = self.tok_emb(idx)
        if prefix_embeds is not None:
            x = torch.cat([prefix_embeds, x], dim=1)
            if targets is not None:
                targets = torch.cat(
                    [targets.new_full((B, P), -100), targets], dim=1)
        new_caches = []
        aux_loss = 0.0
        for i, block in enumerate(self.blocks):
            c = cache[i] if cache is not None else None
            x, c_i, aux = block(x, freqs_cos, freqs_sin, c)
            aux_loss = aux_loss + aux
            new_caches.append(c_i)
        x = rms_norm(x, self.norm_f)
        logits = self.head(x)

        loss = None
        if targets is not None:
            ce = F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size), targets.reshape(-1))
            loss = ce
            if self.cfg.z_loss_coef > 0:  # PaLM logit-z loss: keeps logits bounded
                z = torch.logsumexp(logits.float(), dim=-1).square().mean()
                loss = loss + self.cfg.z_loss_coef * z
            if self.mtp_heads is not None:  # MTP: predict future tokens too
                # targets[t] is already the token AFTER position t, so head j
                # (j = 1, 2, ...) predicting j+1 tokens ahead must line up
                # x[t] with targets[t+j] — not targets[t+j+1].
                for j, head in enumerate(self.mtp_heads, start=1):
                    tgt = targets[:, j:]
                    logits_k = head(x[:, :-j]).reshape(-1, self.cfg.vocab_size)
                    loss = loss + 0.1 * F.cross_entropy(logits_k, tgt.reshape(-1))
            if self.cfg.n_experts > 0:
                loss = loss + self.cfg.moe_aux_loss_coef * aux_loss
        return logits, loss, new_caches
