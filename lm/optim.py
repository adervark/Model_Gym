"""Muon optimizer (module 19): momentum + Newton-Schulz orthogonalization.

Keller Jordan's Muon (used to train Moonlight, 2025): instead of AdamW's
per-parameter variance normalization, Muon keeps a momentum buffer and
orthogonalizes it every step — the update is a proper rotation+scaling of
the weight space. It trained a 16B model with <50% of AdamW's FLOPs.

Recipe: Muon for 2D weight matrices (lr ~0.02 — an order of magnitude above
AdamW's), plain AdamW for 1D params (norm gains, biases). Newton-Schulz
polynomial iteration approximates polar decomposition G -> orthogonal(G).
"""
import torch

_NS_A, _NS_B, _NS_C = 3.4445, -4.7750, 2.0315


def newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate orthogonalization via the Newton-Schulz quintic — faithful
    port of Muon's zeropower (Keller Jordan's Muon post; scaled up in
    arXiv:2502.16982): bf16 arithmetic, quintic coefficients chosen to
    maximize the slope at zero.

    Honest caveat (see module 19): the quintic does NOT drive arbitrary
    matrices to exact orthogonality — it maps the empirical singular-value
    distribution of momentum buffers toward a common scale. The course
    implementation keeps the published constants + bf16, so behavior matches
    the reference."""
    X = G.bfloat16()
    if X.shape[0] > X.shape[1]:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        X = _NS_A * X + (_NS_B * A + _NS_C * A @ A) @ X
    if G.shape[0] > G.shape[1]:
        X = X.T
    return X.float()


class Muon(torch.optim.Optimizer):
    def __init__(self, muon_params, adam_params, lr: float = 0.02,
                 momentum: float = 0.95, weight_decay: float = 0.01,
                 ns_steps: int = 5, adam_lr: float = 3e-4,
                 betas: tuple = (0.9, 0.95), eps: float = 1e-8):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        ns_steps=ns_steps, adam_lr=adam_lr, betas=betas, eps=eps)
        super().__init__([{"params": list(muon_params), "muon": True},
                          {"params": list(adam_params), "muon": False}], defaults)
        # base_lr bookkeeping so the outer training loop's schedule applies
        for g in self.param_groups:
            g["base_lr"] = lr if g["muon"] else adam_lr
            g["adam_lr_base"] = adam_lr

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            if group["muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if "buf" not in state:
                        state["buf"] = torch.zeros_like(p)
                    buf = state["buf"].mul_(group["momentum"]).add_(p.grad)
                    ortho = newton_schulz(buf, group["ns_steps"])
                    # decoupled weight decay (Loshchilov style)
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(ortho, alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if "step" not in state:
                        state["step"] = 0
                        state["m"] = torch.zeros_like(p)
                        state["v"] = torch.zeros_like(p)
                    state["step"] += 1
                    b1, b2 = group["betas"]
                    m = state["m"].mul_(b1).add_(p.grad, alpha=1 - b1)
                    v = state["v"].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                    m_hat = m / (1 - b1 ** state["step"])
                    v_hat = v / (1 - b2 ** state["step"])
                    p.mul_(1 - group["adam_lr"] * group["weight_decay"])
                    p.addcdiv_(m_hat, v_hat.sqrt().add_(group["eps"]),
                               value=-group["adam_lr"])


def split_muon_adam(model) -> tuple[list, list]:
    """2D weight matrices -> Muon; 1D params (norms, biases) -> AdamW."""
    muon, adam = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (muon if p.ndim >= 2 else adam).append(p)
    return muon, adam


class Lion(torch.optim.Optimizer):
    """Lion (arXiv:2302.06675): sign-based update — only signs of a
    momentum-weighted gradient, no magnitude. ~2x memory savings vs AdamW
    (one state instead of two); slightly worse convergence, fine for
    fine-tuning. lr ~ 3-10x SMALLER than AdamW (signs saturate)."""

    def __init__(self, params, lr: float = 1e-4, betas: tuple = (0.9, 0.99),
                 weight_decay: float = 0.1):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__([{"params": list(params), "base_lr": lr}], defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Algorithm 2 of the paper, in order:

            u_t = sign(b1 * m_{t-1} + (1 - b1) * g_t)     # interpolate, THEN sign
            p_t = p_{t-1} - lr * (u_t + wd * p_{t-1})
            m_t = b2 * m_{t-1} + (1 - b2) * g_t           # b2 updates the state

        The two easy ways to get this wrong: taking the sign before the
        interpolation (which yields a blend of signs, not a sign), and using
        one beta for both roles. The update must be exactly +-1 per coordinate
        — that is the entire memory/behaviour argument for Lion."""
        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr, wd = group["lr"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(p)
                g, m = p.grad, state["m"]
                update = m.mul(b1).add_(g, alpha=1 - b1).sign_()   # u_t, out-of-place on m
                p.mul_(1 - lr * wd)                                # decoupled weight decay
                p.add_(update, alpha=-lr)
                m.mul_(b2).add_(g, alpha=1 - b2)                   # state uses b2


class Adafactor(torch.optim.Optimizer):
    """Adafactor (arXiv:1804.04235, PaLM's optimizer): O(1) extra memory per
    tensor via FACTORED second moments (row/col statistics instead of a full
    matrix). Used when AdamW's 8 bytes/param doesn't fit. No beta1 momentum by
    default; update magnitudes are clamped to the root-mean-square scale."""

    def __init__(self, params, lr: float = 1e-2, beta2: float = 0.999,
                 weight_decay: float = 0.0, eps: tuple = (1e-30, 1e-3),
                 clip_thresh: float = 1.0):
        defaults = dict(lr=lr, beta2=beta2, weight_decay=weight_decay,
                        eps=eps, clip_thresh=clip_thresh)
        super().__init__([{"params": list(params), "base_lr": lr}], defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "row" not in state:
                    state["row"] = torch.zeros(p.shape[0], device=p.device)
                    state["col"] = torch.zeros(p.shape[1] if p.ndim > 1 else 1,
                                              device=p.device) if p.ndim > 1 else None
                row, col = state["row"], state["col"]
                b2 = group["beta2"]
                g2 = g.float().square()
                if p.ndim >= 2:
                    row.mul_(b2).add_(g2.mean(1), alpha=1 - b2)
                    col.mul_(b2).add_(g2.mean(0), alpha=1 - b2)
                    v_hat = row[:, None] * col[None, :] / row.mean().clamp(min=1e-12)
                else:
                    row.mul_(b2).add_(g2, alpha=1 - b2)
                    v_hat = row
                # RMS-normalized update with relative step size + cap
                rms = (v_hat / p.numel()).sqrt()
                update = g.float() / (rms + group["eps"][0])
                p_rms = p.float().square().mean().sqrt()
                update_rms = update.square().mean().sqrt()
                scale = min(group["eps"][1] * p_rms, group["clip_thresh"])
                update = update / (update_rms + 1e-12) * scale
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.to(p.dtype), alpha=-group["lr"])
