"""Classical RLHF: reward model + PPO (module 23).

The two-stage pipeline that InstructGPT/ChatGPT used before DPO/GRPO made
single-stage methods practical. Still load-bearing knowledge: PPO-with-KL is
the reference point every newer method (DPO, GRPO) is derived FROM — DPO is
literally the closed form of this objective.

Reward model: the LM's final hidden state + a scalar head, trained with the
Bradley-Terry objective on (prompt, chosen, rejected) triples.

PPO: policy + frozen ref + value model (critic). GAE advantages, clipped
policy update, KL penalty to the ref. This is the loop GRPO replaced the
critic with group statistics (module 10).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Transformer
from .dpo import get_logps


class RewardModel(nn.Module):
    """LM trunk + scalar head: r(x, y) = head(mean-pool(final hidden))."""

    def __init__(self, lm: Transformer, freeze_trunk: bool = False):
        super().__init__()
        self.lm = lm
        self.head = nn.Linear(lm.cfg.hidden, 1, bias=False)
        self.head.weight.data.normal_(mean=0.0, std=0.01)
        if freeze_trunk:
            for p in lm.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: [B, T] prompt+completion; mask: completion tokens.
        Returns per-row reward (mean-pooled hidden over the completion)."""
        hidden = self.lm(x)[0] if False else self._last_hidden(x)
        d = hidden.shape[-1]
        # pool over the completion span
        pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.head(pooled).squeeze(-1)

    def _last_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Run the trunk and return the final (pre-head) hidden states."""
        from .model import rms_norm
        B, T = x.shape
        f = self.lm.freqs[:T]
        freqs_cos = torch.cos(f)[None, :, None, :]
        freqs_sin = torch.sin(f)[None, :, None, :]
        h = self.lm.tok_emb(x)
        for block in self.lm.blocks:
            h, _, _ = block(h, freqs_cos, freqs_sin)
        return rms_norm(h, self.lm.norm_f)


def bradley_terry_loss(rm: RewardModel, x: torch.Tensor, mask: torch.Tensor,
                       margin: bool = False) -> tuple[torch.Tensor, dict]:
    """L = -log sigma(r_w - r_l). x: [2B, T] chosen-then-rejected."""
    rewards = rm(x, mask)
    B = x.shape[0] // 2
    diff = rewards[:B] - rewards[B:]
    loss = -F.logsigmoid(diff if not margin else diff).mean()
    acc = (diff > 0).float().mean().item()
    return loss, {"reward_margin": diff.mean().item(), "pref_accuracy": acc}


class ValueHead(nn.Module):
    """Critic: same trunk, value scalar head (separate from the policy)."""

    def __init__(self, lm: Transformer):
        super().__init__()
        self.lm = lm
        self.head = nn.Linear(lm.cfg.hidden, 1, bias=False)

    def _last_hidden(self, x):
        from .model import rms_norm
        B, T = x.shape
        f = self.lm.freqs[:T]
        freqs_cos = torch.cos(f)[None, :, None, :]
        freqs_sin = torch.sin(f)[None, :, None, :]
        h = self.lm.tok_emb(x)
        for block in self.lm.blocks:
            h, _, _ = block(h, freqs_cos, freqs_sin)
        return rms_norm(h, self.lm.norm_f)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Per-token values [B, T]."""
        hidden = self._last_hidden(x)
        v = self.head(hidden).squeeze(-1)
        return v * mask + (1 - mask.float()) * -1e9  # masked positions: no value


def gae(values: torch.Tensor, rewards: torch.Tensor, mask: torch.Tensor,
        gamma: float = 1.0, lam: float = 0.95) -> torch.Tensor:
    """Generalized Advantage Estimation (Schulman et al. 2015).
    values/rewards: [B, T] per-token; returns advantages [B, T]."""
    B, T = rewards.shape
    adv = torch.zeros_like(rewards)
    last = torch.zeros(B, device=rewards.device)
    for t in reversed(range(T)):
        next_val = values[:, t + 1] if t + 1 < T else 0.0
        done = (mask[:, t] == 0).float()
        delta = rewards[:, t] + gamma * next_val * (1 - done) - values[:, t]
        last = delta + gamma * lam * (1 - done) * last
        adv[:, t] = last
    return adv


def ppo_loss(policy_logps: torch.Tensor, old_logps: torch.Tensor,
             advantages: torch.Tensor, mask: torch.Tensor, eps: float = 0.2) -> torch.Tensor:
    """The clipped surrogate — the core of PPO (and, inherited, GRPO)."""
    ratio = torch.exp(policy_logps - old_logps)
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps)
    loss = -torch.min(ratio * advantages, clipped * advantages)
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def ppo_step(policy: Transformer, ref: Transformer, critic: ValueHead,
             x: torch.Tensor, mask: torch.Tensor, rewards: torch.Tensor,
             old_logps: torch.Tensor, beta: float = 0.04,
             clip: float = 0.2, vf_coef: float = 0.5,
             gamma: float = 1.0, lam: float = 0.95,
             policy_opt=None, critic_opt=None):
    """One PPO iteration on a rollout batch [B, T] (prompt+completion concat).
    Returns (policy_loss, value_loss, stats). Callers step the optimizers."""
    per_tok = policy_logps_sequence(policy, x, mask)          # [B, T-1]
    values = critic(x, mask)
    advantages = gae(values.detach(), rewards, mask, gamma, lam)[:, :-1]
    policy_loss = ppo_loss(per_tok, old_logps, advantages, mask[:, 1:], eps=clip)
    returns = advantages + values[:, :-1].detach()
    value_loss = ((values[:, :-1] - returns) ** 2 * mask[:, 1:]).sum() \
        / mask[:, 1:].sum().clamp(min=1)
    # KL to the frozen ref (k3 estimator on completion logps)
    with torch.no_grad():
        ref_logps = get_logps(ref, x, mask)
    kl = (torch.exp(ref_logps - get_logps(policy, x, mask))
          - (ref_logps - get_logps(policy, x, mask)) - 1).mean()
    total = policy_loss + vf_coef * value_loss + beta * kl
    return total, {"policy_loss": policy_loss.item(),
                   "value_loss": value_loss.item(), "kl": kl.item()}


def policy_logps_sequence(policy: Transformer, x: torch.Tensor,
                          mask: torch.Tensor) -> torch.Tensor:
    """Per-token log-probs of x under the policy, masked to the completion."""
    logits, _, _ = policy(x[:, :-1])
    logp = F.log_softmax(logits, dim=-1)
    per_tok = torch.gather(logp, -1, x[:, 1:].unsqueeze(-1)).squeeze(-1)
    return per_tok * mask[:, 1:]
