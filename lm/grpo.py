"""GRPO / RLVR (module 10): Group Relative Policy Optimization (DeepSeek-R1) and
RL with verifiable rewards (RLVR) — the 2025 replacement for PPO in most LLM labs.

Why GRPO: PPO needs a critic (value model) trained alongside the policy — 2x memory,
hard to stabilize. GRPO replaces the advantage estimator with *group statistics*:
sample G completions per prompt, reward them, normalize within the group:

    A_i = (r_i - mean(r_1..r_G)) / std(r_1..r_G)

Then a clipped policy update + KL penalty to a frozen reference:

    L = -min(rho * A, clip(rho, 1-eps, 1+eps) * A) + beta * KL(pi || pi_ref)

    rho = pi(y|x) / pi_old(y|x)   (importance ratio vs the roll-out policy)

RLVR: rewards come from *verifiable* rules (math answer check, code unit tests),
not a learned reward model. This is the core recipe of R1 and current reasoning
models: no reward hacking, dense signal, scales with compute.
"""
import re
import torch
import torch.nn.functional as F

from .model import Transformer


def group_advantages(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    """Rewards [N]; returns group-normalized advantages. N % group_size == 0.
    (G=1 degenerates to adv 0 — filtered upstream by filter_degenerate.)"""
    r = rewards.view(-1, group_size)
    if group_size > 1:
        mu, sd = r.mean(-1, keepdim=True), r.std(-1, keepdim=True)
    else:
        mu, sd = r, torch.ones_like(r)
    return ((r - mu) / (sd + 1e-6)).view(-1)


def completion_logps(model: Transformer, prompt: torch.Tensor, completion: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
    """Per-token log-probs of the completion region: [N, T_c]."""
    x = torch.cat([prompt, completion], dim=1)
    logits, _, _ = model(x[:, :-1])
    logp = F.log_softmax(logits, dim=-1)
    per_tok = torch.gather(logp, -1, x[:, 1:].unsqueeze(-1)).squeeze(-1)
    comp = per_tok[:, prompt.shape[1] - 1:]   # positions predicting completion tokens
    return comp * mask


def completion_logp(model: Transformer, prompt: torch.Tensor, completion: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
    """Mean per-token log-prob of the completion (masked region), one scalar
    per sequence — a summary statistic for logging and for sequence-level
    objectives like DPO.

    NOT the input to `grpo_loss`: importance ratios are per position, so that
    function needs `completion_logps` (plural, [N, T_c]) and rejects this
    shape."""
    per_tok = completion_logps(model, prompt, completion, mask)
    return per_tok.sum(-1) / mask.sum(-1).clamp(min=1)


def grpo_loss(model: Transformer, ref_model: Transformer, prompt: torch.Tensor,
              completion: torch.Tensor, mask: torch.Tensor, rewards: torch.Tensor,
              old_logp: torch.Tensor, group_size: int,
              eps: float = 0.2, beta: float = 0.04,
              dapo: bool = False, eps_high: float = 0.28,
              token_level: bool = False) -> tuple[torch.Tensor, dict]:
    """completion: [N, T_c], prompt broadcast to [N, T_p].

    old_logp: PER-TOKEN log-probs [N, T_c] under the rollout policy (get them
    from `completion_logps`). Per-token is required: the importance ratio is
    rho_t = pi(y_t|.) / pi_old(y_t|.) at each position, so comparing a token's
    log-prob against a whole-completion average is not a ratio of anything.

    Both variants share the clipped surrogate, per token:

        rho_t   = exp(logp_t - old_logp_t)
        surr_t  = min(rho_t * A, clip(rho_t, 1-eps_low, 1+eps_high) * A)

    and differ ONLY in how surr is normalized:
      token_level=False  GRPO: mean over tokens within a completion, then mean
                         over completions — every completion gets equal weight.
      token_level=True   DAPO: sum over all tokens / total token count — every
                         TOKEN gets equal weight, so long completions carry
                         proportionally more gradient (this is the fix for
                         length collapse, module 19).

    dapo=True raises the UPPER clip bound to 1+eps_high while the lower stays
    at 1-eps ("clip-higher", arXiv:2503.14476 §3.1). Both sides remain clipped:
    the asymmetry widens the trust region for low-probability tokens with
    positive advantage, it does not remove the trust region.
    """
    if old_logp.dim() != 2:
        raise ValueError(
            f"old_logp must be per-token [N, T_c], got shape {tuple(old_logp.shape)}. "
            "Compute it with lm.grpo.completion_logps(model, prompt, completion, mask) "
            "under torch.no_grad() at rollout time.")

    per_tok = completion_logps(model, prompt, completion, mask)
    adv = group_advantages(rewards, group_size)
    adv_tok = adv[:, None].expand_as(per_tok)

    ratio_tok = torch.exp(per_tok - old_logp)
    hi = 1 + (eps_high if dapo else eps)
    clipped = torch.clamp(ratio_tok, 1 - eps, hi)
    surr = torch.min(ratio_tok * adv_tok, clipped * adv_tok)

    denom = mask.sum(-1).clamp(min=1)
    if token_level:
        policy_loss = -(surr * mask).sum() / mask.sum().clamp(min=1)
    else:
        policy_loss = -((surr * mask).sum(-1) / denom).mean()

    with torch.no_grad():
        ref_per_tok = completion_logps(ref_model, prompt, completion, mask)
    # k3 estimator (Schulman), per token: exp(r) - r - 1 with r = log(pi_ref/pi).
    # Unbiased, always >= 0, much lower variance than the naive -logratio.
    r = ref_per_tok - per_tok
    kl_tok = torch.exp(r) - r - 1
    kl = ((kl_tok * mask).sum(-1) / denom).mean()

    logp = (per_tok * mask).sum(-1) / denom
    frac_clipped = ((ratio_tok != clipped).float() * mask).sum() / mask.sum().clamp(min=1)
    stats = {"logp": logp.mean().item(), "kl": kl.item(),
             "adv_std": adv.std().item(), "reward_mean": rewards.mean().item(),
             "frac_clipped": frac_clipped.item()}
    return policy_loss + beta * kl, stats


def filter_degenerate(rewards: torch.Tensor, group_size: int,
                      tol: float = 1e-6) -> torch.Tensor:
    """DAPO dynamic sampling: a group whose G rewards are all equal carries no
    advantage signal (adv ≈ 0). Returns a per-group keep mask [n_groups]."""
    r = rewards.view(-1, group_size)
    return r.std(dim=-1) > tol


def overlong_shaped_reward(reward: float, length: int, max_len: int,
                           is_correct: bool) -> float:
    """Overlong reward shaping: penalize a rollout that is BOTH wrong AND ran
    into the length limit, so "ramble until truncation" stops being free. A
    correct answer is never penalized for being long.

    Simplified vs DAPO §3.4, which applies a *soft* length-dependent penalty
    over a cache interval before max_len regardless of correctness; this
    hard-gated version is the same idea with one threshold (exercise 2)."""
    if not is_correct and length >= max_len:
        return -0.5
    return reward


# --- rule-based rewards (RLVR) ----------------------------------------------

def gsm8k_reward(completion_text: str, answer: str) -> float:
    """Extract the last number in the completion (GSM8K convention: '#### N')."""
    nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", completion_text)
    if not nums:
        return 0.0
    got = nums[-1].replace(",", "")
    want = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", answer)
    return 1.0 if want and float(got) == float(want[-1].replace(",", "")) else 0.0


def code_reward(completion_text: str, test_cases: list[tuple[str, str]]) -> float:
    """Run completion through tests in a subprocess; fraction passed (0/1 for now)."""
    import subprocess, tempfile, os
    code = completion_text + "\n\n" + "\n".join(
        f"assert repr({c}) == {r!r}, f'failed {c}'" for c, r in test_cases)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        subprocess.run(["python", path], capture_output=True, timeout=10, check=True)
        return 1.0
    except Exception:
        return 0.0
    finally:
        os.unlink(path)
