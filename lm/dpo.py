"""Direct Preference Optimization (module 10).

DPO converts RLHF's two-stage (reward model -> PPO) into one cross-entropy-ish
loss on pairs (y_w preferred over y_l):

    L = -log sigma(beta * [log pi(y_w|x)/pi_ref(y_w|x) - log pi(y_l|x)/pi_ref(y_l|x)])

Derivation: Rafailov et al. 2023 (arXiv:2305.18290). With Bradley-Terry reward
model r = beta log(pi/pi_ref), the optimal-policy theorem r = beta log(pi*/pi_ref)
is solved for the RL objective, giving this closed form.
"""
import torch
import torch.nn.functional as F

from .model import Transformer


def get_logps(model: Transformer, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean log-likelihood of tokens under `mask` per row."""
    logits, _, _ = model(x[:, :-1])
    logp = F.log_softmax(logits, dim=-1)
    targets = x[:, 1:]
    per_tok = torch.gather(logp, -1, targets.unsqueeze(-1)).squeeze(-1)
    m = mask[:, 1:]
    return (per_tok * m).sum(-1) / m.sum(-1).clamp(min=1)


def dpo_loss(model: Transformer, ref_model: Transformer, x: torch.Tensor,
             mask: torch.Tensor, beta: float = 0.1) -> tuple[torch.Tensor, dict]:
    """x: [2B, T]; first half chosen, second half rejected (interleaved rows)."""
    with torch.no_grad():
        ref_logps = get_logps(ref_model, x, mask)
    logps = get_logps(model, x, mask)
    B = x.shape[0] // 2
    chosen, rejected = logps[:B], logps[B:]
    ref_chosen, ref_rejected = ref_logps[:B], ref_logps[B:]
    pi_ratio = chosen - rejected
    ref_ratio = ref_chosen - ref_rejected
    losses = -F.logsigmoid(beta * (pi_ratio - ref_ratio))
    stats = {"chosen_logp": chosen.mean().item(), "rejected_logp": rejected.mean().item(),
             "reward_margin": (beta * (pi_ratio - ref_ratio)).mean().item()}
    return losses.mean(), stats


def orpo_loss(model: Transformer, x: torch.Tensor, mask: torch.Tensor,
              beta: float = 0.1) -> tuple[torch.Tensor, dict]:
    """ORPO (arXiv:2403.07691): preference alignment WITHOUT a reference model
    — an SFT NLL term (teaches chosen) + an odds-ratio penalty (depresses
    rejected relative to chosen):

    L = L_sft + beta * max(0, log odds(y_l) - log odds(y_w))
    odds(y) = p(y) / (1 - p(y))
    """
    logps = get_logps(model, x, mask)
    B = x.shape[0] // 2
    chosen, rejected = logps[:B], logps[B:]
    log_odds = lambda lp: lp - torch.log1p(-lp.exp().clamp(max=0.999))
    ratio = log_odds(rejected) - log_odds(chosen)
    # SFT term: NLL on chosen completions
    nll = -chosen.mean()
    loss = nll + beta * F.relu(ratio).mean()
    return loss, {"orpo_ratio": ratio.mean().item(), "nll": nll.item()}


def simpo_loss(model: Transformer, x: torch.Tensor, mask: torch.Tensor,
               gamma: float = 1.0, beta: float = 10.0) -> tuple[torch.Tensor, dict]:
    """SimPO (arXiv:2405.14734): no reference model AND no length penalty.
    Uses length-normalized logps and a target margin gamma between chosen and
    rejected likelihoods:

    L = -log sigma( beta/|y| · [logp_w − logp_l] − gamma )
    """
    logps = get_logps(model, x, mask)
    B = x.shape[0] // 2
    chosen, rejected = logps[:B], logps[B:]
    diff = beta * (chosen - rejected) - gamma
    loss = -F.logsigmoid(diff).mean()
    return loss, {"simpo_margin": diff.mean().item()}


def kto_loss(model: Transformer, ref_model: Transformer, x: torch.Tensor,
             mask: torch.Tensor, beta: float = 0.1, lambda_desired: float = 1.0,
             lambda_undesired: float = 1.0) -> tuple[torch.Tensor, dict]:
    """KTO (arXiv:2402.01306): per-example losses — no PREFERENCE PAIRS needed.
    Desirable examples are pulled up, undesirable pushed down, each against a
    reference point (the KL-scaled implicit reward r = beta·log(pi/pi_ref)):

    L = E_desired[ w · (1 - sigma(beta·r - ref)) ] + E_undesired[ w · sigma(beta·r - ref) ]
    where w = (1-p)/p with p = sigma(ref) is the weighting, ref ~ KL target.
    """
    with torch.no_grad():
        ref_logps = get_logps(ref_model, x, mask)
    logps = get_logps(model, x, mask)
    r = beta * (logps - ref_logps)
    ref = F.kl_div(logps.detach(), ref_logps, reduction="none",
                   log_target=True).sum(-1).mean()  # KL as the reference level
    B = x.shape[0] // 2
    desired, undesired = r[:B], r[B:]
    p_d = torch.sigmoid(ref)
    w_d = (1 - p_d) / p_d.clamp(min=1e-6)
    w_u = p_d / (1 - p_d).clamp(min=1e-6)
    l_desired = lambda_desired * (w_d.detach() * (1 - torch.sigmoid(desired - ref))).mean()
    l_undesired = lambda_undesired * (w_u.detach() * torch.sigmoid(undesired - ref)).mean()
    return l_desired + l_undesired, {"kto_desired": l_desired.item(),
                                     "kto_undesired": l_undesired.item()}


class PreferenceLosses:
    """Registry so runners can switch losses by name (module 23)."""
    LOSSES = {"dpo": dpo_loss, "orpo": orpo_loss, "simpo": simpo_loss, "kto": kto_loss}
