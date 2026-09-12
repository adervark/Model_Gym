"""Post-training runners: shared by scripts/* and `python -m lm.cli sft|dpo|grpo`."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from .config import ModelConfig
from .model import Transformer
from .sft import build_sft_sample, sft_step
from .dpo import dpo_loss
from .tokenizer import get_tokenizer

PAIRS = [
    ("What is 2 + 2?", "2 + 2 equals 4.", "I don't know, maybe ask someone else."),
    ("Name three planets.", "Mercury, Venus, and Earth are three planets.", "Planets are things in space and stuff."),
    ("What is the capital of France?", "The capital of France is Paris.", "France is a country in Europe."),
    ("How many legs does a spider have?", "A spider has eight legs.", "Spiders are bugs with legs."),
]

PROMPTS = [
    ("What is 2 + 2?", "4"), ("What is 7 + 5?", "12"), ("What is 15 - 6?", "9"),
    ("What is 3 * 8?", "24"), ("What is 20 / 4?", "5"), ("What is 9 + 14?", "23"),
    ("What is 50 - 27?", "23"), ("What is 6 * 7?", "42"),
]


def load_base(path: str, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["model_cfg"])
    model = Transformer(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    return model, cfg


def run_lora(ckpt: str, data: str, out: str, steps: int, lr: float,
             batch_size: int, seq_len: int, r: int, alpha: float,
             device: str = "auto"):
    """LoRA SFT (module 14): frozen base + rank-r adapters on attention/MLP."""
    import json
    from .lora import apply_lora, lora_trainable_params, merge_lora, count_lora_params

    device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_base(ckpt, device)
    apply_lora(model, r=r, alpha=alpha)
    trainable, frozen = count_lora_params(model)
    print(f"[lora] r={r} alpha={alpha}: {trainable/1e6:.2f}M trainable / "
          f"{frozen/1e6:.1f}M frozen ({100*trainable/frozen:.2f}%)")

    enc = get_tokenizer()
    samples = [json.loads(l) for l in open(data)]
    batches = []
    for i in range(0, len(samples), batch_size):
        xs, masks = [], []
        for s in samples[i:i + batch_size]:
            x, m = build_sft_sample(enc, s["prompt"], s["response"], seq_len)
            xs.append(x); masks.append(m)
        batches.append((torch.stack(xs).to(device), torch.stack(masks).to(device)))

    opt = torch.optim.AdamW(lora_trainable_params(model), lr=lr, weight_decay=0.1)
    model.train()
    for step in range(steps):
        x, mask = batches[step % len(batches)]
        loss = sft_step(model, x, mask)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            print(f"step {step:>5} | loss {loss.item():.4f}")

    os.makedirs(out, exist_ok=True)
    # the real LoRA artifact: adapters only
    adapters = {k: v for k, v in model.state_dict().items()
                if k.split(".")[-1] in ("A", "B")}
    torch.save({"lora_state": adapters, "r": r, "alpha": alpha,
                "model_cfg": cfg.__dict__, "base_ckpt": ckpt, "step": steps},
               os.path.join(out, "adapters.pt"))
    # merged full checkpoint so lm.cli generate/eval work unchanged
    merge_lora(model)
    save_out(model, cfg, out, steps)
    print(f"adapters: {out}/adapters.pt | merged model: {out}/best.pt")


def save_out(model, cfg, out, step):
    os.makedirs(out, exist_ok=True)
    torch.save({"model": model.state_dict(), "model_cfg": cfg.__dict__,
                "step": step}, os.path.join(out, "best.pt"))
    print(f"saved to {out}/best.pt")


def run_sft(ckpt: str, data: str, out: str, steps: int, lr: float,
            batch_size: int, seq_len: int, device: str = "auto"):
    device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_base(ckpt, device)
    enc = get_tokenizer()
    import json
    samples = [json.loads(l) for l in open(data)]
    batches = []
    for i in range(0, len(samples), batch_size):
        xs, masks = [], []
        for s in samples[i:i + batch_size]:
            x, m = build_sft_sample(enc, s["prompt"], s["response"], seq_len)
            xs.append(x); masks.append(m)
        batches.append((torch.stack(xs).to(device), torch.stack(masks).to(device)))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    model.train()
    for step in range(steps):
        x, mask = batches[step % len(batches)]
        loss = sft_step(model, x, mask)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            print(f"step {step:>5} | loss {loss.item():.4f}")
    save_out(model, cfg, out, steps)


def run_dpo(ckpt: str, out: str, steps: int, lr: float, beta: float,
            seq_len: int, device: str = "auto"):
    device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_base(ckpt, device)
    ref, _ = load_base(ckpt, device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    enc = get_tokenizer()
    batches = []
    for prompt, chosen, rejected in PAIRS:
        x1, m1 = build_sft_sample(enc, prompt, chosen, seq_len)
        x2, m2 = build_sft_sample(enc, prompt, rejected, seq_len)
        batches.append((torch.stack([x1, x2]).to(device),
                        torch.stack([m1, m2]).to(device)))

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for step in range(steps):
        x, mask = batches[step % len(batches)]
        loss, stats = dpo_loss(model, ref, x, mask, beta=beta)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            print(f"step {step:>4} | loss {loss.item():.3f} | "
                  f"chosen lp {stats['chosen_logp']:.2f} | rejected lp {stats['rejected_logp']:.2f}")
    save_out(model, cfg, out, steps)


def run_grpo(ckpt: str, out: str, steps: int, group: int, lr: float,
             beta: float, eps: float, max_new: int, device: str = "auto"):
    device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    import re
    from .generate import generate
    from .grpo import (grpo_loss, completion_logps, filter_degenerate,
                       overlong_shaped_reward)

    model, cfg = load_base(ckpt, device)
    ref, _ = load_base(ckpt, device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    enc = get_tokenizer()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    prompt_ids = [enc.encode_ordinary(p) for p, _ in PROMPTS]
    pad = max(len(p) for p in prompt_ids)
    P = torch.stack([torch.tensor(p + [enc.eot_token] * (pad - len(p)))
                     for p in prompt_ids]).to(device)
    _ans = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")

    def math_reward(text, answer):
        nums = _ans.findall(text)
        if not nums:
            return 0.0
        want = _ans.findall(answer)
        return 1.0 if want and nums[-1].replace(",", "") == want[-1].replace(",", "") else 0.0

    skipped = 0
    for step in range(steps):
        q_idx = step % len(PROMPTS)
        answer = PROMPTS[q_idx][1]
        completions, rewards = [], []
        for _ in range(group):
            ids = generate(model, prompt_ids[q_idx], max_new=max_new,
                           temperature=1.0, top_p=0.95, device=device)
            completions.append(ids)
            r = math_reward(enc.decode(ids), answer)
            # DAPO overlong shaping: wrong AND truncated costs extra
            rewards.append(overlong_shaped_reward(r, len(ids), max_new, r > 0))

        # DAPO dynamic sampling: an all-equal group has zero advantage for every
        # member — the gradient is exactly 0, so the rollouts were wasted. Skip
        # the step rather than diluting the batch with it.
        if not filter_degenerate(torch.tensor(rewards), group).any():
            skipped += 1
            continue

        T_c = max(len(c) for c in completions)
        C = torch.stack([torch.tensor(c + [enc.eot_token] * (T_c - len(c)))
                         for c in completions]).to(device)
        mask = torch.zeros(group, T_c, dtype=torch.bool, device=device)
        for i, c in enumerate(completions):
            mask[i, :len(c)] = True
        prompt = P[q_idx:q_idx + 1].expand(group, -1)
        # per-token log-probs under the ROLLOUT policy (frozen for this step)
        with torch.no_grad():
            old_logp = completion_logps(model, prompt, C, mask)
        rewards_t = torch.tensor(rewards, device=device)
        loss, stats = grpo_loss(model, ref, prompt, C, mask, rewards_t,
                                old_logp, group, eps=eps, beta=beta)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 25 == 0:
            print(f"step {step:>4} | loss {loss.item():.3f} | "
                  f"reward {stats['reward_mean']:.2f} | kl {stats['kl']:.4f} | "
                  f"clip {stats['frac_clipped']:.2f}")
    if skipped:
        print(f"[grpo] skipped {skipped}/{steps} steps with degenerate (all-equal) groups")
    save_out(model, cfg, out, steps)
