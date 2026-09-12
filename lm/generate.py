"""Sampling with KV cache. top-p, temperature, repetition penalty."""
import torch
import torch.nn.functional as F

from .model import Transformer
from .tokenizer import get_tokenizer


@torch.no_grad()
def generate(model: Transformer, prompt_ids: list[int], max_new: int = 128,
             temperature: float = 0.8, top_p: float = 0.95, device: str = "cuda",
             echo: bool = False, repetition_penalty: float = 1.0) -> list[int]:
    """Single-sequence (batch-1) sampling with a KV cache.

    repetition_penalty (Keskar et al., CTRL): divide the logit of every token
    already in the context by this factor (multiply if the logit is negative,
    so the penalty always pushes *down*). 1.0 = disabled."""
    model.eval()
    ids = list(prompt_ids)[-model.cfg.max_seq_len:]  # clamp context to the window
    x = torch.tensor([ids], dtype=torch.long, device=device)
    enc = get_tokenizer()
    # vocab padded to 50304 for GEMM efficiency; gpt2 BPE only defines 50257 —
    # allow only the real ids, mask the padding rows
    vocab = model.cfg.vocab_size
    allowed = torch.zeros(vocab, dtype=torch.bool, device=device)
    allowed[:enc.n_vocab] = True
    caches = None
    for _ in range(max_new):
        if caches is not None and caches[0][0].shape[2] >= model.cfg.max_seq_len:
            break  # context full: the model cannot attend further
        logits, _, caches = model(x, cache=caches)
        logits = logits[:, -1].masked_fill(~allowed, float("-inf"))
        if repetition_penalty != 1.0:
            seen = torch.tensor(sorted(set(ids)), dtype=torch.long, device=device)
            prev = logits[0, seen]
            logits[0, seen] = torch.where(prev > 0, prev / repetition_penalty,
                                          prev * repetition_penalty)
        logits = logits / max(temperature, 1e-6)
        if top_p < 1.0:
            probs = F.softmax(logits, dim=-1)
            sorted_p, sorted_i = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_p, dim=-1)
            over = cum > top_p
            # keep the smallest prefix whose mass exceeds top_p; if rounding
            # leaves the total at/below top_p, keep everything (never just 1).
            cutoff = int(over.float().argmax().item()) + 1 if bool(over.any()) \
                else sorted_p.shape[-1]
            logits = logits.masked_fill(torch.ones_like(logits).scatter(
                1, sorted_i[:, cutoff:], 0).bool(), float("-inf"))
        tok = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
        ids.append(tok)
        x = torch.tensor([[tok]], dtype=torch.long, device=device)
        if tok == get_tokenizer().eot_token:
            break
    return ids if echo else ids[len(prompt_ids):]


def sample(model: Transformer, prompt: str, device: str = "cuda", **kw) -> str:
    enc = get_tokenizer()
    out = generate(model, enc.encode_ordinary(prompt), device=device, **kw)
    return enc.decode(out)


@torch.no_grad()
def speculative_generate(target: Transformer, draft: Transformer, prompt_ids: list[int],
                         max_new: int = 128, gamma: int = 4, device: str = "cuda") -> list[int]:
    """Speculative decoding (module 12): the draft proposes gamma tokens per
    round (cheap autoregression); the target verifies them in ONE parallel
    forward and accepts each with probability min(1, p_target/p_draft), else
    samples from the residual (rejection sampling). Accepted output is
    EXACTLY distributed as the target's own sampling — zero quality loss,
    up to gamma× fewer target forwards.

    target/draft: Transformers over the same gpt2 vocab."""
    enc = get_tokenizer()
    ids = list(prompt_ids)
    draft_ids = list(prompt_ids)
    draft_caches = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def dist(logits):
        return torch.softmax(logits, dim=-1)

    while len(ids) - len(prompt_ids) < max_new:
        # 1. draft gamma tokens, recording the draft's probabilities
        xd = (torch.tensor([[draft_ids[-1]]], dtype=torch.long, device=device)
              if draft_caches is not None
              else torch.tensor([draft_ids], dtype=torch.long, device=device))
        draft_toks, draft_dists = [], []
        for _ in range(gamma):
            logits_d, _, draft_caches = draft(xd, cache=draft_caches)
            p_d = dist(logits_d[0, -1])
            t = torch.multinomial(p_d, 1).item()
            if t >= target.cfg.vocab_size:
                break
            draft_toks.append(t)
            # keep the FULL draft distribution: the rejection step needs the
            # whole vector to form the residual (p_t - p_d), not just p_d[t].
            draft_dists.append(p_d.clone())
            xd = torch.tensor([[t]], dtype=torch.long, device=device)
            if t == enc.eot_token:
                break
        if not draft_toks:
            break

        # 2. target verifies the whole candidate block in ONE parallel forward
        full = torch.tensor([ids + draft_toks], dtype=torch.long, device=device)
        logits_t, _, _ = target(full)
        p_t = dist(logits_t[0])  # [P+g, V]; logits[i] predicts position i+1

        # 3. reject-sample the prefix; positions are relative to the length
        # BEFORE this round's acceptance began (ids grows during the loop)
        base = len(ids)
        accepted_all = True
        for i, t in enumerate(draft_toks):
            pos = base + i                   # token index being predicted
            p_target = p_t[pos - 1]          # target dist at this position [V]
            p_draft = draft_dists[i]         # draft dist at this position  [V]
            if torch.rand(1).item() < min(1.0, p_target[t].item()
                                          / max(p_draft[t].item(), 1e-12)):
                ids.append(t)
                # a round accepts up to gamma tokens, so the budget has to be
                # checked HERE, not only in the outer while — otherwise the
                # function can return up to gamma-1 more tokens than max_new.
                if t == enc.eot_token or len(ids) - len(prompt_ids) >= max_new:
                    accepted_all = False
                    break
            else:
                # rejected: resample from the normalized residual (p_t - p_d)+.
                # This elementwise difference is what makes the accepted stream
                # EXACTLY target-distributed (Leviathan et al., eq. 8); using a
                # scalar p_d[t] here silently breaks the guarantee.
                resid = (p_target - p_draft).clamp(min=0)
                s = resid.sum().item()
                ids.append(torch.multinomial(resid / s, 1).item() if s > 1e-9
                           else torch.multinomial(p_target, 1).item())
                accepted_all = False
                break
        if accepted_all:
            draft_ids = list(ids)   # draft cache already covers the accepted block
        else:
            draft_ids = list(ids)
            draft_caches = None     # rebuild the draft's cache from the accepted prefix
        if ids[-1] == enc.eot_token or len(ids) - len(prompt_ids) >= max_new:
            break
    return ids[len(prompt_ids):len(prompt_ids) + max_new]
