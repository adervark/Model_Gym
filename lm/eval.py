"""Evaluation (module 11): perplexity + few-shot accuracy on benchmark subsets.

Serious eval uses lm-evaluation-harness; this is the minimal version that
teaches the mechanics: loglikelihood-per-character style scoring, few-shot
prompt construction, and the two standard answering modes (continuation vs
multiple-choice probability comparison).
"""
import torch
import torch.nn.functional as F

from .model import Transformer
from .tokenizer import get_tokenizer


def _auto_device(device: str) -> str:
    return device if device != "cuda" or torch.cuda.is_available() else "cpu"


@torch.no_grad()
def perplexity(model: Transformer, text: str, device: str = "cuda",
               seq_len: int | None = None, stride: int | None = None) -> float:
    device = _auto_device(device)
    enc = get_tokenizer()
    ids = enc.encode_ordinary(text)
    seq_len = seq_len or model.cfg.max_seq_len
    stride = stride or seq_len // 2
    model.eval()
    nll, count = 0.0, 0
    for i in range(0, len(ids) - 1, stride):
        chunk = ids[i:i + seq_len]
        x = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([chunk[1:]], dtype=torch.long, device=device)
        logits, _, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        nll += loss.item() * (len(chunk) - 1)
        count += len(chunk) - 1
    return float(torch.exp(torch.tensor(nll / count)))


@torch.no_grad()
def fewshot_logprob(model: Transformer, prompt: str, completion: str,
                    device: str = "cuda") -> float:
    """Log-likelihood of `completion` tokens given `prompt` (continuation mode)."""
    device = _auto_device(device)
    enc = get_tokenizer()
    ids = enc.encode_ordinary(prompt + completion)
    p_len = len(enc.encode_ordinary(prompt))
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    logits, _, _ = model(x)
    logp = F.log_softmax(logits, dim=-1)
    y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
    per_tok = torch.gather(logp, -1, y.unsqueeze(-1)).squeeze(-1)
    return per_tok[0, p_len - 1:].sum().item()


def mc_accuracy(model: Transformer, samples: list[dict], device: str = "cuda") -> float:
    """Multiple-choice via comparing P(choice i | context). samples:
    [{"context": ..., "choices": [..], "answer": idx}]"""
    device = _auto_device(device)
    correct = 0
    for s in samples:
        logps = [fewshot_logprob(model, s["context"], f" {c}") for c in s["choices"]]
        if int(torch.argmax(torch.tensor(logps))) == s["answer"]:
            correct += 1
    return correct / len(samples) if samples else 0.0


@torch.no_grad()
def logit_lens(model: Transformer, prompt: str, device: str = "cuda",
               k: int = 5) -> list[list[tuple[str, float]]]:
    """The logit lens (module 13): decode the unembedding applied to each
    layer's residual stream — what the model would predict if it stopped there.

    Returns per-layer top-k token/probability pairs for the last prompt position.
    The progression through layers shows the model 'refining' its next-token
    prediction — raw interpretability from module 02's residual-stream view.
    """
    device = _auto_device(device)
    from .model import rms_norm
    enc = get_tokenizer()
    ids = enc.encode_ordinary(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    T = x.shape[1]
    f = model.freqs[:T]
    freqs_cos = torch.cos(f)[None, :, None, :]
    freqs_sin = torch.sin(f)[None, :, None, :]

    h = model.tok_emb(x)
    out = []
    for i, block in enumerate(model.blocks):
        h, _, _ = block(h, freqs_cos, freqs_sin)
        logits = model.head(rms_norm(h, model.norm_f))[:, -1]
        probs = torch.softmax(logits, dim=-1)
        top = torch.topk(probs, k)
        out.append([(enc.decode([t.item()]), float(p)) for p, t in zip(top.values[0], top.indices[0])])
    return out
