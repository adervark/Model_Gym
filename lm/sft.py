"""Supervised fine-tuning (module 09): chat template, packing with loss masking,
loss on assistant turns only."""
import torch
import torch.nn.functional as F

from .config import ModelConfig
from .model import Transformer
from .tokenizer import get_tokenizer
from .data import TokenizedFile

CHAT_TEMPLATE = """<|user|>
{prompt}<|end|>
<|assistant|>
{response}<|end|>"""


def build_sft_sample(enc, prompt: str, response: str, seq_len: int):
    """One chat sample: tokenize template, mask loss to the assistant span."""
    # the assistant span starts exactly after this prefix
    prefix = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
    ids = enc.encode_ordinary(prefix + response + "<|end|>") + [enc.eot_token]
    a_start = len(enc.encode_ordinary(prefix))
    ids = ids[:seq_len]
    mask = torch.zeros(len(ids), dtype=torch.bool)
    mask[a_start:] = True
    if len(ids) < seq_len:
        ids += [enc.eot_token] * (seq_len - len(ids))
        mask = F.pad(mask, (0, seq_len - len(mask)), value=False)
    return torch.tensor(ids, dtype=torch.long), mask


def sft_step(model: Transformer, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logits, _, _ = model(x[:, :-1])
    shift = logits.view(-1, logits.size(-1))
    targets = x[:, 1:].contiguous().view(-1)
    losses = F.cross_entropy(shift, targets, reduction="none")
    m = mask[:, 1:].contiguous().view(-1)
    if m.sum() == 0:
        return torch.tensor(0.0, device=x.device, requires_grad=True)
    return losses[m].mean()
