"""BPE from scratch (module 04) + tiktoken bridge for the training pipeline.

Use tiktoken's gpt2 encoder for real runs; the BPE class here is the
reference implementation you build in module 04.
"""
import heapq
from collections import Counter


def get_pairs(ids):
    return Counter(zip(ids, ids[1:]))


def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, text: str, vocab_size: int = 512):
        self.vocab_size = vocab_size
        self.text = text
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}
        self.merges: dict[tuple[int, int], int] = {}
        self.train()

    def train(self):
        ids = [self.stoi[c] for c in self.text]
        for new_id in range(len(self.stoi), self.vocab_size):
            counts = get_pairs(ids)
            if not counts:
                break
            top = max(counts.items(), key=lambda kv: kv[1])[0]
            self.merges[top] = new_id
            self.itos[new_id] = self.itos[top[0]] + self.itos[top[1]]
            ids = merge(ids, top, new_id)

    def encode(self, text: str) -> list[int]:
        ids = [self.stoi[c] for c in text]
        while len(ids) >= 2:
            counts = get_pairs(ids)
            # merge the pair with the lowest learned id (earliest merge = most frequent)
            pair = min(counts, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids) -> str:
        return "".join(self.itos[i] for i in ids)


# --- training-pipeline tokenizer (tiktoken gpt2) -----------------------------
import tiktoken

_VOCAB_SIZE = 50304  # padded to a 128-multiple for tensor-core efficiency


def get_tokenizer():
    return tiktoken.get_encoding("gpt2")


def tokenize_doc(enc, text: str):
    eot = enc.eot_token
    ids = enc.encode_ordinary(text)
    return ids + [eot]
