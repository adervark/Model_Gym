"""Streaming data pipeline: download -> tokenize -> pack -> dataloader.

Packing: concatenate documents separated by <eot>, then chop into seq_len
windows. No padding, no attention masks, no document boundaries needed at
this scale (loss is uniform over tokens; see module 05 for the tradeoffs).
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from .tokenizer import get_tokenizer, tokenize_doc


def load_raw_text(name: str, data_dir: str = "data") -> str:
    path = os.path.join(data_dir, name + ".txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing — run scripts/download_data.sh first (module 05)")
    with open(path) as f:
        return f.read()


def tokenize_to_bin(raw: str, bin_path: str) -> None:
    enc = get_tokenizer()
    ids = enc.encode_ordinary(raw) + [enc.eot_token]
    np.array(ids, dtype=np.uint16).tofile(bin_path)  # 0.5M vocab fits uint16


class TokenizedFile(IterableDataset):
    """Streams packed seq_len+1 windows from a .bin file, one worker per file.
    Shuffling happens at the file level between epochs (enough at this scale)."""

    def __init__(self, bin_path: str, seq_len: int):
        self.bin_path = bin_path
        self.seq_len = seq_len

    def __iter__(self):
        data = np.memmap(self.bin_path, dtype=np.uint16, mode="r")
        n = len(data) - 1
        while True:
            i = torch.randint(0, n - self.seq_len, (1,)).item()
            chunk = np.asarray(data[i:i + self.seq_len + 1])
            yield torch.from_numpy(chunk.astype(np.int64))


def build_dataloaders(cfg, split: float = 0.99) -> tuple[DataLoader, DataLoader]:
    """Assumes {cfg.dataset}_train.bin / _val.bin exist (see download script)."""
    train_path = os.path.join(cfg.data_dir, cfg.dataset + "_train.bin")
    val_path = os.path.join(cfg.data_dir, cfg.dataset + "_val.bin")
    pin = torch.cuda.is_available()
    train = DataLoader(TokenizedFile(train_path, cfg.seq_len), batch_size=cfg.micro_batch_size,
                       num_workers=cfg.num_workers, pin_memory=pin)
    val = DataLoader(TokenizedFile(val_path, cfg.seq_len), batch_size=cfg.micro_batch_size,
                     num_workers=cfg.num_workers, pin_memory=pin)
    return train, val


# --- MinHash fuzzy dedup (module 05) -----------------------------------------

def minhash_signature(tokens: list[int], k: int = 5, m: int = 128,
                      seed: int = 0) -> list[int]:
    """m smallest hashes of all k-shingles — the document's fingerprint.
    Near-duplicate docs share a fraction of signature entries ≈ Jaccard
    similarity (Broder's theorem)."""
    if len(tokens) < k:
        return [0] * m
    rng = np.random.default_rng(seed)
    a = rng.integers(1, 2**32, size=m, dtype=np.uint64)
    b = rng.integers(0, 2**32, size=m, dtype=np.uint64)
    mins = [np.uint64(2**64 - 1)] * m
    for i in range(len(tokens) - k + 1):
        h = hash(tuple(tokens[i:i + k])) & 0xFFFFFFFF
        for j in range(m):
            hv = (a[j] * np.uint64(h) + b[j]) & np.uint64(2**32 - 1)
            if hv < mins[j]:
                mins[j] = hv
    return [int(v) for v in mins]


def deduplicate(docs: list[list[int]], threshold: float = 0.8,
                k: int = 5, m: int = 128) -> list[int]:
    """Return the indices of KEPT documents (first-seen wins; later docs whose
    estimated Jaccard vs any kept doc exceeds threshold are dropped)."""
    sigs = [minhash_signature(d, k, m) for d in docs]
    kept = []
    for i, s in enumerate(sigs):
        dup = False
        for j in kept:
            match = sum(a == b for a, b in zip(s, sigs[j])) / m
            if match >= threshold:
                dup = True
                break
        if not dup:
            kept.append(i)
    return kept
