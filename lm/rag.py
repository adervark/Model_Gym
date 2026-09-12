"""Retrieval-Augmented Generation (module 24): BM25 + dense retrieval + the RAG loop.

Chunk a corpus, index it (lexical BM25 + dense embeddings from your own model),
and generate answers grounded in the retrieved chunks. This is the most
deployed production pattern around LLMs — and every piece here is your code.
"""
import math
from collections import Counter

import torch
import torch.nn.functional as F

from .tokenizer import get_tokenizer


def chunk_text(text: str, chunk_tokens: int = 128, overlap: int = 16) -> list[str]:
    enc = get_tokenizer()
    ids = enc.encode_ordinary(text)
    chunks, i = [], 0
    while i < len(ids):
        chunks.append(enc.decode(ids[i:i + chunk_tokens]))
        i += chunk_tokens - overlap
    return chunks


class BM25:
    """BM25 over tokenized chunks (Okapi weighting, the classic lexical
    retriever — still a strong baseline for keyword queries)."""

    def __init__(self, chunks: list[str], k1: float = 1.5, b: float = 0.75):
        enc = get_tokenizer()
        self.docs = [enc.encode_ordinary(c) for c in chunks]
        self.n = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(1, self.n)
        df = Counter(t for d in self.docs for t in set(d))
        self.idf = {t: math.log(1 + (self.n - c + 0.5) / (c + 0.5))
                    for t, c in df.items()}
        self.k1, self.b = k1, b

    def search(self, query: str, k: int = 5) -> list[int]:
        enc = get_tokenizer()
        q = enc.encode_ordinary(query)
        scores = []
        for i, doc in enumerate(self.docs):
            tf = Counter(doc)
            s = 0.0
            for t in q:
                if t in self.idf:
                    denom = tf[t] + self.k1 * (1 - self.b + self.b * len(doc) / self.avgdl)
                    s += self.idf[t] * tf[t] * (self.k1 + 1) / denom
            scores.append(s)
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]


def embed_chunks(model, chunks: list[str], device: str = "cpu") -> torch.Tensor:
    """Dense embeddings with YOUR model: mean-pool the final hidden states.
    (Frontier systems use dedicated embedders; the model's own representation
    is the instructive baseline.)"""
    enc = get_tokenizer()
    model.eval()
    outs = []
    for c in chunks:
        ids = enc.encode_ordinary(c)[:model.cfg.max_seq_len]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        f = model.freqs[:len(ids)]
        freqs_cos = torch.cos(f)[None, :, None, :]
        freqs_sin = torch.sin(f)[None, :, None, :]
        from .model import rms_norm
        with torch.no_grad():
            h = model.tok_emb(x)
            for block in model.blocks:
                h, _, _ = block(h, freqs_cos, freqs_sin)
            h = rms_norm(h, model.norm_f)
        outs.append(h.mean(1))
    return F.normalize(torch.cat(outs), dim=-1)


def dense_search(embeddings: torch.Tensor, query_emb: torch.Tensor, k: int = 5) -> list[int]:
    sims = embeddings @ query_emb.squeeze(0)
    return sims.topk(k).indices.tolist()


@torch.no_grad()
def rag_answer(model, query: str, chunks: list[str], bm25: BM25,
               embeddings: torch.Tensor, device: str = "cpu",
               k: int = 3, max_new: int = 96, hybrid: bool = True) -> str:
    """The RAG loop: retrieve top-k chunks (hybrid = BM25 ∪ dense), prepend
    as context, generate. This is the entire production pattern."""
    enc = get_tokenizer()
    q_emb = embed_chunks(model, [query], device=device)
    k = min(k, len(chunks))  # never ask for more hits than the corpus holds
    dense_hits = dense_search(embeddings, q_emb, k)
    hits = list(dict.fromkeys(bm25.search(query, k) + dense_hits))[:k] if hybrid \
        else dense_hits
    context = "\n".join(f"[{i}] {chunks[h]}" for i, h in enumerate(hits))
    prompt = (f"Context:\n{context}\n\nQuestion: {query}\n\n"
              f"Answer based on the context:")
    from .generate import generate
    out = generate(model, enc.encode_ordinary(prompt), max_new=max_new,
                   temperature=0.7, top_p=0.9, device=device)
    return enc.decode(out)
