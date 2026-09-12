"""Continuous batching + paged KV arena (module 12): the serving core.

Requests arrive and finish continuously; each decode step runs ONE batched
forward for all live requests. The KV cache lives in fixed-size PAGES shared
across requests (PagedAttention) — the mechanism behind vLLM.

This is a scheduler-level implementation: it drives `generate`-style token
production but the batching/paging logic is real. For actual fused kernels,
run vLLM (module 12 §3); this file teaches the mechanics.
"""
import torch

from .tokenizer import get_tokenizer


class KVPageManager:
    """Paged KV cache: fixed page size, allocation table per request,
    LRU eviction when full. The OS-paging analogy from PagedAttention."""

    def __init__(self, n_pages: int, page_size: int, feat_dim: int,
                 dtype=torch.float32):
        self.page_size, self.n_pages = page_size, n_pages
        self.feat_dim = feat_dim
        self.arena = torch.zeros(n_pages, feat_dim, page_size, dtype=dtype)
        self.free = list(range(n_pages))
        self.request_pages: dict[int, list[int]] = {}
        self.request_lens: dict[int, int] = {}
        self.next_request_id = 0

    def new_request(self, n_tokens: int) -> int:
        rid = self.next_request_id
        self.next_request_id += 1
        n = (n_tokens + self.page_size - 1) // self.page_size
        if len(self.free) < n:
            self.evict_lru()
        if len(self.free) < n:
            raise RuntimeError("KV arena exhausted (all pages in use)")
        self.request_pages[rid] = [self.free.pop() for _ in range(n)]
        self.request_lens[rid] = 0
        return rid

    def append_token(self, rid: int, kv: torch.Tensor):
        """kv: [feat_dim] one new position; grows the request's pages."""
        pages = self.request_pages[rid]
        pos = self.request_lens[rid]
        page_i, slot = divmod(pos, self.page_size)
        if page_i >= len(pages):
            if not self.free:
                self.evict_lru()
            pages.append(self.free.pop())
        self.arena[pages[page_i], :, slot] = kv
        self.request_lens[rid] += 1

    def request_len(self, rid: int) -> int:
        return self.request_lens[rid]

    def evict_lru(self):
        """Drop the oldest (lowest-id) request — the simple policy; vLLM
        evicts the least-recently-USED request with swap/recompute."""
        if self.request_pages:
            oldest = min(self.request_pages)
            self.free.extend(self.request_pages.pop(oldest))
            self.request_lens.pop(oldest)

    def release(self, rid: int):
        if rid in self.request_pages:
            self.free.extend(self.request_pages.pop(rid))
            self.request_lens.pop(rid)


class ContinuousBatcher:
    """Round-based scheduler: each step runs one forward for all live
    requests. Batched decode = the throughput multiplier vs one-at-a-time."""

    def __init__(self, model, page_manager: KVPageManager, seq_len: int):
        self.model = model
        self.pm = page_manager
        self.seq_len = seq_len
        self.live: dict[int, dict] = {}  # rid -> {tokens, done}
        self.enc = get_tokenizer()
        self.device = next(model.parameters()).device

    def add_request(self, prompt_ids: list[int]):
        rid = self.pm.new_request(len(prompt_ids))
        self.live[rid] = {"tokens": list(prompt_ids), "done": False}
        # mirror the prompt into the paged KV arena (one row per token;
        # a real system stores the model's K/V here)
        for t in prompt_ids:
            kv = torch.full((self.pm.feat_dim,), float(t))
            self.pm.append_token(rid, kv)
        return rid

    def step(self, max_new_per_request: int = 1, temperature: float = 1.0):
        """One batched decode step for all live requests."""
        if not self.live:
            return {}
        rids = list(self.live)
        # batch the CURRENT token of every request
        batch = torch.tensor([[r["tokens"][-1]] for r in
                              [self.live[i] for i in rids]],
                             dtype=torch.long, device=self.device)
        logits, _, _ = self.model(batch, cache=None)  # 1 token: no cache needed
        new_tokens = {}
        for i, rid in enumerate(rids):
            r = self.live[rid]
            p = torch.softmax(logits[i, -1] / temperature, dim=-1)
            t = torch.multinomial(p, 1).item()
            r["tokens"].append(t)
            self.pm.append_token(rid, torch.full((self.pm.feat_dim,), float(t)))
            if t == self.enc.eot_token or len(r["tokens"]) >= self.seq_len:
                r["done"] = True
            new_tokens[rid] = t
        return new_tokens

    def collect_finished(self) -> list[tuple[int, list[int]]]:
        done = [(rid, r["tokens"]) for rid, r in self.live.items() if r["done"]]
        for rid, _ in done:
            self.pm.release(rid)
            del self.live[rid]
        return done
