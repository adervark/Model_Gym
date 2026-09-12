# Module 04 — Tokenization

**Goal:** implement byte-pair encoding (BPE) from scratch, understand why it's
*the* design decision that quietly shapes every LLM's behavior, and wire the
production tokenizer (tiktoken/gpt2) into the pipeline.

## 1. Why tokens (not characters, not words)

- Characters: sequences too long — attention is O(T^2), and per-token compute
  wastes layers on spelling. Char models need 4x the context to see the same text.
- Words: vocab explodes (morphology, compounding), OOV forever, no subword reuse.
- Subwords (BPE): compression of frequent sequences. English ends up ~4x fewer
  tokens than characters; rare/misspelled text degrades gracefully to characters.

The tokenizer is the first thing any input passes through. It determines:
effective context length (your "8k context" is ~2k words), per-token compute cost,
and a long tail of failure modes (spelling, arithmetic, non-English, code). This is
why "the tokenizer is the model's alphabet" and why the field moved to byte-level
tokenizers (GPT-2+) and why some frontier models (BYTE, MegaByte, and 2025's
byte-latent work) try to kill subwords entirely.

## 2. BPE from scratch (the whole algorithm is ~40 lines)

BPE: start with bytes/chars as vocab, iteratively merge the most frequent adjacent
pair into a new token, until you hit your vocab budget.

```python
from collections import Counter

def get_pairs(ids): return Counter(zip(ids, ids[1:]))

def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids)-1 and (ids[i], ids[i+1]) == pair:
            out.append(new_id); i += 2
        else:
            out.append(ids[i]); i += 1
    return out

text = open("data/tiny_shakespeare.txt").read()
chars = sorted(set(text))
stoi, itos = {c:i for i,c in enumerate(chars)}, {i:c for i,c in enumerate(chars)}
ids = [stoi[c] for c in text]
merges = {}
for new_id in range(len(stoi), 500):
    pair = max(get_pairs(ids).items(), key=lambda kv: kv[1])[0]
    merges[pair] = new_id
    itos[new_id] = itos[pair[0]] + itos[pair[1]]
    ids = merge(ids, pair, new_id)
```

Encoding: greedily apply merges in the order they were *learned* (earliest = most
frequent = highest priority). Decoding: recursively expand merged ids. Both are in
`lm/tokenizer.py` (`BPETokenizer`) — read and run them.

## 3. The byte-level refinement (what GPT-2 actually did)

- Operate on **UTF-8 bytes** (vocab base = 256) instead of characters: every string
  is encodable, no OOV, no unicode table needed; emoji/multilingual degrade to
  bytes instead of errors.
- **Regex pre-segmentation**: BPE merges never cross word boundaries — pure
  frequency merging otherwise creates absurd tokens across words/punctuation.
- GPT-2 vocab: 50,257 (256 bytes + merges + `<|endoftext|>`).

We use exactly this tokenizer via tiktoken (`lm/tokenizer.py::get_tokenizer`).
tiktoken encodes ~1M tokens/s in Rust; your Python BPE is for understanding, not
production.

## 4. The parts that matter downstream

- **`<|endoftext|>` (eot)**: document separator. During pretraining, documents are
  concatenated with eot between them; the model learns "crossing this token =
  uncorrelated text" and uses it as a generation stop signal. Our data pipeline
  depends on this (module 05).
- **Special/control tokens**: chat formats (`<|user|>`, `<|assistant|>`), tool
  calls, thinking tags (`<|begin_of_thought|>`) — added to the vocab *after* base
  pretraining, in the SFT phase (module 09). Their embeddings are learned from
  scratch while the rest of the model fine-tunes.
- **Token-budget math**: every training decision is denominated in tokens, not
  words or GB. Compute budget C -> tokens D = C/(6N) (module 08). Data pipelines
  report tokens. Get comfortable converting.

## Exercises

1. **Train your BPE** on tiny Shakespeare (target 512 vocab) and show: (a) merge
   rank vs pair frequency curve, (b) the 10 most merged pairs, (c) compression ratio
   (bytes/token) vs a character-level baseline. Then do it on 1MB of code — what
   changes in the merges?
2. **The tokenizer tax.** Take a 100-token English sentence. Translate to a
   language your model doesn't know well (e.g., Arabic, Thai): how many tokens?
   (Thai often 4-6x.) This single measurement explains most multilingual quality
   gaps and the `tokenizer tax` discussions in frontier releases.
3. **Break it.** Feed tiktoken: reversed text, a lone emoji, `"a" * 10000`, and
   `2+2=4` in one string. Decode the tokenization. Explain what you see. (These
   exact artifacts show up in "why can't the model spell strawberry" discourse.)
4. **Chat templates.** Take `<|user|>\nhi<|end|>\n<|assistant|>\nhello<|end|>\n`
   and count tokens. This template overhead is why context windows effectively
   shrink ~10-15% in chat mode. (Full treatment in module 09.)

**Papers:**
- Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units*
  (arXiv:1508.07909) — the original BPE-for-NLP
- Radford et al., GPT-2 (byte-level BPE + regex segmentation)
- Kudo & Richardson, *SentencePiece* (arXiv:1808.06226) — BPE/Unigram, the other
  half of the ecosystem
