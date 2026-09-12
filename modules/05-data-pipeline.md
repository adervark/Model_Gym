# Module 05 — The data pipeline

**Goal:** the real 2025 pretraining corpus (FineWeb-Edu), tokenized, packed, and
streamed into the training loop. Data is the least glamorous and most decisive
part of pretraining: Llama 3's quality jump over Llama 2 was primarily data.

## 1. What the frontier trains on (and why)

| Corpus | Size | Lesson |
|---|---|---|
| Common Crawl raw | 100s of TB | unusable: boilerplate, SEO spam, porn |
| C4 | ~300B tok | filter CC with heuristics + perplexity; 2019 baseline |
| The Pile | 800GB | diversity curation; the "20% niche sources" insight |
| RefinedWeb | 5T tok | scale + dedup, minimal filtering |
| **FineWeb-Edu** | 1.3T tok | **quality classifier (educational-value model) filters CC**; synthetic data era begins |
| Llama 3 / Phi / Qwen mixes | 9-15T tok | mixture ratios tuned; high-quality code/math over-sampled |

The throughline: **data quality beats data quantity** at fixed compute — the
Chinchilla paper's re-analysis and every lab since confirms a curated 1T tokens
outperforms a raw 5T. Modern recipe: start from CC, dedup (fuzzy + exact), filter
with a *trained quality model* (FineWeb-Edu's), upweight code/math/science.

## 2. The pipeline stages (each is one file)

```
download (scripts/download_data.sh)
  -> tokenize (lm/data.py::tokenize_to_bin)       gpt2 BPE, eot-terminated
  -> pack (lm/data.py::TokenizedFile)             concat docs, chop seq_len+1
  -> stream (DataLoader workers)                  random starts, no shuffling buffer
```

### Tokenize to a flat uint16 array

```python
ids = enc.encode_ordinary(raw) + [enc.eot_token]
np.array(ids, dtype=np.uint16).tofile(bin_path)
```

One big binary file of token ids, eot between documents. No JSON, no per-example
files — this is how nanoGPT/FineWeb pipelines ship data; training reads it with
zero parse overhead.

### Packing

Documents vary 1-100k tokens. Instead of padding (waste ~50%+), concatenate
everything and slice windows:

```
...doc A...<eot>...doc B...<eot>...doc C...
|---- seq_len ----||---- seq_len ----|
```

Tradeoffs (know them):
- **Loss is computed over eot tokens and document crossings** — the model learns
  to predict `<eot>` naturally (uniform loss assumption). At 100M-1B scale this
  costs nothing measurable; the Llama models do exactly this.
- **No attention masking** for document boundaries: simplest, standard.
- Cross-document windows inject a few "nonsense" training examples per doc — the
  model is robust to it, and it doubles as regularization.

### Streaming + shuffling

FineWeb-Edu's larger samples run to hundreds of GB of text; you can't hold that
in RAM. Our loader memmaps the .bin and samples random windows — the same trick
nanoGPT uses. At frontier scale the real systems are more elaborate (S3 streaming,
per-epoch shard shuffling, dedup hashing across nodes), but the *shape* — flat
token streams, packed windows — is identical.

## 3. Dedup: the one stage that sounds trivial and isn't

CC is full of near-duplicate text (templates, mirrored articles). Duplicates:
- waste compute (same tokens learned twice)
- cause benchmark contamination (module 11)
- induce memorization (verbatim regurgitation)

MinHash fuzzy dedup is ~10 lines of intuition: hash each doc's k-grams, keep the
smallest m hashes as a signature; near-identical docs share signatures with
probability = Jaccard similarity. Exact substring dedup (Suffix Arrays) catches
the rest. FineWeb already dedups; you'll need this yourself the moment you scrape.

## 4. Run it

```bash
bash scripts/download_data.sh   # tiny_shakespeare + fineweb_edu sample (200k docs)
python -m lm.cli pretrain --dataset fineweb_edu --size 100m --steps 20000
```

FineWeb-Edu ships in named sample sizes (sample-10BT, sample-100BT, ...) where
the number is *tokens*, not bytes — sample-100BT is ~100B tokens, far more than
you want here. Our script streams a small slice of it: ~3GB of text, which
tokenizes to ~1.5GB of uint16 (**~750M tokens**, since uint16 = 2 bytes/token).

Do the budget arithmetic once and it will stop being mysterious:

```
100M params x 20 tokens/param (Chinchilla, module 08) = 2B tokens
2B tokens as uint16  = 2 x 2e9      = 4 GB on disk
2B tokens as raw text ≈ 4 bytes/tok = 8 GB
```

So Chinchilla-optimal for your 100M model is ~2B tokens / **4GB**, and the
slice above is ~1/3 of that — the affordable approximation. You'll see real
English emerge by ~5k steps.

## Exercises

1. **Measure packing efficiency.** Write the padding-based loader (pad each doc to
   seq_len) and compute wasted tokens on FineWeb-Edu docs. The 2-3x difference is
   why everyone packs.
2. **Run the dedup (it's real code).** `lm/data.py::minhash_signature` and
   `deduplicate` implement MinHash (k=5, m=128) — feed 10k CC-style docs,
   find the duplicates. Why does k matter? Try k=3 vs k=8.
3. **Contamination check.** Take 5 MMLU questions (find them online). Check how
   many of their substrings appear verbatim in a FineWeb sample. This is the
   mechanic behind "benchmark contamination" — now you know why eval suites
   (module 11) get rebuilt yearly.
4. **Write the token budget.** Your 100M model: compute FLOPs/token, then tokens
   needed for Chinchilla-optimal training. How much disk would that corpus take
   as uint16? As text? (Answer in module 08 if stuck.)

**Papers:**
- Penedo et al., *The RefinedWeb Dataset* (arXiv:2306.01116) — scale + dedup
- Penedo et al., *The FineWeb Datasets* (arXiv:2406.17557) — quality filtering
- Lee et al., *Deduplicating Training Data Makes LMs Better* (arXiv:2107.06499)
- Broder, *On the resemblance and containment of documents* — MinHash
- Llama 3 paper (arXiv:2407.21783) — §3.1 data mix, the best public writeup
