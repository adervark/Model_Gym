# Module 22 — Ship it: GGUF/llama.cpp + the scaling-law lab

**Goal:** close the loop — run your model outside this repo (GGUF export +
llama.cpp, verified end-to-end) and turn your GPU into a scaling-law
laboratory (the sweep harness). After this module your work runs anywhere
GGUF runs: laptops, phones, servers.

## 1. GGUF: the format that won

GGUF is llama.cpp's container — every "local LLM" app (ollama, LM Studio,
llama.cpp, kobold) loads it. One file: architecture metadata + tensors +
tokenizer, all key-value typed. Our exporter (`lm/gguf_export.py`) writes the
v3 format **by hand with struct.pack** — ~150 lines for the whole format.
The parts that matter:

- **Metadata**: `general.architecture=llama` plus the hyperparams
  (`llama.embedding_length`, `llama.attention.head_count_kv`, ...). Wrong or
  missing metadata = load failure, and llama.cpp validates aggressively
  (e.g., it *requires* `n_heads × head_dim == hidden` — the shape consistency
  the tiny preset enforces).
- **Tensor naming**: `blk.{i}.attn_q.weight`, `ffn_gate`, `output_norm` —
  the llama arch names, in **sorted order**, dims reversed, offsets
  **relative to the 32-byte-aligned data section** (not absolute — the
  classic exporter bug).
- **Tokenizer**: tokens + scores + types + **merges**. gpt2's merge list has
  to be *recovered* from tiktoken's vocab (the min-max-rank split algorithm
  in `gpt2_tokenizer_metadata`), byte tokens are type NORMAL (llama.cpp
  looks them up by raw byte — typing them BYTE crashes `token_to_byte`).
- **Vocab trimming**: our padded 50304 → real 50257 at export (the padding
  rows were never trained).

```bash
python -m lm.cli export-gguf --ckpt checkpoints/base-100m/best.pt \
    --out base-100m.gguf
# then, anywhere:
#   llama-cli -m base-100m.gguf -p "Once upon a time" -n 200
```

**This path is verified end-to-end by the test suite** — `test_llamacpp_inference`
runs your exported checkpoint through a real llama.cpp binary (the course's smoke
chain trained a model, exported it, and llama.cpp generated text).

The test *skips* when it can't find `llama-cli`, so check that it actually ran
before trusting the claim — a skipped test looks identical to a passing one in a
`-q` summary:

```bash
python -m pytest tests/test_gguf_sweep.py -q -rs     # -rs prints skip reasons
LLAMA_CLI=/path/to/llama.cpp/build/bin/llama-cli python -m pytest tests/ -q
```

It searches `$LLAMA_CLI`, then `PATH`, then the usual build directories. (This
is not hypothetical: the check was once pinned to a single hard-coded `/tmp`
path, so it silently skipped for anyone whose build lived elsewhere — or after
any reboot cleared `/tmp`.)

## 2. The scaling-law lab (`scripts/sweep_scaling.py`)

Module 08's exercise (fit Chinchilla yourself) automated: run an (N, D) grid
via subprocesses, collect val losses, fit L = E + A/N^a + B/D^b.

```bash
python scripts/sweep_scaling.py --sizes tiny,50m --steps 500,2000,8000 \
    --dataset tiny_shakespeare --out sweeps
# -> sweeps/fit.json: E, A, B, alpha, beta, per-cell losses, in-sample RMSE
```

The fitter (`fit_law`) exploits that for fixed (a, b) the model is linear in
(E, A, B) — a least-squares solve — and refines the exponents by coordinate
descent. The test suite verifies it recovers known exponents from synthetic
data. What you get: a machine-readable answer to "what's my next model size,
and how many tokens does it need?" — the decision every pretraining run
starts with.

## 3. Why these two tools together

The course's arc: build the model (01-08), align it (09-10), measure it (11),
serve it (12), push the frontier (13-21). This module closes the
**deployment loop**: the same checkpoint that trained here now runs in
llama.cpp at production speed on hardware you don't own — and the scaling
harness tells you what to train *next*. That's the operational loop of
every small lab: measure → train → ship → measure.

## Exercises

1. **Export the full 100M and run it.** Train 20k steps (module 06), export
   GGUF, then compare generation quality in llama.cpp vs `lm.cli generate` —
   they should agree (same weights, same sampling); any drift means a
   conversion bug. The equivalence check is the professional habit.
2. **Quantize the GGUF.** llama.cpp's own quantizer
   (`llama-quantize base-100m.gguf q4_k_m.gguf Q4_K_M`) vs the course's
   GPTQ (module 12). Measure ppl drop and file size for both. Different
   algorithms, same goal — now you can compare them on real inference.
3. **Your first real scaling decision.** Sweep 6 cells overnight. From the
   fit: predict the val loss of (100m, 20M tokens). Train it. How close was
   the prediction? Then: given your GPU's 12-hour budget, what (N, D) does
   the law recommend?
4. **Serve it.** Point ollama (or llama.cpp's server) at your gguf and hit
   it with an API request. You have now shipped an LLM.

**Papers:**
- GGUF spec (github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- Hoffmann et al., *Chinchilla* (arXiv:2203.15556) — the law you're fitting
- Muennighoff et al., *Scaling Data-Constrained LMs* (arXiv:2305.16264) —
  epochs, for when your sweep hits data limits
