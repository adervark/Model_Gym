# Module 18 — Diffusion Language Models (MDLM)

**Goal:** the non-autoregressive frontier — LLaDA, Mercury, D3LM. Train a model
to *denoise* masked text instead of predicting the next token, and generate by
iterative unmasking. Implementation: `lm/mdlm.py` + `causal=False` in
`lm/model.py`.

## 1. Why autoregression isn't final

AR generation makes L sequential decisions, each a full forward pass — the
latency wall behind every LLM API. Diffusion LMs instead:

- **Training**: mask a random fraction t of tokens, ask a *bidirectional*
  model to recover them (absorbing-state discrete diffusion — MDLM).
- **Generation**: start with all-[MASK], repeatedly predict the masked
  positions and reveal the most confident — parallelizable, ~O(T/steps)
  full-sequence passes.

Properties the AR world can't touch: **infilling is native** (mask the middle
of a sentence — the model already does that task), **editing** (mask + rerun),
and **faster sampling** (Mercury: ~10x tokens/s). The 2025 trade: AR still
wins on raw quality per parameter — diffusion is the efficiency frontier, not
yet the quality one.

## 2. The training objective (read `lm/mdlm.py::mdlm_loss`)

```
sample t ~ U(0,1)                       corruption level
mask tokens with probability t          x_m = where(mask, [MASK], x)
L = CE( logits(x_m) at masked positions, true tokens )
```

Every t from 0..1 is one task: at t→0, masked-language-modeling; at t→1,
generate-from-nothing. One model learns the whole noise schedule.

The **absorbing state**: a token meaning "nothing here yet". Our course
implementation uses the gpt2 eot id (50256) for it — real MDLM adds a
dedicated [MASK] token to the vocab (exercise 2). Also note `cfg.causal=False`:
bidirectional attention, because denoising needs to look *both* ways.

## 3. Generation: iterative unmasking (`mdlm_generate`)

```
x = all [MASK]
for i in 1..T:
    p = softmax(model(x))
    reveal the top k = L·(1 - i/T) most confident positions
return x
```

Ancestral sampling in reverse time: the mask rate decreases linearly (the
discrete diffusion's reverse process). Two notes: confidence = max prob is a
cheap heuristic (real MDLM samples per-token then votes); and the final step
must reveal *everything* — a one-line edge case worth knowing exists.

## 4. Run it

```bash
python - <<'EOF'
import sys; sys.path.insert(0, ".")
import torch
from lm.config import ModelConfig
from lm.model import Transformer
from lm.mdlm import mdlm_loss, mdlm_generate
from lm.data import load_raw_text, tokenize_to_bin
import numpy as np

# train a bidirectional model on tiny shakespeare (module 05 data)
cfg = ModelConfig(hidden=192, n_layers=4, n_heads=4, n_kv_heads=2,
                  max_seq_len=256, causal=False)
m = Transformer(cfg)
ids = np.fromfile("data/tiny_shakespeare_train.bin", dtype=np.uint16).astype(np.int64)
opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
for step in range(2000):
    i = torch.randint(0, len(ids) - 256, (1,)).item()
    x = torch.tensor(ids[i:i+256]).unsqueeze(0)
    loss, _ = mdlm_loss(m, x)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 200 == 0:
        print(step, round(loss.item(), 3))

out = mdlm_generate(m, L=64, steps=16)
import tiktoken
print(tiktoken.get_encoding("gpt2").decode(out.tolist()))
EOF
```

## Exercises

1. **The infilling superpower.** Train, then mask the middle 50% of a real
   sentence and denoise. AR models need a special objective for this; yours
   did it during pretraining. Then do the same with a model trained
   *causally* (causal=True) — compare.
2. **Add a real [MASK] token.** Append one row to tok_emb/head (vocab+1),
   freeze the rest, train only the new embeddings + a few steps. Compare
   denoising quality vs using eot. (This is the difference between the
   course hack and LLaDA.)
3. **Schedule matters.** Try reveal schedules: linear (current), cosine,
   "sample-then-correct" (MDLM's full sampler: sample candidates from the
   marginal, then correct with one refinement pass). Plot quality vs steps.
4. **Speed measurement.** Time AR generation (module 12's cache) vs MDLM
   generation for the same output length. Where does diffusion win, and
   where does the AR cache's per-token efficiency beat it? (This curve is
   the whole commercial case for Mercury-class models.)

**Papers:**
- Austin et al., *Structured Denoising Diffusion Models in Discrete State
  Spaces* (arXiv:2107.03006)
- Sahoo et al., *Simple and Effective Masked Diffusion Language Models*
  (arXiv:2406.07524) — MDLM
- Nie et al., *Large Language Diffusion Models* (arXiv:2502.09992) — LLaDA
- Inception Labs, *Mercury* (2025) — the commercial diffusion LLM
