# Module 03 — The modern architecture (Llama 3-style)

**Goal:** upgrade your module-02 GPT to the architecture every frontier lab uses
in 2025 (Llama 3/4, Qwen3, DeepSeek-V3). Each change exists for a measured reason:
stability, context length, memory, or throughput. This module's code *is*
`lm/model.py` — read it in parallel.

## The four upgrades, with the numbers behind each

### 1. RMSNorm replaces LayerNorm

LayerNorm normalizes by `(x - mean) / std`, RMSNorm by `x / rms(x)`, and drops the
bias. Zhang & Sennrich found the mean-subtraction does nothing measurable. You save
memory/FLOPs on every sublayer of every layer.

### 2. RoPE replaces learned positional embeddings

Two problems with learned positions: (a) can't extrapolate past the trained length;
(b) absolute positions don't express *relative* distances, which is what attention
actually needs. Rotary Position Embedding (RoPE) multiplies each q/k vector by a
rotation matrix whose angle is proportional to absolute position:

```
q' = R(pos) q,   k' = R(pos) k        where R(pos) = block-diag of 2x2 rotations
(q' k'^T)_ij = q_i R(pos_i - pos_j) k_j     # the product depends only on RELATIVE position
```

Implemented as element-wise rotation on (x0, x1) pairs of the head dim — see
`precompute_rope_freqs`/`apply_rope` in `lm/model.py`. Frequency `theta^(-2i/d)`:
low dims rotate slowly (long-range position signal), high dims rotate fast.
Because attention only sees relative positions, models can extrapolate beyond
training length (to a point), and frontier labs scale `theta` to 500k-10M for
long context (module 13).

### 3. SwiGLU replaces ReLU/GELU

`swish(x) * gate(x)` on an expanded width, with *three* projections:

```
MLP(x) = down( silu(gate(x)) * up(x) ),   gate/up: d -> h,  down: h -> d
```

**The width matters, and it is the detail everyone gets wrong.** GPT-2's MLP is
two matrices, d→4d→d = 8d² params. SwiGLU needs *three* matrices, so reusing
h = 4d would cost 3·4d² = **12d²** — a 1.5x bigger block, which is not a free
swap. Llama's rule is to shrink the hidden dim to compensate:

```
h = 8d/3   (= 2/3 of 4d),  rounded up to a multiple of 64/256 for tensor cores
=> 3 · d · (8d/3) = 8d²    — exactly a GPT-2 MLP's parameter count
```

That is what Shazeer's GLU-variants result means: same loss at ~2/3 the hidden
width, hence the same params/FLOPs as the ReLU-MLP it replaces. `cfg.ffn_hidden`
implements this rule (`ffn_mult=8/3`, rounded to `ffn_multiple_of`). If you ever
see a "SwiGLU" model with h = 4d, its MLP is 50% larger than the baseline it is
being compared against — check before you believe the ablation.

### 4. GQA replaces MHA

Multi-head attention: `n_kv = n_q = n_heads`. GQA (Ainslie et al.): share K,V across
groups of query heads, `n_kv = n_q / 4` (Llama 3 70B) or `/8` (V3). Effect:
- KV cache memory divided by 4-8 (the cache, not weights, is the binding constraint
  at long context / high batch — module 12)
- fewer K/V heads = fewer memory bandwidth-bound reads at decode = faster serving
- quality loss is negligible (Llama-2 paper measured it)

### Free lunch #5: FlashAttention via SDPA

Your module-02 attention materializes the [B, H, T, T] attention matrix — O(T^2)
HBM traffic, the #1 bottleneck. FlashAttention (Dao et al.) fuses the whole
softmax(QK^T)V into one tiled kernel that never writes intermediates to HBM:
~7.6x fewer memory reads on GPT-2 sizes. You do **not** write this kernel yourself:

```python
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

On Ampere+ with fp16/bf16 it dispatches to the fused kernel automatically. This is
the single biggest one-line speedup in the course. (If you want the kernel details,
read the FlashAttention paper + Triton tutorial — worth one afternoon.)

## Context extension: RoPE scaling / YaRN (the long-context lever)

Trained at length L, want 2L. Options, in increasing quality
(implemented in `precompute_rope_freqs`, selectable via `rope_method`/`rope_scale`):

1. **Linear scaling** (position interpolation): slow all frequencies by s.
   Works, but washes out high-frequency (short-range) position info — quality
   drops fast at s > 4.
2. **NTK-aware**: raise theta by s^(d/(d-2)) — stretches low frequencies (long
   range) far more than high ones. Better than linear; the community default
   for a while.
3. **YaRN** (Peng et al., arXiv:2309.00071): NTK-aware base *plus* a per-
   dimension ramp blending interpolation (low dims) with extrapolation (high
   dims), plus attention-temperature correction. This is what production
   long-context models use, along with continued training at long context.

```python
# your model was trained at 1024; serve it at 2048 with YaRN-rescaled freqs:
import torch
from lm.config import ModelConfig
from lm.model import Transformer

ckpt = torch.load("checkpoints/base-100m/best.pt", map_location="cpu", weights_only=False)
# NOTE: model_cfg already contains max_seq_len/rope_method/rope_scale, so passing
# them again as kwargs is a TypeError ("multiple values for keyword argument").
# Build the dict, then override.
saved = dict(ckpt["model_cfg"])
saved.update(max_seq_len=2048, rope_method="yarn", rope_scale=2.0)
cfg = ModelConfig(**saved)
model = Transformer(cfg)
model.load_state_dict(ckpt["model"])   # freqs buffer recomputed from new cfg; weights load cleanly
```

Mechanics: the rope frequencies are a non-persistent buffer — construct the
model with the new `rope_method`/`rope_scale` and load the trained weights; they
load cleanly (no `freqs` in the state dict), and the model attends 2x further.
Measure it (exercise 2) before believing it: extension without any continued
training degrades gracefully with YaRN but it degrades.

## The assembled model

`lm/model.py` is exactly module-02's structure with these four swaps. Key extra
details to notice while reading:

- **GQA cache layout**: the cache stores K,V in *kv-head* space (small), broadcasts
  to query heads at compute time. This is why the cache is 4-8x smaller.
- **Residual-scaled init** (`_init`): each linear's std divided by `sqrt(2L)` so a
  block's output is small vs. the stream at init — trains deep models without
  needing warmup tricks. muP (module 08) formalizes this.
- **Untied embeddings** (`tie_embeddings=False`): Llama/GPT-3 untie input/output
  embeddings. Tying saves a vocab matrix (~30% of a small model's params) for a
  small quality hit; modern models pay the memory for the quality.
- `vocab_size=50304`: gpt2's 50257 padded to a 128-multiple so tensor-core GEMMs
  don't waste cycles on misaligned shapes.

## Run it

```bash
python -m lm.cli pretrain --out checkpoints/m3 --size tiny --steps 500
python -m lm.cli generate --ckpt checkpoints/m3/best.pt
```

Should be coherent-ish text at 500 steps on tiny Shakespeare (it's a 2M-param
model; don't expect more).

## Exercises

1. **Ablate the four upgrades.** Train tiny models (200 steps each) with
   LayerNorm→RMSNorm, learned-pos→RoPE, GELU-MLP→SwiGLU, MHA→GQA toggled. Rank the
   changes by loss impact *and* by memory/time impact. (The point: most of these
   are efficiency plays, not quality plays.)
2. **RoPE extrapolation.** Train on seq 128, evaluate loss at seq 256 with (a)
   learned positional embeddings (b) RoPE. Explain the curves. Then try doubling
   `rope_theta` — what changes?
3. **Verify KV-cache correctness.** Run `generate()` with cache on and a re-written
   version that recomputes the full sequence each step. Outputs must match bit-for-bit
   after fixing the seed. (Cache bugs are the classic silent-error source; learn the
   test.)
4. **Count it.** For d=768, L=12, H=12, V=50304: compute param count by hand (formula
   in `lm/config.py`), compare with `model.cfg.n_params`. Then: how many bytes is the
   KV cache for batch 64, seq 4096, GQA-4, bf16? How many A100s' worth of HBM is that?

**Papers:**
- Zhang & Sennrich, *Root Mean Square Layer Normalization* (arXiv:1910.07467)
- Su et al., *RoFormer* (arXiv:2104.09864) — RoPE
- Shazeer, *GLU Variants Improve Transformer* (arXiv:2002.05202)
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models* (arXiv:2305.13245)
- Dao et al., *FlashAttention / FlashAttention-2* (arXiv:2205.14135, 2307.08691)
- Peng et al., *YaRN* (arXiv:2309.00071) — context extension
- Touvron et al., *Llama 2* (arXiv:2307.09288) — the architecture you just built
