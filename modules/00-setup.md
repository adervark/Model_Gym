# Module 00 — Setup

**Goal:** working environment. Everything after this assumes it.

## Install

```bash
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt
# GPU (pick your CUDA): uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

`uv` is a fast pip replacement; plain `pip`/`conda` works if you prefer.
torch>=2.4 matters: `F.scaled_dot_product_attention` (FlashAttention dispatch),
`torch.compile`, and FSDP all behave differently below it.

## Verify

```bash
python -c "
import torch
print(torch.__version__)
print('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

## Your hardware: RTX 4070 laptop, 8GB (read this)

Training memory ≈ **16 bytes/param** (bf16 weights+grads + AdamW state) plus
activations. Your usable VRAM is ~7GB (display shares the rest). What fits:

| Size | Total params | Non-embedding | Weights+AdamW | Fits on 8GB | Recommended command |
|---|---|---|---|---|---|
| tiny | 13.7M | 0.8M | 0.22GB | trivially | `--size tiny --batch-size 16 --seq-len 512` |
| 50m | 75.1M | 23.6M | 1.20GB | easily | `--size 50m --batch-size 16 --seq-len 1024` |
| 100m | 152.8M | 75.5M | 2.44GB | yes, batch 8 | `--size 100m --batch-size 8 --seq-len 1024` |
| 100m + grad ckpt | 152.8M | 75.5M | 2.44GB | yes, batch 16 | `--size 100m --batch-size 16 --grad-ckpt` |
| 300m | 288.7M | 185.6M | 4.62GB | yes with `--grad-ckpt` | `--size 300m --batch-size 4 --seq-len 1024 --grad-ckpt` |

**Why the preset names don't match the totals.** The names are nominal size
classes. With an untied 50,304-row vocabulary, the input embedding and the
output head together cost `2 x 50304 x d` — 12.9M params at d=128 and 103M at
d=1024. On small models that *dominates*: `tiny` is 94% embedding. The
non-embedding column is the one that tracks compute (it's what shows up in the
`6 * N * D` FLOP estimate), so watch that one when you reason about scaling;
watch the total when you reason about VRAM. Print either from any checkpoint:

```python
from lm.config import ModelConfig
cfg = ModelConfig(hidden=768, n_layers=12, n_heads=12, n_kv_heads=4)
print(cfg.n_params, cfg.active_params, cfg.ffn_hidden)
```

Notes:
- **7B+ models**: impossible on 8GB for training. Fine-tune those via QLoRA-style
  methods (module 12 explains the quantization underneath).
- Your 4070 supports bf16 — everything here uses it by default.
- Close GPU-hungry apps while training; watch `nvidia-smi`.
- 4070 laptop bf16 peak ≈ **58 TFLOPS** for training, vs A100's 312. Two
  halvings get you there from the marketing number: GeForce tensor cores run at
  half rate when accumulating in FP32 (what training does), and vendor headline
  figures assume structured sparsity. Module 07 §5 explains why this matters for
  MFU. All step counts in this course still apply; expect ~4-6x longer
  wall-clock per step than an A100. Your first 100m run (20k steps) is ~4-6 hours.
- `--grad-ckpt` trades ~20% step time for ~60-70% less activation memory.
  Use it when VRAM is the constraint, not time.

## Smoke test the whole pipeline now

This runs end-to-end (train -> eval -> generate) in ~1 minute even on CPU:

```bash
bash scripts/download_data.sh     # skip the FineWeb part if you're offline
python -m lm.cli pretrain --out checkpoints/smoke --size tiny \
    --steps 30 --seq-len 128 --batch-size 4 --grad-accum 2
python -m lm.cli generate --ckpt checkpoints/smoke/best.pt
```

Output will be garbage. Expected. The point is that every system you will modify
across this course already runs.

## The stack, one screen

| Layer | Tool | Why |
|---|---|---|
| Language | Python 3.12 | dataclass configs, no ceremony |
| Tensors/autograd | PyTorch 2.x | bf16, SDPA, compile, FSDP |
| Data | HuggingFace `datasets` (streaming) | FineWeb-Edu without disk bloat |
| Tokenizer | tiktoken (gpt2 BPE) | byte-level BPE, 50k vocab |
| Logging | terminal | add wandb when you need curves |

## Layout recap

```
lm/          reference implementation (assembled as you go)
modules/     the lessons — start here, in order
scripts/     data + launch scripts
```

## What this course is NOT

- Not a PyTorch tutorial from zero Python. If `torch.no_grad()` is new to you,
  skim the official 60-minute blitz first.
- Not math-free. Every equation here appears because you implement it.
- Not outdated. Where 2023 practice died (learned positional embeddings, PPO for
  RLHF, LayerNorm), the course says so and uses the 2025 replacement.

**Papers:** none yet. Next module.
