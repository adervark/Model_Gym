# LM From Zero: PyTorch Language Model Training

Code-first course. You build a modern (Llama-style) language model stack from scratch,
then push it through the full pipeline to the frontier: pretraining -> distributed scaling
-> SFT -> preference RL (DPO/GRPO/RLVR) -> evals -> inference optimization -> reasoning models.

**Rules of the course:**
- Every module is: goal -> concepts (tight) -> code -> run -> exercises -> papers.
- All code is real and runs. The `lm/` package is the assembled result of modules 01-08.
- Sized for one GPU: an 8GB laptop GPU (RTX 4070) runs everything through the 300M
  model with `--grad-ckpt`; the 100M model is the recommended sweet spot
  (see module 00 for the exact VRAM table). 7B+ training needs more hardware.

## Modules

| #   | Module                                                             | I build                                                    | I run                         |
| --- | ------------------------------------------------------------------ | ---------------------------------------------------------- | ----------------------------- |
| 00  | [Setup](modules/00-setup.md)                                       | environment                                                | first tensors                 |
| 01  | [Tensors, autograd, training loop](modules/01-tensors-autograd.md) | MLP + backprop from first principles                       | first train step              |
| 02  | [Transformer from scratch](modules/02-transformer.md)              | GPT-2-style causal transformer                             | char-level LM                 |
| 03  | [Modern architecture](modules/03-modern-architecture.md)           | RoPE, RMSNorm, SwiGLU, GQA, KV cache, FlashAttention, YaRN | `lm/model.py`                 |
| 04  | [Tokenization](modules/04-tokenization.md)                         | BPE from scratch                                           | tokenizer artifacts           |
| 05  | [Data pipeline](modules/05-data-pipeline.md)                       | packing, dataloaders, token budgets                        | `lm/data.py`                  |
| 06  | [Pretraining](modules/06-pretraining.md)                           | full loop: AdamW/muP, mixed precision, wandb, checkpoints  | train a 100M model            |
| 07  | [Distributed training](modules/07-distributed-training.md)         | DDP, FSDP, MFU math                                        | multi-GPU run                 |
| 08  | [Scaling laws](modules/08-scaling-laws.md)                         | Chinchilla, muP transfer                                   | scaling experiments           |
| 09  | [SFT](modules/09-sft.md)                                           | chat templates, loss masking                               | instruction-tune              |
| 10  | [Preference optimization](modules/10-preference-optimization.md)   | DPO, GRPO, RLVR, reward models                             | RLHF your model               |
| 11  | [Evaluation](modules/11-evaluation.md)                             | perplexity, benchmarks, contamination                      | eval harness                  |
| 12  | [Inference optimization](modules/12-inference-optimization.md)     | quantization, speculative decoding, PagedAttention         | fast serving                  |
| 13  | [The frontier](modules/13-frontier.md)                             | R1-style reasoning, test-time compute, logit lens, agents  | what's next                   |
| 14  | [LoRA / QLoRA](modules/14-lora-qlora.md)                           | low-rank adapters, NF4, merge                              | fine-tune big models on 8GB   |
| 15  | [Mixture of Experts](modules/15-moe.md)                            | router, top-k, load balancing                              | V3-style sparse models        |
| 16  | [Distillation + RLVR](modules/16-distillation.md)                  | CoT trace generation, the R1 pipeline                      | reasoning on your GPU         |
| 17  | [MLA](modules/17-mla.md)                                           | DeepSeek-V2 latent attention                               | compressed KV cache           |
| 18  | [Diffusion LMs](modules/18-diffusion-lm.md)                        | masked diffusion, iterative unmasking                      | the non-AR frontier           |
| 19  | [Muon + DAPO](modules/19-muon-dapo.md)                             | NS-orthogonalized optimizer, GRPO fixes                    | the 2025 training stack       |
| 20  | [SAEs](modules/20-sae.md)                                          | sparse autoencoders, feature reading                       | interpretability              |
| 21  | [Tiny VLM](modules/21-vlm.md)                                      | vision tower + connector + your LM                         | multimodal                    |
| 22  | [Ship it](modules/22-serving-scaling.md)                           | GGUF/llama.cpp export, scaling-law lab                     | your model runs anywhere      |
| 23  | [Classical RLHF](modules/23-classical-rlhf.md)                     | reward models, PPO, ORPO/SimPO/KTO                         | the full alignment genealogy  |
| 24  | [RAG + agents](modules/24-rag-agents.md)                           | BM25/dense retrieval, tool-use loops                       | LMs as systems                |
| 25  | [Mamba](modules/25-mamba.md)                                       | selective state spaces, hybrids                            | the other architecture family |
| 26  | [Production toolkit](modules/26-production-toolkit.md)             | Lion, Adafactor, MTP, z-loss, MinHash, continuous batching | the rest of the stack         |

**[Study Guide](STUDY-GUIDE.md)** — four tracks (builder / alignment / systems /
researcher), the dependency graph, the 10 highest-value exercises, the
canonical 30-paper stack.

## Quick start

```bash
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt   # + torch for your CUDA version

# smoke test (2 min on GPU, ~10 min on CPU)
python -m lm.cli pretrain --out checkpoints/smoke \
    --batch-size 4 --seq-len 128 --steps 10 --grad-accum 2 --dataset tiny_shakespeare

# real run (FineWeb-Edu sample, ~100M params, fits an 8GB GPU at batch 8)
bash scripts/download_data.sh
python -m lm.cli pretrain --out checkpoints/base-100m --steps 20000 \
    --batch-size 8 --seq-len 1024

# or push your 8GB further: bigger model, gradient checkpointing
python -m lm.cli pretrain --out checkpoints/base-300m --size 300m --steps 40000 \
    --batch-size 4 --seq-len 1024 --grad-ckpt

# chat with it
python -m lm.cli generate --ckpt checkpoints/base-100m/step_20000.pt
```

## Layout

```
lm/             the assembled codebase (reference implementation)
  model.py      Llama-style transformer: RMSNorm, RoPE, GQA, SwiGLU, MoE, MLA,
                YaRN, MTP heads, z-loss
  tokenizer.py  BPE trainer + tiktoken bridge
  data.py       streaming, packing, dataloaders, MinHash dedup
  train.py      training loop (AdamW/Muon/muP, bf16, FSDP/DDP, wandb, checkpoints)
  generate.py   sampling, KV cache, speculative decoding
  lora.py       LoRA adapters: wrap, train, merge (module 14)
  sae.py        sparse autoencoders (module 20)
  vlm.py        vision tower + connector + LM (module 21)
  mdlm.py       masked diffusion LMs (module 18)
  optim.py      Muon, Lion, Adafactor (modules 19/26)
  rm.py         reward models + PPO (module 23)
  rag.py        BM25, dense retrieval, RAG loop (module 24)
  agent.py      tool-use agent loop (module 24)
  mamba.py      selective SSM + hybrids (module 25)
  serving.py    continuous batching + paged KV (module 26)
  gguf_export.py  hand-written GGUF v3 exporter (module 22)
  sft.py dpo.py grpo.py eval.py quant.py
  cli.py        one entry point: python -m lm.cli <subcommand>
modules/        the lessons (start here) + STUDY-GUIDE.md for navigation
scripts/        data download, distillation, SAE/VLM training, scaling sweeps
tests/          140 tests: every module, every subcommand, llama.cpp roundtrip
checkpoints/    your runs
```

## Course philosophy

- **No magic.** You write or read every kernel-level concept once. Then you use the
  assembled version. (Same principle as Karpathy's minGPT/nanoGPT, updated to 2025
  practice: RoPE/GQA/SwiGLU/RMSNorm, SDPA/FlashAttention, FSDP, GRPO.)
- **Practice over trivia.** Each module ends with exercises that force you to
  reproduce a result or break something and explain it.
- **Frontier-relevant.** Every concept here is load-bearing in current frontier work
  (Llama 3/4, DeepSeek-R1/V3, Qwen3). Where practice moved on, the course says so
  (e.g., PPO -> GRPO/RLVR; learned positional embeddings -> RoPE).

## Paper stack (read alongside)

Per-module references live in each file. The core list:
[Attention Is All You Need](https://arxiv.org/abs/1706.03762) ·
[GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) ·
[Llama 2](https://arxiv.org/abs/2307.09288) ·
[RoFormer](https://arxiv.org/abs/2104.09864) ·
[FlashAttention-2](https://arxiv.org/abs/2307.08691) ·
[Chinchilla](https://arxiv.org/abs/2203.15556) ·
[ZeRO](https://arxiv.org/abs/1910.02054) ·
[muP](https://arxiv.org/abs/2203.03466) ·
[DPO](https://arxiv.org/abs/2305.18290) ·
[GRPO](https://arxiv.org/abs/2402.03300) ·
[R1](https://arxiv.org/abs/2501.12948) ·
[Scaling test-time compute](https://arxiv.org/abs/2408.03314)
