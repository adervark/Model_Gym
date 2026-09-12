# Module 12 — Inference optimization

**Goal:** serve your model fast and cheap. Inference cost dominates once a model
ships — a model is economically viable or not based on this module. The stack:
KV cache math → quantization → batching/serving → speculative decoding.

## 1. The decode bottleneck: it's memory bandwidth, not compute

Generating T tokens autoregressively: prefill (batch, O(T²) compute) then T
decode steps, each tiny (1 token × P params of matmuls). A decode step on a 7B
model is ~14GB of weight reads to produce ~1 token — the GPU is **memory-bound**
at ~10-20% of peak FLOPS. Every technique below attacks this fact.

**KV cache math** (from module 03): cache = 2 × L × n_kv × head_dim × seq × dtype.
7B model (GQA-8, bf16): 2·32·8·128·4096·2 ≈ 0.5GB @4k ctx — fine. @128k ctx:
16GB. Batch 32 @128k: >500GB — infeasible. This is why long-context serving needs
GQA/MLA (DeepSeek's Multi-head Latent Attention compresses the cache ~10x more,
arXiv:2405.04434).

## 2. Quantization (in `lm/quant.py`)

The loss landscape: weights are near-Gaussian per-channel; activations have
outliers. Three tiers you can run:

**W8A8 int8** (`quantize_linear_int8`): symmetric per-tensor. ~2x memory, near-zero
loss drop. Skip the `head` (logit projection) — standard.

**NF4 4-bit** (`NF4.quantize`): 16 quantiles of N(0,1) per 64-weight block —
near-optimal for Gaussian weights (Dettmers). Storage compression ~4x; used by
QLoRA to *train* 65B on 48GB. Dequantize-on-the-fly for compute.

**GPTQ-lite** (`gptq_layer`): the real algorithm (Frantar et al.) — layerwise,
minimize ||WX - quant(W)X||_F using the inverse-Hessian to correct each column's
error into the remaining columns. No gradients needed; runs in minutes. This is
what GPTQ/AWQ/exllama production pipelines do.

Rules of thumb (measured, not folklore): 8-bit ≈ free; 4-bit ≈ 0.5-2% quality
drop for >1B models; 4-bit below ~500M params hurts more. Quantize the big
matrices, keep norms/heads in fp16. **Always re-eval after quantizing** (module
11) — the loss is architecture-dependent.

## 3. Batching, PagedAttention, and the serving stack

Continuous batching: requests arrive/leave continuously; group decodes into one
big batch (throughput ↑ ~10-30x vs one-at-a-time). **PagedAttention** (vLLM,
arXiv:2309.06180): KV cache in *pages* like OS virtual memory — eliminates the
fragmentation that wasted ~60-80% of KV memory in naive batching. This single
idea is why vLLM is the standard.

```bash
pip install vllm
vllm serve /path/to/exported-hf-model --tensor-parallel-size 1
```

(Export: save your checkpoint as HF-format safetensors — `scripts/export_hf.py`.
Then you get: continuous batching, PagedAttention, tensor parallel, and — the
big one — fused kernels at 2-4x naive PyTorch speed.)

## 4. Speculative decoding (2-3x faster decode, no quality loss)

Draft a few tokens cheaply (a 1-2B draft model, or an n-gram model), then verify
them *in parallel* with the big model (one forward pass checks k drafts).
Accept the prefix that the big model would have sampled (rejection sampling);
accepted tokens were exactly as likely under the target → zero quality loss,
2-3x throughput. This is how every production LLM API is served. The 2025
frontier variant is **self-speculation** (draft from the model's own earlier
layers) and **multi-token prediction** heads (trained to predict 2+ tokens at
once — pretraining-side, DeepSeek/Medusa).

## 5. KV cache tricks at the frontier

- **Prefix caching**: share the KV prefix across requests (system prompts!) —
  up to ~10x savings on chat workloads.
- **KV quantization** (FP8/INT4 cache): cache is 60%+ of memory at scale;
  quantizing it is usually safe (it's already been through softmax).
- **Sliding-window / sparse attention** (Mistral-style): attend only the last W
  tokens + a few global ones → linear context cost. Llama 4/Mistral use hybrids.

## Exercises

1. **The throughput table.** For your 100M model: measure decode latency (ms/token)
   at batch 1 and batch 32, prefill+decode. Compute MFU (module 07 formula).
   Confirm memory-boundness: run the same shape on a smaller model and watch
   latency barely change.
2. **Quantize + re-eval.** int8 → 4-bit GPTQ on your trained model. Eval ppl and
   generation quality at each tier. Find the tier where loss becomes visible for
   a 100M vs (if you can) a 1B model. Explain the size dependence.
3. **Run the speculative decoder (it's real code).** `lm/generate.py::speculative_generate` —
   draft with a small model, verify with rejection sampling. Measure acceptance
   rate and speedup; the test suite verifies the distributional guarantee
   (spec output ≡ target's own sampling distribution).
4. **PagedAttention at the scheduler level.** `lm/serving.py::KVPageManager` +
   `ContinuousBatcher` — the page-table mechanics and batching loop that vLLM
   fuses into kernels. Drive it with staggered requests and watch eviction.

**Papers:**
- Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention*
  (arXiv:2309.06180)
- Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*
  (arXiv:2211.17192)
- Frantar et al., *GPTQ* (arXiv:2210.17323) · Dettmers et al., *QLoRA*
  (arXiv:2305.14314)
- DeepSeek-V2, *MLA* (arXiv:2405.04434)
- Gloeckle et al., *Better & Faster Large Language Models via Multi-token
  Prediction* (arXiv:2404.19737)
