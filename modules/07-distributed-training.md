# Module 07 — Distributed training

**Goal:** go from 1 GPU to N, and understand the memory/bandwidth arithmetic that
decides what's possible. The frontier lives here: 8xH100 nodes, 10k+ GPUs,
tensor+context+expert parallel combined (DeepSeek-V3, Llama 4).

## 1. The memory budget (do this math before any scale-up)

For a model with P params, training on one GPU needs:

| Component | Bytes/param | At bf16+AdamW |
|---|---|---|
| weights (bf16) | 2 | 2P |
| gradients | 2 | 2P |
| AdamW m, v (fp32) | 8 | 8P |
| master weights (fp32) | 4 | 4P |
| **Total** | | **16P** |

P=7B -> 112GB before a single activation. That's why: 1 GPU per 1-2B params at
bf16+AdamW. Add activations (seq × d × L), and a 100M model wants ~3-4GB — fits
anywhere; a 7B model needs 8xH100. Everything in this module is engineering to
beat that budget.

## 2. DDP: data parallel (works up to the budget)

Every GPU gets the full model, a shard of the batch, and an equal optimizer. The
only communication: gradients all-reduced per step.

```bash
torchrun --nproc-per-node=4 -m lm.cli pretrain --ddp --size 300m --steps 50000
```

Key mechanics: gradients averaged *before* the optimizer step (identical math to
one big batch); ring-allreduce means bandwidth cost is ~constant per-GPU as you
scale. DDP's limit is the budget above: you still can't fit P > ~2B on one card.

## 3. FSDP / ZeRO: shard everything

ZeRO's insight (Rajbhandari et al.): AdamW state is 4/5 of the memory and it's
used *sequentially* — shard it across GPUs and fetch what you need when you need it.

Sharding a 7B model over 8 GPUs, starting from the 16 bytes/param = 112GB
baseline above (weights 2P + grads 2P + optimizer 12P):

| ZeRO stage | Shards | Per-GPU arithmetic | 7B mem/GPU |
|---|---|---|---|
| none (DDP) | nothing | 4P + 12P | 112GB |
| 1 | optimizer state (12P) | 4P + 12P/8 | 38.5GB |
| 2 | + gradients (2P) | 2P + 14P/8 | 26.3GB |
| 3 (=FSDP full) | + weights (2P) | 16P/8 | 14GB |

(Every row is `bytes/param x 7e9`; check one by hand — the point of the table is
that you can regenerate it for any model size and GPU count.)

FSDP is ZeRO-3 in PyTorch: weights are gathered before forward, discarded after
backward; optimizer state never materialized whole. Trade: all-gather comm per
layer per step — it's the reason FSDP training is ~5-15% slower than DDP at same
size. Full details in `lm/train.py` (one-line switch):

```python
model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD, use_orig_params=True)
```

Run it:

```bash
torchrun --nproc-per-node=2 -m lm.cli pretrain --fsdp --size 100m --steps 20000
```

(FSDP on a 100M model is silly — it fits on one GPU. Run it to learn the mechanics;
the memory win shows at >1B.)

## 4. The rest of the frontier stack (know the map)

| Technique | What it shards | Used for |
|---|---|---|
| Tensor parallel (Megatron) | weight matrices across GPUs, per-layer allreduce | intra-node, H100 NVLink |
| Pipeline parallel (GPipe) | layers across GPUs, micro-batch bubbles | inter-node |
| Sequence parallel | activations along seq dim | long context |
| Context parallel (RingAttention) | attention across GPUs | 100k+ context |
| MoE (module 13) | expert MLPs across GPUs | V3/Mixtral: 8x capacity at ~1x flops |

Real systems combine them (3D parallel: DP × TP × PP). The orchestration is why
Megatron-LM/DeepSpeed exist; the one to learn deeply in 2025 is **context
parallelism** (RingAttention, arXiv:2310.01889) — it's what made 1M-token context
models trainable.

## 5. Throughput: MFU, the number that exposes everything

MFU = achieved FLOPs / theoretical peak. Compute it for a step:

```
FLOPs/step ≈ 6 * P * tokens_per_step     (2 fwd + 4 bwd per param-token)
MFU = FLOPs/step / (peak_flops * step_time)
```

A100 bf16 peak ≈ 312 TFLOPS dense; H100 ≈ 989 dense (the 1,979 you'll see
quoted is the *with-sparsity* number — halve vendor headline figures unless
they say dense). Careful on consumer cards: GeForce runs tensor ops at **half
rate with FP32 accumulate**, which is the mode training uses, so a 4090's
famous "330 TFLOPS" is ~165 for your actual run. `lm/train.py::_PEAK_BF16`
records the dense/FP32-accumulate numbers for this reason — using the wrong
convention silently doubles or halves every MFU you report. Frontier MFU is ~40-55%;
random PyTorch is ~20-30%. The gap is attention-not-fused, kernel launches,
comm overlap, and the fact that **attention is memory-bound, not compute-bound**.
`lm/train.py` logs an approximate MFU each step — watch it as you add
FlashAttention (jump), compile (jump), FSDP (small drop), bigger batches (rise).

## Exercises

1. **The budget.** Compute the memory for: 100M, 1B, 7B, 70B models (bf16+AdamW).
   How many 80GB A100s each, with DDP? With FSDP? Write it down — you'll reference
   this table forever.
2. **MFU dissection.** Train 100M at seq 256 vs 1024 vs 4096, fixed total tokens.
   Plot MFU. Explain the curve in terms of matmul shapes and memory-bound ops.
3. **torch.compile.** Rerun with `--compile`: measure the step-time delta, then
   inspect one generated kernel's triton code (`TORCH_LOGS=output_code`). The
   "compilation wall" it hits on first step is the future of all LLM serving.
4. **Run FSDP vs DDP** on 2 GPUs, same config. Measure: step time, MFU, and peak
   memory (`torch.cuda.max_memory_allocated`). Confirm ZeRO's trade — comm for memory.

**Papers:**
- Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion
  Parameter Models* (arXiv:1910.02054)
- Shoeybi et al., *Megatron-LM* (arXiv:1909.08053) — tensor parallel
- Liu et al., *Ring Attention with Blockwise Transformers* (arXiv:2310.01889)
- Chowdhery et al., *PaLM* (arXiv:2204.02311) — MFU, the throughput discipline
- PyTorch FSDP docs — the practical reference
