# Module 15 — Mixture of Experts (MoE)

**Goal:** the architecture trick behind DeepSeek-V3, Qwen3-MoE, Mixtral, and
Llama 4: ~4-8x capacity at ~1x active compute. On small hardware MoE is
unexpectedly friendly — you trade VRAM (more total params) for FLOPs you don't
have. Implementation: `MoE` in `lm/model.py`.

## 1. The idea

A dense model spends *all* its MLP FLOPs on every token. But tokens are
specialized — some need math-ish processing, some syntax. MoE: replace the one
MLP per layer with E experts and a **router** that sends each token to the top-k
most relevant:

```
router logits = gate(x)                    # [tokens, E] — cheap d->E linear
(top-k vals, top-k idx) = topk(logits, k)
weights = softmax(top-k vals)              # how much each chosen expert contributes
y = sum_k weight_k · expert_k(x)           # experts are full SwiGLU MLPs
```

- **Capacity**: E experts, each a full SwiGLU MLP at `cfg.ffn_hidden` (= 8d/3,
  module 03) = E× the dense MLP params.
- **Active cost**: only k experts run → ~k/E the MLP FLOPs of the same-capacity
  dense model. V3: 671B total, 37B active (~4.5%).
- Attention stays dense (attending is not token-specialized); embeddings shared.

## 2. The load-balancing problem (the whole game)

Untrained routers collapse: one expert wins everything (its loss gradients are
largest), the rest never train. The fix (Shazeer's GShard, then Switch): an
**aux loss** pushing usage toward uniform:

```
L_aux = E · Σ_e f_e · p_e      f_e = fraction of tokens routed to expert e
                               p_e = mean gate probability for expert e
total loss = L_ce + alpha · L_aux            alpha ~ 0.01
```

L_aux is **minimized** when routing is balanced (its minimum is k, not 0 — for
top-k routing the counts force it) and grows as experts collapse. Gradient flows
through the router only. Two more tricks you'll see in real systems:
- **Router z-loss**: penalize huge gate logits (stability in bf16).
- **Expert capacity limit** (drop tokens when an expert exceeds its quota):
  throughput insurance for training at scale; skipped here for clarity.

`lm/model.py::MoE.forward` implements top-k dispatch, the aux loss (fp32), and
per-expert gather/scatter (`index_add_`). Note the dispatch is a loop over
experts — fine to read; production uses fused kernels / expert-parallel comm
(module 07's map: experts sharded across GPUs, tokens all-to-all'd).

## 3. The design decisions (what the frontier actually does)

| Decision | V3 / Mixtral practice |
|---|---|
| E, k | 8-256 experts; k=2 or 8 (V3: 256/8); small models use 8/2 |
| MoE layers | not every layer — V3: first 3 dense (attention-heavy early layers matter more) |
| Expert size | dense-MLP-sized (8d/3 hidden) — V3's "fine-grained" experts are smaller + shared |
| Router init | small init — avoids early collapse |
| Sparse + dense | Mixtral: every layer MoE; V3: mixed; Qwen3: hybrid |

## 4. Run it

```bash
# 4 experts fits 8GB comfortably (see the table below before you pick E)
python -m lm.cli pretrain --out checkpoints/moe-100m --size 100m \
    --moe-experts 4 --moe-topk 2 --steps 20000
```

Check the log line: MFU now tracks *active* FLOPs (module 07 formula with
`active_params`) — the honest way to compare sparse vs dense. And:
`model.cfg.n_params` vs `model.cfg.active_params` in any checkpoint.

**8GB math — run the numbers before you pick E.** MoE trades VRAM for capacity,
and the VRAM bill is on the *total* parameter count (all experts hold weights,
gradients and AdamW state whether or not they fire this step). At the `100m`
preset (d=768, L=12, ffn_hidden=2048), with top-k=2:

| E | Total params | Active params | Weights+AdamW @16B | 8GB card (~7GB usable) |
|---|---|---|---|---|
| dense | 152.8M | 152.8M | 2.44GB | trivially |
| 2 | 209.4M | 209.4M | 3.35GB | yes |
| 4 | 322.7M | 209.4M | 5.16GB | **yes — the sweet spot** |
| 6 | 436.0M | 209.5M | 6.98GB | at the edge; needs `--grad-ckpt` |
| 8 | 549.2M | 209.5M | 8.79GB | **no — OOM** |

Two things to read off this table:

- **Active params barely move with E.** That's the whole point: capacity grows,
  per-token compute doesn't. Going 2→8 experts is 2.6x the parameters for
  +0.1M active.
- **Active > dense here (209M vs 153M)**, because top-k=2 runs *two* experts per
  token where the dense model runs one. So MoE 8x2 is not "the same step time as
  dense 100m" — it's the same step time as a dense model of ~209M. Compare
  against that, not against the smaller dense baseline, or the ablation flatters
  itself. (Reproduce the numbers: `ModelConfig(..., n_experts=E).n_params` vs
  `.active_params`.)

## Exercises

1. **Watch the router learn.** Log L_aux and per-expert token fractions over
   training. Without aux loss (set `moe_aux_loss_coef=0`): what happens to
   expert distribution by step 500? (Collapse. This is *why* the loss exists —
   reproduce it.)
2. **Capacity beats density.** Dense 300m (288.7M, 4.62GB) vs MoE 4×2 (322.7M,
   5.16GB) — matched on VRAM, which is the constraint that actually binds on
   your card — for the same total tokens. Plot val loss vs *wall-clock step
   time*. MoE should win on the loss-vs-time curve, because its 209M active
   params cost less compute per step than the dense model's 289M. That gap is
   the entire commercial argument. (Match on the binding resource, not on the
   parameter count: the honest comparison is the one where both configs cost
   you the same thing.)
3. **Expert specialization.** After training, feed math text vs prose vs code
   (FineWeb has all three) and histogram router assignments per expert per
   domain. Do experts specialize? (Answer: weakly — this is an open research
   question, not a solved one.)
4. **k matters.** k=1 vs 2 vs 4 at fixed E=8. Measure: loss, router entropy,
   step time (k raises FLOPs ~linearly). Explain V3's choice of k=8 at E=256.

**Papers:**
- Shazeer et al., *Outrageously Large NNs: The Sparsely-Gated MoE Layer*
  (arXiv:1701.06538)
- Fedus et al., *Switch Transformers* (arXiv:2101.03961)
- Jiang et al., *Mixtral of Experts* (arXiv:2401.04088)
- DeepSeek-V3 (arXiv:2412.19437) — fine-grained experts, shared experts,
  aux-loss-free balancing via bias terms
