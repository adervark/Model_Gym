# Module 13 — The frontier: reasoning, test-time compute, agents

**Goal:** the current edge of the field and the map to it. Everything after this
module is papers, not courses. This is where the open problems are — and where
you'd work.

## 1. Reasoning models: the R1 recipe (the 2025 paradigm shift)

DeepSeek-R1 (arXiv:2501.12948) demonstrated that *reasoning is a training
outcome, not an architecture feature*. The recipe:

```
base model
 -> RL (GRPO + rule-based math/code rewards, no SFT)     [R1-Zero]
    => "aha moment": the model starts writing <think> traces, self-verifying
 -> SFT with cold-start long-CoT data (a few thousand curated reasonings)
 -> large RL run
 -> rejection sampling + SFT for general capability
```

Key mechanics to internalize:
- **Long-CoT emergence**: RL on verifiable rewards makes models emit reasoning
  traces *because thinking reduces error probability under the reward* — not
  because anyone taught "think". Compute shifts from training to inference.
- **The <think> budget**: R1 uses thousands of tokens of "thinking" before
  answering. This is the test-time compute law (below) applied inside the model.
- **Distillation**: R1's reasoning transfers to small models via SFT on R1's
  traces (the 1.5B-70B R1-Distill models) — you can distill reasoning into your
  100M model the same way: generate traces from a strong model, SFT (module 09),
  then GRPO (module 10).

o-series (OpenAI) is the same idea with undisclosed details + learned reward
models + search. Qwen3/Kimi/Llama-4 all ship hybrid "thinking mode" — a
capability they train in, not a UI toggle.

## 1.5 The logit lens (interpretability in 20 minutes, on the model you trained)

Module 02's residual-stream mental model has a direct experiment: apply the
unembedding head to *each* layer's residual stream and decode the top tokens —
what would the model predict if it stopped at that depth? `lm/eval.py::logit_lens`:

```python
from lm.eval import logit_lens
for layer, top in enumerate(logit_lens(model, "The capital of France is")):
    print(layer, [t for t, _ in top[:3]])
```

Typical result: early layers predict context-continuation garbage; around the
middle the right token emerges and its probability climbs; late layers sharpen
and stabilize it. This "refinement through depth" is the empirical backbone of
mechanistic interpretability — and the lens generalizes: you can decode *any*
projection of the residual stream (SAE features, probing classifiers), which is
how frontier labs audit reasoning models for hidden intent in <think> traces.

## 2. Test-time compute scaling (the law that replaced parameter scaling)

For hard problems, spend compute at *inference*: it scales accuracy as reliably
as pretraining scales base capability (Snell et al., arXiv:2408.03314). Two
antecedents worth separating: Lightman et al., *Let's Verify Step by Step*
(arXiv:2305.20050) established verifier-guided selection, and Brown et al.,
*Large Language Monkeys* (arXiv:2407.21787) established the repeated-sampling
scaling curve. The techniques, cheapest first:

| Method | How | Cost |
|---|---|---|
| Self-consistency / best-of-N | sample N, vote or pick by verifier | N× |
| Rejection sampling | best-of-N with automated check | N× |
| PRM-guided beam search | process reward model scores each *step* | N×+ |
| Monte-Carlo tree search | explore/backprop over reasoning trees | large |
| Long-CoT (thinking) | let the model reason inside its context | 10-1000× tokens |

**PRMs** (process reward models) deserve special note: reward *each reasoning
step* (vs ORM: outcome only) — trained on step-level labels (Math-Shepherd,
arXiv:2312.08935). The training data problem (step labels are hard) is why most
2025 systems use rule-based verifiers on final answers instead.

## 3. Agents & tool use

The o-style loop generalized: the model plans, calls tools (search, code
execution, browser, files), observes, iterates — until the task is verifiably
done. The training stack:

- **Tool-use SFT**: traces of (call → observation → act) pairs, special tokens
  for tool calls (module 04/09 mechanics again).
- **Agentic RL**: RLVR where the reward is "did the episode accomplish the
  task" (SWE-bench: do the tests pass). DeepSeek-R1 + tool RL is the current
  open-source frontier for coding agents.
- **Sandboxed execution**: code-verifiable rewards are why coding agents lead
  the agentic wave — unit tests are the densest verifiable signal available.

## 4. The open problems (the actual frontier, as of this writing)

1. **RL exploration / sample efficiency.** GRPO wastes ~all its rollouts
   (G=16 samples, most worthless). Better exploration (MCTS, learned world
   priors, self-improvement curricula) is *the* open problem — the next R1-scale
   jump likely comes from here.
2. **Verifiable rewards for general tasks.** Math/code have verifiers. Medicine,
   law, creative work don't. Learned RMs reward-hack; human feedback doesn't
   scale. Bridging this gap is the field's biggest hole.
3. **Data walls.** High-quality human text is nearly exhausted (~3-30T tokens
   used of available). The frontier pivots to: synthetic data, RL self-generated
   data, multimodal data, non-text (code execution traces, agent trajectories).
4. **Long context + memory.** 1M-token context exists (Gemini, Qwen). The open
   part is *using* it: retrieval over lifetime-scale memory, cache-aware
   attention, and the KV-cache economics (module 12) at that scale.
5. **Evaluation collapse.** Benchmarks saturate in months (module 11). The field
   needs groundable, contamination-proof, self-calibrating evals. ARC-AGI-2 is
   the current "hard" bar.
6. **Multimodality + world models.** Text-only reasoning plateaus without
   grounding. Frontier labs are betting on video/sensory data + embodied RL —
   and on *world models* (predict next sensory state, then plan inside).
7. **Safety/alignment under RL scaling.** Reasoning models are harder to
   supervise (they hide intent in <think>). Mechanistic interpretability
   (reading the residual stream, module 02's mental model) is the main
   technical response.

## 5. Where you go from here (the shortest path)

1. **Run the R1 recipe at your scale**: take your trained 100M model, GRPO it
   on GSM8K (module 10 exercises). You will reproduce, at tiny scale, the
   emergence of thinking traces. This is the single most valuable experiment
   you can do right now.
2. **Read the canonical stack, in order**: Chinchilla → Llama 2 → Llama 3 →
   R1 → o1-ish (test-time compute) → DeepSeek-V3 → your subfield's latest.
3. **Pick a lane**: pretraining infra (module 07/08), post-training (09/10),
   serving (12), agents/RL (13). All four hire from the same fundamentals —
   which you now have.
4. **Build one thing end-to-end** — a small model that solves a real task for
   you (your own domain, your own data, your own verifier). The difference
   between reading the frontier and being on it is exactly that.

## Exercises

1. **Reproduce an aha moment.** GRPO your base model on GSM8K (G=4-8, RLVR
   reward, no SFT). Log generations every 100 steps. Do thinking traces appear?
   At what step does answer accuracy exceed the reward rate (self-verification)?
   This is the R1-Zero phenomenon, in miniature.
2. **Run the logit lens.** On your trained 100M model: 10 prompts, find the
   layer where the top-1 prediction becomes the correct next token, averaged.
   Does it move earlier with model size? With task difficulty? (This is the
   kind of measurement frontier interpretability runs at scale.)
3. **Best-of-N curve.** For a fixed model: plot accuracy vs N (1,4,16,64) with
   (a) majority vote, (b) rule verifier. Fit the power law. This is the
   "compute-optimal inference" curve from arXiv:2408.03314.
4. **Distill a reasoner.** Generate 500 long-CoT traces from a strong open model
   (Llama-3.1-8B is enough) on GSM8K, SFT your 100M model on them, compare
   against GRPO-from-scratch. Which wins at your scale, and why does the answer
   flip at frontier scale? (Full pipeline: module 16.)
5. **Build a toy agent.** Tool = a calculator (regex-matchable). Prompt-set:
   20 multi-step arithmetic problems. Loop: generate → parse tool call → execute
   → append result → continue (max 5 turns). Measure solve rate with/without
   the tool. Then RLVR-train on it. You have now built a miniature o-style agent.

**Papers (the canonical stack):**
- DeepSeek-R1 (arXiv:2501.12948) — reasoning via RL
- Snell et al., *Scaling LLM Test-Time Compute Optimally* (arXiv:2408.03314)
- Lightman et al., *Let's Verify Step by Step* (arXiv:2305.20050) — PRMs
- Wang et al., *Math-Shepherd* (arXiv:2312.08935)
- DeepSeek-V3 (arXiv:2412.19437) — MoE at frontier scale, FP8 training
- OpenAI o1 blog / system card — the closed counterpart
- ARC Prize (arcprize.org) — the current capability frontier

That's the course. The rest is compute.
