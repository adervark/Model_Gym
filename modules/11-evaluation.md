# Module 11 — Evaluation

**Goal:** measure what you trained, and understand why LLM evaluation is
genuinely hard (it's ~the hardest open problem in the field, second only to RL
exploration). By the end: your own eval harness, and the judgment to distrust
every number you see.

## 1. The three levels of evaluation

| Level | What | How |
|---|---|---|
| 1. Next-token loss | is the model learning? | perplexity on held-out text |
| 2. Capability benchmarks | can it *do* things? | MMLU, GSM8K, HumanEval, HellaSwag |
| 3. Preference/human evals | is it *good*? | Chatbot Arena, internal human rubrics |

Level 1 is cheap and reliable, and everything else is noisy. You should always
have level 1; frontier labs run all three continuously.

## 2. Perplexity (the foundation — `lm/eval.py::perplexity`)

Perplexity = exp(cross-entropy). On held-out text: "how surprised is the model
by real language". It's the direct training signal on fresh data — the cleanest
measure of *model* quality, though a poor proxy for *assistant* quality (a model
great at essays can be useless as a chat assistant; hence levels 2-3).

Mechanics: run the model over a long text in windows, average NLL. Nothing else
to it — but see exercise 3 for why this can lie.

## 3. Capability benchmarks (the minefield)

The standard battery, and what each measures:

| Bench | Task | Format | Notes |
|---|---|---|---|
| MMLU | knowledge, 57 subjects | 4-way MC | saturated; the "human-level" claims of 2023 aged poorly |
| GSM8K | grade-school math | numeric answer | saturated for frontier; the RLVR training signal |
| MATH / AIME | competition math | numeric | AIME/AMC became the 2025 frontier bar |
| HumanEval / SWE-bench | code / real repos | unit tests | SWE-bench = agentic benchmark (module 13) |
| HellaSwag | commonsense | MC continuation | saturated |
| GPQA | expert science | MC | the "PhD-level" claim source; still hard |
| LiveBench | dynamic, 6 categories | mixed | rebuilt monthly to fight contamination |
| SimpleBench | adversarial commonsense | binary | designed against LLM failure modes |

The traps, in decreasing order of danger:

1. **Contamination.** Your pretraining corpus contains the test questions
   (module 05 exercise). MMLU/SQuAD-class benchmarks are heavily contaminated.
   That's why AIME-2024/2025-style *fresh* problems and LiveBench exist.
2. **Format sensitivity.** MC accuracy swings ±10% depending on how you
   compute P(answer): logprob of choice, logprob of choice-token after "Answer:",
   token-per-character normalized... The evaluation code *is* part of the method.
   Your `lm/eval.py::mc_accuracy` is one convention (prob of choice given context).
3. **Prompt sensitivity.** 5-shot vs 0-shot vs CoT ("think step by step") changes
   scores by more than model-size increments. Report the prompt.
4. **Saturation + overfitting.** When a benchmark saturates (99%), differences
   between models vanish; the field moves on (MMLU → GPQA → AIME → ...).
5. **Benchmark gaming.** If you train on benchmark data, your scores are fake.
   The line between "test-time optimization" and "training on the test" is the
   single most abused line in the industry.

**The production answer**: `lm-evaluation-harness` (EleutherAI) — standardized
prompts, contamination-aware dataset versions, thousands of tasks. Use it for
anything that matters; use `lm/eval.py` to understand the mechanics.

## 4. Preference evals (level 3)

LLM-as-judge: another (stronger) model rates outputs on a rubric
(helpfulness/format/refusal). Correlates ~0.7-0.8 with human preference when
done well (judge ≥ target, position-swapped, temperature-0). Arena-style
human preference (Elo over head-to-head) is the community standard (LMSYS
Chatbot Arena); frontier labs run both. Watch for: judge bias (position, length,
formatting), and *evaluator drift* — judges in 2025 prefer styles that didn't
exist in 2023.

## 5. Run it

```bash
python -m lm.cli eval --ckpt checkpoints/base-100m/best.pt --file data/tiny_shakespeare.txt
# and with the harness:
pip install lm-eval
lm_eval --model hf --model_args pretrained=your/hf/export --tasks mmlu,gsm8k --num_fewshot 5
```

(Export your checkpoint to HF format first — module 13 covers model sharing.)

## Exercises

1. **Build the harness mechanics.** Implement: few-shot formatting, logprob
   scoring, MC accuracy on a 100-question HellaSwag slice (`datasets`). Report
   the number. Then re-run with (a) different few-shot count, (b) shuffled choice
   order, (c) different scoring convention. Report the spread — that's your error
   bar.
2. **Contamination audit.** For 20 GSM8K questions, count verbatim/near-verbatim
   hits in the FineWeb sample (n-gram overlap ≥ 30 tokens). Estimate what
   fraction of a benchmark's score could come from memorization. (The classic
   method; FrontierMath and LiveBench exist to defeat exactly this.)
3. **Perplexity's blind spot.** Two models, same perplexity, wildly different
   generation quality: train one on shuffled-word text (permute within sentences).
   Ppl will look fine; samples will be garbage. Explain from the loss function.
   (This is why labs don't ship on ppl alone.)
4. **Build a judge.** Use a 7B open model as judge on 50 of your model's answers
   vs 50 ChatGPT answers (position-swap every pair). Compare judge ranking to
   your own. Where does it disagree, and why?

**Papers:**
- Gao et al., *A Framework for Few-Shot Language Model Evaluation* — the
  lm-evaluation-harness; cite the Zenodo record (doi:10.5281/zenodo.5371628),
  it has no arXiv id
- Hendrycks et al., *Measuring Massive Multitask Language Understanding*
  (arXiv:2009.03300)
- Zheng et al., *Judging LLM-as-a-Judge* (arXiv:2306.05685)
- White et al., *LiveBench* (arXiv:2406.19314)
- Rein et al., *GPQA* (arXiv:2311.12022)
- Chollet, *On the Measure of Intelligence* (arXiv:1911.01547) — why benchmarks
  keep failing; ARC-AGI
