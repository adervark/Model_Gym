# Study Guide: four tracks through the course

The course is a reference implementation plus 27 modules. This is the
navigation: four tracks depending on your goal, with hours, prerequisites,
and the "what mastery looks like" checkpoint per stage.

## Track A — The Builder (≈60h): train and ship your own model

```
00 setup → 01 tensors → 02 transformer → 03 modern arch → 04 tokenization
→ 05 data → 06 pretraining → [TRAIN THE 100M MODEL] → 11 evals
→ 12 inference → 22 ship it (GGUF/llama.cpp)
```

Checkpoint: a 100M model trained on FineWeb-Edu, evaluated, quantized, and
running in llama.cpp on your laptop. That's the complete lifecycle.

## Track B — The Alignment Engineer (≈40h): post-training is the frontier's open door

```
00 → 01 → 02/03 (skim) → 06 (skim) → 09 SFT → 23 classical RLHF
→ 10 preference RL (DPO/GRPO/RLVR) → 19 Muon+DAPO → 16 distillation
→ 13 frontier (reasoning)
```

Checkpoint: R1's pipeline at course scale — distill traces, SFT, GRPO on
verifiable rewards — and the ability to derive every preference loss from
first principles.

## Track C — The Systems Person (≈35h): efficiency is the moat

```
00 → 01 → 03 → 06 (optimizer parts) → 07 distributed → 17 MLA
→ 15 MoE → 12 inference → 22 serving → 19 optimizers → 26 toolkit
```

Checkpoint: you can compute the memory/bandwidth budget of any training or
serving setup from the architecture, and name the exact lever to improve it.

## Track D — The Researcher (≈80h): everything, plus the open problems

All modules, in order, with the exercises done *honestly* (the ablations,
the derivations, the sweeps). Then: module 13 §4's open-problem list is your
map. The difference between Track D and the others is the last step: pick
ONE open problem and run one experiment on it a week.

## Module dependency graph (fast paths)

```
00 → 01 → 02 → 03 → 17 (MLA)        [architecture spine]
01 → 06 → 07 → 08 → 19              [training spine]
06 → 09 → 10/23 → 16 → 13           [post-training spine]
03 → 12 → 22 → 26                   [systems spine]
02/03/13 → 20 (SAE) → 21 (VLM) → 24 (RAG/agents) → 25 (Mamba)   [breadth]
04 → 05                             [data spine]
14 (LoRA) anywhere after 06         [8GB escape hatch]
18 (diffusion) after 03             [alternative paradigm]
11 (evals) after 06, before 09      [measure everything]
```

## The exercises that matter most (the 20%)

If your time is short, do these and skim the rest:
1. 01-ex1 overfit sanity — the universal check
2. 03-ex3 KV-cache bit-exactness — the universal silent-bug class
3. 08-ex1 muP width sweep — the one scaling experiment that transfers
4. 10-ex3 reward hacking hands-on — the RL failure mode you'll meet daily
5. 13-ex1 reproduce an aha moment — the frontier in miniature
6. 14-ex3 which targets matter — LoRA's real knob
7. 17-ex1 cache byte math — the number every architecture decision follows
8. 19-ex2 length collapse, killed — the current RL frontier
9. 22-ex1 GGUF equivalence check — the shipping discipline
10. 26-ex2 MTP self-speculation — a 2025 trick, assembled from your own parts

## Paper stack by stage (the canonical 30)

- **Foundations**: Attention (1706.03762) · GPT-2 · Llama 2 (2307.09288) ·
  RoFormer (2104.09864) · FlashAttention-2 (2307.08691)
- **Scale**: Chinchilla (2203.15556) · muP (2203.03466) · ZeRO (1910.02054) ·
  PaLM (2204.02311)
- **Alignment**: InstructGPT (2203.02155) · DPO (2305.18290) · GRPO
  (2402.03300) · DAPO (2503.14476) · R1 (2501.12948)
- **Architecture**: DeepSeek-V2/V3 (2405.04434 / 2412.19437) · Mamba
  (2312.00752) · MDLM (2406.07524) · Muon (2502.16982)
- **Systems**: PagedAttention (2309.06180) · Speculative Decoding
  (2211.17192) · GPTQ (2210.17323) · Ring Attention (2310.01889)
- **Frontier**: test-time compute (2408.03314) · PRMs (2305.20050) ·
  scaling monosemanticity · LLaDA (2502.09992)

## How to run the verification

```bash
python -m pytest tests/ -q          # the whole course's correctness
python -m pytest tests/ -q -k "not slow"   # fast unit layer only
```

The suite is the course's spine: every module's claims are locked in as
tests (cache equivalence, gradient math, format roundtrips, llama.cpp
roundtrip). When you modify code, the tests tell you what you broke.
