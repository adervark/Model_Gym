# Module 21 — The tiny VLM: your model gets eyes

**Goal:** multimodal frontier in the smallest possible package — a frozen
vision tower feeding your LM through one connector layer (the LLaVA pattern).
Implementation: `lm/vlm.py`, the `prefix_embeds` hook in `lm/model.py`.

## 1. The architecture (LLaVA reduced to its skeleton)

```
image → vision_tower (frozen) → patches [B, P, v_dim]
      → connector Linear(v_dim, d) → P embedding vectors
      → prepended to the token embeddings: [PATCHES | Caption: ...]
      → your LM attends over both, predicts caption tokens
```

Everything you already built is reused: the LM's attention, norms, RoPE — the
only new machinery is `prefix_embeds` (read `Transformer.forward`): prepend
non-text embeddings, and pad the targets with `-100` over the prefix so the
loss is caption-only. That one flag turns your LM into a multimodal encoder.

Why frozen tower + tiny connector: vision encoders (SigLIP/CLIP) are
already-aligned semantic spaces; training them jointly would need ~100x the
compute. The connector is the whole "vision-language alignment" — a single
matrix, ~768×4096.

## 2. Run it (offline, anywhere)

```bash
python scripts/train_vlm.py --ckpt checkpoints/base-100m/best.pt --steps 2000
```

The default offline mode uses `RandomVisionTower` — a small CNN whose images
encode a color channel (the caption says which). It's synthetic, but every
mechanism is real: tower → patches → connector → prefix → masked loss. The
overfit sanity test (`test_training_improves_captioning`) proves the gradient
path works.

Real towers: `--vision openai/clip-vit-base-patch32` (or
`google/siglip-base-patch16-224`) plus a caption dataset (COCO via
`datasets`) — the loop is identical; module exercise 2 wires it.

## 3. The details that decide quality (all frontier-standard)

- **Patch count P is your context tax.** CLIP-B/32 → 49 patches; SigLIP-16
  → 256. Every patch is one token position your LM can't use for text —
  the same tokenizer-tax logic as module 04, in pixels.
- **Resolution** is the strongest single lever on VLM quality (AnyRes crops
  high-res images into sub-images; Llama-4's biggest practical innovation).
- **The frozen tower's distribution matters**: tower pretraining data ==
  what you should fine-tune on. CLIP was trained on internet image-text —
  fine for COCO captions, wrong for satellite imagery.
- **Generation**: your LM now conditions on images; with the KV cache the
  prefix is cached once and every caption token reuses it (module 12).

## 4. The 2025 map

From this skeleton, the frontier stack is three extensions:
1. **Interleaved any-to-any** (Gemini/Chameleon-style): images *inside* the
   text stream, not just as prefix — the same `prefix_embeds` trick applied
   at arbitrary positions.
2. **Projector depth**: one linear layer → 2-layer MLP (LLaVA-1.5's
   improvement) → perceiver resamplers (Qwen-VL, PaliGemma).
3. **Joint training**: unfreeze the tower late in training (one-pass
   unfreezing), or train from scratch on multimodal data (the frontier
   models are natively multimodal — language is one modality).

## Exercises

1. **The context tax, measured.** VLM with 4 vs 64 patches: same captions,
   same steps. Report loss difference. Then compute: at 49 patches, how much
   of your model's 1024 context is vision? (The number that motivates
   perceiver resamplers.)
2. **Real data.** Wire COCO captions (HuggingFace) into the offline loop,
   with a frozen SigLIP. Compare generated captions pre/post training.
   (First real VLM checkpoint of the course.)
3. **Frozen LM experiment.** Freeze the LM too — train only the connector.
   How much does the connector alone achieve? (The answer shows how much of
   "multimodality" is alignment vs representation.)
4. **Prefix caching.** Generate 20 captions for one image with the KV cache
   vs without: measure the speedup (module 12's prefix-caching argument,
   now real).

**Papers:**
- Liu et al., *Visual Instruction Tuning* (LLaVA, arXiv:2304.08485)
- Radford et al., *CLIP* (arXiv:2103.00020) · Zhai et al., *SigLIP*
  (arXiv:2303.15343)
- Liu et al., *Improved Baselines with Visual Instruction Tuning*
  (LLaVA-1.5, arXiv:2310.03744) — the MLP projector finding
