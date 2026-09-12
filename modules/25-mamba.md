# Module 25 — Mamba: selective state-space models

**Goal:** the other architecture family — the one attention *competes* with.
Mamba replaces attention with a recurrent state that's O(T) in both memory
and time, and makes the state dynamics *input-dependent* so the model can
choose what to remember. Implementation: `lm/mamba.py`.

## 1. State-space models in one equation

```
h_t = A_t h_{t-1} + B_t x_t      (recurrent: summarize the past into h)
y_t = C_t h_t                    (read out)
```

Compare attention: attention stores the *entire* past and re-reads it per
token — O(T) memory, O(T²) compute. An SSM compresses the past into a fixed
state h ∈ R^N and updates it O(N) per token — **linear in T, constant memory**.
The catch: a fixed h is a lossy summary, which is why SSMs historically lost
to attention on language.

**Mamba's move (selectivity):** make A, B, C depend on the input x_t. The
recurrence becomes a *selective* gate: "this token looks like it matters —
let it in; this one doesn't — let it decay out". Selectivity is what lets a
fixed-size state carry long-range information: the model chooses what to
keep, per token, per channel.

## 2. The block (read `MambaBlock` in `lm/mamba.py`)

```
x -> in_proj (2d) ----------------> SiLU -------------------> gate
      |                                                     |
      '-> conv1d (local context) -> selective SSM:          |
           dt = softplus(x W_dt + b)       per-token time step
           B, C = x W_B, x W_C             input-dependent in/out
           A = -exp(A_log)                 learned decay rates
           h_t = exp(dt·A) h_{t-1} + dt·B_t x_t
           y_t = C_t h_t
out = out_proj(y * gate) + x
```

Notes on the pieces:

- **dt is a gate on memory**: large dt = update the state a lot (keep);
  small = the past decays (forget). Softplus keeps it positive.
- **A is the decay**: fixed per channel, learned — the "forgetting rate".
- **conv1d**: SSMs see one token at a time; a small causal conv adds local
  context first (the analog of attention's window).
- The course implementation is a **sequential scan** — readable, correct,
  slow. Production uses parallel associative scans (Blelloch) + fused CUDA
  kernels; that's an implementation detail, not a concept.

## 3. The frontier pattern: hybrids (`MambaHybrid`)

Nobody fielding a frontier model went pure-Mamba. The deployed pattern
(Zamba, Jamba, Griffin, Gemma 3, Falcon-H1) interleaves blocks:

```
[Mamba, Attention, Mamba, Attention, ...]
```

Why: attention excels at precise local recall and retrieval; the SSM excels
at cheap long-range summarization. The hybrid gets both at a fraction of
attention's memory. `MambaHybrid` interleaves `MambaBlock` and the course's
`Block` — train it with the same loop.

```python
from lm.mamba import MambaModel, MambaHybrid
m = MambaModel(ModelConfig(hidden=192, vocab_size=50304, max_seq_len=256),
               n_blocks=4, d_state=16)
# same training loop as every other module: loss -> backward -> step
```

## Exercises

1. **The memory game.** Train a Mamba model and a small attention model on
   a synthetic task: recall a token presented 100 positions back (copying).
   Plot accuracy vs distance. Where does attention beat Mamba? (The answer —
   precise long-range recall — is *why* hybrids exist.)
2. **Watch selectivity.** Forward a trained block on two sequences: one with
   a rare token early, one without. Compare the hidden states 20 steps later
   — the selective gate should have *kept* the rare token's influence in the
   first. Measure with cosine similarity.
3. **Throughput reality check.** Time the Mamba block's sequential scan vs
   attention at T ∈ {64, 256, 1024, 4096} (module 07's MFU discipline).
   Confirm the O(T) vs O(T²) curves — then note the constant factor gap
   that keeps attention competitive at short context.
4. **Hybrid ablation.** MambaHybrid with ratios {0, 1/4, 1/2, 1} attention
   blocks, matched params. Which ratio wins at your scale? (The published
   answer favors ~1/4-1/2 attention; verify why.)

**Papers:**
- Gu et al., *Mamba: Linear-Time Sequence Modeling with Selective State
  Spaces* (arXiv:2312.00752)
- Dao & Gu, *Transformers are SSMs* (arXiv:2405.21060) — the unifying view
- Lieber et al., *Jamba* (arXiv:2403.19887) — the hybrid pattern
- Glorioso et al., *Zamba* (arXiv:2405.16712)
