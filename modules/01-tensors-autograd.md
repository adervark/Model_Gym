# Module 01 — Tensors, autograd, and the training loop

**Goal:** derive backprop manually, then let autograd do it; train an MLP on a real
learning problem (a tiny bigram char-LM). Everything in modules 02-13 is this loop
with different `model`, `data`, and `loss`.

## 1. What a training loop is (5 lines of truth)

```python
for step in range(max_steps):
    x, y = next_batch()
    loss = loss_fn(model(x), y)   # forward: compute the objective
    optimizer.zero_grad()
    loss.backward()               # backward: d(loss)/d(every param) via chain rule
    optimizer.step()              # update: params -= lr * grad (modulo optimizer tricks)
```

That's it. All of ML is this loop plus:
- what `model` is (modules 02-03)
- what `x, y` are (modules 04-05)
- how the update is computed (module 06-08)
- what the objective is (modules 09-10)

## 2. Backprop by hand (do this once, never again)

For a 2-layer net `y = W2 @ relu(W1 @ x)`, `loss = (y - t)^2`, derive
`dL/dW1` by chain rule. Verify against autograd:

```python
import torch
torch.manual_seed(0)
x = torch.randn(4, 3, requires_grad=False)
t = torch.randn(4, 2)
W1 = torch.randn(3, 5, requires_grad=True)
W2 = torch.randn(5, 2, requires_grad=True)
h = (x @ W1).relu()
y = h @ W2
loss = ((y - t) ** 2).mean()
loss.backward()

# manual, read right to left:
#   dL/dy = 2(y - t)/N          (N = t.numel(), because .mean())
#   dL/dh = dL/dy @ W2.T        (backprop through y = h @ W2)
#   dL/d(pre-relu) = dL/dh * (h > 0)
#   dL/dW1 = x.T @ dL/d(pre-relu)
dy = 2 * (y - t) / t.numel()
manual_W1 = x.T @ (dy @ W2.t() * (h > 0).float())
print((manual_W1 - W1.grad).abs().max())   # ~0
```

Key facts to internalize:

- Autograd builds a graph of every op on `requires_grad` tensors; `backward()`
  walks it in reverse applying each op's known local derivative. No magic.
- `zero_grad()` because gradients **accumulate** into `.grad`.
- In-place ops on non-leaf tensors break the graph. Never `h += x` mid-forward.
- `.item()`/`.detach()` detach from the graph (use for logging; a `log_loss` tensor
  that keeps graph refs leaks memory across steps).

## 3. The three things that decide everything: batching, dtype, device

```python
# batching: [B, ...] tensors. B is your one lever for hardware efficiency.
# dtype: fp32 is the safe default, bf16 halves memory and doubles throughput on Ampere+.
# device: cuda vs cpu. Never mix: put everything on the same device explicitly.
x = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
```

## 4. Build: bigram char-level LM (the smallest real LM)

Train a 1-param-class model on tiny Shakespeare text. A "bigram" model is a
lookup table `P(next char | current char)`. In neural form: embed char -> linear ->
softmax over vocab. This is a 0-layer transformer.

```python
import torch, torch.nn.functional as F

text = open("data/tiny_shakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
ids = torch.tensor([stoi[c] for c in text])

V, C = len(chars), 32          # vocab, embedding dim
W = torch.randn(V, C) * 0.1    # input embeddings
O = torch.randn(C, V) * 0.1    # output head
params = [W, O]
for p in params: p.requires_grad_(True)

# optimizer: SGD for now (AdamW in module 06)
lr = 1e-2
for step in range(5000):
    ix = torch.randint(0, len(ids) - 8, (32,))       # B=32, seq 8
    x = ids[ix[:, None] + torch.arange(8)]           # [B, 8]
    y = ids[ix[:, None] + torch.arange(1, 9)]
    # W[x] is an EMBEDDING LOOKUP, not a matmul: x holds integer ids, so
    # `x @ W` would be both a shape error and a dtype error. Indexing a [V, C]
    # table with a [B, 8] id tensor gives [B, 8, C]; then @ O -> [B, 8, V].
    logits = W[x] @ O                                # [B, 8, V] — context-free (bigram)
    loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
    for p in params: p.grad = None
    loss.backward()
    for p in params: p.data -= lr * p.grad
    if step % 500 == 0: print(step, round(loss.item(), 3))
```

Expected: loss ~4.2 -> ~2.5 (perplexity ~12). The starting value is the uniform
baseline ln(V) = ln(65) ≈ 4.17, because the small init makes every logit ≈ 0 —
a useful check in itself: if step 0 is far from ln(vocab), your init or your
loss reduction is wrong. The ~2.5 floor is the information-theoretic ceiling
for predicting the next char from the previous one only.

## 5. Read the actual loop you'll be modifying

Open `lm/train.py`. Find each element of the 5-line loop, plus:
- gradient **accumulation** (`.grad_accum` micro-batches per optimizer step — why?
  module 06)
- `clip_grad_norm_` (why? module 06)
- mixed precision `torch.autocast` (module 06)
- an eval pass that runs the model under `no_grad()` (module 11)

## Exercises (do all three)

1. **Overfit sanity check.** On the bigram model: replace random batches with a
   fixed batch, train 1000 steps. Loss must approach ~0 (memorization). If your
   training loop can't overfit, nothing else matters. (Standard trick, always run
   it first on a new architecture.)
2. **Derive a loss gradient that's actually wrong.** Compute `dL/dW2` manually,
   flip a sign, compare against autograd, see it fail. You should know what a
   broken check looks like.
3. **The importance of temperature.** Add temperature to the softmax at sampling
   time: sample 100 chars at T=0.1, 1.0, 2.0. Explain the entropy difference.

**Papers:** none needed yet. (If you want the math formalized: Rumelhart, Hinton &
Williams, "Learning representations by back-propagating errors", Nature 1986.)
