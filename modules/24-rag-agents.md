# Module 24 — RAG and agents: LMs that reach outside themselves

**Goal:** the two production patterns that turn a language model into a
*system* — retrieval-augmented generation and tool-use agent loops. Both are
in `lm/rag.py` and `lm/agent.py`, and both are built entirely from course
primitives (tokenization, embeddings, generation).

## 1. RAG: ground the model in your documents

A model's weights are frozen knowledge; RAG gives it *your* knowledge at
query time. The loop:

```
index:   chunk corpus -> BM25 (lexical) + embeddings (dense)
query:   embed query -> retrieve top-k -> prepend as context -> generate
```

Three components, each instructive:

- **Chunking** (`chunk_text`): the context window is the constraint — chunks
  must fit, overlap preserves boundary continuity. The single most
  underrated RAG decision.
- **BM25** (`lm/rag.py::BM25`): Okapi weighting — IDF × saturated TF. Still
  the baseline lexical retriever; keyword queries don't need embeddings.
- **Dense retrieval** (`embed_chunks`): mean-pool YOUR model's final hidden
  states as chunk embeddings; cosine nearest-neighbors. Frontier systems use
  dedicated embedders + vector DBs; the mechanics are identical.

**Hybrid retrieval** (`rag_answer`, `hybrid=True`): BM25 hits ∪ dense hits —
the standard production answer to "which retriever?" is both.

## 2. Agents: the loop is the model

```
history = task
loop:
    text = generate(history)
    if "CALC: expr" in text:        history += RESULT: tool(expr); continue
    if "ANSWER: ..." in text:       return it
```

`lm/agent.py::agent_loop` — ReAct at course scale. The three ideas that make
this the frontier pattern (not a hack):

1. **The environment is the memory.** Every observation goes back into the
   context; the model reads its own history. The context window IS the
   agent's working memory (and its limit — hence the sliding window in the
   loop).
2. **Tools are verifiable actions.** The calculator is a *deterministic*
   function — exactly what RLVR rewards can score (module 10): the agent's
   training signal is "did the episode end correctly", nothing else.
3. **The format is a learned behavior.** CALC/ANSWER markers are special
   tokens in disguise — SFT/RL taught the model to emit them (module 09/10
   mechanics, again).

Frontier agents (Claude Code, o-series, R1-tool variants) are this loop with
better tools (code exec, browsers), better memory (vector stores, files), and
RL on episode success (SWE-bench-style unit tests as reward).

## Exercises

1. **Chunk-size sweep.** Index a book; measure retrieval hit rate for
   question answering at chunk sizes {32, 128, 512} × overlap {0, 25%}.
   Report the tradeoff (recall vs precision of retrieval). 
2. **The hybrid wins when.** Build a query set mixing keyword queries and
   paraphrases. Measure BM25-only vs dense-only vs hybrid. Explain the
   pattern per query type.
3. **Multi-turn agent with a filesystem tool.** Add a second tool (write a
   note to a file, read it back). Run a 5-turn task. Where does the model
   fail, and why does the failure mode suggest RLVR training?
4. **RAG hallucination measurement.** Generate answers with k ∈ {0, 1, 4}
   retrieved chunks. Score factual overlap with the source text. The k=0
   vs k=4 gap is *why RAG exists* — quantify it on your model.

**Papers:**
- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP
  Tasks* (arXiv:2005.11401)
- Yao et al., *ReAct* (arXiv:2210.03629)
- Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and
  Beyond* (2009)
