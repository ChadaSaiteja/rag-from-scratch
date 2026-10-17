# 🧱 Module 1: Foundations — Transformers, Tokens & LLM Settings

Before you build a RAG app, you need to understand the engine inside every LLM: the **Transformer**.

---

## 🔤 Tokens — the "atoms" of language

Models don't read words; they read **tokens** (words, sub-words, characters, or byte-pairs).

**Why it matters:** everything is billed, limited, and measured in tokens. Your RAG prompt = user question + retrieved chunks, so chunk size directly controls token cost.

| Key Concept | Plain-English Meaning |
|---|---|
| Simplification | Break text into small learnable units |
| Standardization | Same text → same tokens, every time |

---

## ⚙️ Parameters — the model's "knobs"

Parameters are the learned variables that decide how the model behaves. They change during training to reduce prediction error.

- **Weights (`w`)** — how *important* each feature is.
- **Biases (`b`)** — a baseline shift so a neuron can still fire.

```
z = wx + b
```

- **Learning rate** — how *big* each adjustment step is (too big = unstable, too small = slow).
- **Activation functions** — convert input signal → output signal (e.g., **ReLU**, **sigmoid**).

---

## 🎛️ Inference settings every RAG developer must know

| Setting | What it does | Beginner tip |
|---|---|---|
| **Context window** | Max tokens the model can "see" at once | Keep retrieved chunks small enough to fit |
| **Temperature (0–2)** | Randomness. 0 = accurate, 2 = creative, 1 = balanced | Use **low temp (0–0.3)** for factual RAG answers |
| **Top-K** | Only sample from the K most likely next tokens | K=40 is common |
| **Top-P (nucleus)** | Sample until cumulative probability hits a threshold (e.g. 0.9) | Alternative to Top-K |

---

## 🔮 Next-Token Prediction — how LLMs "talk"

LLMs don't compose sentences; they **predict the next token** from the previous ones, billions of times over.

```
Training loop:
  predict next token → compare with real answer → adjust weights → repeat
```

### 🤥 Why hallucinations happen (Day 1)

- **Confidence in patterns** — plausible-sounding ≠ true.
- **No ground truth** — training never verifies facts.
- **Training data limits** — bad data in, bad data out.
- **Probabilistic nature** — it's guessing, not knowing.
- **No real understanding** — statistics, not semantics.

> 💡 **RAG is the antidote:** give the model real documents as context so it *grounds* its next-token guesses in facts.

---

## 📏 Why Context Windows Have Limits (Day 7)

1. **Quadratic self-attention cost** — every token attends to every other token. 10,000 tokens = **100M comparisons** (O(n²)).
2. **KV cache memory** — Key/Value vectors stored in VRAM for every generated token; long chats need hundreds of GBs.
3. **Training length limit** — quality degrades beyond the length seen in training.

### Solutions (worth knowing for big-document RAG)

- **Sparse attention (BigBird)** — sliding window + global tokens + random links → quadratic → linear.
- **Sliding window attention** — attend only to a fixed neighborhood (O(n·w)); used by Longformer, Mistral.
- **KV cache compression** — evict/quantize/merge less important K/V pairs → O(n²) → O(n).
- **Summary-based memory** — keep a rolling summary of intent/constraints/decisions instead of the full transcript (LangChain's `ConversationSummaryMemory`).

### 🕳️ "Lost in the middle" and context rot

Even when everything fits in the window, models don't use it evenly. Research on long-context models (Liu et al., 2023, *"Lost in the Middle"*) found that facts placed at the **beginning or end** of a long prompt are used far more reliably than facts buried in the **middle**. As prompts get longer, accuracy tends to degrade gradually, an effect often called **context rot**.

What this means for RAG:
- **Fewer, better chunks beat many mediocre ones.** Five relevant chunks outperform twenty loosely related ones.
- **Order matters.** Put the most relevant chunk first (a reranker does this for you, Module 6).
- **A bigger context window is not a substitute for retrieval.** It raises the ceiling on what fits, not on what the model actually pays attention to.

### 🧮 Worked example: a RAG token budget

Say your model costs are measured per token and you want answers to stay fast. A typical RAG prompt for one question:

| Part | Tokens (approx.) |
|---|---|
| System instructions | 150 |
| 5 retrieved chunks × ~250 tokens | 1,250 |
| User question | 30 |
| **Input total** | **~1,430** |
| Model's answer | ~200 |

Rough conversions for English text: **1 token ≈ 4 characters ≈ ¾ of a word**. So a 1,000-*character* chunk (the default in this repo's `config.py`) is roughly **250 tokens**. Doubling `top_k` from 5 to 10 nearly doubles input cost and latency, which is why retrieval quality matters more than quantity.

> 🔗 **Connect to RAG:** chunking exists *because* of the context window. A good chunking strategy (Module 4) keeps the most relevant facts inside the window while they're still cheap to process.

---

## ✅ Key Takeaways

- Tokens are the unit of cost, limit, and measurement.
- Temperature/Top-K/Top-P control *how* the model guesses.
- Hallucinations are inevitable in pure LLMs → **that's why RAG exists**.
- Context windows are finite → chunking + retrieval are mandatory.
- Models attend best to the start and end of a prompt → send fewer, well-ordered chunks.

---

## 🔗 In this repo

- The generation model is set by `GENERATION_MODEL` in [`config.py`](https://github.com/ChadaSaiteja/rag-from-scratch/blob/master/src/pipeline/config.py), and `hybrid_search.py` uses it with `temperature=0` for query rewriting, so results are repeatable.
- `CHUNK_SIZE=1000` characters ≈ 250 tokens, and `RETRIEVAL_K=3` per retriever keeps the prompt small.

---

## 🧠 Self-check

<details markdown="1">
<summary>1. Why is low temperature (0–0.3) recommended for RAG answers?</summary>

RAG answers should restate facts from the context, not invent variations. Low temperature makes the model pick the most likely tokens, so answers are more consistent and less likely to drift from the source.

</details>

<details markdown="1">
<summary>2. A model has a 1M-token context window. Why not skip retrieval and send all documents every time?</summary>

Cost and latency scale with input tokens on every request, and models use information in the middle of very long prompts less reliably ("lost in the middle"). Retrieval sends only what's relevant: cheaper, faster, and usually more accurate.

</details>

<details markdown="1">
<summary>3. Roughly how many tokens is a 2,000-character chunk?</summary>

About 500 tokens, using the rule of thumb 1 token ≈ 4 characters of English text.

</details>

➡️ Next: [Embeddings — turning words into numbers](./02_embeddings_and_vectors.md)
