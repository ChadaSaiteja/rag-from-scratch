# 🎯 Module 6: Retrieval Mechanics — Finding the Needle

> *Source: `AI/day11/retrievalMechanism.md` + production best-practice research (2026)*

**Retrieval** is the process of finding the most relevant pieces of information from a knowledge store, given a user query. It's the "R" in RAG — and the stage that most determines answer quality.

---

## 🔄 The 4 Stages of Retrieval

### 1. Query Processing
- **Tokenization** — split into tokens/words.
- **Lowercasing** — consistent matching.
- **Stop-word removal** — drop "the", "a", "is".
- **Stemming** (sparse) — `running` → `run`.
- **Embedding generation** (dense) — turn the query into a vector.

### 2. Knowledge Base Representation
- **Chunking** — small, self-contained units (Module 4).
- **Indexing** — data structure for fast search:
  - **Inverted index** → keyword→chunk map (sparse)
  - **Vector index** → ANN search over embeddings (dense)
  - **Graph structure** → entities & relationships (nodes/edges)

### 3. Matching / Scoring
- **Sparse** — term overlap; scored by **TF-IDF** or **BM25**.
- **Dense** — vector comparison via **cosine similarity** / dot product.
- **Graph** — traverse connected nodes; score by similarity, path length, relation type.

### 4. Ranking & Selection
Score every chunk, rank, and return the **top-k**.

---

## 🏅 Hybrid Retrieval (Sparse + Dense) ⭐

The 2026 production standard: run **both** methods in parallel and merge.

```
Query ──┬── BM25 (keyword)  ──┐
        └── Vector (semantic) ──┴── Reciprocal Rank Fusion (RRF) ──→ ranked list
```

**Why hybrid?** Dense vectors miss exact identifiers (e.g., `SKU-992`, `ERR_4297`), while BM25 misses synonyms. Each covers the other's blind spot.

**RRF merge formula:** a chunk's fused score = Σ `1 / (k + rank_i)` across both lists (k ≈ 60). Simple, no score normalization needed.

---

## 🔁 Reranking — the "not optional" stage ⭐

First-stage search is a *fast approximation*. A **reranker (cross-encoder)** reads query+chunk *together* and outputs one precise relevance score.

**Two-stage pattern:**
1. Fast retrieval → top ~30 candidates (milliseconds)
2. Reranker scores all 30 (150–300 ms)
3. Top 4–6 chunks go into the LLM prompt

| Reranker type | How it works | Trade-off |
|---|---|---|
| **Syntactic proximity** | Query keywords close together in the chunk score higher | Free, fast, interpretable |
| **Neural cross-encoder** (e.g., `ms-marco-MiniLM-L-6-v2`, Cohere Rerank, Voyage) | Model trained on query-passage relevance | Slower, but far more accurate |
| **LLM reranker** (RankZephyr, Qwen3-Reranker) | "yes/no" relevance probability | Best quality; costly — route only hard queries |

> Reranking also fights **context rot**: it moves the best chunks to the *start/end* of the prompt, where LLMs pay the most attention.

---

## ✍️ Query Rewriting (before you search)

- **Decompose** multi-part questions into sub-queries.
- **HyDE** (Hypothetical Document Embeddings): ask the LLM to *write a hypothetical answer*, embed *that*, and search with it.
- **Multi-query**: generate 2–3 phrasings of the same question, retrieve, merge.

---

## 🧩 Specialized techniques

- **Multi-vector retrieval** — one document gets *several* embeddings (per sentence/aspect) for fine-grained matching.
- **Graph-based retrieval** — follow entity relationships (great for "who worked on X that depends on Y").

---

## 📊 Choosing a Method

| Method | Strength |
|---|---|
| **Sparse (BM25)** | Efficiency + exact keyword matching |
| **Dense (vector)** | Semantic understanding |
| **Hybrid** | Best balance — the default for production |
| **Graph / multi-vector** | Specialized, evolving approaches |

---

## ✅ Key Takeaways

- Retrieval = query processing → indexing → matching → ranking.
- **Hybrid + rerank** is the modern default, not a luxury.
- Rewrite hard queries *before* searching.
- Top-k is a dial: too low = missing facts, too high = noise + cost.

➡️ Next: [Agentic RAG & Tool Calling](./07_agentic_rag_and_tools.md)
