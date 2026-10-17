# 🎯 Module 6: Retrieval Mechanics — Finding the Needle

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

### 🧮 Worked example: RRF by hand

BM25 returns **A, B, C** (ranks 1–3). Vector search returns **C, A, D**. With k = 60:

| Chunk | BM25 rank | Vector rank | Fused score |
|---|---|---|---|
| **A** | 1 | 2 | 1/61 + 1/62 = **0.03252** |
| **C** | 3 | 1 | 1/63 + 1/61 = **0.03227** |
| **B** | 2 | — | 1/62 = 0.01613 |
| **D** | — | 3 | 1/63 = 0.01587 |

Final order: **A, C, B, D**. Chunks that both methods like rise to the top, and RRF never has to compare a BM25 score (unbounded) with a cosine score (−1 to 1). That's why it's the default way to fuse.

### 🔤 BM25 in one paragraph

BM25 scores a chunk higher when it contains the query's words **often** (term frequency, with diminishing returns so 50 repeats isn't 50× better), when those words are **rare** across the corpus (inverse document frequency: "refund" matters more than "the"), and it slightly penalizes **very long** chunks that match only by sheer size. No embeddings, no GPU, and exact identifiers like `ERR_4297` match perfectly.

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

### 💻 Code: add a cross-encoder reranker

```python
# pip install sentence-transformers
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(question, docs, keep=5):
    """docs: LangChain Documents from any retriever (fetch ~20-30 of them first)."""
    scores = reranker.predict([(question, d.page_content) for d in docs])
    ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
    return [d for d, _ in ranked[:keep]]
```

A cross-encoder is slower than vector search because it runs the model once *per (question, chunk) pair*, which is why you only apply it to a short candidate list.

> Reranking also fights **context rot**: it moves the best chunks to the *start/end* of the prompt, where LLMs pay the most attention.

---

## ✍️ Query Rewriting (before you search)

- **Decompose** multi-part questions into sub-queries.
- **HyDE** (Hypothetical Document Embeddings): ask the LLM to *write a hypothetical answer*, embed *that*, and search with it.
- **Multi-query**: generate 2–3 phrasings of the same question, retrieve, merge.

---

## 🏷️ Metadata filtering

Similarity alone can't express rules like "only the 2026 policy" or "only documents this user may see". **Filter first, then rank by similarity**:

```python
store.similarity_search("refund window", k=4,
                        filter={"filename": "refund-policy-2026.md"})
```

Use filters for **freshness** (date ranges), **scope** (product, region, department) and **access control** (only chunks the user is allowed to read; see Module 11). Filtering requires that the metadata was attached at ingestion (Modules 4 and 10).

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
- Fetch wide (20–30), rerank, then send narrow (4–6).
- Metadata filters enforce rules that similarity can't.

---

## 🔗 In this repo

[`hybrid_search.py`](https://github.com/ChadaSaiteja/rag-from-scratch/blob/master/src/pipeline/hybrid_search.py) builds exactly the hybrid pattern described above:
- **BM25:** `BM25Retriever.from_documents(...)` over every chunk pulled back out of Chroma.
- **Multi-query vector search:** `MultiQueryRetriever.from_llm(...)` asks Gemini to rephrase the question several ways, searches with each phrasing, and merges the results.
- **Fusion:** `EnsembleRetriever(retrievers=[bm25, vector], weights=[0.5, 0.5])`, which combines the lists with *weighted* Reciprocal Rank Fusion.
- Compare with pure vector search in [`vector_retrieval.py`](https://github.com/ChadaSaiteja/rag-from-scratch/blob/master/src/pipeline/vector_retrieval.py), and read [`data/raw/06_retrieval.txt`](https://github.com/ChadaSaiteja/rag-from-scratch/blob/master/data/raw/06_retrieval.txt).
- 🧪 **Exercises:** (1) add the cross-encoder `rerank()` above after `hybrid_search()` in `generation.py`; (2) try `weights=[0.3, 0.7]` and see how `evaluation.py` scores change.

---

## 🧠 Self-check

<details markdown="1">
<summary>1. Why can't you simply add a BM25 score to a cosine similarity score?</summary>

They're on different, incompatible scales: BM25 is unbounded and depends on the corpus, while cosine sits between −1 and 1. RRF sidesteps this by using only each result's *rank*.

</details>

<details markdown="1">
<summary>2. Why not run the cross-encoder reranker over the whole corpus instead of vector search?</summary>

It must process each (question, chunk) pair through the model, so scoring a whole corpus per query would take far too long. Use fast retrieval to shortlist ~30 candidates, then rerank only those.

</details>

<details markdown="1">
<summary>3. Users ask "what changed in v2.3?" and get answers about v1.0. What would you add?</summary>

A metadata filter on version or date (which requires that metadata at ingestion), and possibly query rewriting to extract the version from the question into a filter.

</details>

➡️ Next: [Agentic RAG & Tool Calling](./07_agentic_rag_and_tools.md)
