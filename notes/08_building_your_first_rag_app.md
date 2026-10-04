# 🚀 Module 8: Build Your First RAG Application (Capstone)

> *Synthesizes Modules 1–7 + the `AI/day*` code samples*

Everything you've learned, wired into one working pipeline.

---

## 🏗️ The Architecture

```
                  ┌──────────────── OFFLINE (indexing) ────────────────┐
                  │  Documents → Parse → Chunk → Embed → Vector DB     │
                  └────────────────────────────────────────────────────┘

                  ┌──────────────── ONLINE (query time) ───────────────┐
  User Question → │ Query Embed → Hybrid Search → Rerank → Prompt → LLM │ → Answer
                  │              (top-k)      (cross-encoder)  +context│
                  └────────────────────────────────────────────────────┘
```

---

## 📋 Step-by-Step Build

### Step 0 — Setup
```bash
pip install google-genai pinecone langchain-text-splitters python-dotenv
```
```bash
# .env
GOOGLE_API_KEY=your_key
```

### Step 1 — Ingest & Chunk (Module 4)
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text = open("your_document.md").read()
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
chunks = splitter.split_text(text)
```

### Step 2 — Embed & Store (Modules 2 & 5)
```python
import os
from dotenv import load_dotenv
from google import genai
from pinecone import Pinecone, ServerlessSpec
load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
pc = Pinecone(api_key="pclocal", host="http://localhost:5080")

def embed(text):
    return client.models.embed_content(
        model="gemini-embedding-2", contents=text
    ).embeddings[0].values

index = pc.Index(host="http://localhost:5081")
vectors = [
    {
        "id": f"chunk-{i}",
        "values": embed(c),
        "metadata": {"text": c, "source": "your_document.md"},
    }
    for i, c in enumerate(chunks)
]
index.upsert(vectors=vectors)
```

### Step 3 — Retrieve (Module 6)
```python
def retrieve(question, top_k=5):
    qv = embed(question)
    res = index.query(vector=qv, top_k=top_k, include_metadata=True)
    return [m.metadata["text"] for m in res.matches]
```

### Step 4 — Generate with a RAG Prompt (Module 3)
```python
def answer(question):
    contexts = retrieve(question)
    prompt = (
        "Answer using ONLY the context below. "
        "If unsure, say 'I don't know'.\n\n"
        f"CONTEXT:\n{''.join('- ' + c + '\n' for c in contexts)}\n\n"
        f"QUESTION: {question}"
    )
    return client.interactions.create(
        model="gemini-3.5-flash", input=prompt
    ).output_text
```

### Step 5 (Upgrade) — Rerank before prompting (Module 6)
Fetch `top_k=30`, then rerank with a cross-encoder and keep the top 4–6.

### Step 6 (Upgrade) — Tool-calling guardrails (Module 7)
Let the LLM *choose* whether to retrieve, and add a "no-documents-found" tool.

---

## 📏 Evaluating Your RAG (RAGAS) — the missing piece most beginners skip

You can't improve what you don't measure. RAG splits evaluation into **retriever** and **generator**:

| Metric | Measures | Component | Direction |
|---|---|---|---|
| **Faithfulness** | Are claims in the answer supported by retrieved context? (anti-hallucination) | Generator | ↑ higher = better |
| **Answer Relevancy** | Does the answer actually address the question? | Generator | ↑ higher = better |
| **Context Precision** | Are the *relevant* chunks ranked high? | Retriever | ↑ higher = better |
| **Context Recall** | Did retrieval find *all* needed info? | Retriever | ↑ higher = better |

```python
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas import evaluate
from datasets import Dataset

data = {
    "question": ["What is the refund policy?"],
    "answer":   [answer("What is the refund policy?")],
    "contexts": [retrieve("What is the refund policy?")],
    "ground_truth": ["Customers can request a refund within 30 days of purchase."],
}
result = evaluate(
    Dataset.from_dict(data),
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)
print(result)
```

**Diagnosis guide:**
- Low **faithfulness** → strengthen prompt ("answer only from context"), add reranking, cut noisy chunks.
- Low **context recall** → smaller chunks, better chunking strategy, hybrid search.
- Low **answer relevancy** → better prompt + query rewriting.

> 📘 For the full evaluation methodology — test-set generation, LLM-as-a-judge, and CI quality gates — see [Module 9](./09_rag_evaluation.md).

---

## 🏆 Production Checklist (2026 best practices)

- [ ] **Hybrid search** (BM25 + vector) with Reciprocal Rank Fusion
- [ ] **Reranker** (cross-encoder) between retrieval and prompt
- [ ] **Query rewriting / HyDE** for complex questions
- [ ] **Contextual chunk prefixes** (doc title + heading)
- [ ] **Metadata** on every chunk (source, date, section) for filtering
- [ ] **Guardrails** — "answer only from context", citation strings
- [ ] **Semantic cache** for repeated queries (cost/latency win)
- [ ] **Observability** — log query → chunks → scores → answer
- [ ] **Automated evals** (RAGAS) on every change
- [ ] **Domain-adapted embedding model** if your corpus has heavy jargon

---

## 🗺️ Your Full Learning Journey

| # | Topic | Where |
|---|---|---|
| 0 | RAG Overview | `00_introduction_to_rag.md` |
| 1 | Transformers & Tokens | `01_foundations_transformers.md` |
| 2 | Embeddings | `02_embeddings_and_vectors.md` |
| 3 | Prompting | `03_prompting_techniques.md` |
| 4 | Chunking | `04_chunking_strategies.md` |
| 5 | Vector DBs | `05_vector_databases.md` |
| 6 | Retrieval | `06_retrieval_mechanics.md` |
| 7 | Agentic RAG & Tools | `07_agentic_rag_and_tools.md` |
| 8 | **Build the App** | `08_building_your_first_rag_app.md` ← you are here |
| 9 | Evaluation (Deep Dive) | `09_rag_evaluation.md` |
| 10 | Document Ingestion & Parsing | `10_document_ingestion_and_parsing.md` |

**Original docs:** `AI/day1` … `AI/day11` · **Code:** `embedding.py`, `chunking.py`, `pine_cone.py`, `toolcalling.py`

🎉 You now have a complete mental model — from tokens to a production-grade RAG system. Build the capstone, break it, measure it with RAGAS, and iterate!
