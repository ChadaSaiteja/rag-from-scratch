# 🗄️ Module 5: Vector Databases — Your AI's Memory

> *Source: `AI/day8/vectordb.md` + `AI/day8/pine_cone.py`*

A **Vector Database** stores data as **embeddings** — numerical representations of *meaning*, not keywords. It solves the three things a raw LLM can't do:

| Problem | Why it matters |
|---|---|
| **No memory** | LLM forgets between sessions |
| **No private data access** | It only knows its training data |
| **No retrieval** | It can't search a large corpus by itself |

---

## 🔁 The Typical RAG Flow

1. **Split** documents into chunks (Module 4).
2. **Embed** each chunk → vector (Module 2).
3. **Store** vectors + metadata in the vector DB.
4. **Embed** the incoming query the same way.
5. **Search** for the top-k nearest vectors (similarity search).
6. **Feed** the retrieved chunks to the LLM as context (Module 3).

The key trick: a query is compared by **semantic similarity** (e.g., cosine similarity), so it works even when the query and document use *different words*.

---

## 🧭 How to Choose a Vector DB

| Option | Type | Notes |
|---|---|---|
| **pgvector** | Postgres extension | Already on Postgres? Relational + vector in one place |
| **ChromaDB** | Lightweight/embedded | Easy local setup — **best for prototyping** |
| **Pinecone** | Managed/hosted | Fully managed, scales well, zero infra |
| **Weaviate** | Managed/self-hosted | Built-in **hybrid** search (keyword + vector) |
| **FAISS** | Library (in-process) | Blazing fast, no server — but no built-in persistence |

**Consider:** scale (vector count + query volume), hosted vs self-managed, hybrid search support, metadata filtering, and existing infra.

---

## 💻 Code: Embed Documents & Upsert to Pinecone (from `day8/pine_cone.py`)

```python
import os
from dotenv import load_dotenv
from google import genai
from pinecone import Pinecone, ServerlessSpec
load_dotenv()

pc = Pinecone(api_key="pclocal", host="http://localhost:5080")
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

documents = [
    {"id": "doc-1", "text": "Customers can request a refund within 30 days of purchase.", "category": "refund"},
    {"id": "doc-2", "text": "Installation appointments can be rescheduled up to 24 hours before.", "category": "installation"},
    {"id": "doc-3", "text": "Technical support is available Monday–Friday, 9 AM to 6 PM.", "category": "support"},
]

# 1. Embed every document text
embedding_list = [
    client.models.embed_content(
        model="gemini-embedding-2", contents=doc["text"]
    ).embeddings[0].values
    for doc in documents
]

# 2. Create an index (dimension must match your embedding model)
# if "order" not in pc.list_indexes():
#     pc.create_index(
#         name="order",
#         dimension=len(embedding_list[0]),
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1")
#     )

index = pc.Index(host="http://localhost:5081")

# 3. Build vectors: id + values + metadata (keep the raw text for display!)
vectors = [
    {
        "id": doc["id"],
        "values": embedding_list[i],
        "metadata": {"category": doc["category"], "text": doc["text"]},
    }
    for i, doc in enumerate(documents)
]

# 4. Upsert (insert or update)
index.upsert(vectors=vectors)
print(index.describe_index_stats())
```

### 🔍 Then, at query time:
```python
query_vec = client.models.embed_content(
    model="gemini-embedding-2", contents="Can I get a refund?"
).embeddings[0].values

results = index.query(vector=query_vec, top_k=3, include_metadata=True)
# results.matches → [{"id": "doc-1", "score": 0.92, "metadata": {...}}, ...]
```

> 💡 **Beginner tip:** store the **original text in metadata**. The vector is only for matching; you need the readable text to build the LLM prompt.

---

## ✅ Key Takeaways

- Vector DB = semantic search engine over your documents.
- Match dimension to your embedding model; keep raw text as metadata.
- Prototype with ChromaDB/FAISS; scale with Pinecone/pgvector.
- `upsert` = insert-or-update; `top_k` controls how much context you fetch.

➡️ Next: [Retrieval Mechanics — finding the needle](./06_retrieval_mechanics.md)
