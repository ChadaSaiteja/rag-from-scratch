# 🗄️ Module 5: Vector Databases — Your AI's Memory

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

## ⚡ How vector search stays fast: ANN indexes

Comparing a query against *every* stored vector (exact k-nearest-neighbour search) is perfect but slow: a million 3,072-dimension vectors means billions of multiplications per query. Vector databases use **approximate nearest neighbour (ANN)** indexes that trade a tiny bit of accuracy for huge speed-ups.

| Index | Idea (plain English) | Trade-off |
|---|---|---|
| **Flat / brute force** | Compare against everything | 100% accurate; fine up to tens of thousands of vectors |
| **HNSW** (Hierarchical Navigable Small World) | A multi-layer graph: start at a coarse layer, hop toward closer neighbours, zoom in | Very fast and accurate; uses more memory. The default in Chroma, pgvector (optional), Weaviate and many others |
| **IVF** (Inverted File) | Cluster vectors into buckets; search only the few buckets nearest the query | Less memory than HNSW; accuracy depends on how many buckets you probe |
| **PQ** (Product Quantization) | Compress each vector into a short code | Massive memory savings for billions of vectors; some accuracy loss. Often combined with IVF |

> 🧠 **Analogy:** exact search is reading every book in the library. HNSW is asking a librarian who knows which shelf to walk to, then which row, then which book.

**Recall** in this context means *how often the ANN index returns the same results exact search would*. Well-tuned indexes reach 95–99% recall at a fraction of the cost.

---

## 💻 Code: Chroma (local, no server), as used in this repo

Chroma runs inside your Python process and saves to a folder, which makes it ideal for learning and prototypes:

```python
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

docs = [
    Document(page_content="Customers can request a refund within 30 days of purchase.",
             metadata={"category": "refund", "filename": "refund-policy.md"}),
    Document(page_content="Installation appointments can be rescheduled up to 24 hours before.",
             metadata={"category": "installation", "filename": "installation.md"}),
]

store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="demo",
    persist_directory="./chroma_demo_db",
    ids=["refund-1", "install-1"],          # stable IDs → re-running updates instead of duplicating
)

# Similarity search, optionally restricted by metadata
results = store.similarity_search_with_score("Can I get my money back?", k=2,
                                             filter={"category": "refund"})
for doc, score in results:
    print(round(score, 3), doc.metadata["filename"], doc.page_content)
```

> ⚠️ Chroma's default distance metric is **L2**, so the score printed above is a *distance*: lower is better. You can choose cosine when creating the collection; see the Chroma docs for the setting in your version.

---

## 💻 Code: Embed Documents & Upsert to Pinecone

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
        model="gemini-embedding-001", contents=doc["text"]
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
    model="gemini-embedding-001", contents="Can I get a refund?"
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
- ANN indexes (HNSW, IVF, PQ) make search fast by being *approximately* right.
- Give chunks stable IDs so re-ingestion updates instead of duplicating.

---

## 🔗 In this repo

- [`ingest.py`](https://github.com/ChadaSaiteja/rag-from-scratch/blob/master/src/pipeline/ingest.py) creates the store with `Chroma.from_documents(..., persist_directory=PERSIST_DIRECTORY)`.
- [`vector_retrieval.py`](https://github.com/ChadaSaiteja/rag-from-scratch/blob/master/src/pipeline/vector_retrieval.py) reopens it and runs `similarity_search(query, k=RETRIEVAL_K)`.
- ⚠️ **Known issue:** `ingest.py` doesn't pass `ids`, so running it twice stores every chunk twice. 🧪 **Exercise:** generate IDs from `filename` + chunk position (or a content hash, Module 10) and pass `ids=` so re-runs overwrite instead of duplicating.

---

## 🧠 Self-check

<details markdown="1">
<summary>1. Why store the original text in metadata (or alongside the vector)?</summary>

The vector is only used for matching; you can't turn it back into text. To build the LLM prompt and show citations you need the original chunk text and its source.

</details>

<details markdown="1">
<summary>2. Your index has 5,000 vectors. Do you need HNSW?</summary>

Not really. Brute-force search over a few thousand vectors takes milliseconds. ANN indexes matter at hundreds of thousands to billions of vectors. (Many databases use HNSW by default anyway, which is fine.)

</details>

<details markdown="1">
<summary>3. Chroma returns scores 0.41 and 0.93 for two chunks. Which is the better match?</summary>

With Chroma's default L2 metric these are distances, so 0.41 is closer and therefore the better match. Always check whether a score is a distance or a similarity.

</details>

➡️ Next: [Retrieval Mechanics — finding the needle](./06_retrieval_mechanics.md)
