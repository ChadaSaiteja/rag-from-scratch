# 🔢 Module 2: Embeddings — Turning Words into Numbers

Computers don't understand language — they understand **numbers**. An **embedding** is a numerical representation (a vector) that captures the *meaning* of a word, sentence, or document.

---

## 🧭 The Core Idea: Meaning = Position in Space

Every piece of text becomes a point in a multi-dimensional space. **Similar meanings land near each other.**

```
A = "I love football"
B = "I love soccer"
→ A and B point in almost the same direction, even though the words differ.
```

This is the magic that powers RAG retrieval: your query and the right document can use *different words* but still be *semantically close*.

---

## 🏭 Process of Creating Vector Embeddings

1. **Get the raw data** — documents, PDFs, web pages.
2. **Clean the data** — tokenization, noise removal, preprocessing.
3. **Break into tokens** — small individual pieces.
4. **Convert tokens → numerical vectors** — via an embedding model (e.g., Google `gemini-embedding-001`, which this repo uses, OpenAI `text-embedding-3`, Cohere Embed, or open-source BAAI/bge).

---

## 📐 Measuring Similarity: Cosine Similarity

Instead of distance, we measure the **angle** between two vectors:

```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
```

| Score | Interpretation |
|---|---|
| **1.0** | Very similar direction |
| **0.8** | Strongly similar |
| **0.5** | Somewhat related |
| **0.0** | Little/no relationship |
| **-1.0** | Opposite direction |

---

## 🧠 Semantic Relationships

Embeddings capture *meaning*, so math works on concepts:

```
        KING (●)
         ↑  gender relationship
        MAN (●)

  Replace "man" with "woman"  →  KING − MAN + WOMAN ≈ QUEEN
```

---

## 💻 Code: Generate & Compare Embeddings

```python
import os
import numpy as np
from dotenv import load_dotenv
load_dotenv()
from google import genai

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

sentences = [
    "I love playing football.",
    "I really enjoy playing soccer.",
    "Football is my favorite sport.",
    "I like eating pizza.",
    "The weather is very cold today.",
]

# 1. Convert each sentence into a vector
embedding_list = [
    client.models.embed_content(
        model="gemini-embedding-001", contents=sentence
    ).embeddings[0].values
    for sentence in sentences
]

# 2. Compare every pair with cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)

for i in range(len(embedding_list)):
    for j in range(i + 1, len(embedding_list)):
        sim = cosine_similarity(np.array(embedding_list[i]), np.array(embedding_list[j]))
        print(f"{sentences[i]}  ↔  {sentences[j]}  →  {sim:.4f}")
```

**Expected insight:** football/soccer sentences will score **high (~0.8+)**; pizza/weather sentences will score **low**. That's semantic similarity in action.

### 🧪 Try this beginner experiment
Add `"I adore kicking a ball around on the pitch"` to the list. Its similarity to `"I love playing football."` will still be high, even though the two sentences share **no content words**. A keyword search would miss it completely. That's why embeddings beat keyword search for paraphrased questions.

> ⚠️ **Beginner trap:** always embed the **query** and the **chunks** with the *same* model. Mixing models = apples vs. oranges.

---

## 📏 Dimensions, normalization and distance metrics

Every embedding model outputs vectors of a fixed length, its **dimension** (for example 768, 1536 or 3072 numbers). More dimensions can capture finer distinctions but cost more storage and slightly slower search.

| Metric | Formula idea | Use when |
|---|---|---|
| **Cosine similarity** | Angle between vectors, ignores length | Default for text embeddings |
| **Dot product** | Angle *and* length | Vectors are already normalized (then it equals cosine and is faster) |
| **Euclidean (L2) distance** | Straight-line distance; *smaller* = more similar | Some vector DBs default to it; on normalized vectors it ranks results the same as cosine |

> ⚠️ **Beginner trap:** a vector DB's distance setting must match how the model was meant to be compared. Chroma defaults to **L2**, Pinecone asks you to choose. If scores look "backwards", check whether you're reading a *distance* (lower is better) or a *similarity* (higher is better).

### 🎚️ Shrinking vectors (Matryoshka embeddings)

Some modern models, including `gemini-embedding-001`, are trained so the **first N numbers of a vector are still a good embedding on their own**. You can request smaller vectors (for example 768 instead of 3072) to cut storage and speed up search with only a small quality loss:

```python
from google.genai import types

vec = client.models.embed_content(
    model="gemini-embedding-001",
    contents="How do refunds work?",
    config=types.EmbedContentConfig(output_dimensionality=768),
).embeddings[0].values
```

> If you truncate dimensions, re-normalize the vectors before using dot product, and use the same dimension for documents and queries.

### 🎯 Task types: documents vs. queries

Retrieval questions are short ("refund window?") while documents are long. Several embedding APIs let you say which one you're embedding so the model can place them better:

```python
doc_vec = client.models.embed_content(
    model="gemini-embedding-001", contents=chunk_text,
    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
).embeddings[0].values

query_vec = client.models.embed_content(
    model="gemini-embedding-001", contents="refund window?",
    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
).embeddings[0].values
```

### 🏷️ Choosing an embedding model

1. Start with a strong general model from your LLM provider (simple, one API key).
2. Check the public **MTEB leaderboard** (Massive Text Embedding Benchmark) and filter by the *Retrieval* task and your language.
3. **Evaluate on your own questions** (Module 9). Leaderboard rank does not guarantee the best results on your documents.
4. Remember that **changing the model means re-embedding the whole corpus**, because vectors from different models can't be mixed.

---

## ✅ Key Takeaways

- Embeddings = meaning as numbers.
- Cosine similarity measures *directional* closeness: 1 = same direction, 0 = unrelated, −1 = opposite. Text embeddings usually land between 0 and 1.
- Same embedding model for documents AND queries — non-negotiable.
- This is the "Retrieval" fuel in RAG.
- Know whether your DB reports a *distance* or a *similarity*, and which metric it uses.

---

## 🔗 In this repo

- [`ingest.py`](https://github.com/ChadaSaiteja/rag-from-scratch/blob/master/src/pipeline/ingest.py) and [`hybrid_search.py`](https://github.com/ChadaSaiteja/rag-from-scratch/blob/master/src/pipeline/hybrid_search.py) both build `GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)` from the same config value. That one shared setting is what guarantees documents and queries live in the same vector space.
- Want to try another model? Change `EMBEDDING_MODEL` in `.env`, **delete `chroma_gemini_db/`**, and re-run `ingest.py`. Old vectors can't be compared with new ones.

---

## 🧠 Self-check

<details markdown="1">
<summary>1. Two chunks score 0.82 and 0.31 cosine similarity to a query. Which is more relevant?</summary>

The 0.82 chunk. Cosine similarity is higher when vectors point in a more similar direction. (If your database reports a *distance* instead, smaller would be better.)

</details>

<details markdown="1">
<summary>2. You switched embedding models but kept the old vector store. Why did retrieval quality collapse?</summary>

Queries are now embedded by the new model, but the stored chunks were embedded by the old one. The two vector spaces aren't comparable, so the "nearest" chunks are essentially random. Re-embed the whole corpus after changing models.

</details>

<details markdown="1">
<summary>3. Why might keyword search beat embeddings for the query "ERR_4297"?</summary>

Error codes, SKUs and IDs carry no semantic meaning an embedding can generalize from, so similar-looking codes may land close together. Exact keyword matching (BM25) finds the precise string. That's the case for hybrid search (Module 6).

</details>

➡️ Next: [Prompting Techniques — talking to the LLM](./03_prompting_techniques.md)
