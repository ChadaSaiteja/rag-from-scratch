# 🔢 Module 2: Embeddings — Turning Words into Numbers

> *Source: `AI/day2/embedding.md` + `AI/day2/embedding.py`*

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
4. **Convert tokens → numerical vectors** — via an embedding model (e.g., `gemini-embedding-2`, OpenAI `text-embedding-3`, Cohere, BAAI/bge).

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

## 💻 Code: Generate & Compare Embeddings (from `day2/embedding.py`)

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
        model="gemini-embedding-2", contents=sentence
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
Add `"I adore playing soccer"` to the list. Its similarity to `"I love playing football."` will still be high — even though *no word overlaps*. That's why embeddings beat keyword search for RAG.

> ⚠️ **Beginner trap:** always embed the **query** and the **chunks** with the *same* model. Mixing models = apples vs. oranges.

---

## ✅ Key Takeaways

- Embeddings = meaning as numbers.
- Cosine similarity measures *directional* closeness (0→1 scale).
- Same embedding model for documents AND queries — non-negotiable.
- This is the "Retrieval" fuel in RAG.

➡️ Next: [Prompting Techniques — talking to the LLM](./03_prompting_techniques.md)
