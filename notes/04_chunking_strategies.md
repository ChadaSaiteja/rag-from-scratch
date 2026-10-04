# ✂️ Module 4: Chunking Strategies — Breaking Documents into Bites

> *Source: `AI/day9/chunkingStrategies.md` + `AI/day9/chunking.py`*

**Chunking** = breaking a large document into smaller segments. The trick: chunks big enough to hold meaning, small enough to stay focused.

---

## ❓ Why do we need chunking?

- LLMs have a **context limit** (Module 1).
- Loading everything = hallucinations + excessive token usage.
- Retrieval works best when the unit of search matches the unit of answer.

## 🤔 Choosing a strategy — ask yourself:

1. **What kind of data?** (PDF, HTML, Markdown, code, tables)
2. **Which embedding model?** (domain-specific models chunk differently)
3. **How long/complex are user queries?** (short queries → smaller, focused chunks)
4. **How will results be used?** (semantic search vs. RAG vs. agentic workflow)

---

## 🔨 Chunking Methods

### 1. Fixed-size chunking
Pick a size (e.g. 500 tokens) + overlap (e.g. 50) and slice. Fast, but can cut mid-sentence.

### 2. Content-aware chunking
Respects the document's structure:
- **Naive splitter** — splits on `,`, `.`, newlines, whitespace.
- **NLTK** — trained sentence tokenizer → more meaningful chunks.
- **spaCy** — powerful NLP library for sentence/entity-aware splitting.

### 3. Recursive character-level chunking ⭐
LangChain's `RecursiveCharacterTextSplitter` splits on a priority list of separators (e.g. `["\n\n", "\n", ".", " "]`) until chunks fit the size. **Best default for general text.**

### 4. Document structure-based
- **PDF** — headings/subheadings first (LangChain has cleanup utilities).
- **HTML** — parse tags, or use LangChain splitters.
- **Markdown** — chunk by headers/sections.
- **LaTeX** — chunk by its own syntax.

### 5. Semantic chunking
1. Break into sentences.
2. Group sentences with neighbors.
3. Embed each group.
4. Split where consecutive embeddings *diverge* (topic shift).

### 6. Contextual chunking
Add a small context prefix to each chunk (e.g., the doc title + section heading) so chunks make sense *out of context* — a 2024–2026 best practice that boosts retrieval recall.

---

## 💻 Code: Recursive Splitting in Action (from `day9/chunking.py`)

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """# Order Installation Guide

## 1. Order Lifecycle
When a customer places an order, the order is first created in the ordering system...
...
"""

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
texts = text_splitter.split_text(text)

print(texts)          # list of chunks
print(len(texts))     # how many chunks we got
```

**Why overlap?** A sentence split across two chunks can be found from *either* side. Overlap (typically 10–20% of chunk size) prevents "lost information" at boundaries.

### 🧪 Beginner experiment
Run the same text with `chunk_size=100` vs `chunk_size=500` and compare `len(texts)`. Then embed each chunk (Module 2) and search for *"When is an order created?"* — notice which chunk size retrieves the right answer.

---

## 🏆 2026 Best-Practice Stack (from production research)

| Layer | Recommendation |
|---|---|
| Strategy | Structure-aware first (headers/sections), then recursive fallback |
| Size | 300–800 tokens for most docs; smaller for FAQs, larger for narrative |
| Overlap | 10–20% of chunk size |
| Metadata | Attach `source`, `page`, `section`, `date` to every chunk |
| Contextual prefix | Prepend doc title + heading to each chunk before embedding |

> 💡 **Golden rule:** your embedding model's max input length caps your chunk size. Never exceed it.

---

## ✅ Key Takeaways

- Chunk size is the #1 lever of RAG quality.
- Recursive character splitting is the safe default.
- Semantic + contextual chunking = premium retrieval.
- Always carry metadata — it powers filtering later.

➡️ Next: [Vector Databases — your AI's memory](./05_vector_databases.md)
