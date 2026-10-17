# ✂️ Module 4: Chunking Strategies — Breaking Documents into Bites

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

## 💻 Code: Recursive Splitting in Action

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

> ⚠️ **Beginner trap: characters, not tokens.** `chunk_size` in `RecursiveCharacterTextSplitter` counts **characters** by default. `chunk_size=500` is about 125 tokens, not 500. If you want to size chunks in tokens (which is what models and costs care about), use a token-aware length function:

```python
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=400, chunk_overlap=50   # now measured in tokens (needs: pip install tiktoken)
)
```

### 🧮 Worked example: how many chunks?

A 40,000-character document with `chunk_size=1000` and `chunk_overlap=200`:
- Each new chunk advances by roughly `1000 − 200 = 800` characters.
- `40,000 / 800 ≈ 50` chunks (a few more or less, because the splitter prefers to break at paragraphs and sentences).
- Without overlap you'd get about 40 chunks. The extra ~10 chunks are the price of not losing facts at boundaries: about 25% more embedding cost and storage.

### 📑 Structure-aware splitting for Markdown

Splitting on headings first keeps each chunk inside one topic and gives you free metadata:

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "title"), ("##", "section")]
)
sections = header_splitter.split_text(markdown_text)  # Documents with title/section metadata

# Then cap the size of very long sections
size_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = size_splitter.split_documents(sections)
print(chunks[0].metadata)  # {'title': 'Order Installation Guide', 'section': '1. Order Lifecycle'}
```

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

> 💡 **Golden rule:** your embedding model's max input length caps your chunk size. Never exceed it; check your model's documented limit (for example, about 2,048 tokens for `gemini-embedding-001`). Text beyond the limit is typically truncated or rejected.

### 🩺 Symptoms of bad chunking

| Symptom | Likely cause | Try |
|---|---|---|
| Answers miss a detail that's clearly in the document | Chunks too big; the detail is diluted by unrelated text | Smaller chunks, structure-aware splitting |
| Answers lack context ("it", "this policy" with no referent) | Chunks too small or cut mid-thought | Larger chunks, more overlap, contextual prefixes |
| Same passage retrieved several times | Very high overlap or duplicate ingestion | Lower overlap; deduplicate; use deterministic IDs (Module 10) |
| Tables or code come back mangled | Splitter broke structured content | Split tables and code blocks as whole units |

---

## ✅ Key Takeaways

- Chunk size is the #1 lever of RAG quality.
- Recursive character splitting is the safe default.
- Semantic + contextual chunking = premium retrieval.
- Always carry metadata — it powers filtering later.
- `chunk_size` is measured in characters unless you use a token-based length function.

---

## 🔗 In this repo

- [`ingest.py`](https://github.com/ChadaSaiteja/rag-from-scratch/blob/master/src/pipeline/ingest.py) → `chunk()` uses `RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE)` with `CHUNK_SIZE=1000` **characters** (≈250 tokens). Overlap isn't set, so LangChain's default of 200 characters applies.
- Each chunk copies its parent's metadata (`source` and `filename`), which is what makes citations possible later in `generation.py`.
- Read the matching lesson: [`data/raw/03_chunking.txt`](https://github.com/ChadaSaiteja/rag-from-scratch/blob/master/data/raw/03_chunking.txt).
- 🧪 **Exercise:** set `CHUNK_SIZE=400` in `.env`, delete `chroma_gemini_db/`, re-ingest, and compare `evaluation.py` scores before and after.

---

## 🧠 Self-check

<details markdown="1">
<summary>1. You set chunk_size=500 expecting 500-token chunks. What did you actually get?</summary>

About 500 *characters*, roughly 125 tokens, because `RecursiveCharacterTextSplitter` measures length in characters by default. Use `from_tiktoken_encoder()` to size chunks in tokens.

</details>

<details markdown="1">
<summary>2. What problem does chunk overlap solve, and what does it cost?</summary>

It keeps a sentence or idea that straddles a boundary fully present in at least one chunk. The cost is more chunks: more embedding calls, more storage, and some duplicate text in retrieval results.

</details>

<details markdown="1">
<summary>3. Your FAQ document has one question and answer per paragraph. What chunking strategy fits best?</summary>

Split on the natural unit, one Q&A pair per chunk (structure-aware splitting), so each retrieved chunk is a complete answer. Small fixed-size chunks would split answers apart; huge chunks would mix unrelated questions.

</details>

➡️ Next: [Vector Databases — your AI's memory](./05_vector_databases.md)
