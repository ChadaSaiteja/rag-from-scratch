# 📓 Notes — Building a RAG Application

> Beginner-friendly study notes for building a **Retrieval-Augmented Generation (RAG)** application.
> Written from the existing docs in `AI/day1` … `AI/day11`, improved with added explanations, analogies, runnable code, and 2026 production best practices.

---

## 💡 The Idea Behind These Notes

**What this is:** a complete, self-paced path from "what is an LLM?" to "a production-grade RAG system" — one concept at a time, each module building directly on the last.

**Who it's for:** beginners who know basic Python but have never built a RAG app, and practitioners who want the modern (2026) retrieval stack — hybrid search, reranking, evaluation — in one place.

**How it differs from the source docs (`AI/day*`):**
- Plain-English analogies & "beginner trap" warnings in every module
- Runnable code snippets adapted from the `day*` Python files
- Modern topics the original docs skip: **reranking** (cross-encoders), **hybrid search** (BM25 + vector + RRF), **query rewriting/HyDE**, **RAG evaluation** (RAGAS), **CI quality gates** (promptfoo), and **document ingestion & parsing** (layout-aware parsing, OCR, metadata, incremental indexing)
- A **production checklist** and an end-to-end **capstone build** (Module 8)

**The through-line:** RAG = *ingest → chunk → embed → store → retrieve → generate*. Every module owns exactly one stage of that pipeline, so by Module 8 you can see how all the pieces snap together.

---

## 🧭 How to Go Through These Notes

### The golden rule
**Read in order, 00 → 10.** Each module assumes the previous one (e.g., chunking needs context windows from Module 1, embeddings need Module 2, retrieval needs Modules 2 + 4 + 5). Skipping around will leave gaps.

### The study loop (repeat per module)
1. **Read** the concept + analogy.
2. **Run** the code snippet — don't just read it.
3. **Try the 🧪 beginner experiment** at the end of most modules (change a parameter, observe the result).
4. **Review the ✅ Key Takeaways**, then follow the ➡️ Next link.

> Budget ~30–45 min per module; Module 8 (capstone) takes 1.5–2 hours.

### Choose your path

| Your situation | Path |
|---|---|
| **Absolute beginner** | 00 → 01 → … → 10 in strict order; run every code cell |
| **Already know embeddings/LLMs** | Skim 00–02, then 03 → 04 → 05 → 06 → 07 → 08 |
| **Want production RAG now** | 00 → 04 → 06 → 08, then backfill 01–03, 05 as needed; finish with 09 (eval) and 10 (ingestion) |
| **Debugging a broken RAG app** | Go straight to 06 (retrieval) + 09 (evaluation) to learn what to measure, then 04 (chunking) |

### Module map & dependencies

```
00 Intro ──→ 01 Foundations (tokens, context window)
                 │
                 ├──→ 02 Embeddings ──→ 05 Vector DBs ──┐
                 │                                      ├──→ 06 Retrieval ──→ 07 Agentic RAG
                 └──→ 03 Prompting ─────────────────────┘         │
                                                                  ▼
04 Chunking (needs 01, 02) ───────────────────────────────→ 08 Build the App
                                                                 │
                                            09 Evaluation ←──────┘ ← deep dive on RAGAS/CI gates
                                            10 Ingestion & Parsing ← pre-04 stage for real docs
```

---

## 📚 Index

| # | Note | Covers | Source docs |
|---|---|---|---|
| 0 | [Introduction to RAG](./00_introduction_to_rag.md) | What/Why RAG, workflow, benefits | — |
| 1 | [Foundations: Transformers & Tokens](./01_foundations_transformers.md) | Tokens, parameters, settings, context window, hallucinations | `day1`, `day7` |
| 2 | [Embeddings & Vectors](./02_embeddings_and_vectors.md) | Semantic vectors, cosine similarity, embedding code | `day2` |
| 3 | [Prompting Techniques](./03_prompting_techniques.md) | Zero/One/Few-shot, CoT, chaining, RAG prompt template | `day3` |
| 4 | [Chunking Strategies](./04_chunking_strategies.md) | Fixed, recursive, semantic, contextual chunking + code | `day9` |
| 5 | [Vector Databases](./05_vector_databases.md) | Pinecone/Chroma/FAISS/pgvector, upsert + query code | `day8` |
| 6 | [Retrieval Mechanics](./06_retrieval_mechanics.md) | Sparse/dense/hybrid, RRF, reranking, query rewriting | `day11` + research |
| 7 | [Agentic RAG & Tool Calling](./07_agentic_rag_and_tools.md) | Structured output, tool calls, ReAct, agentic RAG | `day4` |
| 8 | [Build Your First RAG App](./08_building_your_first_rag_app.md) | End-to-end capstone, RAGAS evaluation, production checklist | all |
| 9 | [RAG Evaluation (Deep Dive)](./09_rag_evaluation.md) | RAGAS metrics math, test-set generation, LLM-as-judge, CI gates | research |
| 10 | [Document Ingestion & Parsing](./10_document_ingestion_and_parsing.md) | PDF/DOCX/scans/OCR/tables, Docling, metadata, incremental indexing | research |

### Conventions used in every module
- 🧠 **Analogy** — the one mental model that makes the idea stick
- ⚠️ **Beginner trap** — the mistake everyone makes once
- 🧪 **Experiment** — a 5-minute tweak that proves the concept
- ✅ **Key takeaways** — the 3–5 lines worth remembering

---

## ▶️ Quick start

**Prerequisites:** Python 3.10+, a `GOOGLE_API_KEY` in a `.env` file (free Google AI Studio key), and (optionally) a local Pinecone instance for the storage examples.

```bash
pip install google-genai pinecone langchain-text-splitters python-dotenv ragas datasets docling
```

Then follow [Module 8](./08_building_your_first_rag_app.md) step by step — it wires Modules 1–7 into one working pipeline, and you can come back to earlier modules whenever a step is unclear.

**Original docs:** `AI/day1` … `AI/day11` · **Code samples:** `embedding.py`, `chunking.py`, `pine_cone.py`, `toolcalling.py`
