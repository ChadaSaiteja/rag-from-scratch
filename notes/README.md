# 📓 Notes — Building a RAG Application

> Beginner-friendly study notes for building a **Retrieval-Augmented Generation (RAG)** application.
> A self-paced course with plain-English explanations, analogies, runnable code, self-check quizzes, and 2026 production best practices — each module linked to the working pipeline in this repo.

---

## 💡 The Idea Behind These Notes

**What this is:** a complete, self-paced path from "what is an LLM?" to "a production-grade RAG system" — one concept at a time, each module building directly on the last.

**Who it's for:** beginners who know basic Python but have never built a RAG app, and practitioners who want the modern (2026) retrieval stack — hybrid search, reranking, evaluation — in one place.

**What makes it different:**
- Plain-English analogies & "beginner trap" warnings in every module
- Runnable code for each stage, aligned with the stack used in this repo (Gemini + Chroma + LangChain)
- **🔗 In this repo** sections that connect every concept to the real implementation in `src/pipeline/`
- **🧠 Self-check** questions with answers at the end of each module
- Modern topics most tutorials skip: **reranking**, **hybrid search** (BM25 + vector + RRF), **query rewriting/HyDE**, **RAG evaluation** (RAGAS, hit rate, MRR), **CI quality gates**, **document ingestion & incremental indexing**, and **production security, cost and monitoring**
- An end-to-end **capstone build** (Module 8), a [Glossary](./glossary.md) and an [FAQ](./faq.md)

**The through-line:** RAG = *ingest → chunk → embed → store → retrieve → generate*. Every module owns exactly one stage of that pipeline, so by Module 8 you can see how all the pieces snap together.

---

## 🧭 How to Go Through These Notes

### The golden rule
**Read in order, 00 → 11.** Each module assumes the previous one (e.g., chunking needs context windows from Module 1, embeddings need Module 2, retrieval needs Modules 2 + 4 + 5). Skipping around will leave gaps.

### The study loop (repeat per module)
1. **Read** the concept + analogy.
2. **Run** the code snippet — don't just read it.
3. **Try the 🧪 experiment or exercise**, many of them change the real pipeline in this repo.
4. **Answer the 🧠 Self-check** questions before opening the answers.
5. **Review the ✅ Key Takeaways**, then follow the ➡️ Next link.

> Budget ~30–45 min per module; Module 8 (capstone) takes 1.5–2 hours.

### Choose your path

| Your situation | Path |
|---|---|
| **Absolute beginner** | 00 → 01 → … → 10 in strict order; run every code cell |
| **Already know embeddings/LLMs** | Skim 00–02, then 03 → 04 → 05 → 06 → 07 → 08 |
| **Want production RAG now** | 00 → 04 → 06 → 08, then backfill 01–03, 05 as needed; finish with 09 (eval), 10 (ingestion) and 11 (production) |
| **Debugging a broken RAG app** | Go straight to 06 (retrieval) + 09 (evaluation) to learn what to measure, then 04 (chunking); check the [FAQ](./faq.md) |
| **Shipping to real users** | 08 → 09 → 10 → 11 (security, cost, latency, monitoring) |

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
                                            11 Production RAG ← security, cost, latency, monitoring
```

---

## 📚 Index

| # | Note | Covers | In this repo |
|---|---|---|---|
| 0 | [Introduction to RAG](./00_introduction_to_rag.md) | What/why RAG, workflow, RAG vs fine-tuning vs long context | — |
| 1 | [Foundations: Transformers & Tokens](./01_foundations_transformers.md) | Tokens, settings, context window, lost in the middle, token budgets | [`config.py`](https://github.com/ChadaSaiteja/rag-from-scratch/blob/master/src/pipeline/config.py) |
| 2 | [Embeddings & Vectors](./02_embeddings_and_vectors.md) | Semantic vectors, similarity metrics, dimensions, task types, choosing a model | [`ingest.py`](https://github.com/ChadaSaiteja/rag-from-scratch/blob/master/src/pipeline/ingest.py) |
| 3 | [Prompting Techniques](./03_prompting_techniques.md) | Zero/few-shot, CoT, chaining, production RAG prompt, injection defence | [`generation.py`](https://github.com/ChadaSaiteja/rag-from-scratch/blob/master/src/pipeline/generation.py) |
| 4 | [Chunking Strategies](./04_chunking_strategies.md) | Fixed, recursive, structure-aware, semantic, contextual; characters vs tokens | [`ingest.py`](https://github.com/ChadaSaiteja/rag-from-scratch/blob/master/src/pipeline/ingest.py) |
| 5 | [Vector Databases](./05_vector_databases.md) | Choosing a DB, ANN indexes (HNSW/IVF/PQ), Chroma + Pinecone code | [`vector_retrieval.py`](https://github.com/ChadaSaiteja/rag-from-scratch/blob/master/src/pipeline/vector_retrieval.py) |
| 6 | [Retrieval Mechanics](./06_retrieval_mechanics.md) | BM25, hybrid + RRF (worked example), reranking code, query rewriting, filters | [`hybrid_search.py`](https://github.com/ChadaSaiteja/rag-from-scratch/blob/master/src/pipeline/hybrid_search.py) |
| 7 | [Agentic RAG & Tool Calling](./07_agentic_rag_and_tools.md) | Structured output, tool calls, agent loop, guardrails | exercise |
| 8 | [Build Your First RAG App](./08_building_your_first_rag_app.md) | End-to-end capstone (Chroma + Gemini), reranking, RAGAS, production checklist | [`src/pipeline/`](https://github.com/ChadaSaiteja/rag-from-scratch/tree/master/src/pipeline) |
| 9 | [RAG Evaluation (Deep Dive)](./09_rag_evaluation.md) | Hit rate, MRR, nDCG, RAGAS, test sets, LLM-as-judge, CI gates | [`evaluation.py`](https://github.com/ChadaSaiteja/rag-from-scratch/blob/master/src/pipeline/evaluation.py) |
| 10 | [Document Ingestion & Parsing](./10_document_ingestion_and_parsing.md) | PDF/DOCX/scans/OCR/tables, metadata, incremental indexing code | [`ingest.py`](https://github.com/ChadaSaiteja/rag-from-scratch/blob/master/src/pipeline/ingest.py) |
| 11 | [Production RAG](./11_production_rag.md) | Threat model, access control, cost & latency budgets, caching, monitoring | exercises |
| 📖 | [Glossary](./glossary.md) | Every term in plain English, linked to its module | — |
| ❓ | [FAQ](./faq.md) | Short answers to the most common RAG questions | — |

### Conventions used in every module
- 🧠 **Analogy** — the one mental model that makes the idea stick
- ⚠️ **Beginner trap** — the mistake everyone makes once
- 🧪 **Experiment** — a 5-minute tweak that proves the concept
- ✅ **Key takeaways** — the 3–5 lines worth remembering
- 🔗 **In this repo** — where the concept lives in the real pipeline
- 🧠 **Self-check** — questions with hidden answers

---

## ▶️ Quick start

**Prerequisites:** Python 3.10+ and a `GOOGLE_API_KEY` in a `.env` file (free Google AI Studio key).

```bash
pip install google-genai langchain-chroma langchain-google-genai langchain-text-splitters python-dotenv
```

Then follow [Module 8](./08_building_your_first_rag_app.md) step by step — it wires Modules 1–7 into one working pipeline. To run the full pipeline in this repo instead, see the [main README](../README.md#setup); optional extras used in later modules: `sentence-transformers` (reranking), `ragas` (evaluation), `docling` (document parsing).
