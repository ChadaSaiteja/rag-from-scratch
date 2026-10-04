# 📓 Notes — Building a RAG Application

> Beginner-friendly study notes for building a **Retrieval-Augmented Generation (RAG)** application.
> Written from the existing docs in `AI/day1` … `AI/day11`, improved with added explanations, analogies, runnable code, and 2026 production best practices.

## 🗺️ Learning Path (read in order)

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

## 🧭 New in these notes (vs. the original docs)
- Plain-English analogies & "beginner trap" warnings in every module
- Runnable code snippets adapted from the `day*` Python files
- **Reranking** (cross-encoders), **hybrid search** (BM25 + vector + RRF), **query rewriting/HyDE**
- **RAG evaluation** with RAGAS (faithfulness, relevancy, context precision/recall) + **CI quality gates** (promptfoo)
- **Document ingestion & parsing** (layout-aware parsing, OCR, metadata, incremental indexing)
- A **production checklist** and an end-to-end **capstone build**

## ▶️ Quick start
```bash
pip install google-genai pinecone langchain-text-splitters python-dotenv ragas datasets docling
```
Then follow [Module 8](./08_building_your_first_rag_app.md) step by step.
