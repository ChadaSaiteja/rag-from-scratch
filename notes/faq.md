# ❓ RAG FAQ: Answers to Common Questions

Short, direct answers to the questions people most often ask when learning and building RAG. Each answer links to the module that goes deeper.

### What is RAG in simple terms?
RAG (Retrieval-Augmented Generation) lets an AI model look things up before it answers. When a question comes in, the system searches your documents for the most relevant passages, adds them to the prompt, and asks the model to answer using only that material. The result is answers based on your current, private information, with sources you can check. See [Module 0](./00_introduction_to_rag.md).

### Should I use RAG or fine-tuning?
Use RAG when the model needs **knowledge**: facts that change, private documents, or answers that must cite sources. Use fine-tuning when it needs a new **behaviour**: a format, tone or domain-specific style that prompting can't achieve. Many systems use both. See [Module 0](./00_introduction_to_rag.md).

### Do I still need RAG if the model has a huge context window?
Usually yes. Sending every document with every question multiplies cost and latency, and models use information in the middle of very long prompts less reliably. Retrieval sends only what's relevant. Long context is fine for a handful of short documents or one-off analysis. See [Module 1](./01_foundations_transformers.md).

### What chunk size should I use?
Start with roughly 300–800 tokens (about 1,200–3,200 characters) with 10–20% overlap, split along the document's structure (headings, paragraphs) where possible. Then measure: compare retrieval hit rate on your own test questions with two or three sizes. Note that LangChain's `RecursiveCharacterTextSplitter` counts **characters** by default. See [Module 4](./04_chunking_strategies.md).

### Which vector database should I choose?
For learning and prototypes, **Chroma** (runs locally, no server) or FAISS. If you already run Postgres, **pgvector** keeps everything in one place. For managed scale with no infrastructure, a hosted service such as Pinecone. Weaviate and others offer built-in hybrid search. Under a few hundred thousand vectors, almost any option performs well. See [Module 5](./05_vector_databases.md).

### Which embedding model is best?
There's no universal best. Start with a strong general model from your LLM provider, shortlist alternatives on the MTEB leaderboard's retrieval tasks, and pick the one that scores best on **your own** test questions. Remember that switching models means re-embedding your entire corpus. See [Module 2](./02_embeddings_and_vectors.md).

### What is hybrid search and do I need it?
Hybrid search runs keyword search (BM25) and vector search together and merges the results with Reciprocal Rank Fusion. It's worth it whenever users search for exact terms like product codes, names or error messages, which vectors handle poorly. It's cheap to add and is the common production default. See [Module 6](./06_retrieval_mechanics.md).

### Why does my RAG app give wrong or made-up answers?
Check retrieval first: print the chunks that were sent to the model. If the right information isn't there, fix chunking, add hybrid search or a reranker, or adjust top-k. If it is there but the answer is still wrong, tighten the prompt ("use only the documents", "say I don't know"), send fewer chunks, and put the best chunk first. See [Module 9](./09_rag_evaluation.md).

### How do I stop the model from answering when the documents don't contain the answer?
Tell it explicitly to reply with a fixed phrase such as "I don't know based on the provided documents", and add a relevance floor: drop retrieved chunks below a similarity threshold, and skip generation if none remain. See [Module 3](./03_prompting_techniques.md) and [Module 11](./11_production_rag.md).

### How do I evaluate a RAG system?
Score retrieval and generation separately. For retrieval, measure hit rate and MRR against questions with known correct sources. For generation, use faithfulness and answer relevancy, for example with RAGAS or an LLM-as-a-judge rubric. Run the evaluation on every change and block releases that regress. See [Module 9](./09_rag_evaluation.md).

### How many chunks should I send to the LLM?
Typically 3–6 after reranking. Retrieve wider (20–30 candidates), rerank, and keep the best few. More chunks increase cost and latency and can bury the key fact in the middle of the prompt. See [Module 6](./06_retrieval_mechanics.md).

### How do I keep the index up to date when documents change?
Hash each document's content and re-process only documents whose hash changed. Give chunks deterministic IDs so updates overwrite instead of duplicating, and delete chunks whose source file was removed. See [Module 10](./10_document_ingestion_and_parsing.md).

### How do I handle PDFs, scans and tables?
Use a layout-aware parser (such as Docling, Unstructured or LlamaParse) for PDFs and office files, OCR or a vision model for scanned documents, and extract tables separately as Markdown. Always check that extraction actually produced text. See [Module 10](./10_document_ingestion_and_parsing.md).

### Is RAG secure? What is prompt injection?
Prompt injection is text that tries to override the model's instructions; in RAG it can hide inside retrieved documents. Treat retrieved text as untrusted data, keep tools read-only, never place secrets in prompts, and enforce access control through metadata filters during retrieval rather than through prompt instructions. See [Module 11](./11_production_rag.md).

### Do I need LangChain to build RAG?
No. RAG is a few steps you can write with an embedding API, a vector store client and an LLM API, as the capstone in [Module 8](./08_building_your_first_rag_app.md) shows. Frameworks like LangChain or LlamaIndex save time with ready-made loaders, splitters and retrievers; this repo uses LangChain for those pieces while keeping each stage visible.

### What is agentic RAG and when should I use it?
Agentic RAG lets the model decide when and how often to retrieve, calling retrieval as a tool in a loop. Use it for multi-step questions or when answers need several sources or live data. Start with classic RAG; agents add latency, cost and new failure modes. See [Module 7](./07_agentic_rag_and_tools.md).

### Where can I see a complete working example?
This repository's [`src/pipeline/`](https://github.com/ChadaSaiteja/rag-from-scratch/tree/master/src/pipeline) folder contains a full pipeline: ingestion into Chroma, hybrid BM25 + multi-query vector search, citation-grounded answers with Gemini, and a retrieval-vs-generation evaluation script.
