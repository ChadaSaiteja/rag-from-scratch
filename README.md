# rag-pipeline

A minimal Retrieval-Augmented Generation (RAG) pipeline built with LangChain, ChromaDB, and Google Gemini embeddings.

Corpus: 10 short plain-text lessons on RAG itself, in `data/raw/`.

## Status

The **retrieval** half of RAG is implemented. The **generation** half is not built yet.

| Stage | File | Status |
| --- | --- | --- |
| Load + chunk + embed + store | `src/pipeline/ingest.py` | done |
| Vector (semantic) retrieval | `src/pipeline/vectorRetrieval.py` | done |
| Hybrid (BM25 + vector) retrieval | `src/pipeline/hybridSearch.py` | done |
| Prompt construction | — | not started |
| LLM generation | — | not started |
| API layer | — | not started |

## Layout

```
rag-pipeline/
├── .env.example              # template for required secrets
├── .gitignore
├── requirements.txt
├── README.md
├── data/
│   └── raw/                  # source .txt corpus (committed)
└── src/
    ├── __init__.py
    ├── app/
    │   └── __init__.py       # reserved for api.py, llm.py, query.py
    └── pipeline/
        ├── __init__.py
        ├── ingest.py
        ├── vectorRetrieval.py
        └── hybridSearch.py
```

Generated artifacts (`.venv/`, `chroma_gemini_db/`, `__pycache__/`) are gitignored and not part of the source tree.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env          # then paste your GOOGLE_API_KEY
```

## Usage

Run everything from the repo root — see the caveat below.

```bash
# 1. Build the vector store (calls the Gemini embeddings API)
python src/pipeline/ingest.py

# 2. Query it
python src/pipeline/vectorRetrieval.py
python src/pipeline/hybridSearch.py
```

## Known issues

These are real bugs in the current code, tracked here rather than fixed, so they are not lost:

1. **Relative `persist_directory`.** All three scripts use `persist_directory="./chroma_gemini_db"`, which resolves against the current working directory rather than the script location. Ingesting from the repo root writes the store to the root; querying from `src/pipeline` reads a different, possibly empty store. Always run from a consistent directory until this is fixed to an absolute path.
2. **Re-ingest duplicates data.** `ingest_pipeline` appends into the existing `gemini_collection` with no reset or deterministic IDs. Running it twice doubles the corpus. Delete `chroma_gemini_db/` before re-ingesting.
3. **Import-time side effects.** `vectorRetrieval.py` and `hybridSearch.py` execute queries at module level with no `if __name__ == "__main__":` guard, so importing either costs an embedding API call. They are not importable as modules.
4. **No relevance floor.** Retrieval returns top-k unconditionally with no score threshold, so unrelated questions still return chunks.
5. **No tests.** Nothing verifies ingestion or retrieval output.
