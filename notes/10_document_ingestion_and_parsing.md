# 📥 Module 10: Document Ingestion & Parsing — The Forgotten Half

> *Source: Red Hat "Engineering RAG for the enterprise", Omdena "Document Parsing for RAG" (2026)*

> 💡 *"Poor parsing leads to weak chunking, low-quality embeddings, and inaccurate retrieval. Fixing parsing often delivers faster gains than changing models."*

Most RAG tutorials skip this stage and hand you a `.txt` file. Real applications start with **messy enterprise documents** — and that's where RAG pipelines actually fail.

---

## 🔄 Where ingestion fits

```
Raw documents → PARSE → clean/normalize → extract metadata → chunk → embed → store
                ↑
        (Module 4 starts here)
```

Chunking (Module 4) can only work with what parsing gives it. **Garbage in = garbage out.**

---

## 📄 Document types & their quirks

| Type | Challenge | Approach |
|---|---|---|
| **Text-based PDF** | Multi-column layouts, headers/footers, page numbers | Layout-aware parser (preserve reading order) |
| **Scanned PDF** | No text layer — just pixels | **OCR** (Tesseract) or vision model |
| **DOCX/PPTX** | Styles, bullets, speaker notes | Structure-aware parser (headings → hierarchy) |
| **XLSX/CSV** | Tables | Extract tables separately (rows → markdown table) |
| **HTML** | Nav menus, ads, scripts | HTML-to-markdown (drop boilerplate) |
| **Images/figures** | Charts with embedded data | Caption with a vision model (multimodal) |

**Common parsing traps:**
- Reading order destroyed in multi-column PDFs (left column mixed with right)
- Tables flattened into one long run-on sentence
- Headers/footers repeated into every chunk
- Scanned docs silently returning *empty* text — always verify extraction yield!

---

## 🛠️ Tooling (2026 defaults)

| Tool | Best for |
|---|---|
| **Docling** (IBM, open-source; recommended by Red Hat) | PDF/DOCX/PPTX/XLSX/HTML with layout + table structure |
| **Unstructured** | Enterprise pipelines, partition-by-element |
| **LlamaParse** | Hard PDFs, tables, fast managed API |
| **PyPDF / pypdfium2** | Simple text-based PDFs (cheap, no structure) |
| **Tesseract** | OCR for scans |
| **Vision LLM** | Figures, charts, handwritten docs |

> **Rule:** *one parser does not fit all.* A multi-column PDF, a scanned invoice, and a slide deck each need different treatment.

---

## 🏷️ Metadata extraction (do this before chunking)

Every chunk should carry metadata for **filtering, citation, and audit**:

```json
{
  "source": "policies/refund-policy.pdf",
  "page": 12,
  "section": "3.2 Refund Window",
  "doc_date": "2026-01-15",
  "category": "refund"
}
```

Why it matters:
- **Filtering** — "only answer from docs dated after 2025"
- **Citations** — "…refund policy (p.12)" in the final answer
- **Deletion/access control** — find and purge a user's data (see OWASP, Module 11 territory)

---

## 💻 Code: Parse → Chunk → Store (a complete ingestion snippet)

```python
# pip install docling langchain-text-splitters
from docling.document_converter import DocumentConverter
from langchain_text_splitters import RecursiveCharacterTextSplitter

converter = DocumentConverter()
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)

def ingest(pdf_path: str):
    # 1. PARSE — layout-aware, preserves headings and tables
    result = converter.convert(pdf_path)
    text = result.document.export_to_markdown()

    # sanity check: empty extraction = scanned PDF → fall back to OCR
    assert len(text.strip()) > 100, f"Parsing extracted nothing from {pdf_path}"

    # 2. CHUNK
    chunks = splitter.split_text(text)

    # 3. ATTACH METADATA (source, page, section)
    return [
        {"text": c, "metadata": {"source": pdf_path, "doc_date": "2026-01-15"}}
        for c in chunks
    ]

documents = ingest("policies/refund-policy.pdf")
print(f"Parsed {len(documents)} chunks")

# 4. EMBED + STORE  →  see Module 5 (Vector Databases)
```

---

## 🗃️ Storage layout (production pattern)

Keep three things **separate** (Splunk/Red Hat advice) so a re-index never forces a re-encode:

| Data | Store |
|---|---|
| Raw documents | Object storage (S3 / Blob) |
| Chunks + embeddings | Vector DB (Module 5) |
| Metadata / chat history | Postgres (or other relational) |

### ♻️ Incremental indexing
Re-embedding your whole corpus on every change is slow and wasteful. Production systems:
1. **Hash** each document's content.
2. Only process documents whose hash **changed**.
3. **Propagate deletions** — removing a file from the source must remove its chunks from the index (a classic OWASP compliance failure).

---

## ✅ Key Takeaways

- Parsing quality > model choice; fix it first.
- Layout-aware parsing for PDFs; OCR for scans; vision models for figures.
- Attach metadata (source/page/section/date) **before** chunking.
- Separate raw docs, vectors, and metadata — enables incremental updates.
- Always verify extraction yield (empty text = silent failure).

⬅️ Back to [RAG Evaluation](./09_rag_evaluation.md) · 🏠 [Notes Index](./README.md)
