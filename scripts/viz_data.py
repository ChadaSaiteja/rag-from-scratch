"""Data for the 3D visualizations, computed from the real corpus at build time.

- Chunks come from the same splitter and settings the pipeline uses
  (RecursiveCharacterTextSplitter, CHUNK_SIZE=1000, LangChain's default 200 overlap).
- Positions are an OFFLINE STAND-IN for embeddings: TF-IDF vectors reduced to 3D
  with PCA. No API key is needed, and the pages say so.
- Retrieval uses the pipeline's evaluation questions, top-3 per leg (RETRIEVAL_K),
  BM25 for the keyword leg, cosine over TF-IDF for the "vector" leg, and weighted
  Reciprocal Rank Fusion with 50/50 weights and c=60, the same as LangChain's
  EnsembleRetriever.
"""
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200     # LangChain default; ingest.py does not override it
RETRIEVAL_K = 3
RRF_C = 60
WEIGHTS = (0.5, 0.5)    # (bm25, vector), as in hybrid_search.py

# One distinct colour per lesson, readable on a dark background.
PALETTE = ["#5eead4", "#93c5fd", "#fbbf24", "#f472b6", "#a78bfa",
           "#4ade80", "#fb923c", "#38bdf8", "#f87171", "#e879f9"]

STOPWORDS = set("""a an and are as at be by can for from has have how if in into is it its
of on or that the their them then there these this to was what when where which while
who why will with you your we our not but do does more most than so such also each
should could would may might one two use used using too very just only any all
some other like get gets make makes many much own same here example e g""".split())

# The evaluation questions from src/pipeline/evaluation.py
EVAL_CASES = [
    ("What is the difference between chunk size and overlap?", "03_chunking.txt"),
    ("How does a vector database find similar embeddings?", "05_vector_database.txt"),
    ("What does the retriever stage do in a RAG pipeline?", "06_retrieval.txt"),
    ("How should a prompt be constructed for the generation stage?", "07_rag_prompt_and_generation.txt"),
    ("Why should retrieval and generation quality be evaluated separately?", "08_rag_evaluation.txt"),
]


def read_prompt_head(generation_py: Path):
    """The fixed instruction text of the prompt in generation.py (the part before
    'Context:'), read from the source so the visualization never drifts from it."""
    import ast
    lines = generation_py.read_text(encoding="utf-8").splitlines()
    start = next(i for i, ln in enumerate(lines) if "prompt = (" in ln)
    head = []
    for ln in lines[start + 1:]:
        s = ln.strip()
        if s.startswith('f"Context:'):
            break
        head.append(ast.literal_eval(s))
    return "".join(head)


def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS and len(t) > 1]


def lesson_title(text):
    first = text.strip().splitlines()[0].strip()
    small = {"in", "and", "for", "of", "the", "a", "to"}
    words = first.lower().split()
    return " ".join(w if (i and w in small) else ("RAG" if w == "rag" else w.capitalize())
                    for i, w in enumerate(words))


def find_spans(text, chunks):
    """Character span of each chunk inside the original text."""
    spans, cursor = [], 0
    for c in chunks:
        start = text.find(c, cursor)
        if start < 0:                      # should not happen; fall back to a loose search
            start = text.find(c[:80])
        spans.append((start, start + len(c)))
        cursor = start + 1
    return spans


def build(lessons_dir: Path):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    lessons, chunks = [], []
    for li, path in enumerate(sorted(lessons_dir.glob("*.txt"))):
        text = path.read_text(encoding="utf-8")
        pieces = splitter.split_text(text)
        spans = find_spans(text, pieces)
        lessons.append({
            "id": li, "filename": path.name, "title": lesson_title(text),
            "color": PALETTE[li % len(PALETTE)], "chars": len(text), "text": text,
            "chunkIds": [],
        })
        prev_end = None
        for ci, (piece, (start, end)) in enumerate(zip(pieces, spans)):
            overlap = max(0, prev_end - start) if prev_end is not None else 0
            chunk_id = len(chunks)
            lessons[-1]["chunkIds"].append(chunk_id)
            chunks.append({
                "id": chunk_id, "lesson": li, "index": ci, "filename": path.name,
                "start": start, "end": end, "chars": len(piece),
                "tokens": round(len(piece) / 4), "overlapPrev": overlap, "text": piece,
            })
            prev_end = end

    # --- TF-IDF (sublinear tf, smoothed idf, L2-normalised)
    docs = [tokenize(c["text"]) for c in chunks]
    vocab = sorted({t for d in docs for t in d})
    index = {t: i for i, t in enumerate(vocab)}
    n = len(docs)
    df = Counter(t for d in docs for t in set(d))
    idf = np.array([math.log((1 + n) / (1 + df[t])) + 1 for t in vocab])

    def tfidf(tokens):
        v = np.zeros(len(vocab))
        for t, cnt in Counter(tokens).items():
            if t in index:
                v[index[t]] = (1 + math.log(cnt)) * idf[index[t]]
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    X = np.vstack([tfidf(d) for d in docs])

    # --- PCA to 3D
    mean = X.mean(axis=0)
    _, _, vt = np.linalg.svd(X - mean, full_matrices=False)
    comps = vt[:3]
    P = (X - mean) @ comps.T
    scale = 9.0 / np.abs(P).max()

    def project(v):
        return [round(float(x), 3) for x in ((v - mean) @ comps.T) * scale]

    for c, row, vec in zip(chunks, P, X):
        c["pos"] = [round(float(x), 3) for x in row * scale]
        top = np.argsort(vec)[::-1][:8]
        c["topTerms"] = [[vocab[i], round(float(vec[i]), 3)] for i in top if vec[i] > 0]

    # Nearest neighbours by cosine in the full TF-IDF space (for the index graph).
    sims = X @ X.T
    for i, c in enumerate(chunks):
        order = [j for j in np.argsort(sims[i])[::-1] if j != i]
        c["neighbors"] = [[int(j), round(float(sims[i, j]), 3)] for j in order[:3]]

    # --- BM25 (k1=1.5, b=0.75)
    k1, b = 1.5, 0.75
    avgdl = sum(len(d) for d in docs) / n
    bm25_idf = {t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in vocab}

    def bm25(query_tokens):
        scores = []
        for d in docs:
            tf = Counter(d)
            s = 0.0
            for t in set(query_tokens):
                if t in tf:
                    s += bm25_idf[t] * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avgdl))
            scores.append(s)
        return np.array(scores)

    queries = []
    for question, expected in EVAL_CASES:
        q_tokens = tokenize(question)
        qv = tfidf(q_tokens)
        vec_scores = X @ qv
        bm_scores = bm25(q_tokens)
        vec_rank = [int(i) for i in np.argsort(vec_scores)[::-1][:RETRIEVAL_K]]
        bm_rank = [int(i) for i in np.argsort(bm_scores)[::-1][:RETRIEVAL_K] if bm_scores[i] > 0]
        fused = {}
        for w, ranking in ((WEIGHTS[0], bm_rank), (WEIGHTS[1], vec_rank)):
            for r, cid in enumerate(ranking, start=1):
                fused[cid] = fused.get(cid, 0.0) + w / (r + RRF_C)
        fused_list = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        matched = sorted({t for t in q_tokens if t in index})
        queries.append({
            "question": question, "expected": expected, "terms": matched, "pos": project(qv),
            "vector": [[cid, round(float(vec_scores[cid]), 3)] for cid in vec_rank],
            "bm25": [[cid, round(float(bm_scores[cid]), 3)] for cid in bm_rank],
            "fused": [[cid, round(score, 5)] for cid, score in fused_list],
            "hit": any(chunks[cid]["filename"] == expected for cid, _ in fused_list),
        })

    # Everything the browser needs to turn any sentence into the same representation:
    # vocabulary, idf weights, the PCA projection, and each chunk's sparse vector.
    space = {
        "stopwords": sorted(STOPWORDS),
        "vocab": vocab,
        "idf": [round(float(x), 5) for x in idf],
        "mean": [round(float(x), 6) for x in mean],
        "comps": [[round(float(x), 6) for x in row] for row in comps],
        "scale": round(float(scale), 6),
        "chunkVecs": [[[int(i), round(float(v[i]), 5)] for i in np.nonzero(v)[0]] for v in X],
        # BM25 needs raw term counts and lengths, not TF-IDF weights
        "bm25": {
            "k1": k1, "b": b, "avgdl": round(avgdl, 4),
            "idf": [round(bm25_idf[t], 5) for t in vocab],
            "counts": [[[index[t], c] for t, c in sorted(Counter(d).items())] for d in docs],
            "lengths": [len(d) for d in docs],
        },
    }

    generation_py = lessons_dir.parents[1] / "src" / "pipeline" / "generation.py"
    return {
        "settings": {"chunkSize": CHUNK_SIZE, "chunkOverlap": CHUNK_OVERLAP, "k": RETRIEVAL_K,
                     "rrfC": RRF_C, "weights": list(WEIGHTS), "vocab": len(vocab)},
        "lessons": lessons, "chunks": chunks, "queries": queries, "space": space,
        "promptHead": read_prompt_head(generation_py),
    }
