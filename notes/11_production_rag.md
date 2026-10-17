# 🏭 Module 11: Production RAG — Security, Cost, Latency & Monitoring

A demo RAG app answers questions. A production RAG app answers them **safely, quickly, affordably and observably** for many users at once. This module covers what changes between the two.

---

## 🛡️ Security: the RAG threat model

RAG adds a new attack surface: **text you didn't write ends up inside your prompt.** The OWASP Top 10 for LLM Applications lists prompt injection as the top risk and includes weaknesses specific to vector stores and embeddings.

| Threat | What it looks like | Defences |
|---|---|---|
| **Indirect prompt injection** | A document, web page or email in your corpus contains *"Ignore previous instructions and…"* | Delimit retrieved text and tell the model it's data, not instructions (Module 3); keep tools read-only; never put secrets in prompts |
| **Cross-user data leakage** | User A's question retrieves chunks from User B's private files | Store owner/team/permission metadata on every chunk and **filter at query time**; never rely on the prompt to hide data |
| **Sensitive data in the index** | Passwords, personal data or keys ingested from shared drives | Scan and redact during ingestion; exclude sensitive folders; support deletion by source (Module 10) |
| **Corpus poisoning** | Someone adds a document designed to be retrieved for certain questions and mislead the answer | Control who can add documents; record provenance; prefer trusted sources in ranking |
| **Over-powered agents** | An injected instruction triggers a tool that sends, deletes or pays | Allow-list tools, validate arguments, require human confirmation for side effects (Module 7) |

### 💻 Code: permission-aware retrieval

Access control belongs in the **retrieval filter**, not in the prompt. If a chunk is never retrieved, it can't leak:

```python
def retrieve_for_user(store, question, user, k=5):
    # Every chunk was ingested with metadata like {"team": "finance", "source": "..."}
    return store.similarity_search(
        question, k=k,
        filter={"team": {"$in": user.teams}},   # Chroma metadata filter syntax
    )
```

> ⚠️ **Beginner trap:** "The prompt says *don't reveal HR documents*." Prompts are not access control. A clever question or an injected document can override instructions; a metadata filter can't be talked out of its job.

---

## 💸 Cost: where the money goes

| Cost | When it happens | How to reduce it |
|---|---|---|
| **Embedding the corpus** | Once, plus every re-index | Incremental indexing with content hashes (Module 10); smaller embedding dimensions (Module 2) |
| **Embedding each question** | Every request | Cheap per call; cache repeated questions |
| **LLM input tokens** | Every request: instructions + chunks + question | Send fewer, better chunks (rerank, then keep 4–6); keep the system prompt short |
| **LLM output tokens** | Every request | Ask for concise answers; set a max output length |
| **Extra LLM calls** | Query rewriting, multi-query, LLM reranking, agent steps | Use them only where evaluation shows they help; route only hard queries |

### 🧮 Worked example: cost per question

The prices below are **illustrative**; check your provider's current pricing.

```
Input:  1,500 tokens × $0.30 per 1M tokens = $0.00045
Output:   250 tokens × $2.50 per 1M tokens = $0.000625
Total per question                          ≈ $0.0011
× 100,000 questions per month               ≈ $107
```

Now add multi-query expansion (one extra LLM call of ~300 tokens in and ~100 out) and the bill rises by roughly a third. Every "advanced" technique has a price tag, so measure whether it earns its place (Module 9).

---

## ⚡ Latency: a budget for every stage

Users notice delays above a couple of seconds. A typical budget for one question:

| Stage | Typical time | Notes |
|---|---|---|
| Embed the question | 50–200 ms | One API call; cacheable |
| Vector + BM25 search | 10–100 ms | ANN indexes keep this low (Module 5) |
| Rerank ~30 candidates | 100–300 ms | Cross-encoder; skip for simple queries |
| Query rewriting / multi-query | 300–1,000 ms | An extra LLM round trip before search even starts |
| LLM generation | 0.5–5 s | Dominated by output length; **stream** tokens so users see progress immediately |

**Biggest wins:** stream the answer, cache repeated questions, run BM25 and vector search in parallel, and avoid extra LLM calls on the critical path.

### 🗄️ Caching

- **Exact cache:** same normalized question → return the stored answer. Simple and safe.
- **Semantic cache:** a new question whose embedding is very close to a cached one (similarity above a strict threshold) reuses that answer. Bigger hit rate, but a threshold set too low returns answers to *different* questions. Tune it on real examples and expire entries when documents change.

---

## 📈 Monitoring: what to log for every request

You can't fix what you can't see. For each question, record a **trace**:

```json
{
  "request_id": "a1b2",
  "question": "Can I return an item after 3 weeks?",
  "retrieved": [{"id": "refund-policy.md::2", "score": 0.81}, {"id": "shipping.md::0", "score": 0.44}],
  "reranked_top": ["refund-policy.md::2"],
  "prompt_tokens": 1420, "output_tokens": 96,
  "latency_ms": {"retrieve": 85, "rerank": 160, "generate": 1240},
  "answer_said_dont_know": false,
  "user_feedback": "thumbs_up"
}
```

Then watch these over time:

| Signal | Why it matters |
|---|---|
| "I don't know" rate | Rising → missing documents or broken retrieval |
| Top retrieval score distribution | Falling → questions drifting away from your corpus |
| Thumbs-down rate, by topic | Shows which document areas need work |
| p50 / p95 latency per stage | Finds the slow stage before users complain |
| Cost per request | Catches prompt bloat and runaway agent loops |

Tools such as LangSmith, Langfuse or OpenTelemetry-based tracing collect this automatically; a structured log line per request is enough to start.

---

## 🩺 Common production failures

| Symptom | Likely cause | Fix |
|---|---|---|
| Confident answers to questions the docs don't cover | No relevance floor: top-k always returns *something* | Drop chunks below a similarity threshold; if none remain, answer "I don't know" |
| Answers cite outdated policies | Old and new versions both indexed | Date/version metadata + filters; remove superseded documents |
| Good in testing, poor in production | Test questions don't match real user phrasing | Build the eval set from real (anonymized) queries |
| Slow first answer after deploy | Models or indexes loading on first request | Warm up at startup |
| Costs creeping up | Prompt template and top-k grew over time | Track tokens per request; alert on increases |

---

## ✅ Key Takeaways

- Treat retrieved text as untrusted input; enforce access control in retrieval filters, never in prompts.
- Every extra LLM call costs money and latency; keep only what evaluation proves useful.
- Stream answers, cache repeats, and parallelize search to stay within a latency budget.
- Log a full trace per request and watch "I don't know" rate, retrieval scores, feedback, latency and cost.

---

## 🔗 In this repo

The pipeline is a learning project, so several production pieces are deliberately missing. Each one is a good next exercise:
- **No relevance floor:** `hybrid_search()` always returns results. Add a score threshold to the vector leg and a "nothing relevant found" path.
- **No access control:** every chunk is visible to every question. Add a `team` or `visibility` metadata field in `ingest.py` and filter on it.
- **No API layer or tracing:** wrap `generation.py` in a small FastAPI `/ask` endpoint and log one trace line per request.
- **Overload handling:** `call_google_api()` returns a friendly message on HTTP 503; add retry with backoff before giving up.

See the full list in the repository [README's known issues](https://github.com/ChadaSaiteja/rag-from-scratch#known-issues).

---

## 🧠 Self-check

<details markdown="1">
<summary>1. Why is "the system prompt tells the model not to reveal HR documents" not access control?</summary>

Instructions can be overridden by clever questions or injected document text. Real access control filters chunks during retrieval based on the user's permissions, so restricted content never reaches the model at all.

</details>

<details markdown="1">
<summary>2. Your answers take 6 seconds. Name three changes that cut perceived latency.</summary>

Stream the answer so text appears immediately; remove or route around extra LLM calls such as multi-query expansion for simple questions; and cache repeated questions. Running BM25 and vector search in parallel also helps.

</details>

<details markdown="1">
<summary>3. Why is a semantic cache riskier than an exact-match cache?</summary>

It reuses answers for questions that are merely *similar*. If the similarity threshold is too loose, "Can I return a laptop?" might get the cached answer to "Can I return a phone case?" Strict thresholds and expiry when documents change keep it safe.

</details>

⬅️ Back to [Document Ingestion & Parsing](./10_document_ingestion_and_parsing.md) · 📖 [Glossary](./glossary.md) · ❓ [FAQ](./faq.md) · 🏠 [Notes Index](./README.md)
