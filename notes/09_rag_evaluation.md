# 📏 Module 9: Evaluating Your RAG Application (Deep Dive)

> *Expands the evaluation section in Module 8 · Sources: Ragas docs, RAGAS paper (arXiv:2309.15217), promptfoo, NVIDIA/IBM RAG eval guides*

> 💡 *"You can't improve what you don't measure."* — most beginners test RAG by vibes. Production teams **score every stage** and gate releases on those scores.

---

## 🎯 Why evaluation is hard (and important)

A RAG app has **two moving parts**, and either can fail silently:

| Component | Fails when... | Symptom |
|---|---|---|
| **Retriever** | Wrong chunks retrieved | Answer is missing facts ("I don't know" too often) |
| **Generator** | Chunks are right but answer is wrong/unfaithful | Hallucinations despite good context |

If you only measure "does the answer look right?", you can't tell *which* part to fix. So we score them **separately**.

---

## 📐 The Core Metrics (RAGAS)

### Generator-side

**1. Faithfulness** — *anti-hallucination score*

```
Faithfulness = (claims in answer inferable from context) / (total claims in answer)
```

Example: answer has 2 claims; only 1 is supported by context → **faithfulness = 0.5**.

**2. Answer Relevancy** — does the answer address the *question*?

```
Answer Relevancy = mean( similarity(question, generated_questions) )
```
Ragas reverse-engineers questions from the answer and compares them to the original.

**3. Answer Correctness / Similarity** — vs. a known ground-truth answer (factuality + semantic similarity).

### Retriever-side

**4. Context Precision** — are the *relevant* chunks ranked **high**?

> Uses Average Precision: relevant chunks at the top score higher than relevant chunks buried at position 10.

**5. Context Recall** — did retrieval find **all** the info needed?

```
Context Recall ≈ TP / (TP + FN)   (estimated against the ground truth)
```

| Metric | Component | You want... |
|---|---|---|
| Faithfulness | Generator | ↑ high (≥ 0.9) |
| Answer Relevancy | Generator | ↑ high |
| Context Precision | Retriever | ↑ high |
| Context Recall | Retriever | ↑ high |

---

## 🧪 Step 1: Build a test set

You need triplets of **`question` + `context` + `ground_truth`**. Generate them from your own documents:

```python
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context

generator = TestsetGenerator.from_langchain(llm, embeddings)
testset = generator.generate_with_langchain_docs(
    docs,
    test_size=50,
    distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25},
)
testset.to_pandas()  # question, ground_truth, contexts...
```

**Rule of thumb:** mirror your real traffic — include short factoid questions *and* multi-hop reasoning questions.

## 🧪 Step 2: Run the evaluation

```python
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness, answer_relevancy, context_precision, context_recall,
)

data = {
    "question": questions,          # list[str]
    "answer":   answers,            # what your pipeline produced
    "contexts": contexts,           # list[list[str]] actually fed to the LLM
    "ground_truth": truths,         # reference answers
}
result = evaluate(
    Dataset.from_dict(data),
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)
print(result)   # per-metric score 0..1
```

## 🧪 Step 3: LLM-as-a-Judge

For open-ended answers, have a strong LLM score responses against a rubric (0–5) with the ground truth visible. Cheap, scalable, and correlates well with human review — but log a sample for human spot-checks.

## 🧪 Step 4: CI quality gates (block bad releases)

Use **promptfoo** to fail a build when a prompt/model change regresses quality:

```yaml
# promptfooconfig.yaml
prompts:
  - "Answer using ONLY the context: {{context}}\nQuestion: {{question}}"
providers: [gemini-3.5-flash]
tests:
  - vars: {context: "Refunds within 30 days", question: "refund policy?"}
    assert:
      - type: llm-rubric
        value: "must state 30 days"
      - type: faithfulness
        threshold: 0.85
```

**Golden rule of release engineering:** a regression in faithfulness or retrieval hit-rate should **block deployment**, not just appear in a dashboard.

---

## 🌐 Offline vs Online evaluation

| | Offline (pre-release) | Online (production) |
|---|---|---|
| When | Before shipping | On live traffic |
| Data | Curated test set | Real user queries |
| Tools | Ragas, promptfoo | LangSmith, Langfuse, OpenTelemetry |
| Catch | Regressions | Drift, staleness, bad edge cases |

**Operational metrics to track live:** retrieval hit-rate, chunk relevance, TTFT (time-to-first-token, target p90 < 2s), cost-per-request, and thumbs-up/down feedback.

---

## 🩺 Diagnosis matrix — low score → what to fix

| Low metric | Likely cause | Fix |
|---|---|---|
| **Faithfulness** | Noisy/irrelevant chunks in prompt | Add reranking, lower top-k, strengthen prompt ("only from context") |
| **Answer Relevancy** | Answer drifts off-topic | Better prompt, query rewriting |
| **Context Precision** | Relevant chunks ranked low | Reranker, hybrid search, better chunking |
| **Context Recall** | Missing information | Smaller chunks, metadata filters, more retrieval rounds |

> 📊 **Production insight:** most RAG failures concentrate in **retrieval and chunking** (Splunk/Red Hat, 2026) — diagnose those *before* touching the prompt.

---

## ✅ Key Takeaways

- Score **retriever and generator separately** — that's the only way to debug.
- RAGAS: faithfulness + answer relevancy (generator), context precision + recall (retriever).
- Generate a test set from your own docs; don't hand-write 50 questions.
- Gate releases on metrics (promptfoo), monitor live traffic (LangSmith).
- Retrieval problems are the most common root cause — check there first.

⬅️ Back to [Building Your First RAG App](./08_building_your_first_rag_app.md) · ➡️ Next: [Document Ingestion & Parsing](./10_document_ingestion_and_parsing.md)
