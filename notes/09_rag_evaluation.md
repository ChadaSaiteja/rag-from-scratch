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

## 🔢 Retrieval metrics you can compute without an LLM

Before any LLM-judged metric, check retrieval with plain arithmetic. You only need, for each test question, the ID of the chunk or file that *should* be found.

| Metric | Question it answers | How to compute |
|---|---|---|
| **Hit rate @k** (recall@k) | Did the right source appear in the top k at all? | Fraction of questions where it did |
| **MRR** (Mean Reciprocal Rank) | How high did the first right result appear? | Average of `1 / rank` of the first correct result (0 if missing) |
| **nDCG @k** | Are *all* relevant results near the top, weighted by position? | Graded relevance discounted by `log2(rank + 1)`; use when several chunks are relevant |

### 🧮 Worked example

Three test questions, top-5 retrieval:

| Question | Rank of first correct chunk | Hit @5 | Reciprocal rank |
|---|---|---|---|
| Q1 | 1 | ✅ | 1.00 |
| Q2 | 3 | ✅ | 0.33 |
| Q3 | not retrieved | ❌ | 0.00 |

**Hit rate @5 = 2/3 = 0.67**, **MRR = (1 + 0.33 + 0) / 3 = 0.44**. Hit rate tells you whether the answer was available; MRR tells you whether it was near the top, which matters because of "lost in the middle" (Module 1).

```python
def hit_rate_and_mrr(results, k=5):
    """results: list of (expected_id, retrieved_ids_in_rank_order)."""
    hits, rr = 0, 0.0
    for expected, retrieved in results:
        top = retrieved[:k]
        if expected in top:
            hits += 1
            rr += 1 / (top.index(expected) + 1)
    n = len(results)
    return hits / n, rr / n
```

These are fast, free and deterministic, so run them on **every** change to chunking, embeddings or retrieval settings.

---

## 🧪 Step 1: Build a test set

You need triplets of **`question` + `context` + `ground_truth`**. Generate them from your own documents:

```python
# ragas 0.2+ API (older tutorials import from ragas.testset.evolutions, which no longer exists)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator

generator = TestsetGenerator(
    llm=LangchainLLMWrapper(ChatGoogleGenerativeAI(model="gemini-3.5-flash")),
    embedding_model=LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")),
)
testset = generator.generate_with_langchain_docs(docs, testset_size=50)  # docs: LangChain Documents
testset.to_pandas()  # user_input, reference, reference_contexts, ...
```

> ⚠️ Always **read and fix** generated questions before trusting them. Synthetic test sets contain some vague or wrong items; 30 reviewed questions beat 300 unreviewed ones.

**Rule of thumb:** mirror your real traffic — include short factoid questions *and* multi-hop reasoning questions.

## 🧪 Step 2: Run the evaluation

```python
from ragas import EvaluationDataset, evaluate
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy

dataset = EvaluationDataset.from_list([
    {
        "user_input": q,                  # the question
        "retrieved_contexts": ctx,        # list[str] actually fed to the LLM
        "response": ans,                  # what your pipeline produced
        "reference": truth,               # reference answer
    }
    for q, ctx, ans, truth in zip(questions, contexts, answers, truths)
])
result = evaluate(
    dataset=dataset,
    metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(), LLMContextRecall()],
    llm=judge, embeddings=judge_embeddings,   # wrapped LLM + embeddings, as in Step 1
)
print(result)               # per-metric averages, 0..1
result.to_pandas()          # per-question scores: find the worst cases and read them
```

## 🧪 Step 3: LLM-as-a-Judge

For open-ended answers, have a strong LLM score responses against a rubric (0–5) with the ground truth visible. Cheap, scalable, and correlates well with human review — but log a sample for human spot-checks.

## 🧪 Step 4: CI quality gates (block bad releases)

Use **promptfoo** to fail a build when a prompt/model change regresses quality:

```yaml
# promptfooconfig.yaml
prompts:
  - "Answer using ONLY the context: {{context}}\nQuestion: {{question}}"
providers: ["google:gemini-3.5-flash"]
tests:
  - vars: {context: "Refunds within 30 days", question: "refund policy?"}
    assert:
      - type: llm-rubric
        value: "must state 30 days"
      - type: context-faithfulness   # uses the `context` var above
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
- Hit rate and MRR are free, deterministic retrieval checks; run them on every change.

---

## 🔗 In this repo

[`evaluation.py`](https://github.com/ChadaSaiteja/rag-from-scratch/blob/master/src/pipeline/evaluation.py) is a minimal version of this module:
- **Retrieval check:** did the expected source file appear in `hybrid_search()` results? That's a **hit rate @k** over 5 hand-written questions.
- **Generation check:** does the answer contain an expected keyword? A cheap, crude stand-in for faithfulness and correctness.
- Known limitations: only 5 cases, a keyword check that passes easily, and an overloaded-API message that still gets scored. 🧪 **Exercises:** (1) add MRR using `hit_rate_and_mrr()` above; (2) grow the cases to 20+ questions covering every lesson; (3) replace the keyword check with RAGAS `Faithfulness`.

---

## 🧠 Self-check

<details markdown="1">
<summary>1. Hit rate @5 is 0.95 but MRR is 0.35. What does that tell you?</summary>

The right chunk is almost always retrieved, but usually not near the top. Add a reranker so the best chunk moves to position 1, or reduce top-k noise.

</details>

<details markdown="1">
<summary>2. Why score retrieval and generation separately instead of just judging the final answer?</summary>

A bad answer can come from missing context (retrieval) or from misusing good context (generation). Separate scores tell you which component to fix.

</details>

<details markdown="1">
<summary>3. Why should a faithfulness regression block a release instead of just showing on a dashboard?</summary>

Dashboards get ignored, and a faithfulness drop means users get more hallucinated answers. Making it a CI gate forces the regression to be fixed or consciously accepted before it reaches users.

</details>

⬅️ Back to [Building Your First RAG App](./08_building_your_first_rag_app.md) · ➡️ Next: [Document Ingestion & Parsing](./10_document_ingestion_and_parsing.md)
