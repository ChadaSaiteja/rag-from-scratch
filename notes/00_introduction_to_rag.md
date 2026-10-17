# 🚀 Introduction to RAG (Retrieval-Augmented Generation)

Welcome to the world of RAG! If you've ever used ChatGPT and wished it knew about your private documents, or wished it wouldn't "hallucinate" (make things up), then **RAG is the solution you've been looking for.**

## ❓ What is RAG?

**RAG** stands for **Retrieval-Augmented Generation**. 

Think of a Large Language Model (LLM) like a very smart student who has read almost everything on the internet but doesn't have access to your personal files or today's news. If you ask them about your private company policy, they might guess (and potentially lie).

**RAG gives that student an open textbook.**

Instead of relying solely on what they already know (their training data), the student:
1. **Retrieves** relevant information from a provided textbook (your documents).
2. **Augments** their knowledge with that information.
3. **Generates** a response based on the retrieved facts.

---

## 🛠️ The RAG Workflow (The Big Picture)

Here is how a RAG application actually works under the hood:

1.  **Data Ingestion (The Preparation Phase):**
    *   **Load:** Take your documents (PDFs, Text, Markdown, etc.).
    *   **Chunk:** Break long documents into smaller, manageable pieces called "chunks".
    *   **Embed:** Convert these chunks into mathematical vectors (numbers) using an **Embedding Model**.
    *   **Store:** Save these vectors in a special database called a **Vector Database**.

2.  **Retrieval (The Search Phase):**
    *   **User Query:** A user asks a question (e.g., *"What is our company's vacation policy?"*).
    *   **Query Embedding:** Convert the user's question into a vector using the *same* embedding model used before.
    *   **Similarity Search:** Look into the Vector Database to find the chunks whose vectors are "closest" (most similar) to the query vector.

3.  **Generation (The Answer Phase):**
    *   **Prompt Augmentation:** Combine the user's original question + the retrieved chunks into one big prompt.
    *   **LLM Generation:** Send this augmented prompt to the LLM.
    *   **Final Answer:** The LLM reads the context and provides a factual, grounded response.

---

## 🌟 Why use RAG?

| Feature | Without RAG (Standard LLM) | With RAG |
| :--- | :--- | :--- |
| **Knowledge** | Limited to training data (cutoff date) | Access to real-time & private data |
| **Accuracy** | Prone to hallucinations | Grounded in provided facts |
| **Cost** | High (if you try to fine-tune) | Low (just add new documents) |
| **Transparency**| "Trust me, I'm an AI" | "According to document X..." |

---

## 🤔 RAG vs. fine-tuning vs. long context

RAG is not the only way to give a model new knowledge. Knowing when *not* to use it is part of using it well.

| Approach | What it changes | Best for | Weak at |
|---|---|---|---|
| **RAG** | What the model *reads* at question time | Facts that change, private documents, answers that need citations | Tasks that need a new *skill* or style rather than new facts |
| **Fine-tuning** | The model's *weights* | Teaching a format, tone, or domain-specific behaviour | Fresh or frequently changing facts (you'd retrain every time); citing sources |
| **Long context** (paste everything) | Nothing; you send all documents every time | A handful of short documents, one-off analysis | Cost and latency grow with every token; quality drops when the answer is buried in the middle |

> 🧠 **Rule of thumb:** *knowledge* problem → RAG. *Behaviour* problem → fine-tuning (or better prompting). Many production systems combine them: a fine-tuned model that is still fed retrieved context.

---

## 🗺️ How this maps to the code in this repo

This repository contains a working implementation of every stage above, so you can read a concept here and then see it in real code:

| Stage | Concept module | Implementation |
|---|---|---|
| Load, chunk, embed, store | [04](./04_chunking_strategies.md), [02](./02_embeddings_and_vectors.md), [05](./05_vector_databases.md) | [`src/pipeline/ingest.py`](https://github.com/ChadaSaiteja/rag-from-scratch/blob/master/src/pipeline/ingest.py) |
| Retrieve (vector + hybrid) | [06](./06_retrieval_mechanics.md) | [`hybrid_search.py`](https://github.com/ChadaSaiteja/rag-from-scratch/blob/master/src/pipeline/hybrid_search.py) |
| Prompt + generate with citations | [03](./03_prompting_techniques.md) | [`generation.py`](https://github.com/ChadaSaiteja/rag-from-scratch/blob/master/src/pipeline/generation.py) |
| Evaluate | [09](./09_rag_evaluation.md) | [`evaluation.py`](https://github.com/ChadaSaiteja/rag-from-scratch/blob/master/src/pipeline/evaluation.py) |

---

## 📚 What's Next?

In this series of notes, we will break down every single component of this process. We will go from the mathematical foundations of how AI "reads" (Transformers) to building a full-scale agentic RAG system.

**The Learning Path:**
1.  [Foundations: Transformers & Tokens](./01_foundations_transformers.md)
2.  [Embeddings: The Language of Numbers](./02_embeddings_and_vectors.md)
3.  [Prompting: Talking to AI](./03_prompting_techniques.md)
4.  [Chunking: Breaking Down Info](./04_chunking_strategies.md)
5.  [Vector Databases: Your AI's Memory](./05_vector_databases.md)
6.  [Retrieval Mechanics: Finding the Needle](./06_retrieval_mechanics.md)
7.  [Agentic RAG: Tools & Actions](./07_agentic_rag_and_tools.md)
8.  [Final Project: Building Your App](./08_building_your_first_rag_app.md)
9.  [Evaluation: Measuring Quality](./09_rag_evaluation.md)
10. [Document Ingestion & Parsing](./10_document_ingestion_and_parsing.md)
11. [Production RAG: Security, Cost & Monitoring](./11_production_rag.md)

Keep the [Glossary](./glossary.md) open in another tab, and check the [FAQ](./faq.md) when a question comes up.

---

## 🧠 Self-check

<details markdown="1">
<summary>1. What are the three words behind "RAG", and what happens in each step?</summary>

**Retrieve** relevant chunks from your documents, **augment** the prompt with them, and **generate** an answer grounded in that context.

</details>

<details markdown="1">
<summary>2. Why must the question be embedded with the same model as the documents?</summary>

Similarity is only meaningful between vectors from the same embedding space. A different model places text at completely different coordinates, so the nearest neighbours would be noise.

</details>

<details markdown="1">
<summary>3. Your support bot must answer in your brand's tone and also know this week's price list. RAG, fine-tuning, or both?</summary>

Prices are a *knowledge* problem that changes weekly, so use RAG. Tone is a *behaviour* problem, so a good system prompt (or fine-tuning, if prompting isn't enough). Combining both is common.

</details>

**Happy Learning! 🚀**
