# 🗣️ Module 3: Prompting Techniques — Talking to the LLM

> *Source: `AI/day3/prompt.md` + `AI/day3/prompt.py`*

**Prompting** is writing clear instructions so the model generates responses that meet your requirements. In RAG, your prompt is where the retrieved chunks and the user's question finally meet.

---

## 📋 Types of Prompting

### 1. Short Prompting

| Technique | What it is | When to use |
|---|---|---|
| **Zero-Shot** | Instruction only, no examples | Straightforward questions |
| **One-Shot** | Instruction + **1 example** | Need to show a specific pattern/format |
| **Few-Shot** | Instruction + **2–5 examples** (a.k.a. *in-context learning*) | Specific tasks, exact output formats |

### 2. Chain-of-Thought (CoT)

Combine few-shot prompting with **step-by-step reasoning**. Ask the LLM to *explain its thinking*. Great for complex logic and math.

### 3. Prompt Chaining

Break a complex task into **multiple sequential prompts**; the output of one becomes the input of the next.

### 4. ReAct (Reason + Act)

The model **reasons** about what to do, then **acts** (e.g., calls a tool), then reasons about the result. This is the backbone of **Agentic RAG** (Module 7).

---

## 💻 Code: Prompting with the Gemini API (from `day3/prompt.py`)

```python
import os
from dotenv import load_dotenv
load_dotenv()
from google import genai

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# The same base question, reframed for different audiences:
base_prompt = "Explain how machine learning works and its applications"

prompt_simple = "What is machine learning? Explain it in simple words that a 10-year-old can understand."

prompt_technical = ("Provide a detailed technical explanation of supervised learning, "
                    "unsupervised learning, and reinforcement learning with mathematical "
                    "foundations and algorithmic complexity analysis.")

prompt_business = ("How can organizations implement machine learning to improve business "
                   "outcomes? Include real-world examples and ROI considerations.")

interaction = client.interactions.create(
    model="gemini-3.7-flash",
    input=base_prompt
)
print(interaction.output_text)
```

**Beginner lesson:** the *same model*, the *same topic* — but the wording of the prompt changes the answer's depth, tone, and audience. Prompt design is a skill, not a chore.

---

## 🧩 The RAG Prompt Template (how it all comes together)

In a real RAG app, your prompt looks like this:

```
You are a helpful assistant. Answer using ONLY the provided context.

CONTEXT:
{retrieved_chunk_1}
{retrieved_chunk_2}
{retrieved_chunk_3}

QUESTION: {user_question}

If the answer isn't in the context, say "I don't know".
```

### 🔑 Best practices for RAG prompts

- **"Answer only from the context"** → reduces hallucination.
- **Include an "I don't know" escape** → prevents confident guessing.
- **Ask for citations** → e.g., "cite the source document".
- **Keep the system prompt short** → leaves room for retrieved context.

> ⚠️ **Beginner trap:** stuffing too many chunks into the prompt. The LLM suffers "context rot" — relevant info buried in the middle gets ignored (see Module 1). Retrieval quality beats retrieval quantity.

---

## ✅ Key Takeaways

- Zero/One/Few-shot control *format*; CoT controls *reasoning*; Chaining controls *complexity*.
- The RAG prompt = instructions + retrieved context + question.
- Explicit constraints ("only from context", "say I don't know") are your first defense against hallucinations.

➡️ Next: [Chunking Strategies — breaking documents into bites](./04_chunking_strategies.md)
