# 🤖 Module 7: Agentic RAG & Tool Calling

> *Source: `AI/day4/toolcalling.md` + `AI/day4/toolcalling.py`*

**Tool calling** lets an LLM interact with external tools and functions in a controlled, predictable way. Combined with **structured output**, it turns a chatbot into an **agent** — the foundation of *Agentic RAG*.

---

## 📦 What is Structured Output?

Constraining the LLM's output to a predictable, type-safe format instead of free-form text.

**Benefits:** predictable format • type safety • easy extraction (no regex) • fewer errors • seamless app integration.

---

## 🧠 Mental Model: The LLM does NOT execute anything

```
User Input
    ↓
LLM (analyzes and suggests an action)
    ↓
"Call get_weather(city='Hyderabad')"
    ↓
YOUR PYTHON CODE (executes the actual function)
    ↓
get_weather("Hyderabad")
    ↓
Tool Result (weather data)
    ↓
LLM (processes result)
    ↓
Final Response to User
```

**Key insight:** the LLM *proposes* actions via structured output; **your application controls execution.**

---

## 🎯 Common Use Cases

- Weather APIs, database queries, calculations, web search
- **Multi-step workflows** — chaining multiple tool calls
- **RAG-specific:** query a vector store, fetch a document by ID, run a SQL report, call a calculator, hit a live API

---

## 💻 Code: Tool Calling with Gemini (from `day4/toolcalling.py`)

```python
import os, json
from dotenv import load_dotenv
load_dotenv()
from google import genai

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def get_weather(city):
    weather = {
        "hyderabad": {"temperature": "30°C", "condition": "Sunny"},
        "bangalore": {"temperature": "28°C", "condition": "Cloudy"},
    }
    return weather.get(city.lower(), {"temperature": "N/A", "condition": "N/A"})

# 1. Declare the tool (name + description + JSON schema)
tool = {
    "type": "function",
    "name": "get_weather",
    "description": "Get the current weather for a city",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}

# 2. Ask the model — it may decide to call the tool
result = client.interactions.create(
    model="gemini-3.5-flash",
    input="Get the current weather for Hyderabad",
    tools=[tool],
)

# 3. Find the function-call step in the response
fc_step = next((s for s in result.steps if s.type == "function_call"), None)

# 4. YOUR CODE executes it (never trust the LLM to run it)
tool_result = get_weather(fc_step.arguments.get("city")) if fc_step else None

# 5. Send the tool result back for a grounded final answer
final_result = client.interactions.create(
    model="gemini-3.5-flash",
    tools=[tool],
    previous_interaction_id=result.id,
    input=[{
        "type": "function_result",
        "name": fc_step.name,
        "call_id": fc_step.id,
        "result": [{"type": "text", "text": json.dumps(tool_result)}],
    }],
)
print(final_result.output_text)
```

---

## 🔀 From Tools → Agentic RAG

A RAG app becomes **agentic** when the LLM *decides* whether/how to retrieve:

| Classic RAG | Agentic RAG |
|---|---|
| Always retrieves for every question | Decides *if* retrieval is needed |
| Single retrieval pass | Multiple rounds (query → retrieve → check → re-query) |
| Fixed pipeline | LLM plans steps, calls tools, self-reflects |

**When to use agentic RAG:** multi-hop questions ("compare policy X and Y"), mixed data sources (DB + docs + API), and tasks needing computation or live data.

> ⚠️ **Beginner trap:** agentic ≠ better. It adds latency, cost, and failure modes. Start with classic RAG; add agents only when you *need* decisions, not just retrieval.

---

## ✅ Key Takeaways

- LLM proposes → your code executes → result returns.
- Declare tools with clear names, descriptions, and JSON schemas.
- Agentic RAG = LLM controls the retrieve/reason loop.
- Use ReAct-style reasoning for multi-step tasks.

➡️ Next: [Build Your First RAG App — the full project](./08_building_your_first_rag_app.md)
