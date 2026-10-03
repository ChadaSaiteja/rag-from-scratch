"""Generation stage: retrieve context via hybrid search, then ask Gemini to answer."""
from pathlib import Path
from google import genai
from google.genai import errors, types
from hybrid_search import hybrid_search
from config import GENERATION_MODEL, GOOGLE_API_KEY

model = GENERATION_MODEL
# Client-side timeout so a slow/overloaded API fails fast instead of hanging silently.
client = genai.Client(api_key=GOOGLE_API_KEY, http_options=types.HttpOptions(timeout=20000))


def call_google_api(question, retrievedInfo, prompt=None):
    """Call the Google API with the retrieved context and the user's question."""
    if not prompt:
        prompt = (
            "You are a helpful assistant. Answer the question using ONLY the context below.\n"
            "Each context block is tagged with its source filename in [Source: ...] form.\n"
            "After every sentence or claim you make, cite the filename(s) it came from, "
            "e.g. 'Chunking improves recall (Source: 03_chunking.txt).'\n"
            "If the answer isn't in the context, say you don't know instead of guessing.\n\n"
            f"Context:\n{retrievedInfo}\n\n"
            f"Question: {question}"
        )
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
    except errors.ServerError:
        return "The model is temporarily overloaded (HTTP 503). Please try again in a moment."
    except Exception as e:
        print(f"[call_google_api] error: {e}")
        raise
    return response.text or ""


def format_context(docs):
    """Join retrieved chunks, tagging each with its source filename for citation."""
    try:
        tagged = []
        for doc in docs:
            # older chunks may predate the dedicated "filename" field, fall back to "source"
            filename = doc.metadata.get("filename") or Path(doc.metadata.get("source", "")).name or "unknown"
            tagged.append(f"[Source: {filename}]\n{doc.page_content}")
        return "\n\n".join(tagged)
    except Exception as e:
        print(f"[format_context] error: {e}")
        raise


if __name__ == "__main__":
    question = input("Enter your question: ")
    docs = hybrid_search(question)
    retrievedInfo = format_context(docs)
    # print(retrievedInfo)
    answer = call_google_api(question, retrievedInfo)
    print(answer)
