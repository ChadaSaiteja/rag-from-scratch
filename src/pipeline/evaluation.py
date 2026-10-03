"""Lightweight RAG evaluation: scores retrieval and generation separately.

Implements the approach described in data/raw/08_rag_evaluation.txt — don't judge a RAG
system only by "does the answer sound good?"; check whether retrieval actually found the
right source, independently of how fluent the generated answer reads.
"""
from pathlib import Path
from hybrid_search import hybrid_search
from generation import call_google_api, format_context

# Each case: a question, the filename that should be retrieved, and a keyword
# expected to appear in a grounded answer. Extend this list as the corpus grows.
EVAL_CASES = [
    {
        "question": "What is the difference between chunk size and overlap?",
        "expected_source": "03_chunking.txt",
        "expected_keyword": "overlap",
    },
    {
        "question": "How does a vector database find similar embeddings?",
        "expected_source": "05_vector_database.txt",
        "expected_keyword": "vector",
    },
    {
        "question": "What does the retriever stage do in a RAG pipeline?",
        "expected_source": "06_retrieval.txt",
        "expected_keyword": "retriev",
    },
    {
        "question": "How should a prompt be constructed for the generation stage?",
        "expected_source": "07_rag_prompt_and_generation.txt",
        "expected_keyword": "prompt",
    },
    {
        "question": "Why should retrieval and generation quality be evaluated separately?",
        "expected_source": "08_rag_evaluation.txt",
        "expected_keyword": "evaluat",
    },
]


def evaluate_retrieval(case):
    """Check whether the expected source file shows up anywhere in the retrieved chunks."""
    try:
        docs = hybrid_search(case["question"])
        # same fallback as format_context(): older chunks may lack "filename" metadata
        retrieved_sources = {
            doc.metadata.get("filename") or Path(doc.metadata.get("source", "")).name
            for doc in docs
        }
        hit = case["expected_source"] in retrieved_sources
        return hit, docs
    except Exception as e:
        print(f"[evaluate_retrieval] error: {e}")
        raise


def evaluate_generation(case, docs):
    """Check whether the generated answer contains the expected keyword (crude groundedness proxy)."""
    try:
        retrieved_info = format_context(docs)
        answer = call_google_api(case["question"], retrieved_info)
        hit = case["expected_keyword"].lower() in answer.lower()
        return hit, answer
    except Exception as e:
        print(f"[evaluate_generation] error: {e}")
        raise


def run_evaluation(cases=EVAL_CASES):
    """Run every eval case and print a retrieval-vs-generation scorecard."""
    results = []
    for case in cases:
        retrieval_hit, docs = evaluate_retrieval(case)
        generation_hit, answer = evaluate_generation(case, docs)
        results.append({
            "question": case["question"],
            "expected_source": case["expected_source"],
            "retrieval_hit": retrieval_hit,
            "generation_hit": generation_hit,
            "answer": answer,
        })

    retrieval_score = sum(r["retrieval_hit"] for r in results)
    generation_score = sum(r["generation_hit"] for r in results)
    total = len(results)

    print(f"\n{'Question':<60} {'Retrieval':<10} {'Generation':<10}")
    print("-" * 82)
    for r in results:
        print(f"{r['question'][:58]:<60} {str(r['retrieval_hit']):<10} {str(r['generation_hit']):<10}")
    print("-" * 82)
    print(f"Retrieval accuracy:  {retrieval_score}/{total}")
    print(f"Generation accuracy: {generation_score}/{total}")
    return results


if __name__ == "__main__":
    run_evaluation()
