"""Hybrid retrieval: combines lexical (BM25) and semantic (vector) search via an ensemble."""
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config import COLLECTION_NAME, EMBEDDING_MODEL, GENERATION_MODEL, GOOGLE_API_KEY, PERSIST_DIRECTORY, RETRIEVAL_K

embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=PERSIST_DIRECTORY
)

# BM25 needs the raw documents in memory; pull everything back out of Chroma
info= vector_store.get()
doc=[]

for text,metadata in zip(info['documents'], info['metadatas']):
    doc.append(Document(page_content=text, metadata=metadata))
    
bm25retriever = BM25Retriever.from_documents(documents=doc)
bm25retriever.k = RETRIEVAL_K

base_vector_retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})

# MultiQueryRetriever asks the LLM to rephrase the question a few ways and unions the
# results, which helps recall when the user's wording doesn't match the corpus wording.
query_expansion_llm = ChatGoogleGenerativeAI(model=GENERATION_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0)
vector_retriever = MultiQueryRetriever.from_llm(
    retriever=base_vector_retriever,
    llm=query_expansion_llm,
)

# Combine both using the EnsembleRetriever (Hybrid Search)
# weights=[0.5, 0.5] splits the relevance evenly between BM25 and Vector search
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25retriever, vector_retriever],
    weights=[0.5, 0.5]
)


def hybrid_search(query):
    """Run hybrid (BM25 + multi-query-expanded vector) search and return matched documents."""
    try:
        return hybrid_retriever.invoke(query)
    except Exception as e:
        print(f"[hybrid_search] error: {e}")
        raise


if __name__ == "__main__":
    query = "How to retrieve data using vectors?"
    print(hybrid_search(query))