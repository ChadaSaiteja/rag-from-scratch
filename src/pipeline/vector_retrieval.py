"""Pure semantic (vector) retrieval demo against the persisted Chroma store."""
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from config import COLLECTION_NAME, EMBEDDING_MODEL, GOOGLE_API_KEY, PERSIST_DIRECTORY, RETRIEVAL_K

embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=PERSIST_DIRECTORY
)

# Top-k nearest neighbours by embedding similarity
results = vector_store.similarity_search(
    query="How to retrieve data using vectors?",
    k=RETRIEVAL_K
)

print(results)