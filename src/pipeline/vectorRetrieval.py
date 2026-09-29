import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))

vector_store = Chroma(
    collection_name="gemini_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_gemini_db"
)

results = vector_store.similarity_search(
    query="How to retrieve data using vectors?",
    k=3
)

print(results)