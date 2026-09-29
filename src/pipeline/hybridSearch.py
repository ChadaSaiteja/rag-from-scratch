import os
from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))

vector_store = Chroma(
    collection_name="gemini_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_gemini_db"
)

info= vector_store.get()
doc=[]

for text,metadata in zip(info['documents'], info['metadatas']):
    doc.append(Document(page_content=text, metadata=metadata))
    
bm25retriever = BM25Retriever.from_documents(documents=doc)
bm25retriever.k = 3

#Lexical Search
# query = "How to retrieve data using vectors?"
# results = bm25retriever.invoke(query)
    
#Hybrid Search
query = "How to retrieve data using vectors?"
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 5. Combine both using the EnsembleRetriever (Hybrid Search)
# weights=[0.5, 0.5] splits the relevance evenly between BM25 and Vector search
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25retriever, vector_retriever],
    weights=[0.5, 0.5]
)

results = hybrid_retriever.invoke(query)
print(results)