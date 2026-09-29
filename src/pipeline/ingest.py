from pathlib import Path
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()


file_path = str(Path(__file__).resolve().parents[2] / "data" / "raw")

def ingest(file_path: str, is_directory: bool = True):
    paths = sorted(Path(file_path).glob("*.txt")) if is_directory else [Path(file_path)]
    documents = [
        Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": str(path)})
        for path in paths
    ]

    print(f"Loaded {len(documents)} documents from {file_path}")
    return documents

def chunk(documents, chunk_size=1000):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
    chunks = []
    for document in documents:
        for chunk in text_splitter.split_text(document.page_content):
            chunks.append(Document(page_content=chunk, metadata=document.metadata))
    return chunks


def ingest_pipeline(file_path: str = file_path, is_directory: bool = True):
    
    documents=ingest(file_path, is_directory=is_directory)    
    chunks=chunk(documents)
    
    # This can be in-memory or persisted to a directory
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="gemini_collection",
        persist_directory="./chroma_gemini_db"  # Optional: omit if you want in-memory only
    )
    
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks
    

if __name__ == "__main__":
    ingest_pipeline(file_path)