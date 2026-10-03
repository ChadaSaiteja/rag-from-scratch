"""Ingestion stage: load the raw corpus, chunk it, embed it, and persist it to Chroma."""
from pathlib import Path
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, COLLECTION_NAME, EMBEDDING_MODEL, GOOGLE_API_KEY, PERSIST_DIRECTORY


# Default corpus location: <repo_root>/data/raw
file_path = str(Path(__file__).resolve().parents[2] / "data" / "raw")

def ingest(file_path: str, is_directory: bool = True):
    """Load .txt files into Documents, tagging each with its source path as metadata."""
    try:
        paths = sorted(Path(file_path).glob("*.txt")) if is_directory else [Path(file_path)]
        documents = [
            Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": str(path), "filename": str(path.name)})
            for path in paths
        ]

        print(f"Loaded {len(documents)} documents from {file_path}")
        return documents
    except Exception as e:
        print(f"[ingest] error: {e}")
        raise

def chunk(documents, chunk_size=CHUNK_SIZE):
    """Split each document into smaller chunks, preserving the parent's metadata."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
        chunks = []
        for document in documents:
            for chunk in text_splitter.split_text(document.page_content):
                chunks.append(Document(page_content=chunk, metadata=document.metadata))
        return chunks
    except Exception as e:
        print(f"[chunk] error: {e}")
        raise


def ingest_pipeline(file_path: str = file_path, is_directory: bool = True):
    """Load, chunk, embed, and persist the corpus into the Chroma vector store."""
    try:
        documents=ingest(file_path, is_directory=is_directory)    
        chunks=chunk(documents)
        
        # This can be in-memory or persisted to a directory
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=PERSIST_DIRECTORY  # Optional: omit if you want in-memory only
        )
        
        print(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    except Exception as e:
        print(f"[ingest_pipeline] error: {e}")
        raise
    

if __name__ == "__main__":
    ingest_pipeline(file_path)