"""Single source of truth for API keys and tunable constants, loaded from .env.

Every pipeline script imports from here instead of hardcoding values, so the whole
pipeline can be retargeted (model, collection, store location, chunk/retrieval sizes)
by editing .env only.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- Secrets ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Models ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-3.5-flash")

# --- Vector store ---
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "gemini_collection")
PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY", "./chroma_gemini_db")

# --- Chunking / retrieval tuning ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "3"))
