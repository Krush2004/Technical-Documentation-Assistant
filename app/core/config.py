"""
Application settings — loaded from .env file.

All configuration lives here so the rest of the code
just imports `settings` and uses it directly.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central config. Values come from .env file or environment variables."""

    # ---- API Keys ----
    groq_api_key: str = ""
    tavily_api_key: str = ""  # optional, for web search fallback

    # ---- LLM Models ----
    llm_model: str = "llama-3.3-70b-versatile"         # main model for generation
    grading_llm_model: str = "llama-3.1-8b-instant"     # fast model for grading

    # ---- Embedding ----
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ---- Chunking ----
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # ---- Retrieval ----
    top_k: int = 5

    # ---- Self-Correction ----
    max_retries: int = 2

    # ---- ChromaDB ----
    chroma_persist_dir: str = "./chromadb_data"
    chroma_collection_name: str = "tech_docs"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Single instance used everywhere
settings = Settings()
