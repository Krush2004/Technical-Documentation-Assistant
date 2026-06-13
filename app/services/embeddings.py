from sentence_transformers import SentenceTransformer
from app.core.config import settings


# Load the model once and reuse it
_model = None


def get_embedding_model() -> SentenceTransformer:
    """Get or create the embedding model (lazy loading)."""
    global _model
    if _model is None:
        print(f"Loading embedding model: {settings.embedding_model}")
        _model = SentenceTransformer(settings.embedding_model)
        print("Embedding model loaded!")
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Turn a list of text strings into embedding vectors.

    Args:
        texts: List of strings to embed.

    Returns:
        List of embedding vectors (each is a list of floats).
    """
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()


def embed_single(text: str) -> list[float]:
    """Embed a single text string. Convenience wrapper."""
    return embed_texts([text])[0]
