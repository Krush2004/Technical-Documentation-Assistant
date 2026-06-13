"""
ChromaDB vector store setup and operations.

Handles creating the collection, adding documents,
and searching for similar chunks.
"""

import chromadb
from app.core.config import settings
from app.services.embeddings import embed_texts, embed_single


# ChromaDB client — created once, reused everywhere
_client = None
_collection = None


def get_chroma_client() -> chromadb.PersistentClient:
    """Get or create the ChromaDB client."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return _client


def get_collection() -> chromadb.Collection:
    """Get or create the document collection."""
    global _collection
    if _collection is None:
        client = get_chroma_client()
        _collection = client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"description": "Technical documentation chunks"},
        )
    return _collection


def add_documents(
    texts: list[str],
    metadatas: list[dict],
    ids: list[str],
) -> int:
    """
    Add document chunks to the vector store.

    Args:
        texts: The text content of each chunk.
        metadatas: Metadata for each chunk (source, section, etc.).
        ids: Unique ID for each chunk.

    Returns:
        Number of chunks added.
    """
    collection = get_collection()

    # Generate embeddings for all chunks
    embeddings = embed_texts(texts)

    # Add to ChromaDB
    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )

    return len(texts)


def search(query: str, top_k: int = None) -> list[dict]:
    """
    Search the vector store for chunks similar to the query.

    Args:
        query: The search query text.
        top_k: Number of results to return (defaults to settings.top_k).

    Returns:
        List of dicts, each with 'text', 'metadata', and 'score' keys.
    """
    if top_k is None:
        top_k = settings.top_k

    collection = get_collection()

    # Don't search if collection is empty
    if collection.count() == 0:
        return []

    # Embed the query and search
    query_embedding = embed_single(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    # Package results into a clean list of dicts
    documents = []
    for i in range(len(results["ids"][0])):
        documents.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": 1 - results["distances"][0][i],  # convert distance to similarity
            "id": results["ids"][0][i],
        })

    return documents


def get_all_documents() -> list[dict]:
    """Get info about all documents in the collection."""
    collection = get_collection()

    if collection.count() == 0:
        return []

    # Get all items (just metadata, not full text)
    all_items = collection.get(include=["metadatas"])

    return [
        {"id": id, "metadata": meta}
        for id, meta in zip(all_items["ids"], all_items["metadatas"])
    ]


def get_document_count() -> int:
    """How many chunks are in the vector store."""
    return get_collection().count()


def delete_collection():
    """Delete the entire collection (useful for re-ingesting)."""
    global _collection
    client = get_chroma_client()
    try:
        client.delete_collection(settings.chroma_collection_name)
    except ValueError:
        pass  # collection doesn't exist, that's fine
    _collection = None
