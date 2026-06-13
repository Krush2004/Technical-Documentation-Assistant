"""
FastAPI Application Entry Point.

This is the main file that starts the server.
Run with:  uvicorn app.main:app --reload
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.services import ingestion, vector_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs on startup and shutdown.

    On startup:
    - Checks if documents are already indexed
    - If not, auto-ingests from data/docs/ folder (if it exists)
    """
    print("\nStarting RAG Documentation Assistant...")

    # Check if we already have documents
    count = vector_store.get_document_count()
    if count > 0:
        print(f"Vector store has {count} chunks already indexed")
    else:
        # Try auto-ingesting from data/docs/
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        docs_dir = os.path.join(project_root, "data", "docs")

        if os.path.exists(docs_dir) and os.listdir(docs_dir):
            print("Auto-ingesting documents from data/docs/...")
            result = ingestion.ingest_directory(docs_dir)
            print(f"Success: {result['message']}")
        else:
            print("Warning: No documents found. Use POST /ingest to add documents.")

    print("Server is ready!\n")

    yield  # Server runs here

    print("\nShutting down...")


# Create the app
app = FastAPI(
    title="RAG Documentation Assistant",
    description=(
        "A Retrieval-Augmented Generation system for answering questions "
        "about technical documentation. Built with LangGraph, FastAPI, and ChromaDB."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Register routes
app.include_router(router)
