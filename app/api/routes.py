"""
API Routes — all the FastAPI endpoints.

Endpoints:
- POST /query         → Ask a question
- POST /ingest        → Add new documents
- GET  /documents     → List what's indexed
- POST /feedback      → Submit feedback
- GET  /health        → Health check
"""

import uuid
from collections import defaultdict

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.schemas.models import (
    QueryRequest,
    QueryResponse,
    IngestURLRequest,
    IngestResponse,
    DocumentsResponse,
    DocumentInfo,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
)
from app.services import vector_store, ingestion
from app.graph.workflow import run_query


router = APIRouter()

# ---- In-memory stores ----

# Conversation memory: session_id → list of messages
chat_sessions: dict[str, list[dict]] = defaultdict(list)

# Feedback storage: list of feedback entries
feedback_store: list[dict] = []


# -------------------------------------------------------
# POST /query — Ask a question
# -------------------------------------------------------

@router.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    """
    Submit a question and get an answer from the RAG pipeline.

    The system will:
    1. Analyze and rewrite your query
    2. Search for relevant documents
    3. Grade document relevance
    4. Generate an answer with citations
    """
    # Check if there are any documents to search
    if vector_store.get_document_count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Use POST /ingest first.",
        )

    # Get or create session
    session_id = request.session_id or str(uuid.uuid4())
    chat_history = chat_sessions.get(session_id, [])

    # Run the RAG pipeline
    result = run_query(
        question=request.question,
        session_id=session_id,
        chat_history=chat_history,
    )

    # Update chat history
    chat_sessions[session_id].append({"role": "user", "content": request.question})
    chat_sessions[session_id].append({"role": "assistant", "content": result["generation"]})

    # Keep only last 10 messages per session
    if len(chat_sessions[session_id]) > 10:
        chat_sessions[session_id] = chat_sessions[session_id][-10:]

    return QueryResponse(
        answer=result["generation"],
        citations=result.get("citations", []),
        query_type=result.get("query_type", "unknown"),
        rewritten_query=result.get("rewritten_query", request.question),
        web_search_used=result.get("web_search_used", False),
        retry_count=result.get("retry_count", 0),
        session_id=session_id,
    )


# -------------------------------------------------------
# POST /ingest — Add new documents
# -------------------------------------------------------

@router.post("/ingest", response_model=IngestResponse, status_code=201)
def ingest_urls_endpoint(request: IngestURLRequest):
    """
    Ingest documents from URLs.

    Provide a list of URLs and the system will:
    1. Fetch each page
    2. Extract the main content
    3. Split into chunks
    4. Generate embeddings
    5. Store in the vector database
    """
    result = ingestion.ingest_urls(request.urls)

    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])

    return IngestResponse(**result)


@router.post("/ingest/files", response_model=IngestResponse, status_code=201)
async def ingest_files_endpoint(files: list[UploadFile] = File(...)):
    """
    Ingest documents from file uploads.

    Upload one or more text/markdown files.
    """
    total_chunks = 0
    documents_ingested = 0
    errors = []

    for file in files:
        try:
            content = await file.read()
            text = content.decode("utf-8")

            result = ingestion.ingest_file_content(text, file.filename)
            total_chunks += result["chunks_created"]
            documents_ingested += 1

        except Exception as e:
            errors.append({"file": file.filename, "error": str(e)})

    response = IngestResponse(
        status="success" if documents_ingested > 0 else "error",
        documents_ingested=documents_ingested,
        chunks_created=total_chunks,
        message=f"Ingested {documents_ingested} file(s)",
        errors=errors if errors else None,
    )

    if documents_ingested == 0:
        raise HTTPException(status_code=400, detail="No files could be ingested")

    return response


# -------------------------------------------------------
# GET /documents — List indexed documents
# -------------------------------------------------------

@router.get("/documents", response_model=DocumentsResponse)
def list_documents_endpoint():
    """
    List all documents currently in the vector store.

    Shows the source file name and how many chunks each has.
    """
    all_docs = vector_store.get_all_documents()
    total_chunks = len(all_docs)

    # Group by source file
    source_counts: dict[str, int] = defaultdict(int)
    for doc in all_docs:
        source = doc["metadata"].get("source", "unknown")
        source_counts[source] += 1

    documents = [
        DocumentInfo(source=source, chunk_count=count)
        for source, count in sorted(source_counts.items())
    ]

    return DocumentsResponse(
        total_documents=len(documents),
        total_chunks=total_chunks,
        documents=documents,
    )


# -------------------------------------------------------
# POST /feedback — Submit feedback
# -------------------------------------------------------

@router.post("/feedback", response_model=FeedbackResponse)
def feedback_endpoint(request: FeedbackRequest):
    """
    Submit feedback on an answer.

    Rate with 'thumbs_up' or 'thumbs_down' and optionally add a comment.
    """
    feedback_store.append({
        "session_id": request.session_id,
        "question": request.question,
        "rating": request.rating,
        "comment": request.comment,
    })

    return FeedbackResponse(
        status="feedback_recorded",
        message="Thank you for your feedback!",
    )


# -------------------------------------------------------
# GET /health — Health check
# -------------------------------------------------------

@router.get("/health", response_model=HealthResponse)
def health_endpoint():
    """Check if the system is running and ready."""
    return HealthResponse(
        status="healthy",
        vector_store_documents=vector_store.get_document_count(),
        llm_provider="groq",
    )
