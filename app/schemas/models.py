from pydantic import BaseModel, Field
from typing import Optional


# -------------------------------------------------------
# Request Models
# -------------------------------------------------------

class QueryRequest(BaseModel):
    """What the user sends to /query."""
    question: str = Field(..., description="The question to ask", min_length=1)
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for conversation memory (optional)",
    )


class IngestURLRequest(BaseModel):
    """What the user sends to /ingest for URL-based ingestion."""
    urls: list[str] = Field(..., description="List of URLs to ingest", min_length=1)


class FeedbackRequest(BaseModel):
    """What the user sends to /feedback."""
    session_id: Optional[str] = Field(default=None, description="Session ID")
    question: str = Field(..., description="The question that was asked")
    rating: str = Field(
        ...,
        description="Rating: 'thumbs_up' or 'thumbs_down'",
        pattern="^(thumbs_up|thumbs_down)$",
    )
    comment: Optional[str] = Field(
        default=None,
        description="Optional text comment",
    )


# -------------------------------------------------------
# Response Models
# -------------------------------------------------------

class Citation(BaseModel):
    """A source reference in the answer."""
    source: str
    chunk_preview: str


class QueryResponse(BaseModel):
    """What /query returns."""
    answer: str
    citations: list[Citation]
    query_type: str
    rewritten_query: str
    web_search_used: bool
    retry_count: int
    session_id: str


class IngestResponse(BaseModel):
    """What /ingest returns."""
    status: str
    documents_ingested: int
    chunks_created: int
    message: str
    errors: Optional[list[dict]] = None


class DocumentInfo(BaseModel):
    """Info about a single indexed document."""
    source: str
    chunk_count: int


class DocumentsResponse(BaseModel):
    """What /documents returns."""
    total_documents: int
    total_chunks: int
    documents: list[DocumentInfo]


class FeedbackResponse(BaseModel):
    """What /feedback returns."""
    status: str
    message: str


class HealthResponse(BaseModel):
    """What /health returns."""
    status: str
    vector_store_documents: int
    llm_provider: str
