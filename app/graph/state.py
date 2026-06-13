from typing import TypedDict, Annotated
import operator


class GraphState(TypedDict):
    """
    The state that flows through our RAG pipeline.

    Each field tracks a specific piece of data:
    - question: what the user asked (never changes)
    - rewritten_query: improved version for better retrieval
    - query_type: helps tailor the answer format
    - documents: raw results from vector store
    - relevant_documents: filtered after grading
    - retry_count: how many times we've retried (prevents infinite loops)
    - max_retries: the limit (default 2)
    - generation: the final answer text
    - citations: source references for the answer
    - web_search_used: whether we fell back to web search
    - web_results: results from web search (if used)
    - chat_history: past conversation turns (for follow-ups)
    - session_id: identifies the conversation session
    """

    # ---- Input ----
    question: str
    session_id: str

    # ---- Query Analysis ----
    rewritten_query: str
    query_type: str  # "conceptual", "how-to", "troubleshooting", "api-reference"

    # ---- Retrieval ----
    documents: list[dict]

    # ---- Grading ----
    relevant_documents: list[dict]

    # ---- Self-Correction ----
    retry_count: int
    max_retries: int

    # ---- Generation ----
    generation: str
    citations: list[dict]

    # ---- Bonus: Hallucination Check ----
    is_grounded: bool

    # ---- Bonus: Web Search ----
    web_search_used: bool
    web_results: list[dict]

    # ---- Bonus: Conversation Memory ----
    # Using Annotated with operator.add so new messages get appended
    chat_history: Annotated[list[dict], operator.add]
