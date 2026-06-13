
from langgraph.graph import StateGraph, END

from app.graph.state import GraphState
from app.graph.nodes import (
    query_analysis,
    retrieve,
    grade_documents,
    generate,
    rewrite_query,
    check_hallucination,
    web_search,
    fallback_response,
)
from app.graph.edges import (
    route_after_grading,
    route_after_hallucination,
    route_after_web_search,
)


def build_graph() -> StateGraph:
    """
    Build and compile the RAG workflow graph.
    Returns a compiled graph ready to use with graph.invoke().
    """

    # Step 1: Create the graph with our state schema
    graph = StateGraph(GraphState)

    # Step 2: Add all the nodes
    graph.add_node("query_analysis", query_analysis)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("generate", generate)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("check_hallucination", check_hallucination)
    graph.add_node("web_search", web_search)
    graph.add_node("fallback_response", fallback_response)

    # Step 3: Wire up the edges (the flow)

    # Start → query analysis
    graph.set_entry_point("query_analysis")

    # query_analysis → retrieve (always)
    graph.add_edge("query_analysis", "retrieve")

    # retrieve → grade_documents (always)
    graph.add_edge("retrieve", "grade_documents")

    # grade_documents → CONDITIONAL (this is the key routing decision)
    graph.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "web_search": "web_search",
        },
    )

    # rewrite_query → retrieve (loop back for retry)
    graph.add_edge("rewrite_query", "retrieve")

    # generate → check_hallucination (always check generation)
    graph.add_edge("generate", "check_hallucination")

    # check_hallucination → CONDITIONAL (check if grounded/hallucinated)
    graph.add_conditional_edges(
        "check_hallucination",
        route_after_hallucination,
        {
            "end": END,
            "fallback": "fallback_response",
        },
    )

    # web_search → CONDITIONAL
    graph.add_conditional_edges(
        "web_search",
        route_after_web_search,
        {
            "generate": "generate",
            "fallback_response": "fallback_response",
        },
    )

    # fallback → END
    graph.add_edge("fallback_response", END)

    # Step 4: Compile and return
    compiled = graph.compile()
    print("Graph compiled successfully!")
    return compiled


# Build once and reuse
_compiled_graph = None


def get_graph():
    """Get or create the compiled graph (singleton)."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_query(question: str, session_id: str = "", chat_history: list = None) -> dict:
    """
    Run a question through the RAG pipeline.
    This is the main entry point for the graph.

    Args:
        question: The user's question.
        session_id: Optional session ID for conversation memory.
        chat_history: Optional list of past messages.

    Returns:
        The final state dict with the answer and metadata.
    """
    graph = get_graph()

    # Prepare the initial state
    initial_state = {
        "question": question,
        "session_id": session_id or "",
        "rewritten_query": "",
        "query_type": "",
        "documents": [],
        "relevant_documents": [],
        "retry_count": 0,
        "max_retries": 2,
        "generation": "",
        "citations": [],
        "is_grounded": True,
        "web_search_used": False,
        "web_results": [],
        "chat_history": chat_history or [],
    }

    print(f"\n{'='*50}")
    print(f"Processing question: {question}")
    print(f"{'='*50}")

    # Run the graph
    result = graph.invoke(initial_state)

    return result
