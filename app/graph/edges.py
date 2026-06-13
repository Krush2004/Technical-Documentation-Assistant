from app.graph.state import GraphState
from app.core.config import settings


def route_after_grading(state: GraphState) -> str:
    """
    Decides what to do after document grading.
    Three possible outcomes:
    1. Found relevant docs → generate an answer
    2. No relevant docs, but retries left → rewrite query and try again
    3. No relevant docs, no retries left → try web search or give up
    """
    relevant_docs = state.get("relevant_documents", [])
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", settings.max_retries)

    if len(relevant_docs) > 0:
        # We have relevant documents — proceed to generation
        print(f"  [Router] Found {len(relevant_docs)} relevant docs → generate")
        return "generate"

    if retry_count < max_retries:
        # No relevant docs, but we can retry
        print(f"  [Router] No relevant docs, retry {retry_count + 1}/{max_retries} → rewrite")
        return "rewrite_query"

    # No relevant docs and out of retries — try web search
    print("  [Router] No relevant docs, out of retries → web search")
    return "web_search"


def route_after_hallucination(state: GraphState) -> str:
    """
    Decides what to do after checking if the answer is grounded.
    Two outcomes:
    1. Answer is grounded in the context → done (END)
    2. Answer is NOT grounded → route to fallback_response
    """
    is_grounded = state.get("is_grounded", True)

    if is_grounded:
        print("  [Router] Answer is grounded → done")
        return "end"
    else:
        print("  [Router] Answer is NOT fully grounded/hallucinated → route to fallback_response")
        return "fallback"


def route_after_web_search(state: GraphState) -> str:
    """
    Decides what to do after web search.
    Two outcomes:
    1. Web search found results → generate answer from them
    2. Web search found nothing → return fallback message
    """
    web_results = state.get("web_results", [])

    if web_results:
        print(f"  [Router] Web search found {len(web_results)} results → generate")
        return "generate"
    else:
        print("  [Router] Web search found nothing → fallback")
        return "fallback_response"
