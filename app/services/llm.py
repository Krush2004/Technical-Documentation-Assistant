from langchain_groq import ChatGroq
from app.core.config import settings


# Lazy-loaded LLM instances
_main_llm = None
_grading_llm = None


def get_main_llm() -> ChatGroq:
    """
    Get the main LLM for generation and query analysis.
    Uses the larger model for better quality answers.
    """
    global _main_llm
    if _main_llm is None:
        _main_llm = ChatGroq(
            model=settings.llm_model,
            api_key=settings.groq_api_key,
            temperature=0.3,  # low temp for factual answers
        )
    return _main_llm


def get_grading_llm() -> ChatGroq:
    """
    Get the fast LLM for document grading.
    Uses a smaller model — we only need yes/no answers here.
    """
    global _grading_llm
    if _grading_llm is None:
        _grading_llm = ChatGroq(
            model=settings.grading_llm_model,
            api_key=settings.groq_api_key,
            temperature=0,  # zero temp for consistent grading
        )
    return _grading_llm
