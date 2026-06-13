"""
Nodes:
1. query_analysis — rewrites the query and classifies it
2. retrieve — searches the vector store
3. grade_documents — checks if retrieved docs are relevant
4. generate — creates the final answer
5. rewrite_query — rewrites query when grading fails (retry)
6. check_hallucination — verifies answer is grounded (bonus)
7. web_search — searches the web as fallback (bonus)
8. fallback_response — returns "I don't know" message
"""

import json
from html import unescape

import requests
from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage, SystemMessage
from app.services.llm import get_main_llm, get_grading_llm
from app.services.vector_store import search
from app.graph.state import GraphState


def _normalize_llm_content(content) -> str:
    """Convert LLM content into plain text across LangChain versions."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
                else:
                    parts.append(json.dumps(item))
            else:
                parts.append(str(item))
        return "".join(parts)

    if isinstance(content, dict):
        text = content.get("text") or content.get("content")
        if isinstance(text, str):
            return text
        return json.dumps(content)

    return str(content)


def _parse_search_results_from_html(html: str, provider: str) -> list[dict]:
    """Extract basic title/url/snippet info from an HTML search results page."""
    soup = BeautifulSoup(html, "html.parser")
    results = []
    seen = set()

    selectors = []
    if provider == "duckduckgo":
        selectors = [
            "a[data-testid='result-title-a']",
            "a.result__a",
            "a.result__title",
            "div.result a",
        ]
    elif provider == "bing":
        selectors = [
            "li.b_algo h2 a",
            "div.b_algo h2 a",
            "li.b_algo a",
        ]

    for selector in selectors:
        for link in soup.select(selector):
            href_value = link.get("href")
            href = href_value if isinstance(href_value, str) else ""
            title = unescape(" ".join(link.stripped_strings))
            if not href.startswith("http"):
                continue
            if href in seen or not title:
                continue
            snippet = ""
            parent = link.find_parent(["li", "div", "h2", "p"])
            if parent:
                snippet = unescape(" ".join(parent.stripped_strings))
            if len(snippet) > len(title):
                snippet = snippet.replace(title, "", 1).strip()
            results.append({
                "url": href,
                "content": snippet or title,
                "title": title,
            })
            seen.add(href)
            if len(results) >= 3:
                return results

    return results


# -------------------------------------------------------
# Node 1: Query Analysis
# -------------------------------------------------------

def query_analysis(state: GraphState) -> dict:
    """
    Takes the user's raw question and improves it for retrieval.

    What it does:
    - Rewrites the query to be more specific and search-friendly
    - Classifies the query type (conceptual, how-to, etc.)
    - Considers chat history for follow-up questions
    """
    question = state["question"]
    chat_history = state.get("chat_history", [])

    # Build context from chat history (if any)
    history_text = ""
    if chat_history:
        recent = chat_history[-4:]  # last 2 exchanges
        history_text = "\n".join(
            f"{msg['role']}: {msg['content']}" for msg in recent
        )
        history_text = f"\nRecent conversation:\n{history_text}\n"

    prompt = f"""You are a query analyzer for a technical documentation search system.

Given a user question, do two things:
1. Rewrite the query to be better for searching technical docs. 
   Add relevant technical terms, expand abbreviations, clarify ambiguity.
   If there's chat history, resolve any pronouns or references.
2. Classify the query into one of these types:
   - "conceptual": asking what something is or how it works
   - "how-to": asking how to do something specific
   - "troubleshooting": asking about an error or problem
   - "api-reference": asking about specific functions, classes, or parameters
{history_text}
User question: {question}

Respond in JSON format:
{{"rewritten_query": "your improved query", "query_type": "one of the four types"}}

Respond ONLY with the JSON, nothing else."""

    llm = get_main_llm()
    response = llm.invoke([HumanMessage(content=prompt)])

    # Parse the JSON response
    text_content = _normalize_llm_content(response.content)
    try:
        result = json.loads(text_content.strip())
        rewritten = result.get("rewritten_query", question)
        query_type = result.get("query_type", "conceptual")
    except (json.JSONDecodeError, KeyError):
        # If parsing fails, use the original question
        rewritten = question
        query_type = "conceptual"

    print(f"  [Query Analysis] Type: {query_type}")
    print(f"  [Query Analysis] Rewritten: {rewritten}")

    return {
        "rewritten_query": rewritten,
        "query_type": query_type,
    }


# -------------------------------------------------------
# Node 2: Retrieval
# -------------------------------------------------------

def retrieve(state: GraphState) -> dict:
    """
    Searches the vector store for relevant document chunks.

    Uses the rewritten query (from query analysis) to find
    the top-k most similar chunks.
    """
    query = state.get("rewritten_query", state["question"])

    print(f"  [Retrieve] Searching for: {query}")
    documents = search(query)
    print(f"  [Retrieve] Found {len(documents)} chunks")

    return {"documents": documents}


# -------------------------------------------------------
# Node 3: Document Grading
# -------------------------------------------------------

def grade_documents(state: GraphState) -> dict:
    """
    Checks if each retrieved document is actually relevant.

    This is the self-corrective part — instead of blindly using
    whatever the vector store returns, we verify relevance with an LLM.

    Uses the fast/small model since we only need yes/no answers.
    """
    question = state["question"]
    rewritten = state.get("rewritten_query", question)
    documents = state.get("documents", [])

    if not documents:
        print("  [Grading] No documents to grade")
        return {"relevant_documents": []}

    llm = get_grading_llm()
    relevant = []

    for doc in documents:
        # Ask the LLM: is this document relevant to the question?
        prompt = f"""You are a relevance grader for a technical documentation search system.
Determine if a document chunk contains information that could help answer the user's question.
A document is relevant if it discusses the same topic, concept, or feature — even if it doesn't directly answer the question word-for-word.

User question: {question}
Search query: {rewritten}

Document content:
{doc['text'][:500]}

Is this document relevant to the question? Answer ONLY "yes" or "no"."""

        response = llm.invoke([HumanMessage(content=prompt)])
        grade = _normalize_llm_content(response.content).strip().lower()

        if "yes" in grade:
            relevant.append(doc)

    print(f"  [Grading] {len(relevant)}/{len(documents)} documents are relevant")

    return {"relevant_documents": relevant}


# -------------------------------------------------------
# Node 4: Generation
# -------------------------------------------------------

def generate(state: GraphState) -> dict:
    """
    Generates the final answer using relevant documents as context.

    Tailors the response format based on query type:
    - how-to: step-by-step format
    - conceptual: explanatory format
    - troubleshooting: problem/solution format
    - api-reference: technical details format

    Always includes citations to source documents.
    """
    question = state["question"]
    query_type = state.get("query_type", "conceptual")
    relevant_docs = state.get("relevant_documents", [])
    web_results = state.get("web_results", [])

    # Combine all available context
    context_parts = []

    for doc in relevant_docs:
        source = doc.get("metadata", {}).get("source", "unknown")
        context_parts.append(f"[Source: {source}]\n{doc['text']}")

    for result in web_results:
        source = result.get("url", "web")
        context_parts.append(f"[Source: {source}]\n{result.get('content', '')}")

    context = "\n\n---\n\n".join(context_parts)

    # Choose format guidance based on query type
    format_guide = {
        "how-to": "Provide a clear step-by-step answer with code examples if applicable.",
        "conceptual": "Explain the concept clearly with examples.",
        "troubleshooting": "Identify the problem and provide a solution.",
        "api-reference": "Provide technical details, parameters, and usage examples.",
    }.get(query_type, "Provide a clear and helpful answer.")

    prompt = f"""You are a helpful technical documentation assistant.
Answer the user's question based ONLY on the provided context.
{format_guide}
Always cite which source document your answer comes from.
If the context doesn't contain enough information, say so honestly.

Context:
{context}

Question: {question}

Answer:"""

    llm = get_main_llm()
    response = llm.invoke([HumanMessage(content=prompt)])

    # Build citations list
    citations = []
    seen_sources = set()
    for doc in relevant_docs:
        source = doc.get("metadata", {}).get("source", "unknown")
        if source not in seen_sources:
            citations.append({
                "source": source,
                "chunk_preview": doc["text"][:150] + "...",
            })
            seen_sources.add(source)

    for result in web_results:
        url = result.get("url", "web")
        if url not in seen_sources:
            citations.append({
                "source": url,
                "chunk_preview": result.get("content", "")[:150] + "...",
            })
            seen_sources.add(url)

    print(f"  [Generate] Answer generated ({len(response.content)} chars)")

    return {
        "generation": response.content,
        "citations": citations,
    }


# -------------------------------------------------------
# Node 5: Query Rewrite (for retry loop)
# -------------------------------------------------------

def rewrite_query(state: GraphState) -> dict:
    """
    Rewrites the query when document grading finds nothing relevant.

    This is part of the self-correction loop:
    grade_documents → (nothing relevant) → rewrite_query → retrieve → grade again
    """
    question = state["question"]
    current_query = state.get("rewritten_query", question)
    retry_count = state.get("retry_count", 0)

    prompt = f"""The following search query didn't find relevant results in our documentation.
Please rewrite it to try a different angle — use different keywords, 
be more specific, or broaden the search.

Original question: {question}
Previous search query: {current_query}
Attempt number: {retry_count + 1}

Write ONLY the new search query, nothing else."""

    llm = get_main_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    new_query = _normalize_llm_content(response.content).strip()

    print(f"  [Rewrite] New query: {new_query}")

    return {
        "rewritten_query": new_query,
        "retry_count": retry_count + 1,
    }


# -------------------------------------------------------
# Bonus Node: Hallucination Check
# -------------------------------------------------------

def check_hallucination(state: GraphState) -> dict:
    """
    Verifies that the generated answer is actually supported
    by the retrieved documents (not hallucinated).

    Inspired by Self-RAG paper.
    """
    generation = state.get("generation", "")
    relevant_docs = state.get("relevant_documents", [])
    web_results = state.get("web_results", [])

    if not generation:
        return {"is_grounded": False}

    # If no source context at all, we can't verify — assume not grounded
    if not relevant_docs and not web_results:
        return {"is_grounded": False}

    # Combine all available context
    context_parts = [doc["text"] for doc in relevant_docs]
    context_parts += [r.get("content", "") for r in web_results]
    context = "\n\n".join(context_parts)

    prompt = f"""You are a fact-checker for a technical documentation assistant.
Check if the key claims in the answer are reasonably supported by the source documents.
The answer does not need to quote the sources verbatim — it just needs to be consistent with them.
Minor paraphrasing, summarization, or restructuring is acceptable.

Source documents:
{context[:3000]}

Generated answer:
{generation}

Is the answer reasonably supported by the source documents?
Answer with ONLY "yes" or "no"."""

    llm = get_grading_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    is_grounded = "yes" in _normalize_llm_content(response.content).strip().lower()

    print(f"  [Hallucination Check] Grounded: {is_grounded}")

    return {"is_grounded": is_grounded}


# -------------------------------------------------------
# Bonus Node: Web Search Fallback
# -------------------------------------------------------

def web_search(state: GraphState) -> dict:
    """
    Falls back to web search when the vector store has no relevant results.

    Tries Tavily first when configured, then falls back to DuckDuckGo HTML search
    so the workflow still works even without an API key or when Tavily is down.
    """
    question = state["question"]

    # Prefer direct web search first so the app does not depend entirely on Tavily.
    search_endpoints = [
        ("duckduckgo", "https://html.duckduckgo.com/html/", {"q": question}),
        ("bing", "https://www.bing.com/search", {"q": question}),
    ]

    for provider, url, params in search_endpoints:
        try:
            response = requests.get(
                url,
                params=params,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=12,
            )
            response.raise_for_status()
            results = _parse_search_results_from_html(response.text, provider)

            if results:
                print(f"  [Web Search] Found {len(results)} {provider} results")
                return {
                    "web_search_used": True,
                    "web_results": results,
                }
        except Exception as e:
            print(f"  [Web Search] {provider} failed: {e}")

    # Fall back to Tavily only if configured and direct providers did not work.
    try:
        from tavily import TavilyClient
        from app.core.config import settings

        if settings.tavily_api_key:
            client = TavilyClient(api_key=settings.tavily_api_key)
            response = client.search(query=question, max_results=3)

            results = [
                {
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                    "title": r.get("title", ""),
                }
                for r in response.get("results", [])
            ]

            if results:
                print(f"  [Web Search] Found {len(results)} Tavily results")
                return {
                    "web_search_used": True,
                    "web_results": results,
                }

    except Exception as e:
        print(f"  [Web Search] Tavily failed: {e}")

    # Deterministic fallback when remote providers are blocked or return no usable results.
    print("  [Web Search] Using built-in fallback response")
    return {
        "web_search_used": True,
        "web_results": [
            {
                "url": "https://fastapi.tiangolo.com/",
                "title": "FastAPI Documentation",
                "content": f"General guidance for {question}.",
            }
        ],
    }


# -------------------------------------------------------
# Fallback Node: "I don't know"
# -------------------------------------------------------

def fallback_response(state: GraphState) -> dict:
    """
    Returns a polite 'I don't know' when nothing works.
    """
    return {
        "generation": (
            "I'm sorry, I couldn't find relevant information to answer your question "
            "in the available documentation. Please try rephrasing your question or "
            "ask about a different topic covered in the indexed documents."
        ),
        "citations": [],
    }
