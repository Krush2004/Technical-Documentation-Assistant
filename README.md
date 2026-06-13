# RAG-Based Technical Documentation Assistant

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688) ![LangGraph](https://img.shields.io/badge/LangGraph-RAG-green) ![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Search-orange)

A state-of-the-art Retrieval-Augmented Generation (RAG) assistant designed to fetch, process, index, and answer queries regarding technical documentation. The pipeline uses a self-corrective LangGraph workflow to improve retrieval quality, grade relevance, and handle out-of-domain questions with a web-search fallback.

Served through **FastAPI** with a lightweight **Streamlit** frontend for interactive testing.

## 🚀 Quick Start

1. Create and activate a Python environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env` and add your API keys.
4. Start the app with `uvicorn app.main:app --reload`.
5. Send questions to `POST /query` or use the Streamlit UI.

## 🧩 Implementation Summary

This project implements the core assignment requirements:
- a LangGraph workflow with query analysis, retrieval, grading, generation, and retry routing,
- a document ingestion pipeline with chunking and embeddings,
- a FastAPI API with query, ingest, documents, feedback, and health endpoints,
- optional web-search fallback and conversation memory support.

---

## 🏗️ System Architecture

### LangGraph Workflow

The core QA loop uses a stateful LangGraph workflow. It does not blindly feed retrieved context to the LLM; instead, it dynamically evaluates the retrieved documents, corrects query representation, and searches the web if local documentation lacks answers.

```
                  ┌──────────────────────┐
                  │        Start         │
                  └──────────┬───────────┘
                             ▼
                ┌──────────────────────────┐
                │  Node 1: Query Analysis  │
                └────────────┬─────────────┘
                             ▼
                ┌──────────────────────────┐
                │    Node 2: Retrieval     │◀──────────────────┐
                └────────────┬─────────────┘                   │
                             ▼                                 │
                ┌──────────────────────────┐                   │
                │ Node 3: Document Grading │                   │
                └────────────┬─────────────┘                   │ (Loop /
                             │                                 │  Retry)
                             ├───────────[No, Retries Left]──▶││
                             │ (Relevant docs?)                │
                             ├───────────[No, No Retries]──────┼──────────────┐
                             │                                 │              │
                             ▼ [Yes]                           │              │
                ┌──────────────────────────┐                   │              │
                │   Hallucination Check    │                   │              │
                └────────────┬─────────────┘                   │              │
                             ▼                                 │              │
                ┌──────────────────────────┐                   │              │
                │    Node 4: Generation    │                   │              │
                └────────────┬─────────────┘                   │              │
                             ▼                                 │              │
                          [ END ]                              │              │
                                                               ▼              ▼
                                                      ┌────────────────┐ ┌───────────────┐
                                                      │ Query Rewrite  │ │  Web Search   │
                                                      └────────────────┘ └──────┬────────┘
                                                                                ▼
                                                                       ┌────────────────┐
                                                                       │ Generate from  │
                                                                       │  Web or Fail   │
                                                                       └────────┬───────┘
                                                                                ▼
                                                                             [ END ]
```
## 🏷️ Tech Stack

| Category | Technologies |
|---|---|
| Language | Python |
| API Framework | FastAPI, Uvicorn |
| Workflow Engine | LangGraph, LangChain, LangChain Core |
| Vector Database | ChromaDB |
| Embeddings | Sentence-Transformers |
| LLM Providers | Groq |
| Search Fallback | Tavily, Requests, BeautifulSoup |
| Frontend | Streamlit |


---

### Workflow Node details:
1. **Query Analysis:** Expands search terms, resolves pronouns from conversation history, and classifies the question into a query type (`conceptual`, `how-to`, `troubleshooting`, `api-reference`).
2. **Retrieval:** Performs a vector similarity search on ChromaDB.
3. **Document Grading:** Grades each chunk as relevant/irrelevant with a fast binary grader LLM.
4. **Conditional Routing:**
   - If relevant documents exist, proceeds to answer generation.
   - If no relevant documents exist and retries are left, rewrites the query and loops back to Retrieval.
   - If no relevant documents exist and retries are exhausted, falls back to Web Search using Tavily.
5. **Generation:** Synthesizes the final response tailored to the query type, adding inline citations.

---

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.10+
- A Groq API Key (for LLM generation and grading)
- A Tavily API Key (optional, for web search fallback)

### Installation
1. Clone this repository and navigate to the project directory:
   ```bash
   git clone <repository_url>
   cd Assessment
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables. Copy `.env.example` to `.env` and fill in your keys:
   ```bash
   cp .env.example .env
   ```
   Open `.env` and set:
   ```env
   GROQ_API_KEY=gsk_your_groq_api_key
   TAVILY_API_KEY=tvly-dev-your_tavily_api_key
   ```

---

## 🚀 Running the Application

### 1. Ingest Documentation (Auto or Manual)
The system is configured to index **FastAPI documentation** pages by default.

- On startup, the FastAPI app automatically checks the vector store. If empty, it loads and chunks the pre-configured pages from `data/docs/` (or fetches them if missing).
- You can manually trigger fetching of the default FastAPI documentation pages by running:
  ```bash
  python scripts/fetch_docs.py
  ```

### 2. Start the FastAPI Server
Run the API using Uvicorn:
```bash
uvicorn app.main:app --reload
```
The server will run at `http://localhost:8000`. You can access the interactive Swagger documentation at `http://localhost:8000/docs`.

### 3. Start the Streamlit Frontend (Bonus)
In a separate terminal (with the virtual environment activated), start the UI:
```bash
streamlit run streamlit_app.py
```
This launches a browser window showing a premium chat interface.

---

## 📡 API Reference & Example Requests

### 1. Query System
- **Endpoint:** `POST /query`
- **Request:**
  ```json
  {
    "question": "How do I create path parameters?",
    "session_id": "test-session-123"
  }
  ```
- **Response:**
  ```json
  {
    "answer": "In FastAPI, you can declare path parameters or variables with the same syntax used by Python format strings...\n\n```python\nfrom fastapi import FastAPI\n\napp = FastAPI()\n\n@app.get(\"/items/{item_id}\")\ndef read_item(item_id: int):\n    return {\"item_id\": item_id}\n```\n*(Source: fastapi_path_parameters.md)*",
    "citations": [
      {
        "source": "fastapi_path_parameters.md",
        "chunk_preview": "You can declare path parameters or variables with the same syntax..."
      }
    ],
    "query_type": "how-to",
    "rewritten_query": "How to define path parameters in FastAPI and type hints",
    "web_search_used": false,
    "retry_count": 0,
    "session_id": "test-session-123"
  }
  ```

### 2. Ingest New Documents from URLs
- **Endpoint:** `POST /ingest`
- **Request:**
  ```json
  {
    "urls": [
      "https://fastapi.tiangolo.com/tutorial/query-params/"
    ]
  }
  ```
- **Response:**
  ```json
  {
    "status": "success",
    "documents_ingested": 1,
    "chunks_created": 8,
    "message": "Ingested 1 document(s)"
  }
  ```

### 3. List Indexed Documents
- **Endpoint:** `GET /documents`
- **Response:**
  ```json
  {
    "total_documents": 5,
    "total_chunks": 42,
    "documents": [
      {
        "source": "fastapi_getting_started.md",
        "chunk_count": 10
      },
      {
        "source": "fastapi_path_parameters.md",
        "chunk_count": 8
      }
    ]
  }
  ```

### 4. Feedback
- **Endpoint:** `POST /feedback`
- **Request:**
  ```json
  {
    "session_id": "test-session-123",
    "question": "How do I create path parameters?",
    "rating": "thumbs_up",
    "comment": "Perfect answer with code example!"
  }
  ```
- **Response:**
  ```json
  {
    "status": "feedback_recorded",
    "message": "Thank you for your feedback!"
  }
  ```

---

## 🧠 Design Decisions & Tradeoffs

### 1. Vector Store & Local Embeddings
- **Decision:** ChromaDB + `sentence-transformers/all-MiniLM-L6-v2`.
- **Tradeoff:** Using a local 384-dimensional embedding model means we do not rely on paid, network-bound APIs (like OpenAI's) for document indexing and retrieval. It runs entirely on the host CPU/GPU in seconds. While larger models (like `text-embedding-3-large`) provide slightly better retrieval accuracy, MiniLM offers an ideal speed-to-accuracy balance for a 5-document local corpus.

### 2. Chunking Strategy
- **Decision:** `RecursiveCharacterTextSplitter` with 1000 character chunks and 200 character overlap, prioritizing Markdown structures (`\n##`, `\n###`, `\n\n`).
- **Tradeoff:** Technical documentation contains code snippets intertwined with textual explanations. A simple character-based or token-based split can tear a code block in half. By prioritizing Markdown header and paragraph splits, we preserve the syntactic logic of the documents, keeping code snippets and their context within a single chunk.

### 3. Model Specialization (Main vs. Grader)
- **Decision:** Groq `llama-3.3-70b-versatile` for Query Analysis & Generation; `llama-3.1-8b-instant` for Grading.
- **Tradeoff:** Evaluating document relevance is a binary decision task. Using a massive 70B model is slow and computationally wasteful for grading 5 retrieved documents. We route grading requests to a highly-efficient 8B model to keep response times under ~1.5 seconds, reserving the 70B model for nuanced text generation and query restructuring tasks.

### 4. Self-Corrective Retry Limit
- **Decision:** Strict limit of `max_retries = 2`.
- **Tradeoff:** If retrieval yields no relevant context, query rewriting helps find a different formulation. However, letting the graph loop indefinitely could cause infinite execution loops and exhaust API rate limits. The hard cap guarantees we fall back to web search or return a graceful failure quickly.
