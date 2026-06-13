"""
Chunking Strategy:
- We use RecursiveCharacterTextSplitter from LangChain.
- It tries to split at natural boundaries (headings, paragraphs)
  before falling back to character-level splits.
- Chunk size of 1000 chars keeps enough context (code + explanation)
  while staying focused for retrieval.
- 200 char overlap prevents losing info at chunk boundaries.
"""

import os
import uuid
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup
import html2text
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.services import vector_store


def create_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Create the text splitter with our chosen strategy.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=[
            "\n## ",     # H2 headings (major sections)
            "\n### ",    # H3 headings (subsections)
            "\n#### ",   # H4 headings
            "\n\n",      # Paragraphs
            "\n",        # Lines
            " ",         # Words (last resort)
        ],
        length_function=len,
    )


def load_markdown_file(filepath: str) -> tuple[str, dict]:
    """
    Load a markdown file and return its content + metadata.

    Returns:
        Tuple of (text_content, metadata_dict)
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    filename = os.path.basename(filepath)
    metadata = {
        "source": filename,
        "filepath": filepath,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }

    return content, metadata


def load_from_url(url: str) -> tuple[str, dict]:
    """
    Fetch a webpage, extract main content, convert to markdown.

    Returns:
        Tuple of (text_content, metadata_dict)
    """
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    # Parse HTML and find the main content
    soup = BeautifulSoup(response.text, "html.parser")
    article = soup.find("article") or soup.find("div", class_="md-content") or soup.find("body")

    # Convert to markdown
    converter = html2text.HTML2Text()
    converter.ignore_links = False
    converter.ignore_images = True
    converter.body_width = 0

    content = converter.handle(str(article)).strip()

    # Create a filename from the URL
    source_name = url.rstrip("/").split("/")[-1] + ".md"

    metadata = {
        "source": source_name,
        "url": url,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }

    return content, metadata


def chunk_text(text: str, metadata: dict) -> tuple[list[str], list[dict], list[str]]:
    """
    Split text into chunks and prepare for vector store.

    Returns:
        Tuple of (texts, metadatas, ids) ready for vector_store.add_documents()
    """
    splitter = create_text_splitter()
    chunks = splitter.split_text(text)

    texts = []
    metadatas = []
    ids = []

    for i, chunk in enumerate(chunks):
        chunk_id = f"{metadata['source']}_chunk_{i}_{uuid.uuid4().hex[:8]}"

        chunk_metadata = {
            **metadata,
            "chunk_index": i,
            "total_chunks": len(chunks),
        }

        texts.append(chunk)
        metadatas.append(chunk_metadata)
        ids.append(chunk_id)

    return texts, metadatas, ids


def ingest_directory(docs_dir: str = None) -> dict:
    """
    Ingest all markdown files from the docs directory.

    Args:
        docs_dir: Path to directory with .md files.
                  Defaults to project's data/docs/ folder.

    Returns:
        Summary dict with counts.
    """
    if docs_dir is None:
        # Default: project_root/data/docs/
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        docs_dir = os.path.join(project_root, "data", "docs")

    if not os.path.exists(docs_dir):
        return {"status": "error", "message": f"Directory not found: {docs_dir}"}

    # Find all markdown files
    md_files = [f for f in os.listdir(docs_dir) if f.endswith(".md")]

    if not md_files:
        return {"status": "error", "message": "No .md files found in docs directory"}

    total_chunks = 0
    documents_ingested = 0

    for filename in md_files:
        filepath = os.path.join(docs_dir, filename)
        print(f"  Ingesting: {filename}")

        # Load the file
        content, metadata = load_markdown_file(filepath)

        # Split into chunks
        texts, metadatas, ids = chunk_text(content, metadata)

        # Add to vector store
        count = vector_store.add_documents(texts, metadatas, ids)
        total_chunks += count
        documents_ingested += 1

        print(f"    - {count} chunks created")

    return {
        "status": "success",
        "documents_ingested": documents_ingested,
        "chunks_created": total_chunks,
        "message": f"Successfully ingested {documents_ingested} document(s)",
    }


def ingest_urls(urls: list[str]) -> dict:
    """
    Ingest documents from a list of URLs.

    Returns:
        Summary dict with counts.
    """
    total_chunks = 0
    documents_ingested = 0
    errors = []

    for url in urls:
        try:
            print(f"  Ingesting from URL: {url}")

            content, metadata = load_from_url(url)
            texts, metadatas, ids = chunk_text(content, metadata)
            count = vector_store.add_documents(texts, metadatas, ids)

            total_chunks += count
            documents_ingested += 1
            print(f"    - {count} chunks created")

        except Exception as e:
            errors.append({"url": url, "error": str(e)})
            print(f"    [ERROR] Failed: {e}")

    result = {
        "status": "success" if documents_ingested > 0 else "error",
        "documents_ingested": documents_ingested,
        "chunks_created": total_chunks,
        "message": f"Ingested {documents_ingested} document(s)",
    }

    if errors:
        result["errors"] = errors

    return result


def ingest_file_content(content: str, filename: str) -> dict:
    """
    Ingest a single document from its raw text content.
    Used for file upload via the API.

    Returns:
        Summary dict with counts.
    """
    metadata = {
        "source": filename,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }

    texts, metadatas, ids = chunk_text(content, metadata)
    count = vector_store.add_documents(texts, metadatas, ids)

    return {
        "status": "success",
        "documents_ingested": 1,
        "chunks_created": count,
        "message": f"Ingested {filename} ({count} chunks)",
    }
