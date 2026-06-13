"""
Fetch FastAPI documentation pages and save them as markdown files.

This script downloads 5 pages from FastAPI's official docs,
extracts the main content, and saves them in data/docs/.

Run once:  python scripts/fetch_docs.py
"""

import os
import requests
from bs4 import BeautifulSoup
import html2text


# Pages to fetch — each is a (filename, url) pair
PAGES = [
    (
        "fastapi_getting_started.md",
        "https://fastapi.tiangolo.com/tutorial/first-steps/",
    ),
    (
        "fastapi_path_parameters.md",
        "https://fastapi.tiangolo.com/tutorial/path-params/",
    ),
    (
        "fastapi_query_parameters.md",
        "https://fastapi.tiangolo.com/tutorial/query-params/",
    ),
    (
        "fastapi_request_body.md",
        "https://fastapi.tiangolo.com/tutorial/body/",
    ),
    (
        "fastapi_dependencies.md",
        "https://fastapi.tiangolo.com/tutorial/dependencies/",
    ),
]


def fetch_page(url: str) -> str:
    """Download a webpage and return its HTML content."""
    print(f"  Fetching: {url}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text


def html_to_markdown(html_content: str) -> str:
    """
    Extract the main article content from FastAPI docs HTML
    and convert it to clean markdown.
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # FastAPI docs use <article> or <div class="md-content"> for main content
    article = soup.find("article") or soup.find("div", class_="md-content")

    if not article:
        # Fallback: use the whole body
        article = soup.find("body")

    # Convert HTML to markdown
    converter = html2text.HTML2Text()
    converter.ignore_links = False
    converter.ignore_images = True
    converter.body_width = 0  # don't wrap lines

    markdown = converter.handle(str(article))
    return markdown.strip()


def save_docs():
    """Fetch all pages and save them as markdown files."""
    # Figure out where to save (project_root/data/docs/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    docs_dir = os.path.join(project_root, "data", "docs")

    os.makedirs(docs_dir, exist_ok=True)

    print("Fetching FastAPI documentation...\n")

    for filename, url in PAGES:
        try:
            html = fetch_page(url)
            markdown = html_to_markdown(html)

            filepath = os.path.join(docs_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"# Source: {url}\n\n")
                f.write(markdown)

            print(f"  [OK] Saved: {filename} ({len(markdown)} chars)\n")

        except Exception as e:
            print(f"  [ERROR] Failed: {filename} - {e}\n")

    print("Done! Documents saved to data/docs/")


if __name__ == "__main__":
    save_docs()
