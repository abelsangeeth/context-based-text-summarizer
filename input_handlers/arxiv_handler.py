"""
Vision-Augmented LLM Pipeline for Long-Form Document Summarization — arXiv / Research Paper Handler
═══════════════════════════════════════════════
Accepts:
  • arXiv ID         (e.g. "1706.03762")
  • arXiv URL        (e.g. "https://arxiv.org/abs/1706.03762")
  • DOI string       (e.g. "10.1038/s41586-021-03819-2")

Returns structured paper metadata + abstract + optionally full text
from the downloaded PDF.
"""

import re
import tempfile
import os

import arxiv

from input_handlers.cleaner import clean_text, chunk_text


# ═══════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════
def fetch_arxiv(query: str) -> dict:
    """
    Fetch a single arXiv paper by ID, URL, or DOI.

    Returns
    -------
    dict
        type, source, arxiv_id, title, authors, published, abstract,
        full_text (or None), chunks, word_count, summary_target
    """
    arxiv_id = _parse_input(query)
    if not arxiv_id:
        raise ValueError(
            "Could not parse arXiv ID from input. "
            "Provide an arXiv ID (e.g. 1706.03762), URL, or DOI."
        )

    try:
        client = arxiv.Client()
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(client.results(search))
    except StopIteration:
        raise ValueError("arXiv paper not found.")
    except Exception as exc:
        raise ValueError(f"Could not connect to arXiv: {exc}") from exc

    title = paper.title
    authors = [a.name for a in paper.authors]
    published = paper.published.strftime("%Y-%m-%d") if paper.published else None
    abstract = clean_text(paper.summary or "")

    # Try downloading the PDF for full text extraction
    full_text = None
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = paper.download_pdf(dirpath=tmp_dir)
            if pdf_path and os.path.exists(pdf_path):
                from input_handlers.pdf_handler import extract_pdf
                # Pass the path directly to extract_pdf
                pdf_result = extract_pdf(pdf_path)
                full_text = pdf_result.get("sections", {}).get("full_text")
    except Exception as e:
        import traceback
        traceback.print_exc()
        # PDF download/parse failed — fall back to abstract only
        full_text = None

    # Build chunks from full text if available, else from abstract
    text_for_chunks = full_text if full_text else abstract
    chunks = chunk_text(text_for_chunks, max_words=800, overlap=100)
    word_count = len(text_for_chunks.split())

    return {
        "type": "arxiv",
        "source": query,
        "arxiv_id": arxiv_id,
        "title": title,
        "authors": authors,
        "published": published,
        "abstract": abstract,
        "full_text": full_text,
        "chunks": chunks,
        "word_count": word_count,
        "summary_target": abstract,  # abstract is always most info-dense
    }


def search_arxiv(topic: str, max_results: int = 5) -> list[dict]:
    """
    Search arXiv by topic keyword. Used for the topic-based search
    feature in the UI.

    Returns a list of dicts with title, authors, arxiv_id, abstract,
    published.
    """
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=topic,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        results = []
        for paper in client.results(search):
            results.append({
                "title": paper.title,
                "authors": [a.name for a in paper.authors],
                "arxiv_id": paper.entry_id.split("/")[-1],
                "abstract": clean_text(paper.summary or ""),
                "published": (
                    paper.published.strftime("%Y-%m-%d")
                    if paper.published else None
                ),
            })
        return results

    except Exception as exc:
        raise ValueError(f"Could not connect to arXiv: {exc}") from exc


# ═══════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════════════════
def _parse_input(query: str) -> str | None:
    """
    Extract an arXiv ID from various input formats.

    Handles:
      • Direct ID:  "1706.03762" or "2301.07041v2"
      • arXiv URL:  "https://arxiv.org/abs/1706.03762"
      • DOI:        "10.48550/arXiv.1706.03762"
    """
    query = query.strip()

    # 1. arXiv URL → extract from path
    if "arxiv.org" in query:
        # Match patterns like /abs/1706.03762 or /pdf/1706.03762
        match = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", query)
        if match:
            return match.group(0)

    # 2. Direct arXiv ID
    match = re.match(r"^(\d{4}\.\d{4,5})(v\d+)?$", query)
    if match:
        return match.group(0)

    # 3. DOI containing arXiv ID
    if query.startswith("10."):
        match = re.search(r"(\d{4}\.\d{4,5})", query)
        if match:
            return match.group(1)

    return None
