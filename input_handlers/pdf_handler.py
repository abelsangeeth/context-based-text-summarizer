"""
ContextFree-X — PDF Handler
════════════════════════════
Extracts text from uploaded PDFs.
• Detects research papers (abstract / intro / conclusion).
• Falls back to chunked full text for regular documents.
"""

import re

import pdfplumber

from input_handlers.cleaner import clean_text, chunk_text


# ═══════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════
def extract_pdf(uploaded_file) -> dict:
    """
    Extract and structure text from a Streamlit UploadedFile PDF.

    Returns
    -------
    dict
        type, source, is_research_paper, pages, sections, chunks,
        word_count
    """
    filename = getattr(uploaded_file, "name", "unknown.pdf")

    try:
        full_text, page_count = _read_pages(uploaded_file)
    except Exception as exc:
        # pdfplumber raises various errors for encrypted PDFs
        if "password" in str(exc).lower() or "encrypted" in str(exc).lower():
            raise ValueError("PDF is password protected.") from exc
        raise

    if not full_text.strip():
        raise ValueError(
            "Scanned PDF detected. Please upload a text-based PDF."
        )

    is_research = _detect_research_paper(full_text)

    if is_research:
        sections = _extract_sections(full_text)
    else:
        sections = {
            "abstract": None,
            "introduction": None,
            "conclusion": None,
        }

    # Clean every section + full text
    cleaned_full = clean_text(full_text)
    for key in ("abstract", "introduction", "conclusion"):
        if sections.get(key):
            sections[key] = clean_text(sections[key])
    sections["full_text"] = cleaned_full

    chunks = chunk_text(cleaned_full, max_words=800, overlap=100)
    word_count = len(cleaned_full.split())

    return {
        "type": "pdf",
        "source": filename,
        "is_research_paper": is_research,
        "pages": page_count,
        "sections": sections,
        "chunks": chunks,
        "word_count": word_count,
    }


def get_summary_target(extracted: dict) -> str:
    """
    Choose the best text segment to summarise.

    Priority:
      1. Research paper with abstract → abstract
      2. Research paper without abstract → conclusion
      3. Regular doc, long → first chunk
      4. Regular doc, short → full text
    """
    sections = extracted.get("sections", {})

    if extracted.get("is_research_paper"):
        if sections.get("abstract"):
            return sections["abstract"]
        if sections.get("conclusion"):
            return sections["conclusion"]

    # Regular document
    if extracted["word_count"] > 800:
        return extracted["chunks"][0]
    return sections.get("full_text", "")


# ═══════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════════════════
def _read_pages(uploaded_file) -> tuple[str, int]:
    """Return (joined_text, page_count) from all PDF pages."""
    full_text: list[str] = []
    page_count = 0

    with pdfplumber.open(uploaded_file) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text.append(text.strip())

    return "\n\n".join(full_text), page_count


def _detect_research_paper(text: str) -> bool:
    """
    Heuristic: if ≥ 3 of these common academic headers appear
    (case-insensitive) the document is likely a research paper.
    """
    lower = text.lower()
    markers = [
        "abstract",
        "introduction",
        "methodology",
        "related work",
        "conclusion",
        "references",
    ]
    found = sum(1 for m in markers if m in lower)
    return found >= 3


def _extract_sections(text: str) -> dict:
    """
    Use regex to carve out abstract, introduction and conclusion blocks
    from a research paper's raw text.

    Returns dict with keys abstract, introduction, conclusion (each str | None).
    """
    sections: dict[str, str | None] = {
        "abstract": None,
        "introduction": None,
        "conclusion": None,
    }

    patterns = {
        "abstract": (
            r"(?i)\n\s*(?:abstract)[:\s]*\n"
            r"(.*?)"
            r"(?=\n\s*(?:introduction|1[\.\s]|I[\.\s])\s*\n|\Z)"
        ),
        "introduction": (
            r"(?i)\n\s*(?:introduction|1[\.\s]+introduction)[:\s]*\n"
            r"(.*?)"
            r"(?=\n\s*(?:related\s+work|methodology|method|background|"
            r"2[\.\s]|II[\.\s])\s*\n|\Z)"
        ),
        "conclusion": (
            r"(?i)\n\s*(?:conclusions?)[:\s]*\n"
            r"(.*?)"
            r"(?=\n\s*(?:references|bibliography|acknowledgment)\s*\n|\Z)"
        ),
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        if match:
            sections[key] = match.group(1).strip()

    return sections
