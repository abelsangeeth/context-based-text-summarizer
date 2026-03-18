"""PDF text extraction with research paper section detection."""

import re
import pdfplumber


def extract_pdf_text(uploaded_file) -> str:
    """Extract and clean text from a Streamlit UploadedFile or file path."""
    full_text = []
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text.append(text.strip())
    return "\n\n".join(full_text)


def _is_research_paper(text: str) -> bool:
    """Detect if text is likely a research paper by checking section headers."""
    lower = text.lower()
    markers = ["abstract", "introduction", "conclusion", "references", "methodology"]
    count = sum(1 for m in markers if m in lower)
    return count >= 3


def _extract_section(text: str, start_headers: list[str],
                     end_headers: list[str]) -> str:
    """Extract text between section headers using regex."""
    start_pat = "|".join(re.escape(h) for h in start_headers)
    end_pat = "|".join(re.escape(h) for h in end_headers)

    pattern = rf"(?i)(?:^|\n)\s*(?:{start_pat})\s*\n(.*?)(?=\n\s*(?:{end_pat})\s*\n|\Z)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_research_paper(uploaded_file) -> dict:
    """
    Extract structured sections from a research paper PDF.
    Returns dict with abstract, introduction, conclusion, and full text.
    Falls back to chunked text if not a research paper.
    """
    full_text = extract_pdf_text(uploaded_file)

    if not full_text.strip():
        return {"warning": "Could not extract text. This may be a scanned PDF.",
                "full": "", "chunks": []}

    if not _is_research_paper(full_text):
        return {
            "type": "document",
            "full": full_text,
            "chunks": chunk_long_text(full_text),
        }

    abstract = _extract_section(
        full_text,
        ["Abstract", "ABSTRACT"],
        ["Introduction", "INTRODUCTION", "1.", "I."]
    )
    introduction = _extract_section(
        full_text,
        ["Introduction", "INTRODUCTION", "1.", "I."],
        ["Related Work", "RELATED WORK", "Methodology", "METHODOLOGY",
         "Method", "METHOD", "2.", "II.", "Background"]
    )
    conclusion = _extract_section(
        full_text,
        ["Conclusion", "CONCLUSION", "Conclusions", "CONCLUSIONS"],
        ["References", "REFERENCES", "Bibliography", "Acknowledgment"]
    )

    return {
        "type": "research_paper",
        "abstract": abstract,
        "introduction": introduction,
        "conclusion": conclusion,
        "full": full_text,
        "chunks": chunk_long_text(full_text),
    }


def chunk_long_text(text: str, max_words: int = 800) -> list[str]:
    """Split text into overlapping chunks using a sliding window."""
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + max_words]))
        i += max_words - 100  # 100-word overlap
    return chunks
