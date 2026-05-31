"""
Vision-Augmented LLM Pipeline for Long-Form Document Summarization — Master Input Router
════════════════════════════════════
Single entry point for ALL Streamlit pages.

    result = route_input(input_data)
    text   = get_summary_target(result)

Detects input type automatically and delegates to the correct handler.
Always returns a unified dict with a guaranteed set of keys.
"""

import re

from input_handlers.text_handler import extract_text
from input_handlers.pdf_handler import (
    extract_pdf,
    get_summary_target as _pdf_summary_target,
)
from input_handlers.url_handler import scrape_url, validate_url, is_news_article
from input_handlers.youtube_handler import extract_youtube, is_valid_youtube_url
from input_handlers.audio_handler import transcribe_audio
from input_handlers.arxiv_handler import fetch_arxiv
from input_handlers.blueprint_handler import extract_blueprint
from input_handlers.batch_handler import extract_batch
from input_handlers.lang_detector import detect_language


# ── File extension sets ────────────────────────────────────────────────────
_AUDIO_EXTS = {"mp3", "wav", "m4a", "ogg", "flac"}
_BLUEPRINT_EXTS = {"jpg", "jpeg", "png", "webp", "bmp", "tiff", "dxf", "dwg"}
_PDF_EXT = "pdf"
_CSV_EXT = "csv"


# ═══════════════════════════════════════════════════════════════════════════
#  route_input  —  the ONE function Streamlit pages call
# ═══════════════════════════════════════════════════════════════════════════
def route_input(
    input_data,
    input_type: str = "auto",
    *,
    blueprint_mode: bool = False,
) -> dict:
    """
    Route *input_data* to the correct handler and return a unified dict.

    Parameters
    ----------
    input_data
        ``str`` (raw text, URL, arXiv ID) **or** Streamlit UploadedFile.
    input_type : str
        Force a specific handler.  ``"auto"`` (default) detects type
        from the input itself.
    blueprint_mode : bool
        If True, force PDF/image files through the blueprint handler
        instead of the regular PDF handler.

    Returns
    -------
    dict
        Always contains:
            type, source, summary_target, chunks, word_count,
            language, recommended_model, metadata
    """
    # ── Resolve type ────────────────────────────────────────
    if input_type == "auto":
        input_type = _detect_type(input_data, blueprint_mode=blueprint_mode)

    # ── Dispatch ────────────────────────────────────────────
    handler_result: dict = {}
    summary_target: str = ""

    if input_type == "text":
        handler_result = extract_text(input_data)
        summary_target = (
            handler_result["chunks"][0]
            if handler_result.get("length_category") == "long"
            else handler_result["text"]
        )

    elif input_type == "pdf":
        handler_result = extract_pdf(input_data)
        summary_target = _pdf_summary_target(handler_result)

    elif input_type == "url":
        handler_result = scrape_url(input_data)
        summary_target = (
            handler_result["chunks"][0]
            if handler_result["word_count"] > 800
            else handler_result["text"]
        )

    elif input_type == "youtube":
        handler_result = extract_youtube(input_data)
        summary_target = (
            handler_result["chunks"][0]
            if handler_result["word_count"] > 800
            else handler_result["transcript"]
        )

    elif input_type == "audio":
        handler_result = transcribe_audio(input_data)
        summary_target = (
            handler_result["chunks"][0]
            if handler_result["word_count"] > 800
            else handler_result["transcript"]
        )

    elif input_type == "arxiv":
        handler_result = fetch_arxiv(input_data)
        summary_target = handler_result["summary_target"]

    elif input_type == "blueprint":
        handler_result = extract_blueprint(input_data)
        summary_target = handler_result["summary_target"]

    elif input_type == "batch":
        handler_result = extract_batch(input_data)
        # For batches the "summary_target" is not a single text
        summary_target = ""

    else:
        raise ValueError(f"Unknown input type: {input_type}")

    # ── Language detection ──────────────────────────────────
    lang_info = detect_language(summary_target) if summary_target else {
        "code": "unknown",
        "name": "Unknown",
        "recommended_model": "csebuetnlp/mT5_multilingual_XLSum",
        "is_supported": False,
        "flag": "🌐",
    }

    # ── Recommended model ───────────────────────────────────
    recommended_model = lang_info["recommended_model"]

    # Override to BART-CNN for news articles
    if input_type == "url" and is_news_article(
        handler_result.get("source", ""), handler_result.get("text", "")
    ):
        recommended_model = "facebook/bart-large-cnn"

    # ── Build unified output ────────────────────────────────
    return {
        "type": input_type,
        "source": handler_result.get("source", "unknown"),
        "summary_target": summary_target,
        "chunks": handler_result.get("chunks", [summary_target] if summary_target else []),
        "word_count": handler_result.get("word_count", len(summary_target.split())),
        "language": lang_info,
        "recommended_model": recommended_model,
        "metadata": _build_metadata(input_type, handler_result),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  get_summary_target  —  convenience helper
# ═══════════════════════════════════════════════════════════════════════════
def get_summary_target(routed: dict) -> str:
    """
    Return the text to pass into the summariser.

    For long documents (> 800 words) returns the first chunk.
    """
    if routed.get("word_count", 0) > 800:
        chunks = routed.get("chunks", [])
        if chunks:
            return chunks[0]
    return routed.get("summary_target", "")


# ═══════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════════════════
def _detect_type(input_data, *, blueprint_mode: bool = False) -> str:
    """
    Auto-detect the input type from *input_data*.

    Strings are inspected for YouTube URLs, HTTP URLs, arXiv IDs, or
    plain text.  UploadedFile objects are routed by extension.
    """
    if isinstance(input_data, str):
        text = input_data.strip()

        if is_valid_youtube_url(text):
            return "youtube"

        if validate_url(text):
            return "url"

        if re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", text):
            return "arxiv"

        if "arxiv.org" in text:
            return "arxiv"

        return "text"

    # Streamlit UploadedFile (or anything with a .name attribute)
    if hasattr(input_data, "name"):
        ext = input_data.name.rsplit(".", 1)[-1].lower() if "." in input_data.name else ""

        if ext in _AUDIO_EXTS:
            return "audio"

        if ext == _CSV_EXT:
            return "batch"

        if ext == _PDF_EXT:
            return "blueprint" if blueprint_mode else "pdf"

        if ext in _BLUEPRINT_EXTS:
            return "blueprint"

    return "text"


def _build_metadata(input_type: str, handler_result: dict) -> dict:
    """
    Extract handler-specific extra info into a metadata sub-dict.
    This keeps the top-level router result clean and uniform.
    """
    meta: dict = {}

    if input_type == "pdf":
        meta["is_research_paper"] = handler_result.get("is_research_paper", False)
        meta["pages"] = handler_result.get("pages")
        meta["sections"] = handler_result.get("sections")

    elif input_type == "url":
        meta["domain"] = handler_result.get("domain")
        meta["title"] = handler_result.get("title")
        meta["scrape_method"] = handler_result.get("scrape_method")

    elif input_type == "youtube":
        meta["video_id"] = handler_result.get("video_id")
        meta["title"] = handler_result.get("title")
        meta["duration_seconds"] = handler_result.get("duration_seconds")

    elif input_type == "audio":
        meta["duration_seconds"] = handler_result.get("duration_seconds")
        meta["language_detected"] = handler_result.get("language_detected")

    elif input_type == "arxiv":
        meta["arxiv_id"] = handler_result.get("arxiv_id")
        meta["title"] = handler_result.get("title")
        meta["authors"] = handler_result.get("authors")
        meta["published"] = handler_result.get("published")
        meta["abstract"] = handler_result.get("abstract")

    elif input_type == "blueprint":
        meta["format"] = handler_result.get("format")
        meta["pages_detected"] = handler_result.get("pages_detected")
        meta["room_count"] = handler_result.get("room_count")
        meta["total_area_sqft"] = handler_result.get("total_area_sqft")
        meta["extracted_data"] = handler_result.get("extracted_data")

    elif input_type == "batch":
        meta["row_count"] = handler_result.get("row_count")
        meta["text_column"] = handler_result.get("text_column")
        meta["reference_column"] = handler_result.get("reference_column")
        meta["rows"] = handler_result.get("rows")

    return meta
