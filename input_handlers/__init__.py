"""
ContextFree-X — Input Handlers
═══════════════════════════════
Unified input ingestion module.

Usage:
    from input_handlers.router import route_input, get_summary_target

    result = route_input(user_input)       # auto-detect type
    text   = get_summary_target(result)    # text for summarisation
"""

# ── Master router (primary API) ────────────────────────────────────────
from input_handlers.router import route_input, get_summary_target

# ── Cleaner utilities (used everywhere) ────────────────────────────────
from input_handlers.cleaner import (
    clean_text,
    chunk_text,
    detect_length_category,
    truncate_for_model,
)

# ── Language detection ─────────────────────────────────────────────────
from input_handlers.lang_detector import (
    detect_language,
    needs_translation,
    SUPPORTED_LANGUAGES,
    TRANSLATION_MODELS,
)

# ── Individual handlers (for direct use if needed) ─────────────────────
from input_handlers.text_handler import extract_text
from input_handlers.pdf_handler import extract_pdf
from input_handlers.url_handler import scrape_url, validate_url, is_news_article
from input_handlers.youtube_handler import extract_youtube, is_valid_youtube_url
from input_handlers.audio_handler import transcribe_audio
from input_handlers.arxiv_handler import fetch_arxiv, search_arxiv
from input_handlers.blueprint_handler import extract_blueprint
from input_handlers.batch_handler import extract_batch, validate_batch
from input_handlers.image_handler import describe_image
