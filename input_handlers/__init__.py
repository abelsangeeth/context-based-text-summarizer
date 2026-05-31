"""
Vision-Augmented LLM Pipeline for Long-Form Document Summarization — Input Handlers
═══════════════════════════════════════════════════════════════════════════════════════
Unified input ingestion module.

Usage (import directly from submodules to avoid circular imports):
    from input_handlers.router import route_input, get_summary_target
    from input_handlers.pdf_handler import extract_pdf
    from input_handlers.cleaner import clean_text
"""

# ── NOTE ──────────────────────────────────────────────────────────────────────
# This __init__.py is intentionally kept empty of imports.
# Each page imports directly from the relevant submodule to prevent circular
# import deadlocks caused by pdf_handler → layout_classifier → __init__ → router
# → pdf_handler (which is still being initialized).
# ──────────────────────────────────────────────────────────────────────────────
