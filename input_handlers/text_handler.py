"""
Vision-Augmented LLM Pipeline for Long-Form Document Summarization — Raw Text Handler
═════════════════════════════════
Accepts plain text (paste / textarea) and returns a unified dict
for the router pipeline.
"""

from input_handlers.cleaner import (
    clean_text,
    chunk_text,
    detect_length_category,
)


def extract_text(raw_text: str) -> dict:
    """
    Process raw text input for the summarisation pipeline.

    Parameters
    ----------
    raw_text : str
        User-pasted text from the Streamlit textarea.

    Returns
    -------
    dict with keys:
        type, source, text, chunks, word_count, length_category
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("No text provided. Please paste some text to summarise.")

    cleaned = clean_text(raw_text)

    if not cleaned:
        raise ValueError("Text was empty after cleaning. Please provide meaningful content.")

    words = cleaned.split()
    word_count = len(words)
    category = detect_length_category(cleaned)
    chunks = chunk_text(cleaned, max_words=800, overlap=100)

    return {
        "type": "text",
        "source": "direct input",
        "text": cleaned,
        "chunks": chunks,
        "word_count": word_count,
        "length_category": category,
    }


def get_summary_target(extracted: dict) -> str:
    """
    Return the text to feed into the summariser.

    For short/medium documents → full text.
    For long documents → first chunk (to stay within model limits).
    """
    if extracted["length_category"] == "long":
        return extracted["chunks"][0]
    return extracted["text"]
