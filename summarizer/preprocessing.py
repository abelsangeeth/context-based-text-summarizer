"""Text preprocessing: cleaning, tokenization, chunking."""

import re


def clean_text(text: str) -> str:
    """Remove extra whitespace, special chars, normalize text for summarization."""
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\[\d+\]", "", text)  # Remove citation brackets like [1]
    # Remove common PDF artifacts
    text = re.sub(r"[●•▪◦►]", "", text)  # bullet characters
    text = re.sub(r"\b\d+\s*\|\s*P a g e\b", "", text, flags=re.IGNORECASE)  # page numbers
    text = re.sub(r"\.{3,}", ".", text)  # excessive dots
    # Attach spaced punctuation back to words instead of deleting them
    text = re.sub(r"\s+([.,;:?!])", r"\1", text)
    text = re.sub(r"([.,;:?!])\1+", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def chunk_text(text: str, max_words: int = 800, overlap: int = 100) -> list[str]:
    """
    Split long text into overlapping chunks for models with limited context.
    Uses a sliding window with overlap for context continuity.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]

    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)
        if i + max_words >= len(words):
            break  # this chunk reached the end; another would only repeat its tail
        i += max_words - overlap
    return chunks


def prepare_input(text: str, model_key: str) -> str:
    """Prepare input text with the task prefix the original T5 checkpoints expect.

    mT5 (csebuetnlp/mT5_multilingual_XLSum) was fine-tuned on raw articles
    with no prefix, so an English "summarize:" would only add noise there.
    """
    text = clean_text(text)
    if model_key.startswith("t5"):
        if not text.lower().startswith("summarize:"):
            text = "summarize: " + text
    return text
