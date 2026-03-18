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
    # Remove standalone single characters (PDF extraction artifacts)
    text = re.sub(r"\s[^\w\s]\s", " ", text)
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
        i += max_words - overlap
    return chunks


def prepare_input(text: str, model_key: str) -> str:
    """Prepare input text with appropriate prefix for T5 models."""
    text = clean_text(text)
    if "t5" in model_key or "mt5" in model_key:
        if not text.lower().startswith("summarize:"):
            text = "summarize: " + text
    return text
