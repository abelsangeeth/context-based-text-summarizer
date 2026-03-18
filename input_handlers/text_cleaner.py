"""Deep text cleaning utilities."""

import re


def deep_clean(text: str) -> str:
    """Thorough text cleaning for summarization input."""
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", "", text)
    # Remove citation brackets
    text = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text
