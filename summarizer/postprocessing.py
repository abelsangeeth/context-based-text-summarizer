"""Post-processing: deduplication, cleanup of generated summaries."""

import re


def remove_duplicate_sentences(text: str) -> str:
    """Remove duplicate sentences from summary output."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    seen = set()
    unique = []
    for s in sentences:
        normalized = s.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(s.strip())
    return " ".join(unique)


def clean_summary(text: str) -> str:
    """Clean up model output artifacts."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.!?,;:])", r"\1", text)
    # Ensure first letter is capitalized
    if text:
        text = text[0].upper() + text[1:]
    return text


def postprocess_summary(text: str) -> str:
    """Full postprocessing pipeline for a generated summary."""
    text = remove_duplicate_sentences(text)
    text = clean_summary(text)
    return text
