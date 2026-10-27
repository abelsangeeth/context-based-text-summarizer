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


# Fix 4 — Enhanced post-processing pass for longform summary paragraphs
def postprocess_summary(text: str) -> str:
    """Full postprocessing pipeline for a generated summary paragraph."""
    if not text:
        return text

    # Step 0: PEGASUS-CNN/DM marks sentence breaks with a literal "<n>" token
    text = text.replace("<n>", " ")

    # Step 1: Remove duplicate sentences
    text = remove_duplicate_sentences(text)

    # Step 2: Remove repeated consecutive words (common BART artifact)
    # e.g. "the the", "is is", "a a"
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text, flags=re.IGNORECASE)

    # Step 3: Capitalize first letter
    text = text[0].upper() + text[1:] if text else text

    # Step 4: Ensure ends with a sentence-ending punctuation
    if text and text[-1] not in '.!?':
        text += '.'

    # Step 5: Remove trailing ellipsis artifacts from model truncation
    text = re.sub(r'\.{3,}$', '.', text)

    # Step 6: Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text).strip()

    # Step 7: Fix spacing before punctuation
    text = re.sub(r'\s+([.!?,;:])', r'\1', text)

    return text
