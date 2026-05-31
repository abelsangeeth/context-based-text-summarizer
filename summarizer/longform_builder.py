import math
import re
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from summarizer.pipeline import summarize
from summarizer.postprocessing import postprocess_summary as _postprocess


def estimate_reading_time(word_count: int) -> str:
    minutes = math.ceil(word_count / 200)
    return f"{minutes} min read"


# Fix 2.1 — Extractive pre-filter before abstractive summarization
def extractive_prefilter(text: str, keep_ratio: float = 0.5) -> str:
    """
    Keep only the top sentences by TF-IDF centrality before sending
    to BART/PEGASUS. Reduces input noise and improves summary quality.
    """
    try:
        sentences = nltk.sent_tokenize(text)
    except LookupError:
        nltk.download("punkt")
        sentences = nltk.sent_tokenize(text)

    if len(sentences) <= 3:
        return text  # Too short to filter

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences + [text])
    except ValueError:
        return text  # Fallback on empty vocabulary

    doc_vector = tfidf_matrix[-1]
    sentence_vectors = tfidf_matrix[:-1]
    scores = cosine_similarity(sentence_vectors, doc_vector).flatten()

    # Keep top sentences, preserving original order
    n_keep = max(3, int(len(sentences) * keep_ratio))
    top_indices = sorted(np.argsort(scores)[-n_keep:])
    return " ".join(sentences[i] for i in top_indices)


def _summarize_section(
    text: str,
    model_name: str,
    max_length: int,
    min_length: int,
    length_mode: str = "standard",
) -> str:
    """Summarize a single text chunk. Handles the model's token limits."""
    words = text.split()

    # Fix 2.1 — Pre-filter long sections with extractive step
    if len(words) > 100:
        text = extractive_prefilter(text, keep_ratio=0.65)
        words = text.split()

    # Model input cap — BART/PEGASUS have ~1024 token limit
    if len(words) > 700:
        text = " ".join(words[:700])
        words = text.split()

    input_tokens = int(len(words) * 1.3)
    safe_max = min(max_length, max(50, int(input_tokens * 0.40)))

    # Fix 2.2 — Per-mode min_length cap (less aggressive than flat 60)
    mode_min_caps = {"detailed": 120, "standard": 80, "brief": 40}
    raw_safe_min = min(min_length, max(10, int(input_tokens * 0.10)))
    safe_min = min(raw_safe_min, mode_min_caps.get(length_mode, 80))

    # Ensure safe_min is strictly less than safe_max
    if safe_min >= safe_max:
        safe_min = max(10, safe_max - 5)

    # Fix 3.4 — Mode-aware length_penalty
    mode_length_penalty = {"detailed": 2.0, "standard": 1.5, "brief": 0.8}
    lp = mode_length_penalty.get(length_mode, 1.5)

    return summarize(
        text,
        model_key=model_name,
        max_length=safe_max,
        min_length=safe_min,
        length_penalty=lp,
    )


from summarizer.pipeline import _token_safe_chunk_size


def _split_oversized_section(section: dict, model_name: str) -> list:
    """
    If a section has more words than max_section_words, split it into
    sub-chunks at sentence boundaries so each chunk gets its own summary.
    """
    raw = section["raw_text"]
    words = raw.split()
    wc = len(words)
    max_section_words = _token_safe_chunk_size(model_name)

    if wc <= max_section_words:
        return [section]

    sentences = re.split(r'(?<=[.!?])\s+', raw)
    chunks = []
    current_words = []
    current_sents = []

    for sent in sentences:
        sent_words = sent.split()
        if len(current_words) + len(sent_words) > max_section_words and current_sents:
            chunk_text = " ".join(current_sents)
            chunks.append({
                "index": section["index"],
                "raw_text": chunk_text,
                "word_count": len(current_words),
                "section_type": section.get("section_type", "body"),
            })
            current_words = []
            current_sents = []
        current_words.extend(sent_words)
        current_sents.append(sent)

    # Flush remaining
    if current_sents:
        chunk_text = " ".join(current_sents)
        chunks.append({
            "index": section["index"],
            "raw_text": chunk_text,
            "word_count": len(current_words),
            "section_type": section.get("section_type", "body"),
        })

    return chunks if chunks else [section]


# Fix 2.3 — Hierarchical summarization for multi-chunk sections
def _summarize_chunked_section(
    chunks: list,
    model_name: str,
    max_length: int,
    min_length: int,
    length_mode: str = "standard",
) -> str:
    """
    Summarize each chunk briefly, then do a final pass over the combined
    chunk summaries. Prevents context loss / repetition from naive join.
    """
    chunk_summaries = []
    brief_max = max(50, max_length // max(len(chunks), 1))
    brief_min = max(10, min_length // max(len(chunks), 1))

    for chunk in chunks:
        if chunk["word_count"] < 50:
            chunk_summaries.append(chunk["raw_text"])
        else:
            s = _summarize_section(
                chunk["raw_text"], model_name, brief_max, brief_min, "brief"
            )
            chunk_summaries.append(s)

    combined = " ".join(chunk_summaries)

    # Final pass — re-summarize the combined chunk summaries
    if len(combined.split()) > 100:
        return _summarize_section(combined, model_name, max_length, min_length, length_mode)
    return combined


def build_longform_summary(
    sections: list,
    subheadings: list,
    model_name: str = "bart-cnn",
    length_mode: str = "detailed",
) -> dict:

    # Fix 2.2 — Raised min_length values per mode
    # max_length / min_length represent TOKENS, not words (tokens ≈ 1.3 × words)
    bounds = {
        "detailed": {"max_length": 450, "min_length": 250},   # was 200
        "standard": {"max_length": 300, "min_length": 150},   # was 120
        "brief":    {"max_length": 150, "min_length": 60},    # was 50
    }
    mode_cfg = bounds.get(length_mode, bounds["standard"])

    results = []
    total_words = 0

    for sec, heading in zip(sections, subheadings):
        # Split oversized sections into chunks
        sub_chunks = _split_oversized_section(sec, model_name)

        if len(sub_chunks) == 1:
            chunk = sub_chunks[0]
            if chunk["word_count"] < 15:
                summary_paragraph = chunk["raw_text"]
            else:
                summary_paragraph = _summarize_section(
                    chunk["raw_text"],
                    model_name=model_name,
                    max_length=mode_cfg["max_length"],
                    min_length=mode_cfg["min_length"],
                    length_mode=length_mode,
                )
        else:
            # Fix 2.3 — Use hierarchical summarization (not naive join)
            summary_paragraph = _summarize_chunked_section(
                sub_chunks,
                model_name=model_name,
                max_length=mode_cfg["max_length"],
                min_length=mode_cfg["min_length"],
                length_mode=length_mode,
            )

        # Fix 4 — Post-processing pass on every summary paragraph
        summary_paragraph = _postprocess(summary_paragraph)

        p_words = len(summary_paragraph.split())
        total_words += p_words
        results.append({
            "subheading": heading,
            "summary_paragraph": summary_paragraph,
            "word_count": p_words,
            "section_type": sec.get("section_type", "body"),
        })

    return {
        "sections": results,
        "total_word_count": total_words,
        "model_used": model_name,
        "length_mode": length_mode,
    }
