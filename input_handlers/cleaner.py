"""
Vision-Augmented LLM Pipeline for Long-Form Document Summarization — Universal Text Cleaner & Chunker
═════════════════════════════════════════════════
Imported by EVERY handler. Provides:
  • clean_text()           — HTML stripping, unicode normalisation, control-char removal
  • chunk_text()           — overlapping word-window chunking
  • detect_length_category() — short / medium / long classification
  • truncate_for_model()   — tokenizer-aware truncation to model max input
"""

import re
import unicodedata
import os

import yaml
from bs4 import BeautifulSoup


# ── Load model config once ──────────────────────────────────────────────────
_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "model_config.yaml",
)

_MODEL_MAX_INPUT: dict[str, int] = {}
try:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        _cfg = yaml.safe_load(f)
    for key, info in _cfg.get("models", {}).items():
        _MODEL_MAX_INPUT[key] = info.get("max_input", 512)
except FileNotFoundError:
    # Sensible defaults if config file is missing
    _MODEL_MAX_INPUT = {
        "t5-base": 512,
        "t5-large": 512,
        "bart-cnn": 1024,
        "pegasus": 1024,
        "mt5": 512,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  1. clean_text
# ═══════════════════════════════════════════════════════════════════════════
def clean_text(text: str) -> str:
    """
    Deep-clean raw text for the summarisation pipeline.

    Steps:
      1. Strip HTML tags (BeautifulSoup)
      2. Normalise Unicode (NFKC)
      3. Remove null bytes & control characters
      4. Remove citation brackets like [1], [2,3]
      5. Remove bare URLs, emails, DOIs
      6. Remove web-page boilerplate phrases
      7. Remove random number sequences (PDF artifacts)
      8. Collapse multiple whitespace / newlines → single space
      9. Strip leading / trailing whitespace
    """
    if not text:
        return ""

    # 1 — Strip HTML
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")

    # 2 — Unicode normalisation
    text = unicodedata.normalize("NFKC", text)

    # 3 — Null bytes & control chars (keep \n \r \t initially)
    text = text.replace("\x00", "")
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # 4 — Citation brackets
    text = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", text)

    # 5 — URLs, emails, and DOI strings
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"www\.\S+", "", text)
    text = re.sub(r"\S+@\S+\.\S+", "", text)
    text = re.sub(r"\bdoi[:\s]*\S+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b10\.\d{4,}/\S+", "", text)  # bare DOI like 10.1038/...

    # 6 — Web-page boilerplate / navigation cruft
    _BOILERPLATE = [
        r"Back to Mail Online home\.?",
        r"Back to the page you came from\.?",
        r"Click here to (?:go to the article|visit)[^.]*\.?",
        r"originally published in the online edition[^.]*\.?",
        r"For confidential support[^.]*\.?",
        r"call the (?:Samaritans|National)[^.]*\.?",
        r"In the U\.?S\.?[,.]?\s*call[^.]*\.?",
        r"In the U\.?K\.?[,.]?\s*(?:call|contact)[^.]*\.?",
        r"visit a local[^.]*branch[^.]*\.?",
        r"OPEN\s+Deep learning for text summarizing[^.]*\.?",
        r"peer-reviewed,? open-access,? scientific journal[^.]*\.?",
        r"published by the Association for Computing[^.]*\.?",
    ]
    for pattern in _BOILERPLATE:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # 7 — Journal codes and number-sequence artifacts
    # Strip journal volume/issue/page codes like "15:36343" or "(2025) 15:36343"
    text = re.sub(r"\(\d{4}\)\s*\d+:\d+", "", text)          # (2025) 15:36343
    text = re.sub(r"\b\d{1,3}:\d{4,}\b", "", text)            # 15:36343 standalone
    # Only strip LONG runs (8+) of small identical/sequential numbers — PDF index artifacts.
    # Do NOT strip 3-7 numbers; those appear legitimately in statistics, years, and tables.
    text = re.sub(r"(?:\b\d{1,3}\b\s+){8,}", "", text)

    # 8 — Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    # 9 — Clean up orphaned punctuation left after stripping
    text = re.sub(r"\s+([.,;:?!])", r"\1", text)       # " ." → "."
    text = re.sub(r"([.,;:?!])\1+", r"\1", text)       # remove repeated punctuation like ".."
    text = re.sub(r"\b(\d+)\.\s*$", "", text)          # trailing standalone "18."
    text = re.sub(r"\s+", " ", text)

    # 10 — Strip
    text = text.strip()

    return text


# ═══════════════════════════════════════════════════════════════════════════
#  2. chunk_text  — overlapping word-window chunking
# ═══════════════════════════════════════════════════════════════════════════
def chunk_text(
    text: str,
    max_words: int = 800,
    overlap: int = 100,
) -> list[str]:
    """
    Split *text* into overlapping chunks of at most *max_words* words.

    Overlap: the last ``overlap`` words of chunk N become the first words
    of chunk N+1, ensuring context continuity.

    Returns ``[text]`` unchanged if the text is shorter than *max_words*.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]

    chunks: list[str] = []
    start = 0
    step = max_words - overlap  # how far we advance each iteration

    while start < len(words):
        end = start + max_words
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break  # reached the end; a further chunk would only repeat this one's tail
        start += step

    return chunks


# ═══════════════════════════════════════════════════════════════════════════
#  3. detect_length_category
# ═══════════════════════════════════════════════════════════════════════════
def detect_length_category(text: str) -> str:
    """
    Classify text length:
      • "short"  — < 200 words
      • "medium" — 200–800 words
      • "long"   — > 800 words
    """
    wc = len(text.split())
    if wc < 200:
        return "short"
    elif wc <= 800:
        return "medium"
    else:
        return "long"


# ═══════════════════════════════════════════════════════════════════════════
#  4. truncate_for_model — tokenizer-aware truncation
# ═══════════════════════════════════════════════════════════════════════════
def truncate_for_model(
    text: str,
    model_name: str,
    tokenizer,
) -> str:
    """
    Tokenise *text* with the given HuggingFace *tokenizer*, truncate to
    the model's ``max_input`` length (from model_config.yaml), and decode
    back to a clean string.

    Supported model keys: t5-base, t5-large, bart-cnn, pegasus, mt5.
    Falls back to 512 tokens if the key is unknown.
    """
    max_len = _MODEL_MAX_INPUT.get(model_name, 512)

    token_ids = tokenizer.encode(
        text,
        max_length=max_len,
        truncation=True,
        add_special_tokens=False,
    )

    truncated = tokenizer.decode(token_ids, skip_special_tokens=True)
    return truncated
