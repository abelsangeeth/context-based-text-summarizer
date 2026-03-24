"""
ContextFree-X — Batch CSV Handler
══════════════════════════════════
Reads uploaded CSV files containing multiple texts to summarise.
Auto-detects text and reference (ground-truth summary) columns.
"""

import pandas as pd

from input_handlers.cleaner import clean_text


# ── Column name heuristics ─────────────────────────────────────────────────
_TEXT_COLUMN_NAMES = [
    "text", "article", "content", "body", "document", "input",
    "passage", "paragraph", "description",
]
_REFERENCE_COLUMN_NAMES = [
    "summary", "reference", "highlight", "abstract", "target",
    "gold", "expected", "ground_truth",
]


# ═══════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════
def extract_batch(uploaded_csv) -> dict:
    """
    Parse an uploaded CSV file for batch summarisation.

    Parameters
    ----------
    uploaded_csv : streamlit.UploadedFile
        A .csv file with at least one text column.

    Returns
    -------
    dict
        type, source, row_count, text_column, reference_column, rows
    """
    filename = getattr(uploaded_csv, "name", "batch.csv")

    if not filename.lower().endswith(".csv"):
        raise ValueError("Please upload a valid CSV file.")

    try:
        df = pd.read_csv(uploaded_csv)
    except Exception as exc:
        raise ValueError(f"Could not read CSV: {exc}") from exc

    if df.empty:
        raise ValueError("CSV has no data rows.")

    # Detect columns
    text_col = _detect_text_column(df)
    ref_col = _detect_reference_column(df)

    # Clean each text value
    rows: list[dict] = []
    for idx, row in df.iterrows():
        raw_text = str(row[text_col]) if pd.notna(row[text_col]) else ""
        cleaned = clean_text(raw_text)

        if not cleaned:
            continue  # skip empty rows after cleaning

        reference = None
        if ref_col and pd.notna(row.get(ref_col)):
            reference = clean_text(str(row[ref_col]))

        rows.append({
            "id": int(idx),
            "text": cleaned,
            "reference": reference,
        })

    return {
        "type": "batch",
        "source": filename,
        "row_count": len(rows),
        "text_column": text_col,
        "reference_column": ref_col,
        "rows": rows,
    }


def validate_batch(extracted: dict) -> tuple[bool, str]:
    """
    Validate a batch extraction result.

    Returns (is_valid, reason_string).
    """
    row_count = extracted.get("row_count", 0)

    if row_count == 0:
        return False, "CSV has no data rows."

    if row_count > 5000:
        return False, "CSV exceeds 5000 row limit."

    # Check average text length
    rows = extracted.get("rows", [])
    if rows:
        avg_words = sum(len(r["text"].split()) for r in rows) / len(rows)
        if avg_words < 20:
            return False, "Texts too short to summarise."

    return True, "ok"


# ═══════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════════════════
def _detect_text_column(df: pd.DataFrame) -> str:
    """
    Find the column most likely to contain the text to summarise.

    1. Check for known column names (case-insensitive).
    2. Fall back to the column with the longest average string length.
    """
    lower_cols = {c.lower().strip(): c for c in df.columns}

    for candidate in _TEXT_COLUMN_NAMES:
        if candidate in lower_cols:
            return lower_cols[candidate]

    # Fallback: longest average string length
    str_cols = df.select_dtypes(include=["object"]).columns
    if str_cols.empty:
        raise ValueError(
            "Could not detect a text column. "
            "Ensure CSV has a column named 'text', 'article', or 'content'."
        )

    best_col = max(
        str_cols,
        key=lambda c: df[c].astype(str).str.len().mean(),
    )
    return best_col


def _detect_reference_column(df: pd.DataFrame) -> str | None:
    """
    Find the optional reference / ground-truth summary column.
    Returns None if no match found.
    """
    lower_cols = {c.lower().strip(): c for c in df.columns}

    for candidate in _REFERENCE_COLUMN_NAMES:
        if candidate in lower_cols:
            return lower_cols[candidate]

    return None
