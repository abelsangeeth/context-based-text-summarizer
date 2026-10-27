import json
import os
from datetime import datetime

# Absolute path so Streamlit and the API server share one file no matter which
# directory they were launched from.
HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "summary_history.json")

MAX_ENTRIES = 100
# Long PDFs would otherwise make every entry tens of KB; the history page only
# needs enough source text to recognise the document.
MAX_SOURCE_CHARS = 5000


def get_history():
    """Return saved entries, newest first."""
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def save_history(original_text, summary, model):
    history = get_history()
    if len(original_text) > MAX_SOURCE_CHARS:
        original_text = original_text[:MAX_SOURCE_CHARS] + " …"
    entry = {
        "original_text": original_text,
        "summary": summary,
        "model": model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    history.insert(0, entry)

    history = history[:MAX_ENTRIES]

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def clear_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
