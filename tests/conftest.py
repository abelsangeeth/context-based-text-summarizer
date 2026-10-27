import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(autouse=True)
def _no_real_claude_calls(monkeypatch):
    # vlm_extractor loads a local .env on import; keep its key away from tests
    # so nothing here can make a paid API call.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture
def make_pdf(tmp_path):
    """Write a PDF whose pages hold the given lines of text; returns its path."""
    import pymupdf

    def _make(lines, name="doc.pdf", per_page=40):
        doc = pymupdf.open()
        for start in range(0, len(lines), per_page):
            page = doc.new_page()
            y = 72
            for line in lines[start:start + per_page]:
                page.insert_text((72, y), line, fontsize=10, fontname="helv")
                y += 16
        path = tmp_path / name
        doc.save(str(path))
        doc.close()
        return str(path)

    return _make
