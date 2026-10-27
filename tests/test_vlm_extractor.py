import importlib
import json
import pathlib
from types import SimpleNamespace

import pytest
from PIL import Image

import vision_summarizer.vlm_extractor as vlm

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_no_api_key_in_source_tree():
    for path in ROOT.rglob("*.py"):
        if ".git" in path.parts or "tests" in path.parts:
            continue
        assert "sk-ant-" not in path.read_text(encoding="utf-8", errors="ignore"), path


def test_import_does_not_set_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    importlib.reload(vlm)
    import os
    assert "ANTHROPIC_API_KEY" not in os.environ


def test_large_images_are_downscaled_for_claude():
    big = Image.new("RGB", (5100, 6600), "white")       # a 300-DPI letter page
    small = Image.new("RGB", (800, 600), "white")
    fitted = vlm._fit_for_claude(big)
    assert max(fitted.size) == vlm._CLAUDE_MAX_EDGE
    assert fitted.size[0] / fitted.size[1] == pytest.approx(5100 / 6600, rel=0.01)
    assert vlm._fit_for_claude(small) is small


class _FakeAnthropic:
    """Stands in for anthropic.Anthropic and records the request."""
    last_request = None
    response = None

    def __init__(self, **kwargs):
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        _FakeAnthropic.last_request = kwargs
        return _FakeAnthropic.response


def _response(stop_reason, *blocks):
    return SimpleNamespace(stop_reason=stop_reason, content=list(blocks))


def test_claude_extraction_reads_the_text_block_after_thinking(monkeypatch):
    import anthropic
    payload = {"floors": [{"floor_number": 1, "rooms": [{"name": "Kitchen"}]}], "confidence": 0.9}
    _FakeAnthropic.response = _response(
        "end_turn",
        SimpleNamespace(type="thinking", thinking=""),
        SimpleNamespace(type="text", text=json.dumps(payload)),
    )
    monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropic)

    assert vlm.extract_with_claude(Image.new("RGB", (4000, 3000))) == payload
    req = _FakeAnthropic.last_request
    assert req["model"] == vlm.CLAUDE_MODEL
    assert req["fallbacks"] == "default"
    assert req["messages"][0]["content"][0]["type"] == "image"


def test_claude_refusal_raises(monkeypatch):
    import anthropic
    _FakeAnthropic.response = _response("refusal")
    monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropic)
    with pytest.raises(ValueError):
        vlm.extract_with_claude(Image.new("RGB", (100, 100)))


def test_safe_parse_json_recovers_common_llm_mistakes():
    assert vlm._safe_parse_json('```json\n{"a": 1,}\n```') == {"a": 1}
    assert vlm._safe_parse_json('Sure! {"floors": [{"rooms": []}') == {"floors": [{"rooms": []}]}
