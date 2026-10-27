import json

import pytest
from fastapi.testclient import TestClient

import app_history
import api_server

ARTICLE = ("The city council approved a new budget on Monday. It raises spending on "
           "buses and trams and cuts funding for road expansion. Critics say the plan "
           "moves too slowly, while supporters call it overdue.")


@pytest.fixture
def history_file(tmp_path, monkeypatch):
    path = tmp_path / "summary_history.json"
    monkeypatch.setattr(app_history, "HISTORY_FILE", str(path))
    return path


@pytest.fixture
def client(monkeypatch):
    """API client with the models stubbed out, recording how metrics were requested."""
    calls = []

    def fake_evaluate_all(preds, refs, lang="en", against_source=False):
        calls.append({"refs": refs, "against_source": against_source})
        return {"ROUGE-1": 0.5, "ROUGE-2": 0.3, "ROUGE-L": 0.4, "BLEU": 0.2,
                "BERTScore_F1": 0.8, "METEOR": 0.3}

    monkeypatch.setattr(api_server, "summarize", lambda text, model_key, **kw: "Council approved budget.")
    monkeypatch.setattr(api_server, "summarize_long", lambda text, model_key: "Council approved budget.")
    monkeypatch.setattr(api_server, "evaluate_all", fake_evaluate_all)
    c = TestClient(api_server.app)
    c.metric_calls = calls
    return c


# ── History ─────────────────────────────────────────────────────────────────

def test_history_is_newest_first_and_capped(history_file):
    for i in range(app_history.MAX_ENTRIES + 5):
        app_history.save_history(f"source {i}", f"summary {i}", "bart-cnn")
    history = app_history.get_history()
    assert len(history) == app_history.MAX_ENTRIES
    assert history[0]["summary"] == f"summary {app_history.MAX_ENTRIES + 4}"


def test_history_trims_long_sources_but_keeps_full_summary(history_file):
    long_summary = "A full summary paragraph. " * 40
    app_history.save_history("x" * 50_000, long_summary, "bart-cnn")
    entry = app_history.get_history()[0]
    assert len(entry["original_text"]) <= app_history.MAX_SOURCE_CHARS + 2
    assert entry["summary"] == long_summary


def test_history_survives_a_corrupt_file(history_file):
    history_file.write_text("{not json", encoding="utf-8")
    assert app_history.get_history() == []


def test_api_history_returns_the_newest_entries_oldest_first(client, history_file):
    # File order is newest-first: entry 0 is the most recent.
    entries = [{"summary": f"s{i}", "model": "bart-cnn", "timestamp": str(i)} for i in range(60)]
    history_file.write_text(json.dumps(entries), encoding="utf-8")

    data = client.get("/api/history?limit=50").json()
    summaries = [h["summary"] for h in data["history"]]
    assert data["total"] == 60
    assert summaries == [f"s{i}" for i in range(49, -1, -1)]  # s49 … s0 (s0 newest, last)


# ── Metric mode routing ─────────────────────────────────────────────────────

def test_summarize_without_reference_scores_against_source(client):
    body = client.post("/api/summarize", json={"text": ARTICLE}).json()
    assert body["metrics"]["scored_against"] == "source"
    assert client.metric_calls[-1]["against_source"] is True


def test_summarize_with_reference_uses_standard_metrics(client):
    body = client.post("/api/summarize", json={"text": ARTICLE, "reference": "Budget passed."}).json()
    assert body["metrics"]["scored_against"] == "reference"
    assert client.metric_calls[-1] == {"refs": ["Budget passed."], "against_source": False}


def test_compare_and_metrics_endpoints_respect_reference(client):
    body = client.post("/api/compare", json={"text": ARTICLE, "models": ["bart-cnn", "t5-base"],
                                             "reference": "Budget passed."}).json()
    assert {r["metrics"]["scored_against"] for r in body["results"].values()} == {"reference"}

    body = client.post("/api/metrics", json={"text": ARTICLE, "models": ["bart-cnn", "t5-base"]}).json()
    assert body["scored_against"] == "source"
    assert all(c["against_source"] for c in client.metric_calls[-2:])


def test_batch_scores_each_row_by_its_own_reference(client):
    csv = "text,reference\n" + f'"{ARTICLE}",Budget passed.\n' + f'"{ARTICLE}",\n'
    body = client.post("/api/batch", files={"file": ("b.csv", csv, "text/csv")},
                       data={"model": "bart-cnn", "max_rows": "2"}).json()
    assert [r["scored_against"] for r in body["results"]] == ["reference", "source"]


def test_short_text_is_rejected(client):
    assert client.post("/api/summarize", json={"text": "too short"}).status_code == 400


def test_multilingual_accepts_chinese(client, monkeypatch):
    monkeypatch.setattr(api_server, "_translate_to_english", lambda text, lang: "translated")
    monkeypatch.setattr("evaluation.bert_scorer.compute_bertscore", lambda *a, **k: {})
    chinese = ("人工智能正在迅速改变全球各个行业。在医疗领域，人工智能诊断工具能够更早地发现疾病，"
               "显著改善患者的治疗效果。")
    body = client.post("/api/multilingual", json={"text": chinese}).json()
    assert body["lang_code"] == "zh"
    assert body["model_used"] == "mt5"
    assert body["translated_summary"] == "translated"
