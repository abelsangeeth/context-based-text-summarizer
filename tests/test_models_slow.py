"""End-to-end checks with real HuggingFace models. Run with: pytest -m slow"""
import pytest

pytestmark = pytest.mark.slow

ARTICLE = (
    "Artificial intelligence is rapidly transforming industries across the globe. In "
    "healthcare, AI-powered diagnostic tools are enabling earlier detection of diseases such "
    "as cancer and diabetes, improving patient outcomes significantly. In the financial sector, "
    "machine learning algorithms are being used for fraud detection, risk assessment, and "
    "automated trading strategies. The manufacturing industry is leveraging AI for predictive "
    "maintenance, quality control, and supply chain optimization. Despite these advances, "
    "concerns about job displacement, data privacy, and algorithmic bias remain significant "
    "challenges that need to be addressed through thoughtful regulation."
)


@pytest.mark.parametrize("model_key", ["bart-cnn", "t5-base"])
def test_summarize_produces_a_shorter_clean_summary(model_key):
    from summarizer.pipeline import summarize
    summary = summarize(ARTICLE, model_key)
    assert 5 < len(summary.split()) < len(ARTICLE.split())
    assert summary[0].isupper() and summary[-1] in ".!?"
    assert "summarize:" not in summary.lower()


def test_evaluate_all_in_both_modes():
    from evaluation.aggregator import evaluate_all, composite_score
    summary = "AI is transforming healthcare, finance and manufacturing, but raises privacy and bias concerns."
    reference = "AI is changing many industries, though it brings concerns about jobs, privacy and bias."
    by_ref = evaluate_all([summary], [reference])
    by_src = evaluate_all([summary], [ARTICLE], against_source=True)
    for scores in (by_ref, by_src):
        assert set(scores) >= {"ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU", "METEOR", "BERTScore_F1"}
        assert all(0.0 <= v <= 1.0 for v in scores.values())
        assert 0.0 < composite_score(scores) < 1.0
    # Against the long source, precision-style ROUGE stays high for a faithful summary
    assert by_src["ROUGE-1"] > by_ref["ROUGE-1"]


def test_longform_summary_has_one_paragraph_per_section():
    from summarizer.section_detector import detect_sections, merge_short_sections
    from summarizer.longform_builder import build_longform_summary
    text = "\n\n".join([ARTICLE] * 14)   # ~1200 words: more than one 500-word section
    sections = merge_short_sections(detect_sections(text), min_words=80)
    result = build_longform_summary(sections, model_name="bart-cnn", length_mode="brief")
    assert len(result["sections"]) == len(sections) > 1
    assert all(sec["summary_paragraph"] for sec in result["sections"])
