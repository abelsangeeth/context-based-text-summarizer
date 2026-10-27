import pytest

from input_handlers import pdf_handler
from input_handlers.lang_detector import count_words, detect_language
from input_handlers.arxiv_handler import _parse_input
from input_handlers.youtube_handler import _extract_video_id

CHINESE = ("人工智能正在迅速改变全球各个行业。在医疗领域，人工智能诊断工具能够更早地发现疾病，"
           "显著改善患者的治疗效果。在金融领域，机器学习算法被用于欺诈检测和风险评估。")
ENGLISH = ("Artificial intelligence is rapidly transforming industries across the globe. "
           "In healthcare, diagnostic tools are enabling earlier detection of diseases.")


# ── Language detection ──────────────────────────────────────────────────────

def test_count_words_handles_unspaced_scripts():
    assert count_words("这是一个测试") == 6
    assert count_words("hello world") == 2
    assert count_words("BART 模型") == 3
    assert count_words("") == 0
    assert len(CHINESE.split()) == 1          # what the old check saw
    assert count_words(CHINESE) > 60


def test_chinese_paragraph_is_detected_as_chinese():
    # Before: one whitespace "word" < 20 → "unknown" → routed to the English model.
    assert detect_language(CHINESE)["code"] == "zh"


def test_english_detection_and_short_text_guard():
    assert detect_language(ENGLISH * 2)["code"] == "en"
    assert detect_language("too short to tell")["code"] == "unknown"


# ── PDF extraction ──────────────────────────────────────────────────────────

PAPER = (
    ["COMPASS: Comparing Abstractive Summarizers", "Abstract",
     "We compare five transformer summarizers on news text.",
     "BART gives the best ROUGE while PEGASUS is the most abstractive.",
     "1 Introduction",
     "Readers face information overload and need short summaries.",
     "2 Method"]
    + ["Each model summarizes the same articles and we score them."] * 30
    + ["5 Conclusion", "BART-large-CNN offers the best balance overall.",
       "References", "[1] Lewis et al. BART. ACL 2020."]
)


def test_research_paper_sections_are_extracted(make_pdf):
    result = pdf_handler.extract_pdf(make_pdf(PAPER))
    sections = result["sections"]
    assert result["is_research_paper"]
    assert result["pages"] == 2
    assert sections["abstract"].startswith("We compare five transformer summarizers")
    assert "PEGASUS is the most abstractive." in sections["abstract"]
    assert "Introduction" not in sections["abstract"]
    assert sections["introduction"] == "Readers face information overload and need short summaries."
    assert sections["conclusion"] == "BART-large-CNN offers the best balance overall."
    # Lines inside a page are joined into running text
    assert "summaries.\n2 Method" not in sections["full_text"]
    assert "summaries. 2 Method" in sections["full_text"]


def test_pages_are_not_rendered_without_a_layout_model(make_pdf, monkeypatch):
    monkeypatch.setattr("input_handlers.layout_classifier._LP_MODEL", "fallback")

    def boom(*a, **k):
        raise AssertionError("render_pages must not run on the text-layer path")

    monkeypatch.setattr(pdf_handler, "render_pages", boom)
    result = pdf_handler.extract_pdf(make_pdf(PAPER))
    assert result["pages"] == 2 and result["word_count"] > 300


def test_plain_document_mentioning_abstractive_is_not_a_paper(make_pdf):
    lines = ["Notes on abstractive summarization",
             "Abstractive models paraphrase instead of copying sentences."] * 5
    result = pdf_handler.extract_pdf(make_pdf(lines))
    assert not result["is_research_paper"]
    assert list(result["sections"]) == ["full_text"]


def test_ieee_style_inline_abstract():
    raw = ("Title\nAbstract—This paper proposes a summarizer.\nIt is fast.\n"
           "I. INTRODUCTION\nSummaries help readers.\nII. RELATED WORK\nPrior work.\n")
    sections = pdf_handler._extract_sections(raw)
    assert sections["abstract"] == "This paper proposes a summarizer. It is fast."
    assert sections["introduction"] == "Summaries help readers."


def test_body_line_starting_with_2_does_not_end_the_introduction():
    raw = ("Abstract\nShort abstract.\n1 Introduction\nWe study models.\n"
           "2 models were\ncompared in total.\n2 Related Work\nPrior work.\n")
    assert pdf_handler._extract_sections(raw)["introduction"] == \
        "We study models. 2 models were compared in total."


# ── URL / ID parsing ────────────────────────────────────────────────────────

@pytest.mark.parametrize("query, expected", [
    ("1706.03762", "1706.03762"),
    ("2301.07041v2", "2301.07041v2"),
    ("https://arxiv.org/abs/1706.03762", "1706.03762"),
    ("https://arxiv.org/pdf/1706.03762v7", "1706.03762v7"),
    ("10.48550/arXiv.1706.03762", "1706.03762"),
    ("not an id", None),
])
def test_arxiv_id_parsing(query, expected):
    assert _parse_input(query) == expected


@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://youtu.be/dQw4w9WgXcQ?t=42",
    "https://www.youtube.com/shorts/dQw4w9WgXcQ",
    "https://www.youtube.com/embed/dQw4w9WgXcQ",
])
def test_youtube_video_id(url):
    assert _extract_video_id(url) == "dQw4w9WgXcQ"
