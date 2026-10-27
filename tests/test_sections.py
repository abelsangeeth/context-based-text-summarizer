from summarizer.section_detector import merge_short_sections, detect_sections


def _sections(*word_counts):
    return [{"index": i, "raw_text": " ".join(["w"] * n), "word_count": n, "section_type": "body"}
            for i, n in enumerate(word_counts)]


def test_many_short_windows_do_not_snowball_into_one_section():
    # The topic-shift splitter cuts real prose into ~50-word windows; the old
    # merge absorbed all of them into a single section.
    merged = merge_short_sections(_sections(*[50] * 30), min_words=80, max_words=500)
    assert len(merged) >= 3
    assert all(s["word_count"] <= 500 for s in merged)
    assert sum(s["word_count"] for s in merged) == 1500
    assert [s["index"] for s in merged] == list(range(len(merged)))


def test_tiny_sections_never_stand_alone():
    merged = merge_short_sections(_sections(20, 300, 10), min_words=80)
    assert [s["word_count"] for s in merged] == [330]

    merged = merge_short_sections(_sections(200, 200, 30), min_words=80)
    assert [s["word_count"] for s in merged] == [200, 230]


def test_long_sections_are_left_alone():
    merged = merge_short_sections(_sections(400, 450, 600), min_words=80)
    assert [s["word_count"] for s in merged] == [400, 450, 600]


def test_long_plain_text_yields_several_sections():
    # Like real prose, neighbouring 50-word windows share few TF-IDF terms, so
    # the topic-shift splitter cuts at almost every window boundary.
    import random
    rng = random.Random(0)
    vocab = [f"term{i}" for i in range(3000)]
    sentences = [" ".join(rng.sample(vocab, 12)).capitalize() + "." for _ in range(130)]
    text = " ".join(sentences)
    sections = merge_short_sections(detect_sections(text), min_words=80)
    assert len(sections) >= 3                      # old code: 1 section
    # a short tail is folded into the last section rather than left on its own
    assert all(s["word_count"] < 500 + 80 for s in sections)
