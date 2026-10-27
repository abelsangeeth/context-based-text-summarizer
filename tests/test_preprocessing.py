import pytest

from input_handlers.cleaner import chunk_text as cleaner_chunk_text
from summarizer.preprocessing import chunk_text as pipeline_chunk_text, prepare_input


@pytest.mark.parametrize("chunk_text", [cleaner_chunk_text, pipeline_chunk_text])
@pytest.mark.parametrize("n_words", [801, 850, 1450, 1500, 2300])
def test_chunks_cover_text_without_redundant_tail(chunk_text, n_words):
    words = [f"w{i}" for i in range(n_words)]
    chunks = chunk_text(" ".join(words), max_words=800, overlap=100)

    # Every word appears, the last chunk ends on the last word...
    assert chunks[0].split()[0] == "w0"
    assert chunks[-1].split()[-1] == words[-1]
    assert set(" ".join(chunks).split()) == set(words)
    # ...and no chunk is just a copy of the previous chunk's tail.
    for prev, cur in zip(chunks, chunks[1:]):
        assert not set(cur.split()) <= set(prev.split())
        assert prev.split()[-100:] == cur.split()[:100]  # overlap is kept


@pytest.mark.parametrize("chunk_text", [cleaner_chunk_text, pipeline_chunk_text])
def test_short_text_is_one_chunk(chunk_text):
    assert chunk_text("just a few words", max_words=800, overlap=100) == ["just a few words"]


def test_t5_gets_task_prefix_but_mt5_and_bart_do_not():
    text = "The committee approved the budget after a long debate."
    assert prepare_input(text, "t5-base").startswith("summarize: ")
    assert prepare_input(text, "t5-large").startswith("summarize: ")
    # mT5-XLSum was fine-tuned without a prefix
    assert not prepare_input(text, "mt5").lower().startswith("summarize:")
    assert not prepare_input(text, "bart-cnn").lower().startswith("summarize:")


def test_prefix_not_duplicated():
    assert prepare_input("summarize: already prefixed", "t5-base").count("summarize:") == 1
