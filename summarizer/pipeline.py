"""Core summarization pipeline — single function interface for all models."""

import torch
from summarizer.models import load_model, get_model_config, get_device
from summarizer.preprocessing import prepare_input, chunk_text, clean_text
from summarizer.postprocessing import postprocess_summary


def summarize(text: str, model_key: str) -> str:
    """
    Generate a summary using the specified model.
    Handles input preparation, inference, and postprocessing.
    """
    cfg = get_model_config(model_key)
    tokenizer, model = load_model(model_key)
    device = get_device()

    prepared = prepare_input(text, model_key)

    inputs = tokenizer(
        prepared,
        return_tensors="pt",
        max_length=cfg["max_input"],
        truncation=True
    ).to(device)

    # Scale min_length based on input length — longer input = longer summary
    input_word_count = len(text.split())
    if input_word_count > 300:
        min_len = min(cfg["min_output"] * 2, cfg["max_output"] - 20)
    elif input_word_count > 100:
        min_len = cfg["min_output"]
    else:
        min_len = min(30, cfg["max_output"] - 10)

    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_length=cfg["max_output"],
            min_length=min_len,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=2.0,
        )

    raw = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return postprocess_summary(raw)


def _token_safe_chunk_size(model_key: str) -> int:
    """Return a safe word count per chunk so it fits within the model's max_input tokens."""
    cfg = get_model_config(model_key)
    # Conservative: ~1.3 tokens per word on average, leave 10% margin
    return max(100, int(cfg["max_input"] / 1.5))


def summarize_long(text: str, model_key: str) -> str:
    """
    Summarize long documents through iterative chunk-and-merge.
    1. Split text into token-safe chunks
    2. Summarize each chunk
    3. If combined summaries still too long, repeat until it fits in one pass
    """
    text = clean_text(text)
    safe_words = _token_safe_chunk_size(model_key)

    chunks = chunk_text(text, max_words=safe_words, overlap=50)

    if len(chunks) == 1:
        return summarize(chunks[0], model_key)

    # Iterative reduction: keep summarizing until it fits in one shot
    current_texts = chunks
    max_iterations = 5  # safety limit

    for _ in range(max_iterations):
        summaries = []
        for chunk in current_texts:
            if len(chunk.split()) < 20:
                summaries.append(chunk)
                continue
            s = summarize(chunk, model_key)
            summaries.append(s)

        combined = " ".join(summaries)

        # If combined text fits in one pass, do final summary
        if len(combined.split()) <= safe_words:
            return summarize(combined, model_key)

        # If only 2-3 chunks remain, join all summaries as the final result
        if len(current_texts) <= 3 and len(combined.split()) <= safe_words * 2:
            return summarize(combined[:safe_words * 6], model_key)

        # Otherwise, chunk the summaries and repeat
        current_texts = chunk_text(combined, max_words=safe_words, overlap=50)

    # Final pass with whatever we have
    return summarize(" ".join(current_texts)[:safe_words * 5], model_key)
