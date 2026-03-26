import math
from summarizer.pipeline import summarize

def estimate_reading_time(word_count: int) -> str:
    minutes = math.ceil(word_count / 200)
    return f"{minutes} min read"

def _summarize_section(text: str, model_name: str, max_length: int, min_length: int) -> str:
    words = text.split()
    if len(words) > 700:
        text = " ".join(words[:700])
        
    input_tokens = int(len(words) * 1.3)
    # A summarization model physically cannot output an 80% length summary without 
    # repeating n-grams and hallucinating gibberish. Max natural compression boundary is ~35%.
    safe_min = min(min_length, int(input_tokens * 0.35))
    safe_max = min(max_length, int(input_tokens * 0.8))
        
    return summarize(text, model_key=model_name, max_length=safe_max, min_length=safe_min, length_penalty=3.0)

def build_longform_summary(
    sections: list[dict],
    subheadings: list[str],
    model_name: str = "bart-cnn",
    length_mode: str = "detailed"
) -> dict:
    
    # max_length / min_length represent TOKENS, not words (tokens ≈ 1.3 * words)
    bounds = {
        "detailed": {"max_length": 450, "min_length": 250}, # 250 tokens min = ~190 words
        "standard": {"max_length": 250, "min_length": 100},
        "brief": {"max_length": 100, "min_length": 40}
    }
    mode_cfg = bounds.get(length_mode, bounds["standard"])
    
    results = []
    total_words = 0
    
    for sec, heading in zip(sections, subheadings):
        if sec["word_count"] < 50:
            summary_paragraph = sec["raw_text"]
        else:
            summary_paragraph = _summarize_section(
                sec["raw_text"],
                model_name=model_name,
                max_length=mode_cfg["max_length"],
                min_length=mode_cfg["min_length"]
            )
            
        p_words = len(summary_paragraph.split())
        total_words += p_words
        
        results.append({
            "subheading": heading,
            "summary_paragraph": summary_paragraph,
            "word_count": p_words,
            "section_type": sec.get("section_type", "body")
        })
        
    return {
        "sections": results,
        "total_word_count": total_words,
        "model_used": model_name,
        "length_mode": length_mode
    }
