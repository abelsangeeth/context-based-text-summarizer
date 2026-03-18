"""ROUGE metric computation using the evaluate library."""

import evaluate

_rouge = evaluate.load("rouge")


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L scores.
    Returns dict with keys: ROUGE-1, ROUGE-2, ROUGE-L (all float 0-1).
    """
    scores = _rouge.compute(predictions=predictions, references=references)
    return {
        "ROUGE-1": round(scores["rouge1"], 4),
        "ROUGE-2": round(scores["rouge2"], 4),
        "ROUGE-L": round(scores["rougeL"], 4),
    }
