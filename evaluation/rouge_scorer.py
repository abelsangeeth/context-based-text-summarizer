"""ROUGE metric computation using rouge_score directly to avoid PyArrow caching issues in Streamlit."""

from rouge_score import rouge_scorer


def compute_rouge(predictions: list[str], references: list[str],
                  against_source: bool = False) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L scores.
    Returns dict with keys: ROUGE-1, ROUGE-2, ROUGE-L (all float 0-1).

    Against a human reference this is the standard ROUGE F1. When the only
    thing to compare with is the source document (``against_source=True``),
    recall is meaningless — a short summary can never cover a long source —
    so precision is reported instead: the share of summary n-grams that are
    supported by the source.
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    field = "precision" if against_source else "fmeasure"

    r1, r2, rl = [], [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(target=ref, prediction=pred)
        r1.append(getattr(scores["rouge1"], field))
        r2.append(getattr(scores["rouge2"], field))
        rl.append(getattr(scores["rougeL"], field))

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    return {
        "ROUGE-1": round(avg(r1), 4),
        "ROUGE-2": round(avg(r2), 4),
        "ROUGE-L": round(avg(rl), 4),
    }
