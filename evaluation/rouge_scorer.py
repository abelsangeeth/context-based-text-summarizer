"""ROUGE metric computation using rouge_score directly to avoid PyArrow caching issues in Streamlit."""

from rouge_score import rouge_scorer


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L scores.
    Returns dict with keys: ROUGE-1, ROUGE-2, ROUGE-L (all float 0-1).
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    
    r1, r2, rl = [], [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(target=ref, prediction=pred)
        r1.append(scores["rouge1"].precision)
        r2.append(scores["rouge2"].precision)
        rl.append(scores["rougeL"].precision)
        
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    return {
        "ROUGE-1": round(avg(r1), 4),
        "ROUGE-2": round(avg(r2), 4),
        "ROUGE-L": round(avg(rl), 4),
    }
