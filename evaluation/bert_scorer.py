"""BERTScore computation — semantic similarity using contextual embeddings."""

import torch
from bert_score import score as bert_score_fn


def compute_bertscore(predictions: list[str],
                      references: list[str],
                      lang: str = "en") -> dict:
    """
    Compute BERTScore Precision, Recall, F1.
    rescale_with_baseline=True normalizes scores to a meaningful [0,1] range.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    P, R, F1 = bert_score_fn(
        predictions,
        references,
        lang=lang,
        device=device,
    )
    return {
        "BERTScore_P":  round(P.mean().item(), 4),
        "BERTScore_R":  round(R.mean().item(), 4),
        "BERTScore_F1": round(F1.mean().item(), 4),
    }
