"""METEOR metric computation using NLTK."""

import nltk

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

from nltk.translate.meteor_score import meteor_score as _meteor


def compute_meteor(predictions: list[str], references: list[str]) -> dict:
    """
    Compute average METEOR score across prediction-reference pairs.
    METEOR accounts for synonyms and stemming, unlike pure n-gram metrics.
    """
    scores = []
    for pred, ref in zip(predictions, references):
        ref_tokens = ref.split()
        pred_tokens = pred.split()
        score = _meteor([ref_tokens], pred_tokens)
        scores.append(score)
    avg = sum(scores) / len(scores) if scores else 0.0
    return {"METEOR": round(avg, 4)}
