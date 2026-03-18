"""BLEU metric computation using NLTK."""

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def compute_bleu(predictions: list[str], references: list[str]) -> dict:
    """
    Compute average BLEU score across prediction-reference pairs.
    Uses smoothing to handle short sequences gracefully.
    """
    smoother = SmoothingFunction().method1
    scores = []
    for pred, ref in zip(predictions, references):
        ref_tokens = ref.split()
        pred_tokens = pred.split()
        score = sentence_bleu(
            [ref_tokens], pred_tokens, smoothing_function=smoother
        )
        scores.append(score)
    avg = sum(scores) / len(scores) if scores else 0.0
    return {"BLEU": round(avg, 4)}
