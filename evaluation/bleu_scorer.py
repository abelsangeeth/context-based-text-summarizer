"""BLEU metric computation using NLTK."""

from nltk.translate.bleu_score import sentence_bleu, modified_precision, SmoothingFunction


def compute_bleu(predictions: list[str], references: list[str],
                 against_source: bool = False) -> dict:
    """
    Compute average BLEU score across prediction-reference pairs.

    Against a human reference this is standard sentence-level BLEU-4 with
    smoothing (method1) so short summaries with no 4-gram match don't score 0.

    Against the source document (``against_source=True``) the brevity penalty
    would punish every summary simply for being shorter than the source, so
    unigram modified precision is reported instead.
    """
    smoother = SmoothingFunction().method1
    scores = []
    for pred, ref in zip(predictions, references):
        ref_tokens = ref.split()
        pred_tokens = pred.split()
        if not pred_tokens or not ref_tokens:
            scores.append(0.0)
        elif against_source:
            scores.append(float(modified_precision([ref_tokens], pred_tokens, n=1)))
        else:
            scores.append(sentence_bleu([ref_tokens], pred_tokens,
                                        smoothing_function=smoother))
    avg = sum(scores) / len(scores) if scores else 0.0
    return {"BLEU": round(avg, 4)}
