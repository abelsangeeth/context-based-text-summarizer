"""
Aggregator — combines all metric scores, computes composite ranking,
picks the best model, and generates human-readable recommendations.
"""

from evaluation.rouge_scorer import compute_rouge
from evaluation.bleu_scorer import compute_bleu
from evaluation.bert_scorer import compute_bertscore
from evaluation.meteor_scorer import compute_meteor

WEIGHTS = {
    "ROUGE-1":      0.20,
    "ROUGE-2":      0.20,
    "ROUGE-L":      0.15,
    "BLEU":         0.10,
    "BERTScore_F1": 0.25,
    "METEOR":       0.10,
}


def evaluate_all(predictions: list[str], references: list[str],
                 lang: str = "en") -> dict:
    """Run all metrics and return a combined dict."""
    results = {}
    results.update(compute_rouge(predictions, references))
    results.update(compute_bleu(predictions, references))
    results.update(compute_bertscore(predictions, references, lang=lang))
    results.update(compute_meteor(predictions, references))
    return results


def composite_score(metrics: dict) -> float:
    """Weighted composite score from individual metrics."""
    return round(sum(metrics.get(k, 0) * w for k, w in WEIGHTS.items()), 4)


def pick_best_model(all_model_scores: dict) -> str:
    """Return name of model with highest composite score."""
    return max(all_model_scores,
               key=lambda m: composite_score(all_model_scores[m]))


def rank_models(all_model_scores: dict) -> list[tuple[str, float]]:
    """Return sorted list of (model_name, composite_score) descending."""
    ranked = [
        (name, composite_score(scores))
        for name, scores in all_model_scores.items()
    ]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def generate_recommendation_text(all_model_scores: dict) -> str:
    """Generate a human-readable recommendation based on all scores."""
    if not all_model_scores:
        return "No model scores available."

    ranked = rank_models(all_model_scores)
    best_name = ranked[0][0]
    best_scores = all_model_scores[best_name]

    # Find which metric each model leads on
    metric_leaders = {}
    for metric in WEIGHTS:
        leader = max(all_model_scores,
                     key=lambda m: all_model_scores[m].get(metric, 0))
        val = all_model_scores[leader].get(metric, 0)
        metric_leaders[metric] = (leader, val)

    parts = [f"**{best_name}** achieves the highest composite score ({ranked[0][1]:.4f})."]

    semantic_leader, semantic_val = metric_leaders.get("BERTScore_F1", (best_name, 0))
    rouge_leader, rouge_val = metric_leaders.get("ROUGE-1", (best_name, 0))

    if semantic_leader != rouge_leader:
        parts.append(
            f"{semantic_leader} leads on semantic quality "
            f"(BERTScore F1: {semantic_val:.4f}), "
            f"while {rouge_leader} leads on n-gram overlap "
            f"(ROUGE-1: {rouge_val:.4f})."
        )

    return " ".join(parts)


# --- Novel Metrics ---

def abstractiveness_score(summary: str, source: str) -> float:
    """
    Fraction of summary unigrams NOT in source.
    Higher = more abstractive (model paraphrasing, not copying).
    """
    src_tokens = set(source.lower().split())
    sum_tokens = summary.lower().split()
    novel = [t for t in sum_tokens if t not in src_tokens]
    return round(len(novel) / len(sum_tokens), 4) if sum_tokens else 0.0


def compression_ratio(summary: str, source: str) -> float:
    """Words in summary / words in source. Lower = more compressed."""
    src_len = len(source.split())
    if src_len == 0:
        return 0.0
    return round(len(summary.split()) / src_len, 4)
