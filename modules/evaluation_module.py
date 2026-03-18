import evaluate
from nltk.translate.bleu_score import sentence_bleu
from bert_score import score


class EvaluationModule:

    def __init__(self):
        self.rouge = evaluate.load("rouge")

    def evaluate(self, predictions, references):

        rouge_scores = self.rouge.compute(
            predictions=predictions,
            references=references
        )

        bleu_scores = [
            sentence_bleu([ref.split()], pred.split())
            for pred, ref in zip(predictions, references)
        ]

        avg_bleu = sum(bleu_scores) / len(bleu_scores)

        P, R, F1 = score(predictions, references, lang="en")
        bert_f1 = F1.mean().item()

        return {
            "ROUGE-1": rouge_scores["rouge1"],
            "ROUGE-2": rouge_scores["rouge2"],
            "ROUGE-L": rouge_scores["rougeL"],
            "BLEU": avg_bleu,
            "BERTScore": bert_f1
        }