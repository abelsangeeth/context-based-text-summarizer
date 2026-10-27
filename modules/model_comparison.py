from datasets import load_dataset
from modules.model_module import ModelModule
from modules.inference_module import InferenceModule
from modules.evaluation_module import EvaluationModule


class ModelComparison:

    def __init__(self, model_names, sample_size=50):
        self.model_names = model_names
        self.sample_size = sample_size
        self.evaluator = EvaluationModule()

    def compare(self):

        dataset = load_dataset("cnn_dailymail", "3.0.0", split="test")
        dataset = dataset.select(range(self.sample_size))

        results = {}

        for model_name in self.model_names:
            print(f"\nEvaluating {model_name}...")

            model_module = ModelModule(model_name)
            tokenizer, model = model_module.load_model()
            inference = InferenceModule(model, tokenizer, model_name)

            predictions = []
            references = []

            for sample in dataset:
                summary = inference.generate_summary(sample["article"])
                predictions.append(summary)
                references.append(sample["highlights"])

            scores = self.evaluator.evaluate(predictions, references)
            results[model_name] = scores

        return results