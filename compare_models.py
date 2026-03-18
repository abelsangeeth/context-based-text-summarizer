from modules.model_comparison import ModelComparison

models = [
    "facebook/bart-large-cnn",
    "t5-base",
    "google/pegasus-cnn_dailymail"
]

comparison = ModelComparison(models, sample_size=20)
results = comparison.compare()

print("\nFINAL AVERAGE RESULTS:\n")

for model, scores in results.items():
    print(model)
    for metric, value in scores.items():
        print(f"  {metric}: {value:.4f}")
    print()
    
import matplotlib.pyplot as plt
import numpy as np

models = list(results.keys())

metrics = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU", "BERTScore"]

# Prepare values
values = [[results[m][metric] for m in models] for metric in metrics]

x = np.arange(len(models))
width = 0.15

plt.figure()

for i, metric_values in enumerate(values):
    plt.bar(x + i * width, metric_values, width, label=metrics[i])

plt.xticks(x + width * 2, models, rotation=45)
plt.xlabel("Models")
plt.ylabel("Score")
plt.title("Model Comparison Across Evaluation Metrics")
plt.legend()
plt.tight_layout()
plt.show()