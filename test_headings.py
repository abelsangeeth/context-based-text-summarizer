import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from summarizer.subheading_generator import generate_subheading

text1 = """Computational Intelligence SRM Institute, Undergraduate Student Department. Automatic text summarization is a critical challenge in managing the growing volume of digital news content. This study presents COMPASS (Comparative Multi-model Pipeline with Adaptive Semantic Scoring), a multi-model summarization framework. BARTlarge-CNN achieves the highest performance across all metrics, recording ROUGE-1 (0.445) and a composite score of 0.4268."""

print("Calling generate_subheading...")
h = generate_subheading(text1, "body")
print("Returned:", h)
