import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from summarizer.longform_builder import extractive_prefilter

text = """Computational Intelligence SRM Institute, Undergraduate Student Department. Automatic text summarization is a critical challenge in managing the growing volume of digital news content. This study presents COMPASS (Comparative Multi-model Pipeline with Adaptive Semantic Scoring), a multi-model summarization framework. BARTlarge-CNN achieves the highest performance across all metrics, recording ROUGE-1 (0.445) and a composite score of 0.4268."""

try:
    print("Testing extractive_prefilter...")
    res = extractive_prefilter(text, keep_ratio=0.5)
    print("Result:", res)
except Exception as e:
    print("Exception:", str(e))
