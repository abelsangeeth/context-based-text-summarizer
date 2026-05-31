import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from summarizer.subheading_generator import get_best_heading

text = """Compass is a multi-model transformer-based text summarization system. It is designed to handle heterogeneous input formats and evaluate multiple summarization architectures. Users can submit input in supported formats, select a summarization model, generate summaries."""

candidates = ["Compression Ratio And Custom Measures", "Compass Is A Multi-Model Transformer-Based Text"]

print("Best:", get_best_heading(candidates, text))
