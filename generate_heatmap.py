"""
generate_heatmap.py
===================
Produces a publication-quality heatmap for the Context-Based Text Summarizer,
styled exactly like the confusion-matrix figure in the reference paper:
  - Blue-scale colormap (Blues)
  - White text on dark cells, dark-grey text on light cells
  - Thin grid lines between cells
  - Colorbar on the right labelled "Normalised Score"
  - Two-line axis tick labels (matching the paper style)
  - Title formatted as "Metric Score Heatmap — ContextFree-X Framework"

Run with:
    $env:PYTHONUTF8=1; python generate_heatmap.py

Output: bert_graphs/fig7_metric_heatmap.png
"""

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

matplotlib.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   True,
    "axes.spines.right":  True,
    "axes.spines.left":  True,
    "axes.spines.bottom": True,
    "figure.dpi":        150,
})

OUT_DIR = os.path.join(os.path.dirname(__file__), "bert_graphs")
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Data  (same benchmark numbers as generate_bert_graphs.py)
# ─────────────────────────────────────────────────────────────────────────────
MODELS = ["T5-Base", "T5-Large", "BART-CNN", "PEGASUS", "mT5-Multi"]

# Two-line labels to match the paper's "Mel. Nevi" / "Bas. Cell\nCarc." style
MODEL_LABELS = [
    "T5\nBase",
    "T5\nLarge",
    "BART\nCNN",
    "PEGA-\nSUS",
    "mT5\nMulti",
]

METRICS = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU", "METEOR",
           "BERTScore\nPrec.", "BERTScore\nRec.", "BERTScore\nF1"]

# Raw keys in DATA dict mapped to METRICS display order
METRIC_KEYS = ["ROUGE-1", "ROUGE-2", "ROUGE-L",
               "BLEU", "METEOR",
               "BERTScore_P", "BERTScore_R", "BERTScore_F1"]

DATA = {
    "T5-Base": {
        "ROUGE-1": 0.382, "ROUGE-2": 0.164, "ROUGE-L": 0.256,
        "BLEU":    0.078, "METEOR":  0.154,
        "BERTScore_P": 0.847, "BERTScore_R": 0.831, "BERTScore_F1": 0.838,
    },
    "T5-Large": {
        "ROUGE-1": 0.413, "ROUGE-2": 0.191, "ROUGE-L": 0.279,
        "BLEU":    0.096, "METEOR":  0.172,
        "BERTScore_P": 0.861, "BERTScore_R": 0.848, "BERTScore_F1": 0.854,
    },
    "BART-CNN": {
        "ROUGE-1": 0.445, "ROUGE-2": 0.214, "ROUGE-L": 0.309,
        "BLEU":    0.118, "METEOR":  0.193,
        "BERTScore_P": 0.879, "BERTScore_R": 0.862, "BERTScore_F1": 0.870,
    },
    "PEGASUS": {
        "ROUGE-1": 0.438, "ROUGE-2": 0.208, "ROUGE-L": 0.303,
        "BLEU":    0.112, "METEOR":  0.189,
        "BERTScore_P": 0.876, "BERTScore_R": 0.859, "BERTScore_F1": 0.867,
    },
    "mT5-Multi": {
        "ROUGE-1": 0.321, "ROUGE-2": 0.131, "ROUGE-L": 0.218,
        "BLEU":    0.059, "METEOR":  0.128,
        "BERTScore_P": 0.839, "BERTScore_R": 0.821, "BERTScore_F1": 0.829,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
#  Build matrix  (rows = models, cols = metrics)
# ─────────────────────────────────────────────────────────────────────────────
matrix = np.array([
    [DATA[m][k] for k in METRIC_KEYS]
    for m in MODELS
], dtype=float)

# ─────────────────────────────────────────────────────────────────────────────
#  Plot
# ─────────────────────────────────────────────────────────────────────────────
n_rows, n_cols = matrix.shape
fig_w = 2.0 + n_cols * 1.18   # auto-scale width with number of columns
fig_h = 1.6 + n_rows * 1.00

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
fig.subplots_adjust(left=0.18, right=0.84, top=0.88, bottom=0.22)

cmap = plt.cm.Blues
norm = Normalize(vmin=0.0, vmax=1.0)

im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto",
               interpolation="nearest")

# ── Draw cell borders (thin white lines between cells like the reference) ──
for x in np.arange(-0.5, n_cols, 1):
    ax.axvline(x, color="white", linewidth=1.5)
for y in np.arange(-0.5, n_rows, 1):
    ax.axhline(y, color="white", linewidth=1.5)

# ── Annotate each cell ──
thresh_white = 0.55   # cells darker than this get white text
for r in range(n_rows):
    for c in range(n_cols):
        val = matrix[r, c]
        # normalized colour intensity
        norm_val = norm(val)
        txt_color = "white" if norm_val > thresh_white else "#4B5563"
        weight    = "bold"  if norm_val > thresh_white else "normal"
        ax.text(c, r, f"{val:.2f}",
                ha="center", va="center",
                fontsize=11, color=txt_color, fontweight=weight)

# ── Axes ticks ──
ax.set_xticks(range(n_cols))
ax.set_xticklabels(METRICS, fontsize=9.5, ha="center", linespacing=1.25)
ax.xaxis.set_label_position("bottom")
ax.xaxis.tick_bottom()

ax.set_yticks(range(n_rows))
ax.set_yticklabels(MODEL_LABELS, fontsize=10.5, va="center", linespacing=1.2)

# ── Axis labels ──
ax.set_xlabel("Evaluation Metric", fontsize=12, labelpad=14, fontweight="bold")
ax.set_ylabel("Model", fontsize=12, labelpad=14, fontweight="bold")

# ── Outer border (match the paper's single-line box) ──
for spine in ax.spines.values():
    spine.set_linewidth(1.2)
    spine.set_color("#555555")

# ── Colorbar — right side, matching paper style ──
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.04)
cbar.set_label("Normalised Score", fontsize=10, labelpad=10)
cbar.ax.tick_params(labelsize=8.5)
cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

# ── Title ──
ax.set_title(
    "Metric Score Heatmap — ContextFree-X Summarization Framework\n"
    "(Rows = Models, Cols = Evaluation Metrics; BERTScore captures semantic similarity)",
    fontsize=10.5, pad=10, color="#111827", fontweight="bold"
)

# ── Highlight the BERTScore columns with a subtle top annotation ──
bert_cols = [METRIC_KEYS.index(k) for k in ["BERTScore_P", "BERTScore_R", "BERTScore_F1"]]
for bc in bert_cols:
    ax.add_patch(plt.Rectangle(
        (bc - 0.5, -0.5), 1, n_rows,
        fill=False, edgecolor="#1D4ED8", linewidth=2.2, zorder=5
    ))

# Small bracket label above BERTScore group
mid_bert = np.mean(bert_cols)
ax.annotate(
    "BERT-based",
    xy=(mid_bert, -0.51), xycoords="data",
    xytext=(mid_bert, -1.45), textcoords="data",
    fontsize=8, color="#1D4ED8", ha="center", style="italic",
    arrowprops=dict(arrowstyle="-", color="#1D4ED8", lw=0.8),
)

plt.savefig(
    os.path.join(OUT_DIR, "fig7_metric_heatmap.png"),
    dpi=180, bbox_inches="tight",
    facecolor="white",
)
plt.close()
print("Saved: bert_graphs/fig7_metric_heatmap.png")
