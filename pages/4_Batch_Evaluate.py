"""Page 4 — Batch Evaluate: CSV bulk processing with export."""

import streamlit as st
import pandas as pd
import io
from summarizer.models import list_models, get_model_label
from summarizer.pipeline import summarize
from input_handlers.cleaner import clean_text as deep_clean
from evaluation.aggregator import evaluate_all, composite_score
from components.export_utils import export_csv_button

from components.stitch_theme import apply_stitch_theme, page_header, sidebar_brand, sidebar_nav, sidebar_footer

st.set_page_config(page_title="Batch Evaluate — COMPASS", page_icon="📦", layout="wide")
apply_stitch_theme()

with st.sidebar:
    sidebar_brand()
    sidebar_nav(active="batch")
    sidebar_footer()

page_header("Batch Evaluate", "Upload a CSV with text samples, run summarization and evaluation at scale.")

# --- Instructions ---
with st.expander("CSV format requirements", expanded=False):
    st.markdown("""
    Your CSV should have at least one of these columns:
    - **text** — The source text to summarize
    - **reference** *(optional)* — Human-written reference summary for scoring
    
    Example:
    | text | reference |
    |------|-----------|
    | Artificial intelligence is transforming... | AI transforms industries... |
    """)

# --- Upload ---
uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.dataframe(df.head(), width='stretch')
    st.caption(f"**{len(df)}** rows loaded")

    if "text" not in df.columns:
        st.error("CSV must have a 'text' column.")
        st.stop()

    has_ref = "reference" in df.columns

    # Model selection
    available = [m for m in list_models() if m != "mt5"]
    model_key = st.selectbox("Model", available, format_func=get_model_label)

    max_rows = st.slider("Max rows to process", 1, min(len(df), 500), min(len(df), 50))

    run = st.button("Run Batch Evaluation", type="primary")

    if run:
        results = []
        progress = st.progress(0, text="Processing...")

        for idx in range(max_rows):
            row = df.iloc[idx]
            text = deep_clean(str(row["text"]))
            ref = str(row.get("reference", "")) if has_ref else text

            progress.progress((idx + 1) / max_rows,
                              text=f"Processing row {idx + 1}/{max_rows}...")

            summary = summarize(text, model_key)
            scores = evaluate_all([summary], [ref if ref.strip() else text])
            scores["Composite"] = composite_score(scores)

            result_row = {
                "Row": idx + 1,
                "Text (preview)": text[:100] + "...",
                "Summary": summary,
            }
            result_row.update(scores)
            results.append(result_row)

        progress.empty()

        result_df = pd.DataFrame(results)

        st.success(f"Processed {max_rows} samples with {get_model_label(model_key)}")

        # Summary stats
        metric_cols = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU", "BERTScore_F1", "METEOR", "Composite"]
        available_metrics = [c for c in metric_cols if c in result_df.columns]

        st.markdown("#### Average Scores")
        avg_cols = st.columns(len(available_metrics))
        for col, m in zip(avg_cols, available_metrics):
            with col:
                st.metric(m, f"{result_df[m].mean():.4f}")

        st.markdown("#### Detailed Results")
        st.dataframe(result_df, width='stretch', height=400)

        export_csv_button(result_df, "batch_results.csv")
