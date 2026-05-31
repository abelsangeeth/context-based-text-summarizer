"""Page 3 — Metrics Dashboard: Full evaluation dashboard with all metrics."""

import streamlit as st
from summarizer.models import list_models, get_model_label
from summarizer.pipeline import summarize
from input_handlers.cleaner import clean_text as deep_clean
from evaluation.aggregator import (
    evaluate_all, composite_score, rank_models,
    generate_recommendation_text, abstractiveness_score, compression_ratio,
)
from components.radar_chart import render_radar_chart, render_ranking_bar, render_scores_table
from components.export_utils import export_csv_button, export_json_button, scores_to_dataframe

from components.stitch_theme import apply_stitch_theme, page_header, sidebar_brand, sidebar_nav, sidebar_footer

st.set_page_config(page_title="Metrics Dashboard — COMPASS", page_icon="📊", layout="wide")
apply_stitch_theme()

with st.sidebar:
    sidebar_brand()
    sidebar_nav(active="dashboard")
    sidebar_footer()

page_header("Metrics Dashboard", "Comprehensive evaluation with ROUGE, BLEU, BERTScore, METEOR, and novel metrics.")

# --- Input ---
input_text = st.text_area("Input text", height=180,
                          placeholder="Paste article text to evaluate summaries against...")
ref_text = st.text_area("Reference summary (optional)", height=100,
                        placeholder="Human-written reference for scoring...")

available = [m for m in list_models() if m != "mt5"]
selected = st.multiselect("Models to evaluate", available, default=available[:4],
                          format_func=get_model_label)

run = st.button("Run Full Evaluation", type="primary",
                disabled=len(input_text.split()) < 10 or len(selected) < 2)

st.divider()

if run and input_text.strip():
    cleaned = deep_clean(input_text)
    reference = ref_text.strip() if ref_text.strip() else cleaned

    all_scores = {}
    all_summaries = {}
    novel_metrics = {}

    progress = st.progress(0, text="Evaluating models...")

    for idx, model_key in enumerate(selected):
        progress.progress((idx + 1) / len(selected),
                          text=f"Evaluating {get_model_label(model_key)}...")

        summary = summarize(cleaned, model_key)
        all_summaries[model_key] = summary

        scores = evaluate_all([summary], [reference])
        all_scores[model_key] = scores

        novel_metrics[model_key] = {
            "Abstractiveness": abstractiveness_score(summary, cleaned),
            "Compression": compression_ratio(summary, cleaned),
        }

    progress.empty()

    # --- Recommendation ---
    st.info(generate_recommendation_text(all_scores))

    # --- Radar + Ranking ---
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### Metric Radar")
        st.plotly_chart(render_radar_chart(all_scores), width='stretch')
    with col2:
        st.markdown("#### Composite Ranking")
        st.plotly_chart(render_ranking_bar(all_scores), width='stretch')

    # --- Detailed table ---
    st.markdown("#### Score Breakdown")
    st.dataframe(render_scores_table(all_scores), width='stretch')

    # --- Novel metrics ---
    st.markdown("#### Research Metrics")
    nm_cols = st.columns(len(selected))
    for col, model_key in zip(nm_cols, selected):
        with col:
            st.markdown(f"**{get_model_label(model_key)}**")
            nm = novel_metrics[model_key]
            st.metric("Abstractiveness", f"{nm['Abstractiveness']:.2%}",
                      help="% of summary words not in source. Higher = more paraphrasing.")
            st.metric("Compression", f"{nm['Compression']:.2%}",
                      help="Summary length / source length. Lower = more compressed.")

    # --- Summaries preview ---
    with st.expander("View generated summaries"):
        for model_key, summary in all_summaries.items():
            st.markdown(f"**{get_model_label(model_key)}**")
            st.write(summary)
            st.divider()

    # --- Export ---
    st.markdown("#### Export")
    e1, e2 = st.columns(2)
    with e1:
        export_csv_button(scores_to_dataframe(all_scores), "evaluation_results.csv")
    with e2:
        export_json_button({"scores": all_scores, "novel_metrics": novel_metrics},
                           "full_evaluation.json")
