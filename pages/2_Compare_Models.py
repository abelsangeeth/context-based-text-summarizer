"""Page 2 — Compare Models: Side-by-side 4-model comparison with metrics."""

import streamlit as st
from summarizer.models import list_models, get_model_label, get_model_color
from summarizer.pipeline import summarize, summarize_long
from input_handlers.text_cleaner import deep_clean
from evaluation.aggregator import (
    evaluate_all, composite_score, pick_best_model,
    rank_models, generate_recommendation_text,
    abstractiveness_score, compression_ratio,
)
from components.model_card import render_model_card
from components.export_utils import export_csv_button, export_summary_text, scores_to_dataframe

st.set_page_config(page_title="Compare Models — ContextFree-X", page_icon="⚔️", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 2rem; max-width: 1400px; }
    div[data-testid="metric-container"] {
        background: rgba(100,100,100,0.08);
        border-radius: 8px;
        padding: 8px 12px;
    }
    .best-badge {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚔️ Compare Models")
st.caption("Run the same text through multiple models and compare quality side-by-side.")

# --- Input ---
input_text = st.text_area(
    "Enter text to summarize",
    height=200,
    placeholder="Paste your article text here...",
)

# --- Model selection ---
available = [m for m in list_models() if m != "mt5"]
selected = st.multiselect(
    "Select models to compare",
    available,
    default=available[:4],
    format_func=get_model_label,
)

col_run, col_ref = st.columns([1, 3])
with col_run:
    run = st.button("Compare", type="primary", use_container_width=True,
                    disabled=len(input_text.split()) < 10 or len(selected) < 2)
with col_ref:
    ref_text = st.text_input("Reference summary (optional, for scoring)",
                             placeholder="Paste a human-written reference summary...")

st.divider()

# --- Run comparison ---
if run and input_text.strip():
    cleaned = deep_clean(input_text)
    word_count = len(cleaned.split())
    use_long = word_count > 900

    summaries = {}
    all_metrics = {}

    progress = st.progress(0, text="Generating summaries...")

    for idx, model_key in enumerate(selected):
        progress.progress((idx + 1) / len(selected),
                          text=f"Running {get_model_label(model_key)}...")
        with st.spinner(f"Generating with {get_model_label(model_key)}..."):
            if use_long:
                s = summarize_long(cleaned, model_key)
            else:
                s = summarize(cleaned, model_key)
            summaries[model_key] = s

            # Compute metrics if reference is provided
            reference = ref_text.strip() if ref_text.strip() else cleaned
            metrics = evaluate_all([s], [reference])
            metrics["Abstractiveness"] = abstractiveness_score(s, cleaned)
            metrics["Compression"] = compression_ratio(s, cleaned)
            all_metrics[model_key] = metrics

    progress.empty()

    # Determine best model
    best = pick_best_model(all_metrics) if all_metrics else None

    # Recommendation
    st.info(generate_recommendation_text(all_metrics))

    # --- Model cards grid ---
    num_models = len(selected)
    cols_per_row = min(num_models, 2)
    cols = st.columns(cols_per_row)

    for idx, model_key in enumerate(selected):
        with cols[idx % cols_per_row]:
            is_best = model_key == best
            label = get_model_label(model_key)
            color = get_model_color(model_key)

            # Card header
            if is_best:
                st.markdown(f'<span class="best-badge">🏆 Best Model</span>',
                            unsafe_allow_html=True)

            st.markdown(f"**{label}**")
            st.markdown(
                f'<div style="border-left: 4px solid {color}; padding: 12px 16px; '
                f'background: rgba(100,100,100,0.04); border-radius: 0 8px 8px 0; '
                f'margin-bottom: 12px; font-size: 0.92em; line-height: 1.65;">'
                f'{summaries[model_key]}</div>',
                unsafe_allow_html=True,
            )

            s = summaries[model_key]
            st.caption(f"{len(s.split())} words")

            # Key metric badges
            m = all_metrics[model_key]
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Composite", f"{composite_score(m):.4f}")
            mc2.metric("BERTScore", f"{m.get('BERTScore_F1', 0):.4f}")
            mc3.metric("ROUGE-1", f"{m.get('ROUGE-1', 0):.4f}")

            with st.expander("All metrics"):
                for metric, val in m.items():
                    st.write(f"**{metric}:** {val:.4f}")
            st.divider()

    # --- Export ---
    st.markdown("### Export Results")
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        df = scores_to_dataframe(all_metrics)
        export_csv_button(df, "model_comparison.csv")
    with col_e2:
        export_summary_text(summaries, "summaries.txt")
