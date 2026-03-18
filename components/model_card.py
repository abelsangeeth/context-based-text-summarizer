"""Reusable summary card UI component for Streamlit."""

import streamlit as st
from summarizer.models import get_model_label, get_model_color
from evaluation.aggregator import composite_score


def render_model_card(model_key: str, summary: str, metrics: dict = None,
                      is_best: bool = False, source_text: str = None):
    """
    Render a single model summary card with optional metric badges.
    """
    label = get_model_label(model_key)
    color = get_model_color(model_key)

    border = f"2px solid {color}"
    if is_best:
        border = f"3px solid #10B981"

    st.markdown(
        f"""<div style="
            border: {border};
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 12px;
        ">""",
        unsafe_allow_html=True,
    )

    header_cols = st.columns([4, 1])
    with header_cols[0]:
        st.markdown(f"**{label}**")
    with header_cols[1]:
        if is_best:
            st.success("Best", icon="🏆")

    st.markdown(
        f'<div style="font-size: 0.92em; line-height: 1.6;">{summary}</div>',
        unsafe_allow_html=True,
    )

    word_count = len(summary.split())
    st.caption(f"{word_count} words")

    if metrics:
        badge_cols = st.columns(len(metrics))
        for col, (metric, value) in zip(badge_cols, metrics.items()):
            with col:
                st.metric(metric, f"{value:.4f}")

    st.markdown("</div>", unsafe_allow_html=True)


def render_comparison_grid(results: dict, source_text: str = None):
    """
    Render all model cards in a responsive grid.
    results: {model_key: {"summary": str, "metrics": dict}}
    """
    if not results:
        return

    # Determine best model
    scores_only = {k: v["metrics"] for k, v in results.items() if v.get("metrics")}
    best_model = None
    if scores_only:
        best_model = max(scores_only, key=lambda m: composite_score(scores_only[m]))

    cols = st.columns(min(len(results), 2))
    for idx, (model_key, data) in enumerate(results.items()):
        with cols[idx % 2]:
            render_model_card(
                model_key=model_key,
                summary=data.get("summary", ""),
                metrics=data.get("metrics"),
                is_best=(model_key == best_model),
                source_text=source_text,
            )
