"""Plotly radar chart + bar chart + styled dataframe for the metrics dashboard."""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import streamlit as st
from summarizer.models import get_model_color, get_model_label
from evaluation.aggregator import composite_score, rank_models

RADAR_METRICS = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU", "BERTScore_F1", "METEOR"]


def render_radar_chart(all_scores: dict):
    """Render a filled radar/spider chart with one trace per model."""
    fig = go.Figure()
    for model_key, scores in all_scores.items():
        values = [scores.get(m, 0) for m in RADAR_METRICS]
        values.append(values[0])  # close the polygon
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=RADAR_METRICS + [RADAR_METRICS[0]],
            fill="toself",
            name=get_model_label(model_key),
            line=dict(color=get_model_color(model_key)),
            opacity=0.7,
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=60, r=60, t=40, b=60),
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_ranking_bar(all_scores: dict):
    """Horizontal bar chart showing composite score ranking."""
    ranked = rank_models(all_scores)
    labels = [get_model_label(name) for name, _ in ranked]
    values = [score for _, score in ranked]
    colors = [get_model_color(name) for name, _ in ranked]

    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.4f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        xaxis=dict(range=[0, max(values) * 1.2] if values else [0, 1],
                    title="Composite Score"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=80, t=20, b=40),
        height=max(200, len(ranked) * 60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_scores_table(all_scores: dict):
    """Styled dataframe with conditional formatting — green=best, red=worst per column."""
    rows = {}
    for model_key, scores in all_scores.items():
        row = {m: scores.get(m, 0) for m in RADAR_METRICS}
        row["Composite"] = composite_score(scores)
        rows[get_model_label(model_key)] = row

    df = pd.DataFrame(rows).T
    df.index.name = "Model"

    def highlight(s):
        """Green for max, red for min per column."""
        styles = [""] * len(s)
        if s.max() != s.min():
            styles[s.values.argmax()] = "background-color: rgba(16,185,129,0.2); font-weight: bold"
            styles[s.values.argmin()] = "background-color: rgba(239,68,68,0.15)"
        return styles

    styled = df.style.apply(highlight).format("{:.4f}")
    return styled


def render_metrics_dashboard(all_scores: dict):
    """Full metrics dashboard: radar + bar + table in a single call."""
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### Metric Comparison")
        st.plotly_chart(render_radar_chart(all_scores), use_container_width=True)
    with col2:
        st.markdown("#### Composite Ranking")
        st.plotly_chart(render_ranking_bar(all_scores), use_container_width=True)

    st.markdown("#### Detailed Scores")
    st.dataframe(render_scores_table(all_scores), use_container_width=True)
