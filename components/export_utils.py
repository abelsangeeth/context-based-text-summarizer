"""Export utilities — CSV and text-based export for results."""

import csv
import io
import json
import streamlit as st
import pandas as pd


def scores_to_dataframe(all_scores: dict) -> pd.DataFrame:
    """Convert nested scores dict to a flat DataFrame."""
    rows = []
    for model, metrics in all_scores.items():
        row = {"Model": model}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def export_csv_button(df: pd.DataFrame, filename: str = "results.csv"):
    """Render a download button for a DataFrame as CSV."""
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    st.download_button(
        label="Download CSV",
        data=csv_buf.getvalue(),
        file_name=filename,
        mime="text/csv",
        icon="📥",
    )


def export_json_button(data: dict, filename: str = "results.json"):
    """Render a download button for dict data as JSON."""
    st.download_button(
        label="Download JSON",
        data=json.dumps(data, indent=2, ensure_ascii=False),
        file_name=filename,
        mime="application/json",
        icon="📥",
    )


def export_summary_text(summaries: dict, filename: str = "summaries.txt"):
    """Export all summaries as a plain text file."""
    buf = io.StringIO()
    for model, summary in summaries.items():
        buf.write(f"=== {model} ===\n{summary}\n\n")
    st.download_button(
        label="Download Summaries",
        data=buf.getvalue(),
        file_name=filename,
        mime="text/plain",
        icon="📄",
    )
