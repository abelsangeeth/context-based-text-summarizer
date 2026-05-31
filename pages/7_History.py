"""Page 7 — History: View past summaries."""

import streamlit as st
from components.stitch_theme import apply_stitch_theme, page_header, sidebar_brand, sidebar_nav, sidebar_footer
from app_history import get_history, clear_history

st.set_page_config(page_title="History — COMPASS", page_icon="🕒", layout="wide")
apply_stitch_theme()

with st.sidebar:
    sidebar_brand()
    sidebar_nav(active="history")
    sidebar_footer()

page_header(
    "History",
    "View your past generated summaries. Only the last 100 entries are saved."
)

history = get_history()

col1, col2 = st.columns([6, 2])
with col2:
    if st.button("Clear History", width='stretch', type="primary"):
        clear_history()
        st.rerun()

st.write("")

if not history:
    st.info("No history found. Generate a summary to populate this list.")
else:
    for idx, entry in enumerate(history):
        # We cap the original text shown in the header
        model_name = entry.get('model', 'Unknown')
        timestamp = entry.get('timestamp', 'Unknown')
        
        with st.expander(f"🕒 {timestamp} — 🧠 {model_name}"):
            st.markdown("### Original Content")
            st.text(entry.get('original_text', ''))
            st.markdown("### Generated Summary")
            st.info(entry.get('summary', ''))
