"""COMPASS — Comparative Multi-model Pipeline with Adaptive Semantic Scoring."""

import streamlit as st
from components.stitch_theme import (
    apply_stitch_theme, sidebar_brand, sidebar_nav, sidebar_footer, page_header,
)

st.set_page_config(
    page_title="COMPASS — Comparative Multi-model Pipeline with Adaptive Semantic Scoring",
    page_icon="⧡",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_stitch_theme()

# --- Sidebar ---
with st.sidebar:
    sidebar_brand()
    sidebar_nav(active="")
    sidebar_footer()

# --- Hero ---
st.markdown(
    """
    <div style="margin-bottom: 6px;">
        <div class="cfx-page-title">
            <span class="cfx-gradient-text">COMPASS</span>
        </div>
    </div>
    <div class="cfx-page-subtitle">
        Comparative Multi-model Pipeline with Adaptive Semantic Scoring —
        a research-grade, multi-domain text summarizer with 5 transformer models,
        6 evaluation metrics, multilingual support, and research paper analysis —
        all running locally with full transparency.
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# --- Feature cards ---
cols = st.columns(3)

features = [
    ("summarize",    "Multi-Input",       "Text, PDF, and URL input with automatic research paper section detection."),
    ("compare_arrows","5 Models",         "T5-Base, T5-Large, BART CNN, PEGASUS, and mT5 — compared side-by-side."),
    ("dashboard",    "6+ Metrics",        "ROUGE, BLEU, BERTScore, METEOR, Abstractiveness, Compression Ratio."),
    ("language",     "8 Languages",       "Auto-detect language and route to the best model. Tamil, Hindi, French, and more."),
    ("layers",       "Batch Mode",        "Process hundreds of samples from CSV with full metric export."),
    ("science",      "Research-Grade",    "Semantic BERTScore, abstractiveness scoring, and composite model ranking."),
]

for idx, (icon_name, title, desc) in enumerate(features):
    with cols[idx % 3]:
        st.markdown(
            f"""<div class="cfx-card" style="margin-bottom:16px; transition: box-shadow 0.2s;">
                  <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
                    <span class="material-symbols-outlined"
                          style="font-size:22px;color:#0058be;
                                 font-variation-settings:'FILL' 1,'wght' 400,'GRAD' 0,'opsz' 24;">
                      {icon_name}
                    </span>
                    <span style="font-family:'Manrope',sans-serif;font-weight:800;
                                 font-size:0.98rem;letter-spacing:-0.02em;color:#191b23;">
                      {title}
                    </span>
                  </div>
                  <p style="margin:0;font-size:0.875rem;color:#727785;line-height:1.6;">{desc}</p>
                </div>""",
            unsafe_allow_html=True,
        )

st.write("")

# --- Comparison table ---
st.markdown(
    '<div class="cfx-section-title">How COMPASS Compares</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
    .vs-table {
        width: 100%; border-collapse: collapse;
        font-family: 'Inter', sans-serif; font-size: 0.875rem;
        background: #ffffff; border-radius: 16px; overflow: hidden;
        border: 1px solid rgba(194,198,214,0.25);
        box-shadow: 0 1px 6px rgba(0,0,0,0.04);
    }
    .vs-table th {
        text-align: left; padding: 12px 16px;
        background: #f2f3fd;
        font-size: 10px; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.1em; color: #424754;
        border-bottom: 1px solid rgba(194,198,214,0.3);
    }
    .vs-table td {
        padding: 10px 16px;
        border-bottom: 1px solid rgba(194,198,214,0.15);
        color: #424754;
    }
    .vs-table tr:last-child td { border-bottom: none; }
    .vs-table tr:hover td { background: rgba(0,88,190,0.03); }
    </style>
    <table class="vs-table">
    <tr><th>Capability</th><th>ChatGPT / Gemini</th><th>COMPASS</th></tr>
    <tr><td>Evaluation scores</td><td>❌ None</td><td>✅ ROUGE + BLEU + BERTScore + METEOR</td></tr>
    <tr><td>Semantic scoring</td><td>❌</td><td>✅ BERTScore F1 with baseline rescaling</td></tr>
    <tr><td>Model comparison</td><td>❌ Single model</td><td>✅ 5 models simultaneously</td></tr>
    <tr><td>Abstractiveness metric</td><td>❌</td><td>✅ Novel n-gram novelty score</td></tr>
    <tr><td>Multi-language</td><td>⚠️ Prompt-based</td><td>✅ Auto-detect + mT5 routing</td></tr>
    <tr><td>PDF / URL input</td><td>⚠️ Limited</td><td>✅ Deep extraction + chunking</td></tr>
    <tr><td>Offline / reproducible</td><td>❌ Cloud black box</td><td>✅ Local HuggingFace weights</td></tr>
    <tr><td>Research paper mode</td><td>❌</td><td>✅ Extracts Abstract / Intro / Conclusion</td></tr>
    <tr><td>Batch evaluation</td><td>❌</td><td>✅ CSV bulk processing with export</td></tr>
    </table>
    """,
    unsafe_allow_html=True,
)

st.write("")
st.markdown(
    '<div class="cfx-divider"><span class="cfx-divider-label">Get Started</span></div>',
    unsafe_allow_html=True,
)
st.caption("Select a page from the sidebar to begin.  →  Summarize is a great first step.")