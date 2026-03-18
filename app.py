"""ContextFree-X — Research-Grade Multi-Domain Text Summarizer."""

import streamlit as st

st.set_page_config(
    page_title="ContextFree-X",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Global minimalist CSS ---
st.markdown("""
<style>
    /* Clean typography */
    .block-container { padding-top: 2.5rem; max-width: 1100px; }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }
    section[data-testid="stSidebar"] .stMarkdown h1 {
        font-size: 1.3rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    /* Metric cards */
    div[data-testid="metric-container"] {
        background: rgba(100,100,100,0.08);
        border-radius: 8px;
        padding: 10px 14px;
    }
    
    /* Feature cards */
    .feature-card {
        padding: 24px;
        border-radius: 12px;
        background: rgba(100,100,100,0.05);
        border: 1px solid rgba(100,100,100,0.12);
        height: 100%;
        transition: border-color 0.2s;
    }
    .feature-card:hover { border-color: rgba(59,130,246,0.5); }
    .feature-card h3 { margin: 0 0 8px 0; font-size: 1.05rem; }
    .feature-card p { margin: 0; font-size: 0.88rem; opacity: 0.8; line-height: 1.5; }
    
    /* Hero section */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.15;
        margin-bottom: 8px;
    }
    .hero-sub {
        font-size: 1.15rem;
        opacity: 0.7;
        line-height: 1.6;
        max-width: 640px;
    }
    
    /* Comparison table */
    .vs-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
    .vs-table th { text-align: left; padding: 10px 12px; border-bottom: 2px solid rgba(100,100,100,0.2); }
    .vs-table td { padding: 8px 12px; border-bottom: 1px solid rgba(100,100,100,0.08); }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("# ⚡ ContextFree-X")
    st.caption("Research-Grade Summarizer")
    st.divider()
    st.markdown("""
    **Navigate**
    - ✍️ **Summarize** — Quick single summary
    - ⚔️ **Compare** — Side-by-side models
    - 📊 **Dashboard** — Full metrics
    - 📦 **Batch** — CSV processing
    - 🌍 **Multilingual** — 8 languages
    """)
    st.divider()
    st.markdown("""
    <div style="font-size: 0.78rem; opacity: 0.5;">
    Built with HuggingFace Transformers<br>
    T5 · BART · PEGASUS · mT5
    </div>
    """, unsafe_allow_html=True)

# --- Hero ---
st.markdown('<div class="hero-title">ContextFree-X</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">'
    'A research-grade, multi-domain text summarizer with 5 transformer models, '
    '6 evaluation metrics, multilingual support, and research paper analysis — '
    'all running locally with full transparency.'
    '</div>',
    unsafe_allow_html=True,
)

st.write("")

# --- Feature cards ---
cols = st.columns(3)

features = [
    ("✍️ Multi-Input", "Text, PDF, and URL input with automatic research paper section detection."),
    ("⚔️ 5 Models", "T5-Base, T5-Large, BART CNN, PEGASUS, and mT5 Multilingual — compared side-by-side."),
    ("📊 6+ Metrics", "ROUGE, BLEU, BERTScore, METEOR, Abstractiveness, Compression Ratio."),
    ("🌍 8 Languages", "Auto-detect language and route to the best model. Tamil, Hindi, French, and more."),
    ("📦 Batch Mode", "Process hundreds of samples from CSV with full metric export."),
    ("🔬 Research-Grade", "Semantic BERTScore, abstractiveness scoring, and composite model ranking."),
]

for idx, (title, desc) in enumerate(features):
    with cols[idx % 3]:
        st.markdown(
            f'<div class="feature-card"><h3>{title}</h3><p>{desc}</p></div>',
            unsafe_allow_html=True,
        )

st.write("")
st.write("")

# --- Comparison table ---
st.markdown("#### How ContextFree-X Compares")
st.markdown("""
<table class="vs-table">
<tr><th>Capability</th><th>ChatGPT / Gemini</th><th>ContextFree-X</th></tr>
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
""", unsafe_allow_html=True)

st.write("")
st.divider()
st.caption("Select a page from the sidebar to get started. → ✍️ Summarize is a great first step.")