"""Page 1 — Summarize: Input text, PDF, or URL and get a single summary."""

import streamlit as st
from summarizer.models import list_models, get_model_label, load_model
from summarizer.pipeline import summarize, summarize_long
from input_handlers.pdf_reader import extract_pdf_text, extract_research_paper
from input_handlers.url_scraper import scrape_url
from input_handlers.text_cleaner import deep_clean
from evaluation.aggregator import evaluate_all, abstractiveness_score, compression_ratio

st.set_page_config(page_title="Summarize — ContextFree-X", page_icon="✍️", layout="wide")

# --- Custom CSS for minimalist look ---
st.markdown("""
<style>
    .block-container { padding-top: 2rem; max-width: 1100px; }
    .stTextArea textarea { font-size: 0.95rem; line-height: 1.6; }
    div[data-testid="metric-container"] {
        background: rgba(100,100,100,0.08);
        border-radius: 8px;
        padding: 10px 14px;
    }
    .summary-box {
        background: rgba(100,100,100,0.05);
        border-left: 4px solid #3B82F6;
        padding: 16px 20px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
        line-height: 1.7;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("✍️ Summarize")
st.caption("Paste text, upload a PDF, or enter a URL — get a high-quality summary instantly.")

# --- Input method ---
tab_text, tab_pdf, tab_url = st.tabs(["📝 Text", "📄 PDF", "🔗 URL"])

input_text = ""

with tab_text:
    input_text = st.text_area(
        "Paste your text",
        height=250,
        placeholder="Enter or paste the article / document text here...",
        key="text_input",
    )

with tab_pdf:
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"], key="pdf_input")
    if uploaded:
        with st.spinner("Extracting text from PDF..."):
            result = extract_research_paper(uploaded)
            if result.get("warning"):
                st.warning(result["warning"])
            elif result.get("type") == "research_paper":
                st.success("Research paper detected — extracted sections")
                section = st.radio(
                    "Summarize which section?",
                    ["Full text", "Abstract", "Introduction", "Conclusion"],
                    horizontal=True,
                )
                mapping = {
                    "Full text": result.get("full", ""),
                    "Abstract": result.get("abstract", ""),
                    "Introduction": result.get("introduction", ""),
                    "Conclusion": result.get("conclusion", ""),
                }
                input_text = mapping[section]
            else:
                input_text = result.get("full", "")
            if input_text:
                with st.expander("Preview extracted text", expanded=False):
                    st.text(input_text[:2000] + ("..." if len(input_text) > 2000 else ""))

with tab_url:
    url = st.text_input("Enter article URL", placeholder="https://...", key="url_input")
    if url:
        with st.spinner("Scraping article..."):
            input_text = scrape_url(url)
            if not input_text:
                st.error("Could not extract text from this URL.")
            else:
                with st.expander("Preview scraped text", expanded=False):
                    st.text(input_text[:2000] + ("..." if len(input_text) > 2000 else ""))

st.divider()

# --- Model selection + Generate ---
col_model, col_btn = st.columns([3, 1])

available = [m for m in list_models() if m != "mt5"]  # mt5 on multilingual page

with col_model:
    model_key = st.selectbox(
        "Model",
        available,
        format_func=get_model_label,
    )

# Determine if long-document mode needed
word_count = len(input_text.split()) if input_text.strip() else 0

with col_btn:
    st.write("")  # vertical align hack
    st.write("")
    generate = st.button("Generate Summary", type="primary", use_container_width=True,
                         disabled=word_count < 10)

if word_count > 0:
    st.caption(f"Input: **{word_count}** words · **{len(input_text)}** characters")

# --- Generation ---
if generate and input_text.strip():
    cleaned = deep_clean(input_text)

    with st.spinner(f"Generating summary with {get_model_label(model_key)}..."):
        if word_count > 500:
            st.info(f"Long document detected ({word_count} words). Using iterative chunk-and-merge strategy.")
            summary = summarize_long(cleaned, model_key)
        else:
            summary = summarize(cleaned, model_key)

    st.markdown("### Summary")
    st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)

    s_words = len(summary.split())
    m1, m2, m3 = st.columns(3)
    m1.metric("Words", s_words)
    m2.metric("Compression", f"{compression_ratio(summary, cleaned):.2%}")
    m3.metric("Abstractiveness", f"{abstractiveness_score(summary, cleaned):.2%}")

    # Quick evaluation if user wants it
    with st.expander("Run evaluation metrics", expanded=False):
        if st.button("Compute metrics", key="eval_btn"):
            with st.spinner("Computing ROUGE, BLEU, BERTScore, METEOR..."):
                scores = evaluate_all([summary], [cleaned])
            for metric, val in scores.items():
                st.metric(metric, f"{val:.4f}")
