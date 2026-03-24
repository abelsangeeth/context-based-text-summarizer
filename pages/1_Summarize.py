"""Page 1 — Summarize: Input text, PDF, URL, YouTube, or arXiv and get a summary."""

import streamlit as st
from summarizer.models import list_models, get_model_label, load_model
from summarizer.pipeline import summarize, summarize_long
from input_handlers.pdf_handler import extract_pdf
from input_handlers.url_handler import scrape_url
from input_handlers.youtube_handler import extract_youtube, is_valid_youtube_url
from input_handlers.arxiv_handler import fetch_arxiv
from input_handlers.cleaner import clean_text
from evaluation.aggregator import evaluate_all, abstractiveness_score, compression_ratio
from components.stitch_theme import apply_stitch_theme, page_header, sidebar_brand, sidebar_nav, sidebar_footer

st.set_page_config(page_title="Summarize — ContextFree-X", page_icon="✍️", layout="wide")
apply_stitch_theme()

with st.sidebar:
    sidebar_brand()
    sidebar_nav(active="summarize")
    sidebar_footer()

page_header(
    "Summarize",
    "Paste text, upload a PDF, enter a URL, paste a YouTube link, or fetch an arXiv paper — get a high-quality summary instantly.",
)

# --- Input method ---
tab_text, tab_pdf, tab_url, tab_yt, tab_arxiv = st.tabs(
    ["📝 Text", "📄 PDF", "🔗 URL", "🎬 YouTube", "📚 arXiv"]
)

input_text = ""

# ══════════════════════════════════════════════════════════════════════
#  TAB 1 — Raw text
# ══════════════════════════════════════════════════════════════════════
with tab_text:
    input_text = st.text_area(
        "Paste your text",
        height=250,
        placeholder="Enter or paste the article / document text here...",
        key="text_input",
    )

# ══════════════════════════════════════════════════════════════════════
#  TAB 2 — PDF upload (with research-paper detection)
# ══════════════════════════════════════════════════════════════════════
with tab_pdf:
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"], key="pdf_input")
    if uploaded:
        with st.spinner("Extracting text from PDF..."):
            try:
                result = extract_pdf(uploaded)
            except ValueError as e:
                st.error(str(e))
                result = None

            if result:
                sections = result.get("sections", {})
                if result.get("is_research_paper"):
                    st.success("Research paper detected — extracted sections")
                    section = st.radio(
                        "Summarize which section?",
                        ["Full text", "Abstract", "Introduction", "Conclusion"],
                        horizontal=True,
                    )
                    mapping = {
                        "Full text": sections.get("full_text", ""),
                        "Abstract": sections.get("abstract", ""),
                        "Introduction": sections.get("introduction", ""),
                        "Conclusion": sections.get("conclusion", ""),
                    }
                    input_text = mapping[section] or sections.get("full_text", "")
                else:
                    input_text = sections.get("full_text", "")
                if input_text:
                    with st.expander("Preview extracted text", expanded=False):
                        st.text(input_text[:2000] + ("..." if len(input_text) > 2000 else ""))

# ══════════════════════════════════════════════════════════════════════
#  TAB 3 — URL scraper
# ══════════════════════════════════════════════════════════════════════
with tab_url:
    url = st.text_input("Enter article URL", placeholder="https://...", key="url_input")
    if url:
        with st.spinner("Scraping article..."):
            try:
                url_result = scrape_url(url)
                input_text = url_result.get("text", "")
            except ValueError as e:
                input_text = ""
                st.error(str(e))
            if not input_text:
                st.error("Could not extract text from this URL.")
            else:
                with st.expander("Preview scraped text", expanded=False):
                    st.text(input_text[:2000] + ("..." if len(input_text) > 2000 else ""))

# ══════════════════════════════════════════════════════════════════════
#  TAB 4 — YouTube transcript
# ══════════════════════════════════════════════════════════════════════
with tab_yt:
    yt_url = st.text_input(
        "YouTube video URL",
        placeholder="https://www.youtube.com/watch?v=...",
        key="yt_input",
    )
    if yt_url:
        if not is_valid_youtube_url(yt_url):
            st.error("Please enter a valid YouTube URL.")
        else:
            with st.spinner("Fetching transcript..."):
                try:
                    yt_result = extract_youtube(yt_url)
                except ValueError as e:
                    yt_result = None
                    st.error(str(e))

            if yt_result:
                input_text = yt_result["transcript"]

                # Metadata display
                meta_cols = st.columns(3)
                meta_cols[0].metric("Words", f"{yt_result['word_count']:,}")

                if yt_result.get("duration_seconds"):
                    mins = yt_result["duration_seconds"] // 60
                    secs = yt_result["duration_seconds"] % 60
                    meta_cols[1].metric("Duration", f"{mins}m {secs}s")

                lang = yt_result.get("language", "en")
                meta_cols[2].metric("Language", lang.upper())

                with st.expander("Preview transcript", expanded=False):
                    st.text(input_text[:3000] + ("..." if len(input_text) > 3000 else ""))

# ══════════════════════════════════════════════════════════════════════
#  TAB 5 — arXiv research paper
# ══════════════════════════════════════════════════════════════════════
with tab_arxiv:
    arxiv_query = st.text_input(
        "arXiv ID, URL, or DOI",
        placeholder="e.g. 1706.03762  or  https://arxiv.org/abs/1706.03762",
        key="arxiv_input",
    )
    if arxiv_query:
        with st.spinner("Fetching paper from arXiv..."):
            try:
                arxiv_result = fetch_arxiv(arxiv_query)
            except ValueError as e:
                arxiv_result = None
                st.error(str(e))

        if arxiv_result:
            # Paper metadata card
            st.markdown(
                f"""<div class="cfx-card" style="margin-bottom: 16px;">
                    <div style="font-family:'Manrope',sans-serif;font-weight:800;font-size:1.05rem;
                                color:#191b23;margin-bottom:6px;">
                        {arxiv_result['title']}
                    </div>
                    <div style="font-size:0.82rem;color:#727785;margin-bottom:4px;">
                        {', '.join(arxiv_result['authors'][:5])}
                        {'  +' + str(len(arxiv_result['authors']) - 5) + ' more'
                         if len(arxiv_result['authors']) > 5 else ''}
                    </div>
                    <div style="font-size:0.78rem;color:#0058be;font-weight:600;">
                        arXiv:{arxiv_result['arxiv_id']}
                        {'  ·  ' + arxiv_result['published'] if arxiv_result.get('published') else ''}
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

            # Choose what to summarise
            has_full = bool(arxiv_result.get("full_text"))
            options = ["Abstract only"]
            if has_full:
                options.append("Full paper")

            source_choice = st.radio(
                "Summarize from:",
                options,
                horizontal=True,
                key="arxiv_source",
            )

            if source_choice == "Full paper" and has_full:
                input_text = arxiv_result["full_text"]
            else:
                input_text = arxiv_result["abstract"]

            # Info about word counts
            meta_cols = st.columns(2)
            meta_cols[0].metric("Abstract words", len(arxiv_result["abstract"].split()))
            if has_full:
                meta_cols[1].metric("Full paper words", len(arxiv_result["full_text"].split()))

            with st.expander("Preview text", expanded=False):
                st.text(input_text[:3000] + ("..." if len(input_text) > 3000 else ""))


# ══════════════════════════════════════════════════════════════════════
#  MODEL SELECTION + GENERATION  (shared across all tabs)
# ══════════════════════════════════════════════════════════════════════
st.divider()

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
    cleaned = clean_text(input_text)

    with st.spinner(f"Generating summary with {get_model_label(model_key)}..."):
        if word_count > 500:
            st.info(f"Long document detected ({word_count} words). Using iterative chunk-and-merge strategy.")
            summary = summarize_long(cleaned, model_key)
        else:
            summary = summarize(cleaned, model_key)

    st.markdown('<div class="cfx-section-title">Summary</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="cfx-output-card"><div class="cfx-summary-text">{summary}</div></div>',
        unsafe_allow_html=True,
    )

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
