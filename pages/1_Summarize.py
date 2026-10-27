"""Page 1 — Summarize: Input text, PDF, URL, YouTube, arXiv, or image and get a summary."""

import html
import streamlit as st
from summarizer.models import list_models, get_model_label, load_model
from summarizer.pipeline import summarize, summarize_long
from input_handlers.pdf_handler import extract_pdf
from input_handlers.url_handler import scrape_url
from input_handlers.youtube_handler import extract_youtube, is_valid_youtube_url
from input_handlers.arxiv_handler import fetch_arxiv
from input_handlers.image_handler import describe_image
from input_handlers.cleaner import clean_text
from evaluation.aggregator import evaluate_all, abstractiveness_score, compression_ratio
from components.stitch_theme import apply_stitch_theme, page_header, sidebar_brand, sidebar_nav, sidebar_footer
from app_history import save_history

st.set_page_config(page_title="Summarize — COMPASS", page_icon="✍️", layout="wide")
apply_stitch_theme()

with st.sidebar:
    sidebar_brand()
    sidebar_nav(active="summarize")
    sidebar_footer()

page_header(
    "Summarize",
    "Paste text, upload a PDF, enter a URL, paste a YouTube link, fetch an arXiv paper, or upload an image — get a high-quality summary instantly.",
)

# --- Input method ---
tab_text, tab_pdf, tab_url, tab_yt, tab_arxiv, tab_img = st.tabs(
    ["📝 Text", "📄 PDF", "🔗 URL", "🎬 YouTube", "📚 arXiv", "🖼️ Image"]
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
                        {html.escape(arxiv_result['title'])}
                    </div>
                    <div style="font-size:0.82rem;color:#727785;margin-bottom:4px;">
                        {html.escape(', '.join(arxiv_result['authors'][:5]))}
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
#  TAB 6 — Image summarization (BLIP + DETR)
# ══════════════════════════════════════════════════════════════════════
with tab_img:
    img_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        key="img_input",
    )
    if img_file:
        # Show image preview
        st.image(img_file, caption=img_file.name, width='stretch')

        with st.spinner("Analyzing image with AI (first run downloads models ~2 GB)..."):
            try:
                img_result = describe_image(img_file)
            except ValueError as e:
                img_result = None
                st.error(str(e))

        if img_result:
            # Caption card
            st.markdown(
                f"""<div class="cfx-card" style="margin-bottom: 16px;">
                    <div class="cfx-metric-label">AI Caption</div>
                    <div style="font-family:'Inter',sans-serif;font-size:1.05rem;
                                color:#191b23;line-height:1.6;margin-top:6px;">
                        {html.escape(img_result['caption'])}
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

            # Object detection metrics
            obj_count = img_result["object_count"]
            unique_labels = len(img_result["objects"])

            mc1, mc2 = st.columns(2)
            mc1.metric("Objects Detected", obj_count)
            mc2.metric("Unique Labels", unique_labels)

            # Tags for detected objects
            if img_result["objects"]:
                tags_html = ""
                for obj in img_result["objects"]:
                    count_str = f" ×{obj['count']}" if obj['count'] > 1 else ""
                    conf_pct = f"{obj['max_confidence']:.0%}"
                    tags_html += (
                        f'<span class="cfx-tag cfx-tag-primary" style="margin:3px;">'
                        f'{obj["label"]}{count_str}'
                        f'<span style="opacity:0.6;font-size:10px;margin-left:4px;">{conf_pct}</span>'
                        f'</span>'
                    )
                st.markdown(
                    f'<div style="margin:12px 0;">{tags_html}</div>',
                    unsafe_allow_html=True,
                )

            # Set the summary text as input for optional text summarization
            input_text = img_result["summary_text"]

            with st.expander("Full description text", expanded=False):
                st.text(input_text)


# ══════════════════════════════════════════════════════════════════════
#  STRUCTURED LONG-FORM GENERATION  (shared across all tabs)
# ══════════════════════════════════════════════════════════════════════
st.divider()

from summarizer.section_detector import detect_sections, merge_short_sections
from summarizer.longform_builder import build_longform_summary
from input_handlers.cleaner import clean_text as deep_clean
from components.longform_renderer import (render_longform_summary,
                                          render_section_controls,
                                          export_as_markdown,
                                          export_as_docx)

# Sidebar controls
with st.sidebar:
    st.markdown("### Rendering Controls")
    length_mode = st.radio(
        "Summary length",
        ["detailed", "standard", "brief"],
        index=0,
        help="Detailed = long paragraphs per section"
    )
    available = [m for m in list_models() if m != "mt5"]
    model_name = st.selectbox(
        "Model",
        available,
        format_func=get_model_label
    )
    show_wc = st.checkbox("Show word counts", value=True)

# ── Detect model / length_mode changes → invalidate stale summary ──────────────
_prev_model  = st.session_state.get("_last_model", None)
_prev_mode   = st.session_state.get("_last_length_mode", None)

if _prev_model is not None and (_prev_model != model_name or _prev_mode != length_mode):
    # Settings changed since last generation — wipe cached result
    st.session_state.longform_summary = None

# Always persist current selections so the next run can compare
st.session_state["_last_model"]       = model_name
st.session_state["_last_length_mode"] = length_mode

word_count = len(input_text.split()) if input_text.strip() else 0

if word_count > 0:
    st.caption(f"Input: **{word_count}** words · **{len(input_text)}** characters")

    # Warn when model/mode changed and old summary was wiped
    if (
        word_count >= 10
        and st.session_state.get("longform_summary") is None
        and _prev_model is not None
    ):
        st.warning(
            f"⚠️ Settings changed to **{get_model_label(model_name)} / {length_mode}**. "
            "Click **Generate Structured Summary** to regenerate.",
            icon="🔄",
        )

    if st.button("Generate Structured Summary", type="primary", width='stretch', disabled=word_count < 10):
        # Reset any past generation
        st.session_state.longform_summary = None

        # Detect sections on ORIGINAL text (preserves \n\n paragraph breaks)
        # Then clean each section's text individually.
        with st.spinner("Detecting sections..."):
            sections = detect_sections(input_text, source_type="auto")

            # Deep-clean each section's raw text individually
            for sec in sections:
                sec["raw_text"] = deep_clean(sec["raw_text"])
                sec["word_count"] = len(sec["raw_text"].split())

            # Drop empty sections that became blank after cleaning
            sections = [s for s in sections if s["word_count"] >= 10]

            # Fix 3.3 — Merge sections under 80 words into their neighbour
            sections = merge_short_sections(sections, min_words=80)

            if not sections:
                st.error("Not enough readable text left after cleaning to summarize.")
                st.stop()

            st.toast(f"✅ {len(sections)} sections detected")

        # Pre-load model explicitly so user sees it loading (cache handles reuse)
        with st.spinner(f"Loading **{get_model_label(model_name)}** model..."):
            load_model(model_name)

        with st.spinner(f"Summarizing {len(sections)} sections with **{get_model_label(model_name)}**..."):
            # Subheadings are not rendered (paragraph view), so they are not
            # generated either: that saves one model call per section.
            longform = build_longform_summary(
                sections=sections,
                model_name=model_name,
                length_mode=length_mode
            )
            # Save the whole summary, not just the start of the first paragraph
            save_history(
                input_text,
                "\n\n".join(sec["summary_paragraph"] for sec in longform["sections"]),
                model_name,
            )

            # Save into session state
            st.session_state.longform_summary = longform

# Render if available in state
if "longform_summary" in st.session_state and st.session_state.longform_summary is not None:
    lf = st.session_state.longform_summary

    st.markdown('<div class="cfx-section-title">Structured Long-Form Summary</div>', unsafe_allow_html=True)

    # Model + mode badge so the user can always see what produced this summary
    _used_model = get_model_label(lf.get("model_used", ""))
    _used_mode  = lf.get("length_mode", "")
    _used_wc    = lf.get("total_word_count", 0)
    st.caption(
        f"🤖 **{_used_model}** · 📏 {_used_mode.capitalize()} · 📝 {_used_wc} words output"
    )

    selected_sec = render_section_controls(lf)

    # Styled card background
    st.markdown('<div class="cfx-output-card" style="padding: 24px; background: #ffffff;">', unsafe_allow_html=True)
    render_longform_summary(lf, selected_sec=selected_sec, show_word_count=show_wc)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇️ Download Markdown",
            data=export_as_markdown(lf),
            file_name="structured_summary.md",
            mime="text/markdown",
            width='stretch'
        )
    with col2:
        st.download_button(
            "⬇️ Download Word Doc",
            data=export_as_docx(lf),
            file_name="structured_summary.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            width='stretch'
        )
