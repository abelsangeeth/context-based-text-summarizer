import html
import streamlit as st
import io

def render_longform_summary(longform: dict, selected_sec: str = "-- Document Overview --", show_word_count: bool = True, show_reading_time: bool = True):
    from summarizer.longform_builder import estimate_reading_time

    sections = longform["sections"]
    n_total = len(sections)

    html_sections = ""
    for idx, sec in enumerate(sections, start=1):
        para_label = f"Para {idx}"
        if selected_sec != "-- Document Overview --" and para_label != selected_sec:
            continue

        badge = ""
        if show_word_count:
            badge = f'<span style="font-size: 11px; color: #888; margin-top: 4px; display: inline-block;">{sec["word_count"]} words</span>'

        sec_html = f"""<div style="margin-bottom: 28px;">
<p style="font-size: 15px; line-height: 1.8; color: var(--text-color); margin: 0;">{html.escape(sec['summary_paragraph'])}</p>
{badge}
</div>"""
        html_sections += sec_html

    header = ""
    if show_reading_time and selected_sec == "-- Document Overview --":
        header = f"""<p style="font-size: 12px; color: #888; margin-bottom: 20px; border-left: 3px solid #000; padding-left: 10px;">
{longform['total_word_count']} words &middot; {estimate_reading_time(longform['total_word_count'])} &middot; {n_total} paragraphs &middot; summarized with {longform['model_used']}
</p>"""

    wrapper = f"""<div style="max-width: 860px; padding: 8px 0;">
{header}
{html_sections}
</div>"""

    st.markdown(wrapper, unsafe_allow_html=True)

def render_section_controls(longform: dict) -> str:
    # Jump nav using paragraph numbers instead of AI subheadings
    para_labels = [f"Para {i}" for i in range(1, len(longform["sections"]) + 1)]
    jump = st.selectbox("Jump to paragraph", ["-- Document Overview --"] + para_labels)
    return jump

def export_as_markdown(longform: dict) -> str:
    md = ""
    for sec in longform["sections"]:
        md += f"{sec['summary_paragraph']}\n\n"
    return md

def export_as_docx(longform: dict) -> bytes:
    try:
        from docx import Document
    except ImportError:
        # Fallback if python-docx not installed
        return b"Error: python-docx not installed"

    import re
    doc = Document()
    for sec in longform["sections"]:
        # python-docx requires strict XML-compatible strings; filter out control chars like NULL bytes
        clean_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', str(sec.get('summary_paragraph', '')))
        doc.add_paragraph(clean_text)
    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()
