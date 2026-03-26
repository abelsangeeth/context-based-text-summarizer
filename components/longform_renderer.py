import streamlit as st
import io

def render_longform_summary(longform: dict, selected_sec: str = "-- Document Overview --", show_word_count: bool = True, show_reading_time: bool = True):
    from summarizer.longform_builder import estimate_reading_time
    
    html_sections = ""
    for sec in longform["sections"]:
        if selected_sec != "-- Document Overview --" and sec["subheading"] != selected_sec:
            continue
            
        badge = ""
        if show_word_count:
            badge = f'<span style="font-size: 11px; color: #888; margin-top: 4px; display: inline-block;">{sec["word_count"]} words</span>'
            
        sec_html = f"""<div style="margin-bottom: 28px;">
<h3 style="color: #000000 !important; font-weight: 700; font-size: 18px; margin-bottom: 8px; border-bottom: 2px solid #000000; padding-bottom: 4px;">{sec['subheading']}</h3>
<p style="font-size: 15px; line-height: 1.8; color: var(--text-color); margin: 0;">{sec['summary_paragraph']}</p>
{badge}
</div>"""
        html_sections += sec_html
        
    header = ""
    if show_reading_time and selected_sec == "-- Document Overview --":
        header = f"""<p style="font-size: 12px; color: #888; margin-bottom: 20px; border-left: 3px solid #000; padding-left: 10px;">
{longform['total_word_count']} words &middot; {estimate_reading_time(longform['total_word_count'])} &middot; {len(longform['sections'])} sections &middot; summarized with {longform['model_used']}
</p>"""
        
    wrapper = f"""<div style="max-width: 860px; padding: 8px 0;">
{header}
{html_sections}
</div>"""
    
    st.markdown(wrapper, unsafe_allow_html=True)

def render_section_controls(longform: dict) -> str:
    # A tiny jump nav for UX
    headings = [s["subheading"] for s in longform["sections"]]
    jump = st.selectbox("Jump to section", ["-- Document Overview --"] + headings)
    return jump

def export_as_markdown(longform: dict) -> str:
    md = ""
    for sec in longform["sections"]:
        md += f"## {sec['subheading']}\n\n{sec['summary_paragraph']}\n\n"
    return md

def export_as_docx(longform: dict) -> bytes:
    try:
        from docx import Document
        from docx.shared import RGBColor
    except ImportError:
        # Fallback if python-docx not installed
        return b"Error: python-docx not installed"
        
    doc = Document()
    for sec in longform["sections"]:
        heading = doc.add_heading(sec['subheading'], level=2)
        for run in heading.runs:
            run.font.color.rgb = RGBColor(0, 0, 0)
            run.bold = True
            
        doc.add_paragraph(sec['summary_paragraph'])
        
    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()
