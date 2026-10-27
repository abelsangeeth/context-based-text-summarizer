"""Vision-Augmented LLM Pipeline for Long-Form Document Summarization — PDF Handler (Vision Pipeline)"""
import os
import re
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import logging

from input_handlers.layout_classifier import classify_layout
from input_handlers.region_handlers import (
    extract_text_region,
    extract_table_region,
    caption_figure_region,
    extract_equation_region
)

logger = logging.getLogger(__name__)

def render_pages(pdf_path: str, dpi: int = 200) -> list[dict]:
    """Render each page of the PDF into a high-resolution pixmap."""
    fitz.TOOLS.mupdf_display_errors(False) # Suppress mupdf structural warnings
    doc = fitz.open(pdf_path)
    pages = []

    # Check if doc has any text elements to gauge if it's research/dense
    for i, page in enumerate(doc):
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)

        # Convert to PIL immediately
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        pages.append({
            "page_num": i + 1,
            "image": img,
            "width": pix.width,
            "height": pix.height
        })
    doc.close()
    return pages

def _assemble_document(pdf_path: str, pages_data: list[dict] | None) -> str:
    """Classify regions and extract content, assembling it in reading order.

    Returns text with the PDF's line breaks still in place so section headings
    can be found on their own lines; run _collapse_lines() on it afterwards.
    ``pages_data`` is only needed (and only rendered) when a layout model is installed.
    """
    full_text_blocks = []

    if not pages_data:
        # Fallback to PyMuPDF native extraction which perfectly handles two-column layouts
        doc = fitz.open(pdf_path)
        for page in doc:
            # get_text("text") natively understands column layouts and returns properly ordered text
            text = page.get_text("text")
            if text and text.strip():
                full_text_blocks.append(text.strip())
        doc.close()
    else:
        # Open pdfplumber once per document to avoid heavy I/O
        with pdfplumber.open(pdf_path) as pl_doc:
            for pdata in pages_data:
                img = pdata["image"]
                page_num = pdata["page_num"]
                pl_page = pl_doc.pages[page_num - 1]

                regions = classify_layout(img, page_num)

                # Sort reading order (y0 primary, x0 secondary for multi-column)
                # Rough column grouping: if x0s are far apart, they are different columns
                regions.sort(key=lambda b: (round(b["bbox"][0] / 300), b["bbox"][1]))

                for region in regions:
                    rtype = region["type"]
                    bbox = region["bbox"]

                    content = ""
                    if rtype in ["text", "title"]:
                        content = extract_text_region(pl_page, bbox)
                        # Clean markdown injected headers
                        if rtype == "title" and content:
                            content = f"\n## {content.strip()}\n"
                    elif rtype == "table":
                        # Crop image for img2table
                        x0, y0, x1, y1 = bbox
                        crop_img = img.crop((x0, y0, x1, y1))
                        content = extract_table_region(pl_page, bbox, crop_img)
                    elif rtype == "figure":
                        x0, y0, x1, y1 = bbox
                        crop_img = img.crop((x0, y0, x1, y1))
                        content = caption_figure_region(crop_img)
                    elif rtype == "equation":
                        x0, y0, x1, y1 = bbox
                        crop_img = img.crop((x0, y0, x1, y1))
                        content = extract_equation_region(crop_img)

                    if content and content.strip():
                        full_text_blocks.append(content)

    return "\n\n".join(full_text_blocks)


def _collapse_lines(text: str) -> str:
    """Join PDF line breaks into running text (keeps '## ' heading markers)."""
    text = re.sub(r'-\n', '', text) # Join hyphenated splits
    return re.sub(r'\n(?!\n|#)', ' ', text) # Remove single stray newlines

def _extract_sections(text: str) -> dict:
    """Isolate Abstract, Introduction, and Conclusion from research papers.

    *text* must still contain the PDF's line breaks: headings are matched as
    whole lines, which is impossible once lines are joined into running text.
    Each returned section is collapsed into running text.
    """
    sections = {
        "abstract": "",
        "introduction": "",
        "conclusion": "",
        "full_text": text
    }

    # Headings on their own line, optionally numbered (1 / I.) and optionally
    # prefixed by the layout path's '## ' markers. The abstract may also be
    # inline, IEEE style: "Abstract—This paper ...".
    abstract_start = re.search(r'(?im)^(?:#+\s*)?abstract\s*(?:$|[\u2014\u2013:.-]\s*)', text)
    intro_start = re.search(r'(?im)^(?:#+\s*)?(?:(?:1|I)\.?\s*)?introduction\s*$', text)
    conc_start = re.search(r'(?im)^(?:#+\s*)?(?:(?:\d{1,2}|[IVX]{1,4})\.?\s*)?conclusions?(?:\s+and\s+future\s+work)?\s*$', text)
    refs_start = re.search(r'(?im)^(?:#+\s*)?(?:references|bibliography)\s*$', text)

    # Slice out Abstract
    if abstract_start and intro_start:
        sections["abstract"] = text[abstract_start.end():intro_start.start()].strip()
    elif abstract_start:
        sections["abstract"] = text[abstract_start.end():abstract_start.end()+2000].strip()

    # Slice out Intro
    if intro_start:
        # Section 2's heading: a short line starting "2" / "II" and a capital letter
        next_sec = re.search(r'(?m)^(?:#+\s*)?(?:2|II)\.?\s+[A-Z][^\n]{0,80}$', text[intro_start.end():])
        if next_sec:
            sections["introduction"] = text[intro_start.end():intro_start.end()+next_sec.start()].strip()
        else:
            sections["introduction"] = text[intro_start.end():intro_start.end()+3000].strip()

    # Slice out Conclusion
    if conc_start:
        if refs_start:
            sections["conclusion"] = text[conc_start.end():refs_start.start()].strip()
        else:
            sections["conclusion"] = text[conc_start.end():].strip()

    return {k: _collapse_lines(v).strip() for k, v in sections.items()}

def extract_pdf(pdf_input) -> dict:
    """Main entrypoint for PDF handling logic. Utilizes the vision pipeline."""
    import tempfile

    is_file_obj = hasattr(pdf_input, "read")
    source_name = getattr(pdf_input, "name", "unknown.pdf") if is_file_obj else os.path.basename(pdf_input)

    logger.info(f"Extracting PDF: {source_name}")

    if is_file_obj:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_input.read())
            pdf_path = tmp.name
    else:
        pdf_path = pdf_input

    try:
        from input_handlers.layout_classifier import get_layout_model

        # Page images are only consumed by the layout model. Without one, PyMuPDF
        # reads the text layer directly, and rendering every page at 200 DPI would
        # only cost ~15 MB of RAM per page for nothing.
        pages_data = None
        if get_layout_model() != "fallback":
            pages_data = render_pages(pdf_path, dpi=200)
        raw_text = _assemble_document(pdf_path, pages_data)

        from input_handlers.cleaner import chunk_text

        sections = _extract_sections(raw_text)
        assembled_text = sections["full_text"]
        is_research = bool(sections["abstract"] or sections["introduction"] or sections["conclusion"])
        if not is_research:
            sections = {"full_text": assembled_text}
        chunks = chunk_text(assembled_text)

        if pages_data is not None:
            page_count = len(pages_data)
        else:
            with fitz.open(pdf_path) as doc:
                page_count = doc.page_count

        return {
            "type": "pdf",
            "source": source_name,
            "is_research_paper": is_research,
            "pages": page_count,
            "sections": sections,
            "chunks": chunks,
            "word_count": len(assembled_text.split())
        }
    finally:
        if is_file_obj and os.path.exists(pdf_path):
            try:
                os.unlink(pdf_path)
            except Exception as e:
                logger.warning(f"Could not delete temp PDF file {pdf_path}: {e}")

def get_summary_target(result: dict) -> str:
    """Determine the primary text segment to send for initial summarization."""
    sections = result.get("sections", {})
    if result.get("is_research_paper") and "abstract" in sections and sections["abstract"]:
        return sections["abstract"]

    chunks = result.get("chunks", [])
    if chunks:
        return chunks[0]
    return sections.get("full_text", "")
