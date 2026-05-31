"""Vision-Augmented LLM Pipeline for Long-Form Document Summarization — Region Handlers"""

import logging
import tempfile
import os
from PIL import Image

logger = logging.getLogger(__name__)

def extract_text_region(page, bbox: tuple) -> str:
    """Extract text from a cropped pdfplumber page using a padded bounding box. strict=False ensures letters aren't chopped."""
    x0, y0, x1, y1 = bbox
    pad = 4 # Padding prevents dropping the first letter of words if bbox is slightly tight (like "Text" -> "Ext")
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(page.width, x1 + pad)
    y1 = min(page.height, y1 + pad)
    
    crop = page.within_bbox((x0, y0, x1, y1), strict=False)
    text = crop.extract_text()
    return text if text else ""

def caption_figure_region(image: Image.Image) -> str:
    from input_handlers.image_handler import _generate_caption
    try:
        desc = _generate_caption(image)
        return f"[FIGURE: {desc}]"
    except Exception as e:
        logger.error(f"Figure captioning failed: {e}")
        return "[FIGURE: Uncaptioned Image]"

def extract_table_region(page, bbox: tuple, img_crop) -> str:
    # First try img2table if installed
    try:
        from img2table.document import Image as I2TImage
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img_crop.save(f.name)
            tmp_path = f.name
        
        doc = I2TImage(tmp_path)
        tables = doc.extract_tables()
        os.unlink(tmp_path)
        if tables:
            t = tables[0]
            if hasattr(t, 'df'):
                return "[TABLE]\n" + t.df.to_markdown() + "\n[/TABLE]"
    except ImportError:
        logger.debug("img2table not available — trying pdfplumber fallback")
    except Exception as e:
        logger.debug(f"img2table extraction failed: {e} — trying pdfplumber fallback")
        
    # Fallback to pdfplumber
    try:
        # Increase padding slightly for table boundaries to ensure cell text isn't lost
        x0, y0, x1, y1 = bbox
        pad = 2
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(page.width, x1 + pad)
        y1 = min(page.height, y1 + pad)
        
        crop = page.within_bbox((x0, y0, x1, y1), strict=False)
        table = crop.extract_table()
        if table:
            rows = []
            for i, r in enumerate(table):
                clean_r = [str(x).replace('\n', ' ') if x else "" for x in r]
                rows.append("| " + " | ".join(clean_r) + " |")
                if i == 0:
                    rows.append("|" + "|".join(["---"] * len(r)) + "|")
            return "[TABLE]\n" + "\n".join(rows) + "\n[/TABLE]"
        else:
            return extract_text_region(page, bbox)
    except Exception as e:
        return "[TABLE: Extraction Failed]"

def extract_equation_region(img_crop: Image.Image) -> str:
    try:
        from pix2tex.cli import LatexOCR
        model = LatexOCR()
        latex = model(img_crop)
        return f"[EQUATION: {latex}]"
    except Exception:
        return "[EQUATION: Math Formula]"
