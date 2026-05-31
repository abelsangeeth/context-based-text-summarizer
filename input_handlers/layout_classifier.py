"""Vision-Augmented LLM Pipeline for Long-Form Document Summarization — Layout Classifier"""
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

_LP_MODEL = None

def get_layout_model():
    global _LP_MODEL
    if _LP_MODEL is None:
        try:
            import layoutparser as lp
            # Try Detectron2 first (Preferred but hard to install on Windows)
            try:
                import detectron2
                _LP_MODEL = lp.models.Detectron2LayoutModel(
                    'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
                    label_map={0: "text", 1: "title", 2: "list", 3: "table", 4: "figure"}
                )
                logger.info("Loaded LayoutParser Detectron2 (PubLayNet) model")
            except (ImportError, Exception) as e:
                logger.info(f"Detectron2 not available ({e}), trying EfficientDet...")
                # Fallback to EfficientDet (Easier to install on Windows: pip install effdet)
                try:
                    _LP_MODEL = lp.models.EfficientDetLayoutModel(
                        'lp://PubLayNet/tf_efficientdet_d1/config',
                        label_map={0: "text", 1: "title", 2: "list", 3: "table", 4: "figure"}
                    )
                    logger.info("Loaded LayoutParser EfficientDet model")
                except (ImportError, Exception) as e2:
                    logger.warning(f"No layout backend found. Detectron2 error: {e}. EfficientDet error: {e2}")
                    logger.warning("To fix: Install a backend like 'pip install effdet' or find Detectron2 Windows wheels.")
                    _LP_MODEL = "fallback"
        except ImportError:
            logger.error("layoutparser is not installed. Run: pip install layoutparser")
            _LP_MODEL = "fallback"
    return _LP_MODEL

def classify_layout(page_image, page_index=0) -> List[Dict[str, Any]]:
    """Detect regions in a PyMuPDF page pixmap."""
    model = get_layout_model()
    
    # If fallback, treat entire page as text
    if model == "fallback":
        width, height = page_image.size
        return [{"type": "text", "bbox": (0, 0, width, height), "score": 1.0}]
        
    layout = model.detect(page_image)
    regions = []
    for block in layout:
        x0, y0, x1, y1 = block.coordinates
        regions.append({
            "type": block.type.lower(),
            "bbox": (x0, y0, x1, y1),
            "score": block.score
        })
    return regions
