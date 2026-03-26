"""
ContextFree-X — Image Summarization Handler
════════════════════════════════════════════
Generates a natural-language description of any image and
detects all visible objects with confidence scores.

Uses two HuggingFace models (downloaded on first use):
  • Salesforce/blip-image-captioning-large  — image captioning
  • facebook/detr-resnet-50                 — object detection
"""

import io
from collections import Counter
from PIL import Image

import torch
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    DetrImageProcessor,
    DetrForObjectDetection,
)


# ── Model cache (lazy-loaded singletons) ──────────────────────────────
_blip_processor = None
_blip_model = None
_detr_processor = None
_detr_model = None


def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_blip():
    global _blip_processor, _blip_model
    if _blip_processor is None:
        _blip_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        )
        _blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        ).to(_get_device())
        _blip_model.eval()
    return _blip_processor, _blip_model


def _load_detr():
    global _detr_processor, _detr_model
    if _detr_processor is None:
        _detr_processor = DetrImageProcessor.from_pretrained(
            "facebook/detr-resnet-50"
        )
        _detr_model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50"
        ).to(_get_device())
        _detr_model.eval()
    return _detr_processor, _detr_model


# ═══════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════
def describe_image(uploaded_file, confidence_threshold: float = 0.7) -> dict:
    """
    Analyze an uploaded image: generate a caption and detect objects.

    Parameters
    ----------
    uploaded_file : UploadedFile or file-like
        Image file (JPG, PNG, WEBP, BMP).
    confidence_threshold : float
        Minimum confidence for DETR detections (0.0–1.0). Default 0.7.

    Returns
    -------
    dict with keys:
        type, source, caption, objects, object_count,
        raw_detections, summary_text, word_count
    """
    filename = getattr(uploaded_file, "name", "image")

    # Read image
    try:
        img = Image.open(uploaded_file).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Could not open image: {exc}") from exc

    # ── Step 1: Generate caption with BLIP ────────────────────────
    caption = _generate_caption(img)

    # ── Step 2: Detect objects with DETR ──────────────────────────
    raw_detections = _detect_objects(img, confidence_threshold)

    # ── Step 3: Deduplicate and summarise detections ──────────────
    objects_summary = _summarise_detections(raw_detections)
    object_count = sum(obj["count"] for obj in objects_summary)

    # ── Step 4: Build readable summary text ───────────────────────
    summary_text = _build_summary_text(caption, objects_summary)
    word_count = len(summary_text.split())

    return {
        "type": "image",
        "source": filename,
        "caption": caption,
        "objects": objects_summary,
        "object_count": object_count,
        "raw_detections": raw_detections,
        "summary_text": summary_text,
        "word_count": word_count,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════════════
def _generate_caption(img: Image.Image) -> str:
    """Generate a natural-language caption using BLIP."""
    processor, model = _load_blip()
    device = _get_device()

    # Unconditional captioning
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=80,
            num_beams=5,
            early_stopping=True,
        )
    caption = processor.decode(out[0], skip_special_tokens=True).strip()

    # Capitalize first letter
    if caption:
        caption = caption[0].upper() + caption[1:]

    return caption


def _detect_objects(
    img: Image.Image,
    threshold: float = 0.7,
) -> list[dict]:
    """
    Detect objects using DETR. Returns list of dicts:
    [{"label": str, "confidence": float, "box": [x1, y1, x2, y2]}, ...]
    """
    processor, model = _load_detr()
    device = _get_device()

    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process: convert outputs to COCO API format
    target_sizes = torch.tensor([img.size[::-1]]).to(device)
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=threshold
    )[0]

    detections = []
    for score, label_id, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        detections.append({
            "label": model.config.id2label[label_id.item()],
            "confidence": round(score.item(), 3),
            "box": [round(c.item(), 1) for c in box],
        })

    # Sort by confidence descending
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


def _summarise_detections(raw_detections: list[dict]) -> list[dict]:
    """
    Group detections by label, count occurrences, track max confidence.
    Returns a list sorted by count descending, then confidence descending.
    """
    counter = Counter()
    max_conf: dict[str, float] = {}

    for det in raw_detections:
        label = det["label"]
        counter[label] += 1
        max_conf[label] = max(max_conf.get(label, 0.0), det["confidence"])

    summary = [
        {
            "label": label,
            "count": count,
            "max_confidence": max_conf[label],
        }
        for label, count in counter.most_common()
    ]
    return summary


def _build_summary_text(caption: str, objects_summary: list[dict]) -> str:
    """Combine caption + object list into a cohesive paragraph."""
    parts = [f"Image description: {caption}."]

    if objects_summary:
        obj_strs = []
        for obj in objects_summary:
            if obj["count"] > 1:
                obj_strs.append(
                    f"{obj['count']}x {obj['label']} "
                    f"(confidence: {obj['max_confidence']:.0%})"
                )
            else:
                obj_strs.append(
                    f"{obj['label']} "
                    f"(confidence: {obj['max_confidence']:.0%})"
                )

        parts.append(
            f"Detected {sum(o['count'] for o in objects_summary)} object(s): "
            + ", ".join(obj_strs)
            + "."
        )
    else:
        parts.append("No specific objects detected above the confidence threshold.")

    return " ".join(parts)
