"""
Extracts structured room data from blueprint image.
Tier 1: Claude Vision API (best accuracy)
Tier 2: LLaVA via Ollama (free, offline)
Tier 3: pytesseract OCR (text-only fallback)
"""

import os
import io
import base64
import json
import time
import re
from PIL import Image


# ── Shared prompt ───────────────────────────────────────────
EXTRACTION_PROMPT = """
You are an expert architectural drawing analyst.
Analyze this house blueprint image carefully.

Extract ALL visible information and return ONLY this exact JSON structure
(no markdown, no explanation, just raw JSON):

{
  "floors": [
    {
      "floor_number": 1,
      "floor_name": "Ground Floor",
      "rooms": [
        {
          "name": "Living Room",
          "width_ft": 14.0,
          "length_ft": 18.0,
          "area_sqft": 252.0,
          "features": ["bay window", "fireplace", "door to hallway"]
        }
      ],
      "total_area_sqft": 1200.0
    }
  ],
  "structure_type": "Single-family residential",
  "num_bedrooms": 3,
  "num_bathrooms": 2,
  "num_floors": 1,
  "has_garage": false,
  "has_basement": false,
  "has_garden": false,
  "scale_detected": "1/4 inch = 1 foot",
  "special_features": ["open floor plan", "attached garage"],
  "orientation": "North-facing",
  "confidence": 0.90
}

Rules:
- If dimension is not readable, use null (not 0)
- area_sqft = width_ft * length_ft (calculate it yourself)
- List ALL rooms visible including hallways, bathrooms, storage
- confidence = your estimate of how clearly you could read the blueprint
"""


# ── Tier 1: Claude Vision ───────────────────────────────────
def extract_with_claude(img: Image.Image) -> dict:
    """Extract room data using Claude Vision API with retry logic."""
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    img_b64 = _pil_to_base64(img)

    for attempt in range(3):
        try:
            response = client.messages.create(
                model="claude-opus-4-5",
                max_tokens=3000,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image",
                         "source": {"type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": img_b64}},
                        {"type": "text", "text": EXTRACTION_PROMPT}
                    ]
                }]
            )
            raw = response.content[0].text
            return _safe_parse_json(raw)

        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)   # exponential backoff
            else:
                raise e


# ── Tier 2: LLaVA via Ollama (local, free) ──────────────────
def extract_with_llava(img: Image.Image) -> dict:
    """Extract room data using LLaVA model via local Ollama server."""
    import ollama

    img_b64 = _pil_to_base64(img)
    response = ollama.chat(
        model="llava:13b",
        messages=[{
            "role": "user",
            "content": EXTRACTION_PROMPT,
            "images": [img_b64]
        }]
    )
    return _safe_parse_json(response["message"]["content"])


# ── Tier 3: pytesseract OCR fallback ───────────────────────
def extract_with_ocr(img: Image.Image) -> dict:
    """
    Text-only fallback. Extracts room labels and numbers
    but cannot infer spatial layout.
    """
    import pytesseract

    text = pytesseract.image_to_string(img, config="--psm 6")
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    room_keywords = [
        "bedroom", "bathroom", "kitchen", "living", "dining",
        "hallway", "garage", "study", "laundry", "storage",
        "lounge", "toilet", "pantry", "entrance", "porch"
    ]
    rooms_found = []
    for line in lines:
        lower = line.lower()
        for kw in room_keywords:
            if kw in lower:
                dims = re.findall(
                    r"(\d+\.?\d*)\s*[x\u00d7'\"]\s*(\d+\.?\d*)", lower
                )
                w = float(dims[0][0]) if dims else None
                l = float(dims[0][1]) if dims else None
                rooms_found.append({
                    "name": line.title(),
                    "width_ft": w,
                    "length_ft": l,
                    "area_sqft": round(w * l, 1) if w and l else None,
                    "features": []
                })
                break

    return {
        "floors": [{"floor_number": 1,
                     "floor_name": "Floor 1",
                     "rooms": rooms_found,
                     "total_area_sqft": None}],
        "structure_type": "Unknown (OCR fallback)",
        "num_bedrooms": sum(1 for r in rooms_found
                            if "bedroom" in r["name"].lower()),
        "num_bathrooms": sum(1 for r in rooms_found
                             if "bathroom" in r["name"].lower()),
        "confidence": 0.40,
        "_extraction_method": "ocr_fallback"
    }


# ── Master extractor — auto-selects best available ─────────
def extract_with_best_available(img: Image.Image,
                                 force_method: str = "auto") -> dict:
    """
    Tries extraction engines in order of quality.
    force_method: "auto" | "claude" | "llava" | "ocr"
    """
    if force_method == "claude":
        result = extract_with_claude(img)
        result["_extraction_method"] = "claude_vision"
        return result

    if force_method == "llava":
        result = extract_with_llava(img)
        result["_extraction_method"] = "llava_local"
        return result

    if force_method == "ocr":
        return extract_with_ocr(img)

    # AUTO — try in order
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            result = extract_with_claude(img)
            result["_extraction_method"] = "claude_vision"
            return result
        except Exception:
            pass

    try:
        import ollama
        ollama.list()   # check if ollama is running
        result = extract_with_llava(img)
        result["_extraction_method"] = "llava_local"
        return result
    except Exception:
        pass

    result = extract_with_ocr(img)
    return result


# ── Helpers ─────────────────────────────────────────────────
def _pil_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64-encoded JPEG string."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


def _safe_parse_json(text: str) -> dict:
    """Strips markdown fences, finds JSON object, parses it."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1:
        raise ValueError("No JSON found in VLM response")
    return json.loads(text[start:end])
