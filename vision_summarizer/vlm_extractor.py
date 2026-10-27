"""
Extracts structured room data from blueprint image.
Tier 1: Claude Vision API (best accuracy) — needs ANTHROPIC_API_KEY in the environment
Tier 2: LLaVA via Ollama (free, offline)
Tier 3: pytesseract OCR (text-only fallback)
"""

import os
import io
import base64
import json
import re
from PIL import Image

CLAUDE_MODEL = os.environ.get("COMPASS_CLAUDE_MODEL", "claude-opus-5-5")
# Claude Opus 5.5 reads images up to 2576 px on the long edge; larger
# images only cost upload time, so downscale before encoding.
_CLAUDE_MAX_EDGE = 2576

# ── Shared prompt ───────────────────────────────────────────
EXTRACTION_PROMPT = """
You are an expert architectural drawing analyst.
Analyze this house blueprint image carefully and extract the real data you see.

Return ONLY a single valid JSON object — no markdown, no explanation, no code fences.

CRITICAL RULES:
- NEVER output 0 or 0.0 as a placeholder. If a value is unknown, use null.
- NEVER copy the example field names/values — replace them with what you actually see.
- DO NOT add any text before or after the JSON.
- DO NOT use angle-bracket placeholders like <Room Name>; write the actual name.

JSON format (fill in real values from the image):
{
  "floors": [
    {
      "floor_number": 1,
      "floor_name": "Ground Floor",
      "rooms": [
        {
          "name": "Living Room",
          "width_ft": null,
          "length_ft": null,
          "area_sqft": null,
          "features": []
        }
      ],
      "total_area_sqft": null
    }
  ],
  "structure_type": "Residential",
  "num_bedrooms": null,
  "num_bathrooms": null,
  "num_floors": 1,
  "has_garage": false,
  "has_basement": false,
  "has_garden": false,
  "scale_detected": null,
  "special_features": [],
  "orientation": null,
  "confidence": 0.85
}

Additional extraction rules:
- If a room is visible but its dimensions are not legible, still list the room with null dimensions.
- area_sqft = width_ft * length_ft (calculate yourself if both are available).
- If dimensions are in millimetres (e.g. 3625x3375), convert to feet by dividing by 304.8.
- confidence = your estimate (0.0-1.0) of how clearly you could read this blueprint.
- List ALL rooms: bedrooms, bathrooms, kitchen, living room, hallways, storage, etc.
"""


# ── Tier 1: Claude Vision ───────────────────────────────────
def extract_with_claude(img: Image.Image) -> dict:
    """Extract room data using the Claude Vision API.

    The key is read from ANTHROPIC_API_KEY by the SDK; never hardcode it.
    The SDK already retries rate limits, 5xx errors and dropped connections.
    """
    import anthropic

    client = anthropic.Anthropic(max_retries=3)
    img_b64 = _pil_to_base64(_fit_for_claude(img))

    response = client.beta.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=16000,
        output_config={"effort": "high"},
        # If a safety classifier declines, re-run on Anthropic's
        # recommended fallback model instead of failing the extraction.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
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
    if response.stop_reason == "refusal":
        raise ValueError("Claude declined to analyse this image.")

    # Thinking blocks come before the answer, so pick the text block explicitly.
    raw = next((b.text for b in response.content if b.type == "text"), "")
    return _safe_parse_json(raw)


# ── Tier 2: LLaVA via Ollama (local, free) ──────────────────
def extract_with_llava(img: Image.Image) -> dict:
    """Extract room data using LLaVA model via local Ollama server."""
    import ollama

    img_b64 = _pil_to_base64(img)

    # Try progressively harder prompts if LLaVA returns the template verbatim
    for attempt, extra_hint in enumerate([
        "",
        "\n\nIMPORTANT: Look at the image carefully. List every room label you can read.",
        "\n\nList the room names you see printed on the floor plan, one per line, then provide the JSON.",
    ]):
        response = ollama.chat(
            model="llava:13b",
            messages=[{
                "role": "user",
                "content": EXTRACTION_PROMPT + extra_hint,
                "images": [img_b64]
            }],
            stream=False,
            options={
                "num_predict": 4096,   # prevent truncated JSON output
                "temperature": 0.1,    # lower temperature = more faithful extraction
            },
        )
        raw_text = response["message"]["content"]
        print(f"[LLaVA attempt {attempt+1} raw output]:", raw_text[:500])

        parsed = _safe_parse_json(raw_text)

        # Check if LLaVA actually extracted something useful
        # (not just a template stub with 0 rooms and 0.0 values)
        rooms_found = sum(
            len(f.get("rooms", [])) for f in parsed.get("floors", [])
        )
        confidence = parsed.get("confidence", 0)
        if rooms_found > 0 or (confidence and float(confidence) > 0.5):
            return parsed   # good extraction — use it

        if attempt < 2:
            print(f"[LLaVA] Attempt {attempt+1} returned empty/template result, retrying...")

    # All attempts returned an empty extraction — still return the last parse
    return parsed


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
def _fit_for_claude(img: Image.Image) -> Image.Image:
    """Downscale so the long edge is at most what Claude reads (keeps aspect)."""
    if max(img.size) <= _CLAUDE_MAX_EDGE:
        return img
    scaled = img.copy()
    scaled.thumbnail((_CLAUDE_MAX_EDGE, _CLAUDE_MAX_EDGE), Image.LANCZOS)
    return scaled


def _pil_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64-encoded JPEG string."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


def _safe_parse_json(text: str) -> dict:
    """
    Robust multi-strategy JSON parser for LLM outputs.
    Handles: markdown fences, trailing commas, truncated JSON, 
    extra prose before/after the JSON block.
    """
    import json

    # 1. Strip markdown code fences
    text = re.sub(r"```(?:json)?", "", text).strip()
    text = text.replace("```", "").strip()

    # 2. Isolate the JSON object (from first { to last })
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in LLaVA response")
    
    # Find the matching closing brace
    depth = 0
    end = start
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    else:
        # JSON was truncated — try to auto-close it
        end = len(text)

    candidate = text[start:end]

    # 3. Strategy A — standard json.loads
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # 4. Strategy B — fix trailing commas (common LLM mistake)
    #    e.g. {"a": 1,} or [1, 2,]
    fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # 5. Strategy C — try json5 library if available (very permissive)
    try:
        import json5
        return json5.loads(fixed)
    except (ImportError, Exception):
        pass

    # 6. Strategy D — auto-close truncated JSON by counting brackets
    try:
        open_braces   = fixed.count("{") - fixed.count("}")
        open_brackets = fixed.count("[") - fixed.count("]")
        repaired = fixed.rstrip(",\n\r\t ")
        repaired += "]" * max(0, open_brackets)
        repaired += "}" * max(0, open_braces)
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # 7. Final fallback — return a minimal stub so the UI never crashes
    print(f"[_safe_parse_json] All strategies failed. Raw snippet: {candidate[:300]}")
    return {
        "floors": [{"floor_number": 1, "floor_name": "Floor 1", "rooms": [],
                    "total_area_sqft": None}],
        "structure_type": "Unknown",
        "num_bedrooms": 0, "num_bathrooms": 0, "num_floors": 1,
        "has_garage": False, "has_basement": False, "has_garden": False,
        "confidence": 0.1,
        "_parse_error": True,
    }
