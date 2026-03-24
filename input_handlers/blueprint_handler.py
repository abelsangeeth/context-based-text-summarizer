"""
ContextFree-X — Blueprint Handler
══════════════════════════════════
Wraps the ``vision_summarizer/`` module as a unified input handler
so that architectural blueprints plug cleanly into the router.

Supported formats: JPG, JPEG, PNG, PDF (blueprint), DXF, DWG
"""

from input_handlers.cleaner import clean_text, chunk_text


# ── Format classification ──────────────────────────────────────────────────
_IMAGE_EXTS = {"jpg", "jpeg", "png", "webp", "bmp", "tiff"}
_CAD_EXTS = {"dxf", "dwg"}
_PDF_EXT = {"pdf"}


# ═══════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════
def extract_blueprint(
    uploaded_file,
    extraction_method: str = "auto",
) -> dict:
    """
    Process an uploaded blueprint image/PDF/CAD file through the full
    vision summariser pipeline.

    Parameters
    ----------
    uploaded_file : streamlit.UploadedFile
        Blueprint file (JPG/PNG/PDF/DXF/DWG).
    extraction_method : str
        VLM method override: "auto", "claude", "llava", "ocr".

    Returns
    -------
    dict
        type, source, format, pages_detected, extracted_data,
        plain_english, summary_target, chunks, word_count,
        room_count, total_area_sqft
    """
    filename = getattr(uploaded_file, "name", "blueprint")
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    # Determine format category
    if ext in _IMAGE_EXTS:
        fmt = "image"
    elif ext in _CAD_EXTS:
        fmt = "cad"
    elif ext in _PDF_EXT:
        fmt = "pdf"
    else:
        raise ValueError(
            f"Unsupported blueprint format '.{ext}'. "
            "Supported: JPG, PNG, PDF, DXF, DWG."
        )

    # ── Import vision_summarizer components ─────────────────
    from vision_summarizer.blueprint_reader import load_blueprint
    from vision_summarizer.image_preprocessor import preprocess_blueprint
    from vision_summarizer.vlm_extractor import extract_with_best_available
    from vision_summarizer.area_calculator import calculate_all_areas
    from vision_summarizer.blueprint_summarizer import json_to_plain_english

    # ── Load blueprint → list of PIL Images ─────────────────
    try:
        images = load_blueprint(uploaded_file)
    except Exception as exc:
        raise ValueError(
            f"Could not load blueprint: {exc}"
        ) from exc

    pages_detected = len(images)

    # ── Process the first page (primary blueprint) ──────────
    img = images[0]
    processed_img = preprocess_blueprint(img)

    # ── Extract structured data with VLM ────────────────────
    try:
        extracted_data = extract_with_best_available(
            processed_img, force_method=extraction_method
        )
    except Exception as exc:
        raise ValueError(
            "Could not extract room data. "
            "Try a clearer image or different engine."
        ) from exc

    # ── Calculate areas ─────────────────────────────────────
    extracted_data = calculate_all_areas(extracted_data)

    # ── Generate plain English description ──────────────────
    plain_english = json_to_plain_english(extracted_data)
    plain_english = clean_text(plain_english)

    # ── Count rooms ─────────────────────────────────────────
    room_count = 0
    total_area_sqft = extracted_data.get("grand_total_sqft")
    for floor in extracted_data.get("floors", []):
        room_count += len(floor.get("rooms", []))

    # ── Chunk for downstream NLP ────────────────────────────
    chunks = chunk_text(plain_english, max_words=800, overlap=100)
    word_count = len(plain_english.split())

    return {
        "type": "blueprint",
        "source": filename,
        "format": fmt,
        "pages_detected": pages_detected,
        "extracted_data": extracted_data,
        "plain_english": plain_english,
        "summary_target": plain_english,
        "chunks": chunks,
        "word_count": word_count,
        "room_count": room_count,
        "total_area_sqft": total_area_sqft,
    }
