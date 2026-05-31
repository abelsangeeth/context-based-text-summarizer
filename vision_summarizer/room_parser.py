"""
JSON cleaner + validator for VLM extraction output.
Ensures all required fields exist and have sane values.
Handles LLaVA quirks: placeholder text, metric (mm) dimensions.
"""

# Strings that indicate LLaVA copied the schema template rather than reading the image
_PLACEHOLDER_STRINGS = {
    "<floor name>", "<actual room name>", "<feature_1>", "<feature_2>",
    "<feature>", "<e.g., single-family, apartment>", "<scale string if visible>",
    "<e.g. north-facing>",
}

_MM_TO_FT = 1 / 304.8  # 1 mm = 0.00328084 ft


def _is_placeholder(value) -> bool:
    """Return True if a string value looks like an unfilled template placeholder."""
    if not isinstance(value, str):
        return False
    v = value.strip().lower()
    return v in _PLACEHOLDER_STRINGS or (v.startswith("<") and v.endswith(">"))


def _sanitize_numeric(val, mm_scale: bool = False):
    """
    Return a float or None.
    If mm_scale=True and the value looks like millimetres (> 100), convert to feet.
    """
    if val is None:
        return None
    try:
        v = float(val)
    except (TypeError, ValueError):
        return None
    if v == 0.0:
        return None
    if mm_scale and v > 100:          # almost certainly mm, not feet
        v = round(v * _MM_TO_FT, 2)
    return v


def validate_and_clean(data: dict) -> dict:
    """
    Validates VLM JSON output, fills missing fields with defaults,
    strips LLaVA template placeholders, auto-converts mm→ft,
    removes malformed rooms, and returns a cleaned dict.
    """
    if not isinstance(data, dict):
        return _empty_blueprint()

    # Ensure floors list exists
    if "floors" not in data or not isinstance(data["floors"], list):
        data["floors"] = []

    cleaned_floors = []
    for floor in data["floors"]:
        if not isinstance(floor, dict):
            continue

        floor.setdefault("floor_number", len(cleaned_floors) + 1)
        # Strip placeholder floor names
        floor_name = floor.get("floor_name", "")
        if _is_placeholder(floor_name) or not floor_name:
            floor["floor_name"] = f"Floor {floor['floor_number']}"
        floor.setdefault("total_area_sqft", None)

        clean_rooms = []
        for room in floor.get("rooms", []):
            if not isinstance(room, dict):
                continue

            name = room.get("name", "")
            # Skip rooms whose name is a template placeholder
            if not name or _is_placeholder(name):
                continue

            # Sanitize numeric dimensions — detect if units are mm
            w_raw = room.get("width_ft")
            l_raw = room.get("length_ft")
            a_raw = room.get("area_sqft")

            w = _sanitize_numeric(w_raw, mm_scale=True)
            l = _sanitize_numeric(l_raw, mm_scale=True)

            # Recalculate area from dimensions if available
            if w and l:
                a = round(w * l, 1)
            else:
                a = _sanitize_numeric(a_raw, mm_scale=False)

            room["width_ft"]   = w
            room["length_ft"]  = l
            room["area_sqft"]  = a

            features = room.get("features", [])
            if not isinstance(features, list):
                features = []
            # Strip placeholder feature strings
            room["features"] = [f for f in features if not _is_placeholder(f)]

            clean_rooms.append(room)

        floor["rooms"] = clean_rooms
        cleaned_floors.append(floor)

    data["floors"] = cleaned_floors

    # ── Sanitize top-level string fields that LLaVA copies from template ──
    for field in ("structure_type", "orientation", "scale_detected"):
        val = data.get(field, "")
        if _is_placeholder(val) or not val:
            data[field] = None   # replaced below with proper defaults if needed

    if not data.get("structure_type"):
        data["structure_type"] = "Residential"

    # ── Auto-count bedrooms/bathrooms from room names if missing or null ──
    all_rooms = [r for f in cleaned_floors for r in f.get("rooms", [])]

    def _count_rooms(*keywords):
        return sum(
            1 for r in all_rooms
            if any(kw in r.get("name", "").lower() for kw in keywords)
        )

    # Trigger auto-count when value is None/null OR 0 (LLaVA default)
    if not data.get("num_bedrooms"):   # catches None, 0, missing
        data["num_bedrooms"] = _count_rooms("bedroom", "bed room", "master bed")
    if not data.get("num_bathrooms"):  # catches None, 0, missing
        data["num_bathrooms"] = _count_rooms("bathroom", "bath", "toilet", "wc")
    if not data.get("has_garage"):
        data["has_garage"] = bool(_count_rooms("garage"))

    # Top-level defaults
    data.setdefault("structure_type", "Residential")
    data.setdefault("num_floors", len(cleaned_floors) or 1)
    if not data.get("num_floors"):
        data["num_floors"] = len(cleaned_floors) or 1
    data.setdefault("has_basement", False)
    data.setdefault("has_garden", False)

    # Strip placeholder special_features
    feats = data.get("special_features", [])
    if isinstance(feats, list):
        data["special_features"] = [f for f in feats if not _is_placeholder(f)]
    else:
        data["special_features"] = []

    data.setdefault("confidence", 0.0)

    return data


def count_rooms(data: dict) -> int:
    """Total number of rooms across all floors."""
    return sum(len(f.get("rooms", [])) for f in data.get("floors", []))


def _empty_blueprint() -> dict:
    """Returns a minimal valid blueprint dict."""
    return {
        "floors": [],
        "structure_type": "Unknown",
        "num_bedrooms": 0,
        "num_bathrooms": 0,
        "num_floors": 0,
        "has_garage": False,
        "has_basement": False,
        "has_garden": False,
        "special_features": [],
        "confidence": 0.0,
    }
