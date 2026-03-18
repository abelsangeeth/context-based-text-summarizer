"""
JSON cleaner + validator for VLM extraction output.
Ensures all required fields exist and have sane values.
"""


_REQUIRED_ROOM_FIELDS = {"name", "width_ft", "length_ft", "area_sqft", "features"}
_REQUIRED_FLOOR_FIELDS = {"floor_number", "floor_name", "rooms", "total_area_sqft"}


def validate_and_clean(data: dict) -> dict:
    """
    Validates VLM JSON output, fills missing fields with defaults,
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
        floor.setdefault("floor_name", f"Floor {floor['floor_number']}")
        floor.setdefault("total_area_sqft", None)

        clean_rooms = []
        for room in floor.get("rooms", []):
            if not isinstance(room, dict):
                continue
            if not room.get("name"):
                continue
            room.setdefault("width_ft", None)
            room.setdefault("length_ft", None)
            room.setdefault("area_sqft", None)
            room.setdefault("features", [])
            if not isinstance(room["features"], list):
                room["features"] = []
            clean_rooms.append(room)

        floor["rooms"] = clean_rooms
        cleaned_floors.append(floor)

    data["floors"] = cleaned_floors

    # Top-level defaults
    data.setdefault("structure_type", "Unknown")
    data.setdefault("num_bedrooms", 0)
    data.setdefault("num_bathrooms", 0)
    data.setdefault("num_floors", len(cleaned_floors))
    data.setdefault("has_garage", False)
    data.setdefault("has_basement", False)
    data.setdefault("has_garden", False)
    data.setdefault("special_features", [])
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
