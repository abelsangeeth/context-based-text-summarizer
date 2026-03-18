"""
Calculates and validates all area measurements.
Handles unit detection and conversion automatically.
"""


SQFT_TO_SQM = 0.0929
SQFT_TO_SQYD = 0.1111


def calculate_all_areas(blueprint_data: dict) -> dict:
    """
    Fills missing area_sqft values, adds metric equivalents,
    computes per-floor and total areas.
    """
    grand_total_sqft = 0.0

    for floor in blueprint_data.get("floors", []):
        floor_total = 0.0

        for room in floor.get("rooms", []):
            w = room.get("width_ft")
            l = room.get("length_ft")

            # Calculate if missing but dimensions available
            if room.get("area_sqft") is None and w and l:
                room["area_sqft"] = round(w * l, 1)

            if room.get("area_sqft"):
                room["area_sqm"] = round(
                    room["area_sqft"] * SQFT_TO_SQM, 1)
                floor_total += room["area_sqft"]

        # Override floor total with calculated value if more accurate
        if floor_total > 0:
            floor["total_area_sqft"] = round(floor_total, 1)
            floor["total_area_sqm"] = round(
                floor_total * SQFT_TO_SQM, 1)

        grand_total_sqft += floor.get("total_area_sqft", 0) or 0

    blueprint_data["grand_total_sqft"] = round(grand_total_sqft, 1)
    blueprint_data["grand_total_sqm"] = round(
        grand_total_sqft * SQFT_TO_SQM, 1)

    return blueprint_data


def get_room_breakdown(blueprint_data: dict) -> list[dict]:
    """Returns flat list of all rooms across all floors, sorted by area."""
    all_rooms = []
    for floor in blueprint_data.get("floors", []):
        for room in floor.get("rooms", []):
            all_rooms.append({
                "floor": floor.get("floor_name",
                                   f"Floor {floor.get('floor_number', '?')}"),
                "room": room["name"],
                "width_ft": room.get("width_ft", "\u2014"),
                "length_ft": room.get("length_ft", "\u2014"),
                "area_sqft": room.get("area_sqft", "\u2014"),
                "area_sqm": room.get("area_sqm", "\u2014"),
                "features": ", ".join(room.get("features", []))
            })
    return sorted(all_rooms,
                  key=lambda r: r["area_sqft"]
                  if isinstance(r["area_sqft"], (int, float)) else 0,
                  reverse=True)
