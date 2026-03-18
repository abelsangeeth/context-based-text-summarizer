"""
Confidence and coverage scores for blueprint extraction quality.
"""


def extraction_confidence(data: dict) -> float:
    """Returns the VLM's self-reported confidence (0.0–1.0)."""
    return float(data.get("confidence", 0.0))


def room_coverage_score(data: dict) -> float:
    """
    Measures how completely rooms were extracted.
    Penalises rooms missing dimensions or area.
    Returns 0.0–1.0.
    """
    total_rooms = 0
    complete_rooms = 0

    for floor in data.get("floors", []):
        for room in floor.get("rooms", []):
            total_rooms += 1
            has_dims = (room.get("width_ft") is not None
                        and room.get("length_ft") is not None)
            has_area = room.get("area_sqft") is not None
            if has_dims and has_area:
                complete_rooms += 1
            elif has_area:
                complete_rooms += 0.5

    if total_rooms == 0:
        return 0.0
    return round(complete_rooms / total_rooms, 2)


def dimension_completeness(data: dict) -> dict:
    """
    Returns per-room completeness: which rooms have full dimensions,
    partial, or none.
    """
    results = []
    for floor in data.get("floors", []):
        for room in floor.get("rooms", []):
            w = room.get("width_ft")
            l = room.get("length_ft")
            a = room.get("area_sqft")
            if w and l and a:
                status = "complete"
            elif a:
                status = "area_only"
            elif w or l:
                status = "partial"
            else:
                status = "missing"
            results.append({
                "room": room.get("name", "Unknown"),
                "floor": floor.get("floor_name", "?"),
                "status": status,
            })
    return results


def overall_quality_score(data: dict) -> float:
    """
    Composite quality score combining confidence and room coverage.
    Returns 0.0–1.0.
    """
    conf = extraction_confidence(data)
    coverage = room_coverage_score(data)
    # Weight confidence 40%, coverage 60%
    return round(0.4 * conf + 0.6 * coverage, 2)
