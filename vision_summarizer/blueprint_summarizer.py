"""
Converts structured blueprint JSON -> plain English description
-> feeds into existing NLP summarizer (BART CNN).
"""


def json_to_plain_english(data: dict) -> str:
    """
    Produces a detailed natural language description
    of the blueprint for NLP summarization.
    """
    floors = data.get("floors", [])
    n_floors = data.get("num_floors", len(floors)) or len(floors) or 1
    n_bed = data.get("num_bedrooms", "?") or "?"
    n_bath = data.get("num_bathrooms", "?") or "?"
    total_sqft = data.get("grand_total_sqft",
                          data.get("total_area_sqft")) or "unknown"
    total_sqm = data.get("grand_total_sqm") or "unknown"
    struct = data.get("structure_type") or "residential property"
    orientation = data.get("orientation") or ""
    extras = data.get("special_features", []) or []

    parts = []

    # Opening
    parts.append(
        f"This blueprint shows a {struct} spanning "
        f"{total_sqft} square feet ({total_sqm} square meters) "
        f"across {n_floors} floor(s)."
    )

    # Bedroom / bathroom summary
    parts.append(
        f"The property has {n_bed} bedroom(s) and "
        f"{n_bath} bathroom(s)."
    )

    # Per-floor room description
    for floor in floors:
        fname = floor.get("floor_name",
                          f"Floor {floor.get('floor_number', '?')}")
        rooms = floor.get("rooms", [])
        f_area = floor.get("total_area_sqft", "?")

        room_descs = []
        for room in rooms:
            w = room.get("width_ft")
            l = room.get("length_ft")
            area = room.get("area_sqft")
            name = room["name"]

            if w and l:
                desc = f"a {name} measuring {w} ft by {l} ft"
                if area:
                    desc += f" ({area} sq ft)"
            elif area:
                desc = f"a {name} of {area} sq ft"
            else:
                desc = f"a {name}"

            feats = room.get("features", [])
            if feats:
                desc += f" with {', '.join(feats)}"
            room_descs.append(desc)

        if room_descs:
            room_list = (", ".join(room_descs[:-1])
                         + (" and " if len(room_descs) > 1 else "")
                         + room_descs[-1])
            parts.append(
                f"The {fname} (total {f_area} sq ft) "
                f"includes {room_list}."
            )

    # Extras
    if data.get("has_garage"):
        parts.append("The property includes a garage.")
    if data.get("has_basement"):
        parts.append("A basement level is present.")
    if data.get("has_garden"):
        parts.append("A garden area is indicated on the plan.")
    if extras:
        parts.append(
            "Notable features include: " + ", ".join(extras) + "."
        )
    if orientation:
        parts.append(f"The building is {orientation}.")

    return " ".join(parts)


def generate_short_summary(plain_english: str,
                           summarizer_fn) -> str:
    """
    Feeds the plain English description into the
    existing BART CNN summarizer for a concise summary.
    """
    if len(plain_english.split()) < 60:
        return plain_english    # already short, no need to summarize

    return summarizer_fn(plain_english, model_key="bart-cnn")
