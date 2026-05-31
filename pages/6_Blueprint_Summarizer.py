import streamlit as st
import pandas as pd
import json

from vision_summarizer.blueprint_reader import load_blueprint
from vision_summarizer.image_preprocessor import preprocess_blueprint
from vision_summarizer.vlm_extractor import extract_with_best_available
from vision_summarizer.room_parser import validate_and_clean
from vision_summarizer.area_calculator import (calculate_all_areas,
                                                get_room_breakdown)
from vision_summarizer.blueprint_summarizer import (json_to_plain_english,
                                                     generate_short_summary)
from vision_summarizer.blueprint_metrics import (extraction_confidence,
                                                  room_coverage_score,
                                                  overall_quality_score,
                                                  dimension_completeness)
from summarizer.pipeline import summarize

from components.stitch_theme import apply_stitch_theme, page_header, sidebar_brand, sidebar_nav, sidebar_footer

st.set_page_config(page_title="Blueprint Summarizer", page_icon="🏠", layout="wide")
apply_stitch_theme()

with st.sidebar:
    sidebar_brand()
    sidebar_nav(active="blueprint")
    st.divider()
    st.markdown('<div class="cfx-section-title" style="padding:0 8px;">Settings</div>', unsafe_allow_html=True)
    method = st.radio(
        "Extraction Engine",
        ["auto", "claude", "llava", "ocr"],
        help="Auto tries Claude first, then LLaVA, then OCR",
    )
    unit = st.radio("Display Units", ["sq ft", "sq m"])
    show_enhanced = st.checkbox("Show preprocessed image", value=True)
    sidebar_footer()

page_header("Blueprint Summarizer", "Upload any floor plan — JPG, PNG, PDF, DXF, or DWG")

# ── File upload ─────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload Blueprint",
    type=["jpg", "jpeg", "png", "pdf", "dxf", "dwg"],
)

if not uploaded:
    st.info("\U0001f446 Upload a blueprint to get started. "
            "Supports images, PDFs, and CAD files.")
    st.stop()

# ── Load + preprocess ───────────────────────────────────────
with st.spinner("Loading blueprint..."):
    try:
        pages = load_blueprint(uploaded)
    except ValueError as e:
        st.error(str(e))
        st.stop()

# Floor selector for multi-page PDFs
floor_idx = 0
if len(pages) > 1:
    floor_idx = st.selectbox(
        f"Select Floor ({len(pages)} pages detected)",
        range(len(pages)),
        format_func=lambda i: f"Floor {i + 1}"
    )

selected_img = pages[floor_idx]

# Preview columns
col_orig, col_proc = st.columns(2)
with col_orig:
    st.subheader("Original")
    st.image(selected_img, width='stretch')

with col_proc:
    if show_enhanced:
        st.subheader("Enhanced for AI Analysis")
        with st.spinner("Enhancing image..."):
            processed = preprocess_blueprint(selected_img)
        st.image(processed, width='stretch')
    else:
        processed = preprocess_blueprint(selected_img)

# ── Analyze button ──────────────────────────────────────────
if st.button("\U0001f50d Analyze Blueprint", type="primary",
             width='stretch'):

    # Step 1: Extract
    with st.spinner(f"Extracting room data "
                    f"({'auto-detecting engine' if method == 'auto' else method})..."):
        try:
            raw_data = extract_with_best_available(processed,
                                                    force_method=method)
        except Exception as e:
            st.error(f"Extraction failed: {e}")
            st.stop()

    # Step 1.5: Validate and clean
    raw_data = validate_and_clean(raw_data)

    # Warn the user if LLaVA returned unparseable JSON
    if raw_data.get("_parse_error"):
        st.warning(
            "⚠️ LLaVA returned a response that could not be fully parsed as JSON. "
            "The blueprint structure could not be extracted. "
            "Try switching to **OCR** engine or uploading a clearer image.",
            icon="🤖"
        )

    # Step 2: Calculate areas
    data = calculate_all_areas(raw_data)
    engine_used = data.get("_extraction_method", "unknown")
    confidence = extraction_confidence(data)
    coverage = room_coverage_score(data)
    quality = overall_quality_score(data)

    st.success(
        f"\u2705 Extracted using **{engine_used}** "
        f"\u2014 Confidence: **{confidence:.0%}** "
        f"| Coverage: **{coverage:.0%}** "
        f"| Quality: **{quality:.0%}**"
    )
    if confidence < 0.6:
        st.warning("\u26a0\ufe0f Low confidence score. "
                   "Try a clearer image or a different engine.")

    # ── Metrics row ─────────────────────────────────────────
    st.subheader("\U0001f4ca Property Overview")
    m1, m2, m3, m4, m5 = st.columns(5)

    area_val = (data.get("grand_total_sqm") if unit == "sq m"
                else data.get("grand_total_sqft"))
    unit_label = unit

    m1.metric("Total Area", f"{area_val} {unit_label}")
    m2.metric("Bedrooms", data.get("num_bedrooms", "?"))
    m3.metric("Bathrooms", data.get("num_bathrooms", "?"))
    m4.metric("Floors", data.get("num_floors",
                                  len(data.get("floors", []))))
    m5.metric("Rooms Detected", sum(len(f["rooms"])
                                     for f in data.get("floors", [])))

    # ── Room breakdown table ─────────────────────────────────
    st.subheader("\U0001f5c2\ufe0f Room Breakdown")
    breakdown = get_room_breakdown(data)

    if breakdown:
        df = pd.DataFrame(breakdown)
        area_col = "area_sqm" if unit == "sq m" else "area_sqft"
        display_cols = ["floor", "room",
                        "width_ft", "length_ft", area_col, "features"]
        df_display = df[[c for c in display_cols if c in df.columns]]
        df_display.columns = (["Floor", "Room",
                                "Width (ft)", "Length (ft)",
                                f"Area ({unit_label})", "Features"]
                               [:len(df_display.columns)])

        # Only highlight numeric area columns
        numeric_area_col = f"Area ({unit_label})"
        styled = df_display.style
        if numeric_area_col in df_display.columns:
            numeric_mask = pd.to_numeric(
                df_display[numeric_area_col], errors="coerce"
            ).notna()
            if numeric_mask.any():
                styled = styled.highlight_max(
                    subset=[numeric_area_col], color="#d4edda"
                ).highlight_min(
                    subset=[numeric_area_col], color="#fff3cd"
                )

        st.dataframe(styled, width='stretch')
    else:
        st.warning("No rooms detected. Try a different engine.")

    # ── Dimension completeness ──────────────────────────────
    dim_status = dimension_completeness(data)
    if dim_status:
        with st.expander("Dimension Completeness Details"):
            for item in dim_status:
                icon = {"complete": "\u2705", "area_only": "\U0001f7e1",
                        "partial": "\U0001f7e0", "missing": "\u274c"}.get(
                    item["status"], "\u2753")
                st.write(
                    f"{icon} **{item['room']}** ({item['floor']}) "
                    f"\u2014 {item['status']}"
                )

    # ── Plain English + NLP Summary ─────────────────────────
    st.subheader("\U0001f4dd AI Summary")

    plain = json_to_plain_english(data)

    tab1, tab2 = st.tabs(["Concise Summary", "Full Description"])

    with tab1:
        with st.spinner("Generating summary with BART CNN..."):
            short = generate_short_summary(plain, summarize)
        st.success(short)

    with tab2:
        st.info(plain)

    # ── Special features ────────────────────────────────────
    feats = data.get("special_features", [])
    if any([data.get("has_garage"), data.get("has_basement"),
            data.get("has_garden"), feats]):
        st.subheader("\u2728 Special Features")
        tags = []
        if data.get("has_garage"):
            tags.append("\U0001f697 Garage")
        if data.get("has_basement"):
            tags.append("\u2b07\ufe0f Basement")
        if data.get("has_garden"):
            tags.append("\U0001f33f Garden")
        for f in feats:
            tags.append(f"\u2022 {f}")
        st.write("  |  ".join(tags))

    # ── Export ───────────────────────────────────────────────
    st.subheader("\U0001f4be Export")
    ex1, ex2, ex3 = st.columns(3)

    with ex1:
        st.download_button(
            "\U0001f4e5 Download JSON",
            data=json.dumps(data, indent=2),
            file_name="blueprint_analysis.json",
            mime="application/json"
        )
    with ex2:
        if breakdown:
            st.download_button(
                "\U0001f4e5 Download Room CSV",
                data=pd.DataFrame(breakdown).to_csv(index=False),
                file_name="room_breakdown.csv",
                mime="text/csv"
            )
    with ex3:
        st.download_button(
            "\U0001f4e5 Download Summary TXT",
            data=short + "\n\n---\n\n" + plain,
            file_name="blueprint_summary.txt",
            mime="text/plain"
        )
