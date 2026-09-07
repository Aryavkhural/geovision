from datetime import date
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from geopy.geocoders import Nominatim

from live_sentinel import get_sentinel2_live_features

st.set_page_config(
    page_title="GeoVision | Land-cover analysis",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #0b1320;
        --panel: #111c2d;
        --panel-soft: #17263a;
        --line: #2c4058;
        --text: #f2f6f8;
        --muted: #9fb0be;
        --terrain: #8fb339;
        --water: #5fa8d3;
        --sand: #d8b365;
    }

   #MainMenu, footer {visibility: hidden;}
button[data-testid="stExpandSidebarButton"] {
    position: fixed;
    top: 0.75rem;
    left: 0.75rem;
    z-index: 999999;
    width: 2.5rem;
    height: 2.5rem;
    border: 1px solid rgba(143, 179, 57, 0.55);
    border-radius: 10px;
    background: #15283a;
    color: #eaf1f5;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.28);
}

button[data-testid="stExpandSidebarButton"]:hover {
    border-color: #b6d66e;
    background: #1b3249;
    color: #ffffff;
}
    .block-container {
        max-width: 1540px;
        padding-top: 2rem;
        padding-bottom: 3.5rem;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 2.75rem 2.8rem;
        border: 1px solid rgba(143, 179, 57, 0.35);
        border-radius: 16px;
        background:
            radial-gradient(circle at 86% 14%, rgba(95, 168, 211, 0.16), transparent 30%),
            linear-gradient(135deg, #101d2b 0%, #15283a 100%);
        color: var(--text);
        margin-bottom: 1.8rem;
    }

    .hero::after {
        position: absolute;
        right: -50px;
        bottom: -85px;
        width: 260px;
        height: 260px;
        border: 1px solid rgba(143, 179, 57, 0.22);
        border-radius: 50%;
        content: "";
    }

    .hero-kicker {
        position: relative;
        z-index: 1;
        color: #b6d66e;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.09em;
        text-transform: uppercase;
    }

    .hero h1 {
        position: relative;
        z-index: 1;
        margin: 0.7rem 0;
        color: #f6f8fb;
        font-size: clamp(2.1rem, 4vw, 3.65rem);
        letter-spacing: -0.055em;
        line-height: 0.98;
    }

    .hero p {
        position: relative;
        z-index: 1;
        max-width: 720px;
        margin: 0;
        color: #bbcad5;
        font-size: 1.04rem;
        line-height: 1.65;
    }

    .section-heading {
        color: #eaf1f5;
        font-size: 1.35rem;
        font-weight: 750;
        letter-spacing: -0.02em;
        margin: 1.8rem 0 0.35rem 0;
    }

    .section-copy {
        color: var(--muted);
        margin-bottom: 1.05rem;
        line-height: 1.55;
    }

    .location-card {
        margin: 0.65rem 0 0.95rem;
        padding: 0.75rem 0.85rem;
        border: 1px solid rgba(95, 168, 211, 0.30);
        border-radius: 10px;
        background: rgba(95, 168, 211, 0.08);
        color: #cce8f6;
        font-size: 0.88rem;
        line-height: 1.45;
    }

    .location-card span {
        display: block;
        margin-bottom: 0.18rem;
        color: #8fc5e2;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .meta-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem 1.2rem;
        padding: 0.85rem 1rem;
        border: 1px solid rgba(143, 179, 57, 0.22);
        border-radius: 10px;
        background: rgba(23, 38, 58, 0.72);
        color: #c6d4dc;
        font-size: 0.88rem;
        margin-bottom: 1.25rem;
    }

    .map-shell {
        padding: 0.65rem;
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 14px;
        background: #101a28;
    }

    .legend-card {
        display: flex;
        align-items: center;
        gap: 10px;
        min-height: 64px;
        padding: 10px 12px;
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 12px;
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.96), rgba(15, 23, 42, 0.96));
        box-shadow: 0 8px 18px rgba(0, 0, 0, 0.14);
        margin-bottom: 0.6rem;
        transition: transform 160ms ease, border-color 160ms ease;
    }

    .legend-card:hover {
        transform: translateY(-2px);
        border-color: rgba(182, 214, 110, 0.58);
    }

    .legend-dot {
        flex: 0 0 18px;
        width: 18px;
        height: 18px;
        border: 2px solid rgba(255, 255, 255, 0.78);
        border-radius: 50%;
        box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.07);
    }

    .legend-label {
        display: flex;
        flex-direction: column;
        gap: 3px;
        line-height: 1.1;
    }

    .legend-label strong {
        color: #f8fafc;
        font-size: 0.83rem;
        font-weight: 750;
    }

    .legend-label small {
        color: #94a3b8;
        font-size: 0.66rem;
        font-weight: 700;
        letter-spacing: 0.07em;
        text-transform: uppercase;
    }

    .notice {
        margin-top: 1rem;
        padding: 0.9rem 1rem;
        border-left: 4px solid #d8b365;
        border-radius: 8px;
        background: rgba(216, 179, 101, 0.10);
        color: #e8d3a0;
        line-height: 1.55;
    }

    .app-card {
        min-height: 150px;
        padding: 1.1rem 1.15rem;
        border: 1px solid rgba(148, 163, 184, 0.20);
        border-radius: 13px;
        background: linear-gradient(145deg, rgba(23, 38, 58, 0.88), rgba(15, 23, 35, 0.94));
    }

    .app-card h3 {
        margin: 0 0 0.55rem 0;
        color: #edf4f7;
        font-size: 1rem;
    }

    .app-card p {
        margin: 0;
        color: #aebfcb;
        font-size: 0.91rem;
        line-height: 1.55;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_MAPS_DIR = PROJECT_ROOT / "outputs" / "maps"
OUTPUTS_REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
LIVE_MODEL_PATH = PROJECT_ROOT / "models" / "rf_6class_synthetic.pkl"
MAP_PATH = OUTPUTS_MAPS_DIR / "geovision_lulc_classified.tif"
METRICS_PATH = OUTPUTS_REPORTS_DIR / "model_metrics.json"
AREA_STATS_PATH = OUTPUTS_REPORTS_DIR / "land_cover_area_statistics.csv"
CONFUSION_MATRIX_PATH = OUTPUTS_REPORTS_DIR / "confusion_matrix.png"
REPORT_PATH = OUTPUTS_REPORTS_DIR / "classification_report.csv"

CLASS_NAMES = {
    0: "Agriculture",
    1: "Built-up",
    2: "Forest",
    3: "Water",
    4: "Wasteland",
    5: "Roads",
}

CLASS_COLORS = {
    0: (234, 217, 76),
    1: (239, 62, 58),
    2: (32, 140, 52),
    3: (47, 115, 221),
    4: (176, 128, 76),
    5: (189, 189, 189),
}

COLOR_NAMES = {
    0: "Crop yellow",
    1: "Built-up red",
    2: "Vegetation green",
    3: "Water blue",
    4: "Bare-land brown",
    5: "Road grey",
}


@st.cache_data
def load_classification_map(path_string):
    with rasterio.open(path_string) as src:
        return src.read(1), src.crs, src.transform


@st.cache_data
def make_rgb_classification(classification):
    image = np.zeros(
        (classification.shape[0], classification.shape[1], 3),
        dtype=np.uint8,
    )
    for class_id, color in CLASS_COLORS.items():
        image[classification == class_id] = color
    return image


@st.cache_data
def load_table(path_string):
    return pd.read_csv(path_string)


@st.cache_data(ttl=86400, show_spinner=False)
def get_location_name(latitude, longitude):
    try:
        geolocator = Nominatim(user_agent="geovision-srm-ist-student-project")
        result = geolocator.reverse(
            f"{latitude}, {longitude}",
            exactly_one=True,
            language="en",
            zoom=10,
            timeout=10,
        )

        if result is None:
            return "Location name not found"

        address = result.raw.get("address", {})
        place = (
            address.get("city")
            or address.get("town")
            or address.get("village")
            or address.get("municipality")
            or address.get("county")
            or address.get("state_district")
        )
        state = address.get("state")
        country = address.get("country")
        parts = [part for part in [place, state, country] if part]

        return ", ".join(parts) if parts else result.address

    except (GeocoderTimedOut, GeocoderServiceError):
        return "Location lookup is temporarily unavailable"
    except Exception:
        return "Location lookup could not be completed"


def show_color_legend():
    st.markdown('<div class="section-heading">Map legend</div>', unsafe_allow_html=True)
    legend_columns = st.columns(6)

    for column, (class_id, class_name) in zip(legend_columns, CLASS_NAMES.items()):
        red, green, blue = CLASS_COLORS[class_id]
        color_name = COLOR_NAMES[class_id]
        with column:
            st.markdown(
                f"""
                <div class="legend-card">
                    <span class="legend-dot" style="background: rgb({red}, {green}, {blue});"></span>
                    <div class="legend-label">
                        <strong>{class_name}</strong>
                        <small>{color_name}</small>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def show_applications():
    st.markdown('<div class="section-heading">Where this helps</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">GeoVision is built for quick screening—not final regulatory or planning decisions.</p>',
        unsafe_allow_html=True,
    )

    city_col, farm_col, environment_col = st.columns([1.05, 0.95, 1.2], gap="large")

    with city_col:
        st.markdown(
            """
            <div class="app-card">
                <h3>Track city growth</h3>
                <p>Spot expansion of built-up land and potential conversion of agricultural or open land around a selected area.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with farm_col:
        st.markdown(
            """
            <div class="app-card">
                <h3>Review agricultural patterns</h3>
                <p>Estimate crop-area coverage and compare vegetation patterns across locations or imagery dates.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with environment_col:
        st.markdown(
            """
            <div class="app-card">
                <h3>Check environmental change</h3>
                <p>Screen forests, water bodies, barren areas, and land-cover shifts before conducting a detailed field assessment.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# Sidebar
st.sidebar.markdown("## GeoVision")
st.sidebar.caption("Land-cover analysis for Indian locations")
st.sidebar.markdown("---")
st.sidebar.markdown("### Study area")

location = st.sidebar.text_input(
    "Location name",
    value="Custom study area",
    placeholder="Example: Pune, Maharashtra",
    help="You can type a name, or use the coordinate lookup below.",
)

coordinate_col1, coordinate_col2 = st.sidebar.columns(2)
with coordinate_col1:
    latitude = st.number_input(
        "Latitude",
        min_value=6.0,
        max_value=37.0,
        value=18.5204,
        step=0.0001,
        format="%.4f",
    )
with coordinate_col2:
    longitude = st.number_input(
        "Longitude",
        min_value=68.0,
        max_value=98.0,
        value=73.8567,
        step=0.0001,
        format="%.4f",
    )

lookup_location = st.sidebar.button("Resolve location from coordinates", use_container_width=True)
coordinate_key = f"{latitude:.4f},{longitude:.4f}"

if (
    "resolved_coordinate_key" not in st.session_state
    or st.session_state.resolved_coordinate_key != coordinate_key
):
    st.session_state.resolved_location_name = ""
    st.session_state.resolved_coordinate_key = coordinate_key

if lookup_location:
    with st.sidebar.spinner("Finding nearby place..."):
        st.session_state.resolved_location_name = get_location_name(latitude, longitude)

resolved_location_name = st.session_state.get("resolved_location_name", "")

if resolved_location_name:
    st.sidebar.markdown(
        f"""
        <div class="location-card">
            <span>Coordinates resolve to</span>
            {resolved_location_name}
        </div>
        """,
        unsafe_allow_html=True,
    )

analysis_width_km = st.sidebar.select_slider(
    "Analysis extent",
    options=[1, 2, 3, 5],
    value=1,
    format_func=lambda value: f"{value} km x {value} km",
)

image_end_date = st.sidebar.date_input(
    "Search imagery up to",
    value=date.today(),
    max_value=date.today(),
    help="GeoVision searches the preceding 60 days and selects a least-cloudy Sentinel-2 mosaic.",
)

run_live = st.sidebar.button("Run analysis", type="primary", use_container_width=True)
st.sidebar.caption("Sentinel-2 Level-2A · 10 m output · Least-cloudy mosaic · 60-day search")
st.sidebar.markdown("---")
st.sidebar.markdown("### Visible classes")

show_agriculture = st.sidebar.checkbox("Agriculture", value=True)
show_builtup = st.sidebar.checkbox("Built-up", value=True)
show_forest = st.sidebar.checkbox("Forest", value=True)
show_water = st.sidebar.checkbox("Water", value=True)
show_wasteland = st.sidebar.checkbox("Wasteland", value=True)
show_roads = st.sidebar.checkbox("Roads", value=True)

selected_layers = {
    0: show_agriculture,
    1: show_builtup,
    2: show_forest,
    3: show_water,
    4: show_wasteland,
    5: show_roads,
}

st.sidebar.markdown("---")
st.sidebar.caption(
    "Prototype output only. Validate with field data or expert labels before using this map for operational decisions."
)

st.markdown(
    """
    <section class="hero">
        <div class="hero-kicker">Sentinel-2 · 10 m land-cover mapping</div>
        <h1>See what covers the ground.</h1>
        <p>
            Select a location in India, fetch recent satellite imagery, and map
            agriculture, built-up land, forest, water, wasteland, and roads in one view.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

if run_live:
    if not LIVE_MODEL_PATH.exists():
        st.error(f"The model file was not found: {LIVE_MODEL_PATH}")
        st.stop()

    if "CDSE_CLIENT_ID" not in st.secrets or "CDSE_CLIENT_SECRET" not in st.secrets:
        st.error(
            "Live analysis needs Copernicus credentials. Add CDSE_CLIENT_ID and CDSE_CLIENT_SECRET in the app's secret settings."
        )
        st.stop()

    display_location = resolved_location_name if resolved_location_name else location

    try:
        with st.spinner("Fetching imagery and classifying the selected area..."):
            live_features, valid_mask, live_bbox, window_start = get_sentinel2_live_features(
                latitude=latitude,
                longitude=longitude,
                width_km=analysis_width_km,
                end_date=image_end_date,
                client_id=st.secrets["CDSE_CLIENT_ID"],
                client_secret=st.secrets["CDSE_CLIENT_SECRET"],
            )

            if not np.any(valid_mask):
                st.error("No valid Sentinel-2 pixels were returned. Try another date or location.")
                st.stop()

            rf_live = joblib.load(LIVE_MODEL_PATH)
            height, width, _ = live_features.shape
            feature_matrix = live_features.reshape(-1, 9)
            mask_flat = valid_mask.reshape(-1)
            predictions = np.full(height * width, -1, dtype=np.int16)
            predictions[mask_flat] = rf_live.predict(feature_matrix[mask_flat])
            predicted_map = predictions.reshape(height, width)

            valid_count = int(np.sum(valid_mask))
            pixel_area_ha = 0.01
            rows = []
            for class_id, class_name in CLASS_NAMES.items():
                count = int(np.sum(predicted_map == class_id))
                area_ha = count * pixel_area_ha
                share = 100 * count / valid_count if valid_count else 0
                rows.append(
                    {
                        "Class": class_name,
                        "Area (ha)": round(area_ha, 2),
                        "Area (km2)": round(area_ha / 100, 3),
                        "Share (%)": round(share, 2),
                    }
                )

            live_stats = pd.DataFrame(rows)
            dominant_live = live_stats.loc[live_stats["Area (ha)"].idxmax()]

    except Exception as error:
        st.error(f"Analysis could not be completed: {error}")
        st.stop()

    st.markdown('<div class="section-heading">Analysis result</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <p class="section-copy">
            GeoVision processed the selected Sentinel-2 scene for <strong>{display_location}</strong>
            and classified each valid 10 m pixel into one of six land-cover categories.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="meta-strip">
            <span><strong>Location:</strong> {display_location}</span>
            <span><strong>Centre:</strong> {latitude:.4f}° N, {longitude:.4f}° E</span>
            <span><strong>Extent:</strong> {analysis_width_km} km x {analysis_width_km} km</span>
            <span><strong>Search window:</strong> {window_start} to {image_end_date}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rgb_map = np.zeros((height, width, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        if selected_layers.get(class_id, True):
            rgb_map[predicted_map == class_id] = color
    rgb_map[~valid_mask] = (24, 31, 43)

    live_map_col, live_stats_col = st.columns([1.22, 0.78], gap="large")
    with live_map_col:
        st.markdown('<div class="map-shell">', unsafe_allow_html=True)
        st.image(rgb_map, use_container_width=True, caption=f"Live Sentinel-2 classification — {display_location}")
        st.markdown("</div>", unsafe_allow_html=True)

    with live_stats_col:
        st.metric("Dominant cover", dominant_live["Class"], f"{dominant_live['Share (%)']}%")
        st.dataframe(live_stats, use_container_width=True, hide_index=True)
        st.bar_chart(live_stats.set_index("Class")["Share (%)"], color="#8fb339")
        st.download_button(
            "Download area statistics",
            data=live_stats.to_csv(index=False).encode("utf-8"),
            file_name="geovision_live_lulc_statistics.csv",
            mime="text/csv",
            use_container_width=True,
        )

    show_color_legend()
    st.markdown(
        '<div class="notice"><strong>Interpret with care.</strong> This prototype supports exploratory screening. Field data or expert-labelled samples are required before operational use.</div>',
        unsafe_allow_html=True,
    )

else:
    st.markdown('<div class="section-heading">Demo analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">Start with this prepared Sentinel-2 classification, or enter your own coordinates in the sidebar and run a live analysis.</p>',
        unsafe_allow_html=True,
    )

    if MAP_PATH.exists():
        classification, crs, transform = load_classification_map(str(MAP_PATH))
        rgb_classification = make_rgb_classification(classification)

        for class_id, is_selected in selected_layers.items():
            if not is_selected:
                rgb_classification[classification == class_id] = (43, 52, 67)

        map_col, stats_col = st.columns([1.22, 0.78], gap="large")
        with map_col:
            st.markdown('<div class="map-shell">', unsafe_allow_html=True)
            st.image(rgb_classification, use_container_width=True, caption="Prepared land-cover classification")
            st.markdown("</div>", unsafe_allow_html=True)

        with stats_col:
            if AREA_STATS_PATH.exists():
                area_stats = load_table(str(AREA_STATS_PATH)).copy()
                rename_map = {
                    "class_id": "Class ID",
                    "class_name": "Class",
                    "pixel_count": "Pixel count",
                    "area_hectares": "Area (ha)",
                    "area_km2": "Area (km2)",
                    "percentage": "Share (%)",
                }
                area_stats.rename(columns=rename_map, inplace=True)
                preferred_columns = [
                    column
                    for column in ["Class", "Area (ha)", "Area (km2)", "Share (%)"]
                    if column in area_stats.columns
                ]
                st.dataframe(
                    area_stats[preferred_columns] if preferred_columns else area_stats,
                    use_container_width=True,
                    hide_index=True,
                )
                if "Class" in area_stats.columns and "Share (%)" in area_stats.columns:
                    st.bar_chart(area_stats.set_index("Class")["Share (%)"], color="#8fb339")
            else:
                st.info("Area statistics are not available yet. Run the notebook export cells to create them.")
    else:
        st.info("The demo map is not available yet. Run the notebook export cells, or use live analysis from the sidebar.")

    show_color_legend()

    st.markdown('<div class="section-heading">Model check</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">These metrics come from the current model-training workflow.</p>',
        unsafe_allow_html=True,
    )

    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            metrics = json.load(file)

        evaluation_col1, evaluation_col2, evaluation_col3 = st.columns([1, 1, 1.4])
        evaluation_col1.metric("Overall accuracy", f"{float(metrics.get('overall_accuracy', 0)) * 100:.2f}%")
        evaluation_col2.metric("Kappa score", f"{float(metrics.get('kappa_score', 0)):.3f}")
        evaluation_col3.metric("Model", metrics.get("model", "Random Forest"))

        if CONFUSION_MATRIX_PATH.exists() and REPORT_PATH.exists():
            evaluation_map_col, evaluation_table_col = st.columns([0.92, 1.08], gap="large")
            with evaluation_map_col:
                st.image(str(CONFUSION_MATRIX_PATH), caption="Confusion matrix", use_container_width=True)
            with evaluation_table_col:
                report_df = load_table(str(REPORT_PATH))
                st.dataframe(report_df, use_container_width=True, hide_index=True)
        else:
            st.info("The confusion matrix and classification report have not been exported yet.")
    else:
        st.info("Model metrics are not available yet. Run the notebook export cells to create them.")

    st.markdown(
        '<div class="notice"><strong>Prototype note.</strong> The displayed metrics are linked to the current training workflow. Independent field or expert-labelled ground-truth samples are needed before treating them as operational accuracy.</div>',
        unsafe_allow_html=True,
    )
    show_applications()

st.caption("GeoVision · Sentinel-2 Level-2A · Random Forest classification · 10 m spatial resolution")
