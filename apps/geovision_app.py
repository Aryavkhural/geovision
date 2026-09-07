from datetime import date
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import rasterio
import streamlit as st

from live_sentinel import get_sentinel2_live_features

st.set_page_config(
    page_title="GeoVision | Geospatial Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .hero {
        padding: 2.2rem 2.4rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #1b1f3b 0%, #2a2f6b 100%);
        color: #f5f6ff;
        margin-bottom: 1.6rem;
    }
    .eyebrow {
        display: inline-block;
        background: rgba(46, 204, 113, 0.18);
        color: #2ecc71;
        border-radius: 999px;
        padding: 0.25rem 0.9rem;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.9rem;
    }
    .hero h1 {
        margin: 0 0 0.6rem 0;
        font-size: 2.6rem;
    }
    .hero p {
        max-width: 760px;
        font-size: 1.02rem;
        line-height: 1.55;
        color: #d7d9f5;
    }
    .section-heading {
        font-size: 1.35rem;
        font-weight: 700;
        margin: 1.6rem 0 0.3rem 0;
    }
    .section-copy {
        color: #9aa0c3;
        margin-bottom: 1rem;
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
    0: "Sunflower Yellow",
    1: "Urban Red",
    2: "Forest Green",
    3: "Water Blue",
    4: "Earth Brown",
    5: "Road Gray",
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


def show_color_legend():
    st.markdown(
        '<div class="section-heading">Land-cover colour key</div>',
        unsafe_allow_html=True,
    )
    legend_columns = st.columns(6)

    for column, (class_id, class_name) in zip(legend_columns, CLASS_NAMES.items()):
        red, green, blue = CLASS_COLORS[class_id]
        color_name = COLOR_NAMES[class_id]

        with column:
            st.markdown(
                f"""
                <div class="legend-card">
                    <span
                        class="legend-dot"
                        style="background: rgb({red}, {green}, {blue});"
                    ></span>
                    <div class="legend-label">
                        <strong>{class_name}</strong>
                        <small>{color_name}</small>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# Sidebar
st.sidebar.markdown("## 🛰️ GeoVision")
st.sidebar.caption("Geospatial Intelligence Workspace")
st.sidebar.markdown("---")
st.sidebar.markdown("### Analysis parameters")

location = st.sidebar.text_input(
    "Area label",
    value="Custom study area",
    placeholder="Example: Pune, Maharashtra",
    help="A user-defined name for this analysis. It does not alter the coordinates.",
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

analysis_width_km = st.sidebar.select_slider(
    "Analysis extent",
    options=[1, 2, 3, 5],
    value=1,
    format_func=lambda value: f"{value} km x {value} km",
)

image_end_date = st.sidebar.date_input(
    "Latest imagery date",
    value=date.today(),
    max_value=date.today(),
    help="The system searches the preceding 60 days and uses a least-cloudy Sentinel-2 mosaic.",
)

run_live = st.sidebar.button(
    "Run satellite analysis",
    type="primary",
    use_container_width=True,
)

st.sidebar.caption(
    "Source: Sentinel-2 Level-2A - 10 m output - Least-cloudy mosaic - 60-day search window"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Layer visibility")
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
    "Prototype model: results are intended for exploratory analysis. Independent ground-truth validation is required for operational use."
)

st.markdown(
    """
    <div class="hero">
        <span class="eyebrow">Live Sentinel-2 analysis enabled</span>
        <h1>GeoVision</h1>
        <p>
            Choose an area anywhere in India, retrieve recent Sentinel-2
            imagery, and generate a six-class land-use and land-cover report
            at 10 m output resolution.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Live-analysis page
if run_live:
    if not LIVE_MODEL_PATH.exists():
        st.error(f"Live model was not found: {LIVE_MODEL_PATH}")
        st.stop()

    try:
        with st.spinner("Retrieving Sentinel-2 imagery, preparing features, and classifying pixels..."):
            live_features, valid_mask, live_bbox, window_start = get_sentinel2_live_features(
                latitude=latitude,
                longitude=longitude,
                width_km=analysis_width_km,
                end_date=image_end_date,
                client_id=st.secrets["CDSE_CLIENT_ID"],
                client_secret=st.secrets["CDSE_CLIENT_SECRET"],
            )

            if not np.any(valid_mask):
                st.error("No valid Sentinel-2 pixels were returned. Choose another date or location.")
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

    except KeyError:
        st.error(
            "Copernicus credentials are missing. Add CDSE_CLIENT_ID and CDSE_CLIENT_SECRET to .streamlit/secrets.toml, then restart Streamlit."
        )
        st.stop()
    except Exception as error:
        st.error(f"Live classification failed: {error}")
        st.stop()

    st.markdown(
        '<div class="section-heading">Live classification result</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="section-copy">Sentinel-2 surface-reflectance imagery was processed into six land-cover classes using the GeoVision Random Forest inference pipeline. Each valid 10 m pixel is assigned one of six land-cover categories.</p>',
        unsafe_allow_html=True,
    )

    rgb_map = np.zeros((height, width, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        if selected_layers.get(class_id, True):
            rgb_map[predicted_map == class_id] = color
    rgb_map[~valid_mask] = (24, 31, 43)

    live_map_col, live_stats_col = st.columns([1.22, 0.78], gap="large")

    with live_map_col:
        st.image(
            rgb_map,
            use_container_width=True,
            caption=f"{location} - {window_start} to {image_end_date}",
        )

    with live_stats_col:
        st.metric(
            "Dominant class",
            dominant_live["Class"],
            f"{dominant_live['Share (%)']}%",
        )
        st.dataframe(live_stats, use_container_width=True, hide_index=True)

    show_color_legend()

else:
    st.markdown(
        '<div class="section-heading">Reference study area</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="section-copy">AI-assisted land-use and land-cover intelligence for India. Explore a prepared study area or run a new satellite analysis using coordinates in the control panel. This is a previously processed Sentinel-2 study area, retained for reference and model demonstration.</p>',
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
            st.image(
                rgb_classification,
                use_container_width=True,
                caption="Reference classification",
            )

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
            else:
                st.info("Area statistics will appear after the notebook export cells have been run.")
    else:
        st.info("No reference classification map was found yet. Run the notebook pipeline once, or use the live satellite analysis in the sidebar.")

    show_color_legend()

    st.markdown(
        '<div class="section-heading">Model evaluation</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="section-copy">Prototype evaluation metrics exported from the model-training workflow.</p>',
        unsafe_allow_html=True,
    )

    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            metrics = json.load(file)

        evaluation_col1, evaluation_col2, evaluation_col3 = st.columns(3)
        evaluation_col1.metric(
            "Overall accuracy",
            f"{float(metrics.get('overall_accuracy', 0)) * 100:.2f}%",
        )
        evaluation_col2.metric(
            "Kappa score",
            f"{float(metrics.get('kappa_score', 0)):.3f}",
        )
        evaluation_col3.metric("Classifier", metrics.get("model", "Random Forest"))

        if CONFUSION_MATRIX_PATH.exists() and REPORT_PATH.exists():
            evaluation_map_col, evaluation_table_col = st.columns([1, 1], gap="large")
            with evaluation_map_col:
                st.image(
                    str(CONFUSION_MATRIX_PATH),
                    caption="Confusion matrix",
                    use_container_width=True,
                )
            with evaluation_table_col:
                report_df = load_table(str(REPORT_PATH))
                st.dataframe(report_df, use_container_width=True, hide_index=True)
        else:
            st.info("Evaluation images and tables will appear after the notebook metrics-export cells have been run.")
    else:
        st.info("Model metrics will appear after the notebook metrics-export cells have been run.")

    st.markdown(
        '<div class="section-heading">Applications</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="section-copy">GeoVision turns satellite imagery into interpretable land-cover insights for planning and monitoring.</p>',
        unsafe_allow_html=True,
    )

    application_col1, application_col2, application_col3 = st.columns(3, gap="large")
    with application_col1:
        st.markdown(
            "**Urban planning**\n\nTrack built-up expansion, identify land-conversion patterns, and support infrastructure and zoning discussions."
        )
    with application_col2:
        st.markdown(
            "**Agriculture monitoring**\n\nEstimate agricultural extent, observe vegetation patterns, and support crop and irrigation monitoring workflows."
        )
    with application_col3:
        st.markdown(
            "**Environmental tracking**\n\nMonitor tree cover, water bodies, wasteland change, and indicators useful for conservation planning."
        )
