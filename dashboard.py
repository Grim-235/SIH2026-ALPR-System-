#!/usr/bin/env python
"""City-Wide ANPR Traffic Analytics Dashboard — Streamlit App."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

import cv2
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from alpr.database import (
    init_db,
    load_cameras_from_json,
    load_blacklist_from_file,
    upsert_camera,
    get_detection_stats,
    get_detections_over_time,
    get_camera_heatmap_data,
    get_all_plates,
    query_plate_history,
    get_top_routes,
    get_blacklist,
    get_alerts,
    acknowledge_alert,
    add_to_blacklist,
    remove_from_blacklist,
    check_blacklist,
    insert_alert,
    insert_detection,
)
from alpr.detector import load_detector, resolve_device, ensure_model, DEFAULT_MODEL_PATH
from alpr.ocr import load_ocr
from alpr.tracker import process_video_with_tracking

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="City-Wide ANPR Analytics",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
BG = "#0B0B0C"
SURFACE = "#17171A"
SURFACE_ALT = "#1F1F23"
BORDER = "#F2F2ED"
ACCENT = "#C6FF3D"      # acid green — primary accent
ACCENT_DIM = "#8FBF2C"
DANGER = "#FF4D4D"
TEXT = "#F2F2ED"
TEXT_DIM = "#9A9A99"

# ---------------------------------------------------------------------------
# Custom CSS — brutalist accents layered on top of the dark theme in
# .streamlit/config.toml (which handles native widget colors: inputs,
# sliders, selects, expanders, dataframes — CSS alone cannot reach those
# reliably across Streamlit versions).
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
    .stApp {{
        background-color: {BG};
        font-family: "JetBrains Mono", "Helvetica Neue", monospace, sans-serif;
    }}

    /* Metric cards */
    div[data-testid="stMetric"] {{
        background: {SURFACE};
        border: 2px solid {BORDER};
        border-radius: 0px;
        padding: 14px 18px;
        box-shadow: 4px 4px 0px {ACCENT};
    }}
    div[data-testid="stMetric"] label {{
        color: {TEXT_DIM} !important;
        font-size: 0.7rem !important;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }}
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: {TEXT} !important;
        font-size: 1.9rem !important;
        font-weight: 800;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {SURFACE} !important;
        border-right: 2px solid {BORDER};
    }}
    section[data-testid="stSidebar"] div[data-testid="stMetric"] {{
        box-shadow: 3px 3px 0px {ACCENT_DIM};
        margin-bottom: 6px;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0;
        background: transparent;
        border-bottom: 2px solid {BORDER};
        flex-wrap: wrap;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 0;
        color: {TEXT_DIM};
        padding: 10px 18px;
        border: 2px solid transparent;
        border-bottom: none;
    }}
    .stTabs [aria-selected="true"] {{
        background: {SURFACE} !important;
        color: {ACCENT} !important;
        border: 2px solid {BORDER};
        border-bottom: 2px solid {SURFACE};
        margin-bottom: -2px;
        font-weight: 800;
    }}

    h1, h2, h3, h4, h5 {{
        color: {TEXT} !important;
        font-weight: 700;
        letter-spacing: -0.02em;
    }}

    /* Alert banner */
    .alert-banner {{
        background-color: {DANGER};
        color: #0B0B0C;
        padding: 12px 20px;
        border: 2px solid {BORDER};
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 16px;
        box-shadow: 5px 5px 0px {BORDER};
    }}

    /* Buttons */
    .stButton > button {{
        border-radius: 0px;
        font-weight: 700;
        background-color: {ACCENT};
        color: #0B0B0C;
        border: 2px solid {BORDER};
        transition: none;
    }}
    .stButton > button:hover {{
        background-color: {BG};
        color: {ACCENT};
        box-shadow: 3px 3px 0px {ACCENT};
        border-color: {ACCENT};
    }}
    .stDownloadButton > button {{
        border-radius: 0px;
        font-weight: 700;
        border: 2px solid {BORDER};
    }}

    /* Bordered containers (settings groups) */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        border: 2px solid {BORDER} !important;
        border-radius: 0px !important;
        background: {SURFACE};
    }}

    /* Dataframes */
    .stDataFrame {{
        border: 2px solid {BORDER};
        border-radius: 0px;
    }}

    /* Section captions */
    .stCaption, [data-testid="stCaptionContainer"] {{
        color: {TEXT_DIM} !important;
    }}

    hr {{
        border-color: {BORDER} !important;
        opacity: 0.3;
    }}
</style>
""", unsafe_allow_html=True)


def style_plotly(fig, height=350, showlegend=True):
    """Apply consistent dark-brutalist styling to a plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(color=TEXT, family="JetBrains Mono, monospace"),
        title_font=dict(color=TEXT, size=16),
        height=height,
        showlegend=showlegend,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    fig.update_xaxes(gridcolor=SURFACE_ALT, zerolinecolor=BORDER, linecolor=BORDER)
    fig.update_yaxes(gridcolor=SURFACE_ALT, zerolinecolor=BORDER, linecolor=BORDER)
    return fig


# ---------------------------------------------------------------------------
# Database Connection (cached)
# ---------------------------------------------------------------------------
DB_PATH = Path("data/alpr.db")
CAMERAS_JSON = Path("cameras.json")
BLACKLIST_FILE = Path("blacklist.txt")


@st.cache_resource
def get_db():
    """Get a persistent database connection."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = init_db(str(DB_PATH))
    load_cameras_from_json(conn, str(CAMERAS_JSON))
    load_blacklist_from_file(conn, str(BLACKLIST_FILE))
    return conn


conn = get_db()

# ---------------------------------------------------------------------------
# Check for unacknowledged alerts — show banner at top
# ---------------------------------------------------------------------------
unack_alerts = get_alerts(conn, only_unacknowledged=True)
if unack_alerts:
    st.markdown(
        f'<div class="alert-banner">⚠ {len(unack_alerts)} UNACKNOWLEDGED ALERT(S) — '
        f'BLACKLISTED VEHICLE(S) DETECTED. SEE ALERTS TAB.</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## CITY ANPR")
    st.caption("Real-time traffic intelligence")
    st.divider()

    stats = get_detection_stats(conn)
    st.metric("Total Detections", f"{stats.get('total_detections', 0):,}")
    st.metric("Unique Vehicles", f"{stats.get('unique_plates', 0):,}")
    st.metric("Active Cameras", f"{stats.get('unique_cameras', 0):,}")

    if stats.get("min_time") and stats.get("max_time"):
        st.caption(f"{stats['min_time'][:16]}  →  {stats['max_time'][:16]}")

    st.divider()
    st.caption("YOLOv8 · EasyOCR · ByteTrack")
    st.caption("SIH 2026 · PS #26127")

# ---------------------------------------------------------------------------
# Main Tabs (Overview first — land on the data, not the upload form)
# ---------------------------------------------------------------------------
tab_overview, tab_process, tab_lookup, tab_heatmap, tab_routes, tab_alerts = st.tabs(
    ["Overview", "Process Video", "Vehicle Lookup", "Traffic Heatmap", "Top Routes", "Alerts"]
)

# ========================== TAB 1: OVERVIEW ================================
with tab_overview:
    st.markdown("### Detection Overview")

    time_data = get_detections_over_time(conn, bucket_minutes=5)
    if time_data:
        df_time = pd.DataFrame(time_data)
        fig_time = px.area(
            df_time,
            x="bucket_time",
            y="count",
            title="Detections Over Time (5-min buckets)",
            labels={"bucket_time": "Time", "count": "Detections"},
        )
        fig_time.update_traces(
            fill="tozeroy",
            line=dict(color=ACCENT, width=2),
            fillcolor="rgba(198,255,61,0.12)",
        )
        st.plotly_chart(style_plotly(fig_time, showlegend=False), use_container_width=True)
    else:
        st.info("No detection data yet. Run `process_videos.py` or `simulate_cameras.py` to populate.")

    heatmap_data = get_camera_heatmap_data(conn)
    if heatmap_data:
        df_cam = pd.DataFrame(heatmap_data)
        fig_cam = px.bar(
            df_cam,
            x="name",
            y="count",
            title="Detections per Camera",
            labels={"name": "Camera", "count": "Total Detections"},
            color="count",
            color_continuous_scale=[SURFACE_ALT, ACCENT_DIM, ACCENT],
        )
        st.plotly_chart(style_plotly(fig_cam, showlegend=False), use_container_width=True)

# ======================== TAB 2: PROCESS VIDEO =============================
with tab_process:
    st.markdown("### Upload & Process Video")
    st.caption("Upload a traffic video to detect and track license plates using YOLOv8 + ByteTrack + EasyOCR")

    col_upload, col_config = st.columns([2, 1])

    with col_config:
        with st.container(border=True):
            st.markdown("**Camera**")

            try:
                with open("cameras.json", "r") as f:
                    cameras_list = json.load(f)
                camera_names = {c["camera_id"]: c["name"] for c in cameras_list}
            except Exception:
                cameras_list = []
                camera_names = {}

            cam_option = st.selectbox(
                "Assign to camera",
                options=["— New Camera —"] + [f"{c['camera_id']} – {c['name']}" for c in cameras_list],
                help="Select an existing camera node or create a new one",
            )

            if cam_option == "— New Camera —":
                new_cam_name = st.text_input("Camera name", placeholder="e.g. Hosur Road Toll")
                new_cam_lat = st.number_input("Latitude", value=12.9716, format="%.4f")
                new_cam_lon = st.number_input("Longitude", value=77.5946, format="%.4f")
            else:
                selected_cam_id = cam_option.split(" – ")[0]

        with st.expander("Advanced processing settings", expanded=False):
            proc_conf = st.slider("Detection confidence", 0.1, 0.9, 0.35, 0.05)
            proc_ocr_n = st.slider("OCR every N frames", 1, 10, 3, help="Lower = more accurate but slower")
            proc_max_frames = st.number_input("Max frames (0 = all)", value=0, min_value=0, step=100)

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload a video file",
            type=["mp4", "avi", "mkv", "mov", "webm", "mpeg"],
            help="Supported: MP4, AVI, MKV, MOV, WebM, MPEG",
        )

        if uploaded_file is not None:
            st.video(uploaded_file)

            if st.button("PROCESS VIDEO", type="primary", use_container_width=True):
                if cam_option == "— New Camera —":
                    if not new_cam_name:
                        st.error("Please enter a camera name.")
                        st.stop()
                    cam_id = f"CAM_{new_cam_name.upper().replace(' ', '_')[:10]}"
                    upsert_camera(conn, cam_id, new_cam_name, new_cam_lat, new_cam_lon,
                                  description="Added via dashboard")
                    st.info(f"Created camera node: **{new_cam_name}** (`{cam_id}`)")
                else:
                    cam_id = selected_cam_id

                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                cap = cv2.VideoCapture(tmp_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                duration_sec = total_frames / fps if fps > 0 else 0
                cap.release()

                frames_to_process = proc_max_frames if proc_max_frames > 0 else total_frames

                st.caption(f"Video info: {total_frames} frames · {fps:.1f} FPS · {duration_sec:.1f}s")

                progress_bar = st.progress(0, text="Loading models...")
                status_text = st.empty()

                try:
                    @st.cache_resource
                    def get_models():
                        ensure_model(DEFAULT_MODEL_PATH, download=True)
                        device = resolve_device("auto")
                        model = load_detector(DEFAULT_MODEL_PATH)
                        reader = load_ocr("easyocr", device)
                        return model, reader, device

                    model, reader, device = get_models()
                    progress_bar.progress(10, text="Models loaded. Processing video...")
                except Exception as e:
                    st.error(f"Failed to load models: {e}")
                    st.stop()

                try:
                    output_dir = Path("results/tracked")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    out_video = str(output_dir / f"{cam_id}_{datetime.now().strftime('%H%M%S')}.mp4")

                    status_text.info("Running YOLO detection + ByteTrack tracking + OCR... this may take a while.")
                    progress_bar.progress(15, text="Processing frames...")

                    detections = process_video_with_tracking(
                        model=model,
                        reader=reader,
                        video_path=tmp_path,
                        conf=proc_conf,
                        iou=0.5,
                        imgsz=640,
                        device=device,
                        ocr_every_n=proc_ocr_n,
                        max_frames=proc_max_frames if proc_max_frames > 0 else 0,
                        show=False,
                        output_video=out_video,
                    )

                    progress_bar.progress(85, text="Saving to database...")

                    start_time = datetime.now()
                    alert_count = 0
                    for det in detections:
                        ts = (start_time + timedelta(seconds=det.first_frame / fps)).isoformat()
                        insert_detection(
                            conn, det.plate_text, cam_id, ts,
                            det.detection_confidence, det.plate_confidence,
                            det.bbox, det.track_id, det.first_frame,
                        )
                        reason = check_blacklist(conn, det.plate_text)
                        if reason:
                            insert_alert(conn, det.plate_text, cam_id, ts, reason)
                            alert_count += 1

                    progress_bar.progress(100, text="Complete")
                    status_text.empty()

                    st.success(
                        f"Processing complete. Found **{len(detections)}** unique vehicle(s) "
                        f"with readable plates across {frames_to_process} frames."
                    )

                    if alert_count > 0:
                        st.error(f"{alert_count} BLACKLISTED vehicle(s) detected. Check the Alerts tab.")

                    if detections:
                        st.markdown("#### Detected Vehicles")
                        results_data = []
                        for det in detections:
                            bl = check_blacklist(conn, det.plate_text)
                            results_data.append({
                                "Track ID": det.track_id,
                                "Plate": det.plate_text,
                                "Confidence": f"{det.plate_confidence:.0%}",
                                "Det. Conf": f"{det.detection_confidence:.0%}",
                                "Frames": f"{det.first_frame}–{det.last_frame}",
                                "Duration": f"{(det.last_frame - det.first_frame) / fps:.1f}s",
                                "Status": "BLACKLISTED" if bl else "Clear",
                            })
                        df_results = pd.DataFrame(results_data)
                        st.dataframe(df_results, use_container_width=True, hide_index=True)

                    out_path = Path(out_video)
                    if out_path.exists() and out_path.stat().st_size > 0:
                        st.markdown("#### Processed Video")
                        with open(out_video, "rb") as vf:
                            st.download_button(
                                "Download tracked video (with bounding boxes)",
                                data=vf.read(),
                                file_name=out_path.name,
                                mime="video/mp4",
                                use_container_width=True,
                            )

                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"Processing failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())

                finally:
                    try:
                        Path(tmp_path).unlink(missing_ok=True)
                    except Exception:
                        pass

# ========================= TAB 3: VEHICLE LOOKUP ==========================
with tab_lookup:
    st.markdown("### Vehicle Trajectory Lookup")

    all_plates = get_all_plates(conn)

    col_input, col_info = st.columns([2, 1])
    with col_input:
        plate_query = st.text_input(
            "Enter plate number",
            placeholder="e.g. MH12JC2813",
            help="Type a plate number to see its movement history",
        )
        if all_plates:
            st.caption(f"{len(all_plates)} plates in database. Try: {', '.join(all_plates[:5])}")

    if plate_query:
        plate_query = plate_query.strip().upper()
        history = query_plate_history(conn, plate_query)

        if not history:
            st.warning(f"No records found for plate **{plate_query}**")
        else:
            with col_info:
                bl_reason = check_blacklist(conn, plate_query)
                if bl_reason:
                    st.error(f"BLACKLISTED: {bl_reason}")
                else:
                    st.success("Not on blacklist")
                st.metric("Sightings", len(history))
                cameras_seen = len(set(h["camera_id"] for h in history))
                st.metric("Cameras", cameras_seen)

            df_hist = pd.DataFrame(history)
            display_cols = ["timestamp", "camera_name", "detection_conf", "ocr_conf"]
            available_cols = [c for c in display_cols if c in df_hist.columns]
            st.dataframe(
                df_hist[available_cols] if available_cols else df_hist,
                use_container_width=True,
                hide_index=True,
            )

            if cameras_seen >= 1:
                try:
                    from trajectory import generate_trajectory_map
                    traj_map = generate_trajectory_map(conn, plate_query)
                    if traj_map:
                        try:
                            from streamlit_folium import st_folium
                            st_folium(traj_map, width=None, height=450, returned_objects=[])
                        except ImportError:
                            import tempfile, os
                            with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
                                traj_map.save(f.name)
                                with open(f.name, "r") as hf:
                                    st.components.v1.html(hf.read(), height=450)
                                os.unlink(f.name)
                except ImportError:
                    st.info("Install `folium` and `streamlit-folium` for interactive trajectory maps.")
                except Exception as e:
                    st.error(f"Map generation error: {e}")

# ========================= TAB 4: TRAFFIC HEATMAP =========================
with tab_heatmap:
    st.markdown("### Camera Network Heatmap")

    heatmap_data = get_camera_heatmap_data(conn)
    if heatmap_data and any(h.get("latitude") for h in heatmap_data):
        try:
            from trajectory import generate_overview_map
            overview_map = generate_overview_map(conn)
            if overview_map:
                try:
                    from streamlit_folium import st_folium
                    st_folium(overview_map, width=None, height=550, returned_objects=[])
                except ImportError:
                    import tempfile, os
                    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
                        overview_map.save(f.name)
                        with open(f.name, "r") as hf:
                            st.components.v1.html(hf.read(), height=550)
                        os.unlink(f.name)
        except ImportError:
            st.info("Install `folium` and `streamlit-folium` for the interactive heatmap.")
        except Exception as e:
            st.error(f"Map error: {e}")

        df_heat = pd.DataFrame(heatmap_data)
        st.dataframe(df_heat, use_container_width=True, hide_index=True)
    else:
        st.info("No camera data available. Ensure `cameras.json` exists and data has been ingested.")

# ========================== TAB 5: TOP ROUTES ==============================
with tab_routes:
    st.markdown("### Most Common Vehicle Routes")

    routes = get_top_routes(conn, limit=15)
    if routes:
        df_routes = pd.DataFrame(routes)

        labels = list(set(df_routes["from_name"].tolist() + df_routes["to_name"].tolist()))
        label_idx = {name: i for i, name in enumerate(labels)}

        fig_sankey = go.Figure(go.Sankey(
            arrangement="snap",
            node=dict(
                pad=20,
                thickness=25,
                line=dict(color=BORDER, width=2),
                label=labels,
                color=[ACCENT_DIM] * len(labels),
            ),
            link=dict(
                source=[label_idx[r["from_name"]] for r in routes],
                target=[label_idx[r["to_name"]] for r in routes],
                value=[r["count"] for r in routes],
                color=["rgba(198,255,61,0.18)"] * len(routes),
            ),
        ))
        fig_sankey.update_layout(title="Vehicle Flow Between Cameras")
        st.plotly_chart(style_plotly(fig_sankey, height=400, showlegend=False), use_container_width=True)

        st.markdown("#### Route Details")
        for r in routes:
            avg_secs = r.get("avg_travel_seconds")
            time_str = f"{avg_secs / 60:.1f} min" if avg_secs and avg_secs > 0 else "N/A"
            st.markdown(
                f"**{r['from_name']}** → **{r['to_name']}**: "
                f"`{r['count']}` vehicle(s), avg travel: `{time_str}`"
            )
    else:
        st.info("No route data yet. Need vehicles detected at multiple cameras.")

# ========================== TAB 6: ALERTS ==================================
with tab_alerts:
    st.markdown("### Alert System")

    col_alerts, col_manage = st.columns([3, 1])

    with col_manage:
        st.markdown("#### Manage Blacklist")

        with st.form("add_blacklist", clear_on_submit=True):
            new_plate = st.text_input("Plate to blacklist", placeholder="e.g. KA01XX9999")
            new_reason = st.text_input("Reason", placeholder="e.g. Stolen vehicle")
            if st.form_submit_button("Add to Blacklist"):
                if new_plate:
                    add_to_blacklist(conn, new_plate.strip().upper(), new_reason or "Manual entry")
                    st.success(f"Added {new_plate.upper()} to blacklist")
                    st.rerun()

        bl = get_blacklist(conn)
        if bl:
            st.markdown("#### Current Blacklist")
            for entry in bl:
                col_p, col_r = st.columns([2, 1])
                col_p.code(entry["plate_text"])
                if col_r.button("Remove", key=f"rm_{entry['plate_text']}"):
                    remove_from_blacklist(conn, entry["plate_text"])
                    st.rerun()

    with col_alerts:
        show_all = st.checkbox("Show acknowledged alerts", value=False)
        alerts = get_alerts(conn, only_unacknowledged=not show_all)

        if alerts:
            for alert in alerts:
                severity = "●" if not alert.get("acknowledged") else "○"
                with st.expander(
                    f"{severity} {alert['plate_text']} @ {alert.get('timestamp', 'Unknown time')}",
                    expanded=not alert.get("acknowledged"),
                ):
                    st.markdown(f"**Plate:** `{alert['plate_text']}`")
                    st.markdown(f"**Camera:** {alert.get('camera_id', 'Unknown')}")
                    st.markdown(f"**Time:** {alert.get('timestamp', 'Unknown')}")
                    st.markdown(f"**Reason:** {alert.get('reason', 'Unknown')}")

                    if not alert.get("acknowledged"):
                        if st.button("Acknowledge", key=f"ack_{alert['id']}"):
                            acknowledge_alert(conn, alert["id"])
                            st.rerun()
        else:
            if show_all:
                st.info("No alerts recorded.")
            else:
                st.success("No pending alerts — all clear.")
