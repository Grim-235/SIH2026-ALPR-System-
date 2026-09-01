#!/usr/bin/env python
"""City-Wide ANPR Traffic Analytics Dashboard — Streamlit App."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import cv2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Custom module imports with graceful degradation
# ---------------------------------------------------------------------------
_ALPR_OK = False
try:
    from alpr.database import (
        acknowledge_alert,
        add_to_blacklist,
        check_blacklist,
        create_job,
        get_alerts,
        get_all_plates,
        get_blacklist,
        get_camera_heatmap_data,
        get_detection_stats,
        get_detections_over_time,
        get_job_status,
        get_top_routes,
        init_db,
        load_blacklist_from_file,
        load_cameras_from_json,
        query_plate_history,
        remove_from_blacklist,
        upsert_camera,
    )
    _ALPR_OK = True
except ImportError as _imp_err:
    _ALPR_ERR = str(_imp_err)

_TRAJ_OK = False
try:
    from trajectory import generate_overview_map, generate_trajectory_map
    _TRAJ_OK = True
except ImportError:
    generate_overview_map = None
    generate_trajectory_map = None

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
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
    .stApp {
        background-color: #F4F4F0;
        color: #111111;
        font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    }
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 2px solid #111111;
        border-radius: 0px;
        padding: 16px 20px;
        box-shadow: 3px 3px 0px #111111;
    }
    div[data-testid="stMetric"] label {
        color: #555555 !important;
        font-size: 0.75rem !important;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #111111 !important;
        font-size: 2rem !important;
        font-weight: 800;
    }
    section[data-testid="stSidebar"] {
        background-color: #EBEBE6 !important;
        border-right: 2px solid #111111;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: transparent;
        border-bottom: 2px solid #111111;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 0;
        color: #555555;
        padding: 8px 16px;
        border: 2px solid transparent;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background: #FFFFFF !important;
        color: #111111 !important;
        border: 2px solid #111111;
        border-bottom: 2px solid #FFFFFF;
        margin-bottom: -2px;
        font-weight: 800;
    }
    h1, h2, h3, h4, h5 {
        color: #111111 !important;
        font-weight: 700;
    }
    .alert-banner {
        background-color: #D92D20;
        color: #FFFFFF;
        padding: 12px 24px;
        border: 2px solid #111111;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 16px;
        box-shadow: 4px 4px 0px #111111;
    }
    .stButton > button {
        border-radius: 0px;
        font-weight: 700;
        background-color: #111111;
        color: #FFFFFF;
        border: 2px solid #111111;
        transition: none;
    }
    .stButton > button:hover {
        background-color: #FFFFFF;
        color: #111111;
        box-shadow: 2px 2px 0px #111111;
        border-color: #111111;
    }
    .stDataFrame {
        border: 2px solid #111111;
        border-radius: 0px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Early exit if core backend is missing
# ---------------------------------------------------------------------------
if not _ALPR_OK:
    st.error("🚨 Critical backend module `alpr.database` is missing.")
    st.code(_ALPR_ERR, language="bash")
    st.info("Please ensure your ALPR package is installed or on PYTHONPATH.")
    st.stop()

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
    if CAMERAS_JSON.exists():
        try:
            load_cameras_from_json(conn, str(CAMERAS_JSON))
        except Exception as e:
            st.warning(f"Could not load cameras.json: {e}")
    if BLACKLIST_FILE.exists():
        try:
            load_blacklist_from_file(conn, str(BLACKLIST_FILE))
        except Exception as e:
            st.warning(f"Could not load blacklist.txt: {e}")
    return conn


try:
    conn = get_db()
except Exception as e:
    st.error(f"Database initialization failed: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# Check for unacknowledged alerts — show banner at top
# ---------------------------------------------------------------------------
try:
    unack_alerts = get_alerts(conn, only_unacknowledged=True)
except Exception as e:
    st.warning(f"Alert check failed: {e}")
    unack_alerts = []

if unack_alerts:
    st.markdown(
        f'<div class="alert-banner">🚨 {len(unack_alerts)} UNACKNOWLEDGED ALERT(S) — '
        f"Blacklisted vehicle(s) detected! Go to Alert Panel.</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🚗 City ANPR")
    st.markdown("##### Real-Time Traffic Intelligence")
    st.divider()

    try:
        stats = get_detection_stats(conn)
    except Exception as e:
        st.warning(f"Stats error: {e}")
        stats = {}

    st.metric("Total Detections", f"{stats.get('total_detections', 0):,}")
    st.metric("Unique Vehicles", f"{stats.get('unique_plates', 0):,}")
    st.metric("Active Cameras", f"{stats.get('unique_cameras', 0):,}")

    if stats.get("min_time") and stats.get("max_time"):
        st.caption(f"📅 {stats['min_time'][:16]} → {stats['max_time'][:16]}")

    st.divider()
    st.caption("Powered by YOLOv8 + EasyOCR + ByteTrack")
    st.caption("SIH 2026 • PS #26127")

# ---------------------------------------------------------------------------
# Main Tabs
# ---------------------------------------------------------------------------
tab_process, tab_overview, tab_lookup, tab_heatmap, tab_routes, tab_alerts = st.tabs(
    [
        "📹 Process Video",
        "📊 Overview",
        "🔍 Vehicle Lookup",
        "🗺️ Traffic Heatmap",
        "🛣️ Top Routes",
        "🚨 Alerts",
    ]
)

# ======================== TAB 0: PROCESS VIDEO =============================
with tab_process:
    st.markdown("### 📹 Upload & Process Video (Async)")
    st.caption(
        "Upload videos for background processing. The UI stays responsive while videos are analyzed."
    )

    col_upload, col_config = st.columns([2, 1])

    with col_config:
        st.markdown("#### Camera Settings")

        cameras_list = []
        camera_names = {}
        try:
            if CAMERAS_JSON.exists():
                with open(CAMERAS_JSON, "r", encoding="utf-8") as f:
                    cameras_list = json.load(f)
                camera_names = {c["camera_id"]: c["name"] for c in cameras_list}
        except Exception as e:
            st.warning(f"Could not read cameras.json: {e}")

        cam_option = st.selectbox(
            "Assign to camera",
            options=["— New Camera —"]
            + [f"{c['camera_id']} – {c['name']}" for c in cameras_list],
            help="Select an existing camera node or create a new one",
            key="cam_select",
        )

        new_cam_name = ""
        new_cam_lat = 12.9716
        new_cam_lon = 77.5946
        selected_cam_id = None

        if cam_option == "— New Camera —":
            new_cam_name = st.text_input(
                "Camera name", placeholder="e.g. Hosur Road Toll", key="new_cam_name"
            )
            new_cam_lat = st.number_input(
                "Latitude", value=12.9716, format="%.4f", key="new_cam_lat"
            )
            new_cam_lon = st.number_input(
                "Longitude", value=77.5946, format="%.4f", key="new_cam_lon"
            )
        else:
            selected_cam_id = cam_option.split(" – ")[0]

        st.markdown("#### Processing Settings")
        proc_conf = st.slider(
            "Detection confidence", 0.1, 0.9, 0.35, 0.05, key="proc_conf"
        )
        proc_ocr_n = st.slider(
            "OCR every N frames",
            1,
            10,
            3,
            help="Lower = more accurate but slower",
            key="proc_ocr",
        )
        proc_max_frames = st.number_input(
            "Max frames (0 = all)", value=0, min_value=0, step=100, key="proc_frames"
        )

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload a video file",
            type=["mp4", "avi", "mkv", "mov", "webm", "mpeg"],
            help="Supported: MP4, AVI, MKV, MOV, WebM, MPEG",
            key="video_uploader",
        )

        if uploaded_file is not None:
            # Ensure we read from the start for st.video
            uploaded_file.seek(0)
            st.video(uploaded_file)

            if st.button(
                "🚀 Submit for Processing", type="primary", use_container_width=True
            ):
                # Determine camera ID
                if cam_option == "— New Camera —":
                    name_clean = new_cam_name.strip()
                    if not name_clean:
                        st.error("Please enter a camera name.")
                        st.stop()
                    cam_id = f"CAM_{name_clean.upper().replace(' ', '_')[:10]}"
                    try:
                        upsert_camera(
                            conn,
                            cam_id,
                            name_clean,
                            new_cam_lat,
                            new_cam_lon,
                            description="Added via dashboard",
                        )
                    except Exception as e:
                        st.error(f"Failed to create camera: {e}")
                        st.stop()
                    st.info(f"Created camera node: **{name_clean}** (`{cam_id}`)")
                else:
                    if selected_cam_id is None:
                        st.error("Camera selection error. Please re-select.")
                        st.stop()
                    cam_id = selected_cam_id

                # Save uploaded file to temp location
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                except Exception as e:
                    st.error(f"Failed to save upload: {e}")
                    st.stop()

                # Get video info
                cap = cv2.VideoCapture(tmp_path)
                if not cap.isOpened():
                    st.error("Could not open video file. It may be corrupt or unsupported.")
                    os.unlink(tmp_path)
                    st.stop()

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                duration_sec = total_frames / fps if fps > 0 else 0
                cap.release()

                st.markdown(
                    f"**Video info:** {total_frames} frames, {fps:.1f} FPS, {duration_sec:.1f}s duration"
                )

                # Create a job in the database
                job_id = str(uuid.uuid4())[:8]
                try:
                    create_job(conn, job_id, cam_id, tmp_path, total_frames)
                except Exception as e:
                    st.error(f"Failed to create job: {e}")
                    os.unlink(tmp_path)
                    st.stop()

                st.success(
                    f"✅ **Job {job_id} submitted!** Processing in background..."
                )
                st.info(
                    "You can navigate other tabs while processing continues. Results will appear automatically when complete."
                )

                # Spawn worker subprocess
                worker_script = Path(__file__).parent / "worker.py"
                if not worker_script.exists():
                    st.error(f"Worker script not found: {worker_script}")
                    os.unlink(tmp_path)
                    st.stop()

                cmd = [
                    sys.executable,
                    str(worker_script),
                    "--job-id",
                    job_id,
                    "--db",
                    str(DB_PATH),
                    "--video",
                    tmp_path,
                    "--camera",
                    cam_id,
                    "--conf",
                    str(proc_conf),
                    "--ocr-every-n",
                    str(proc_ocr_n),
                    "--max-frames",
                    str(proc_max_frames),
                ]

                try:
                    # Capture stderr to a log so you can debug worker crashes
                    log_path = Path(tempfile.gettempdir()) / f"anpr_worker_{job_id}.log"
                    with open(log_path, "w") as log_f:
                        subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
                    st.write(f"📌 **Job ID:** `{job_id}` — Use this to track progress")
                    st.caption(f"Worker log: `{log_path}`")
                except Exception as e:
                    st.error(f"Failed to start background worker: {e}")
                    os.unlink(tmp_path)

    # Show job status
    st.divider()
    st.markdown("### 📊 Processing Jobs")

    tracker_col1, tracker_col2, tracker_col3 = st.columns(3)
    with tracker_col1:
        job_id_input = st.text_input(
            "Track job by ID", placeholder="e.g. abc123de", key="job_tracker"
        )

    if job_id_input:
        job_id_clean = job_id_input.strip()
        try:
            job_status = get_job_status(conn, job_id_clean)
        except Exception as e:
            st.error(f"Could not fetch job status: {e}")
            job_status = None

        if job_status:
            with tracker_col2:
                status_badge = {
                    "pending": "⏳ Pending",
                    "processing": "🔄 Processing",
                    "completed": "✅ Completed",
                    "failed": "❌ Failed",
                }.get(job_status.get("status"), job_status.get("status", "Unknown"))
                st.metric("Status", status_badge)

            with tracker_col3:
                if job_status.get("status") == "completed":
                    st.metric("Detections", job_status.get("detections_found", 0))
                elif job_status.get("status") == "processing":
                    progress_pct = job_status.get("progress", 0)
                    st.metric("Progress", f"{progress_pct}%")

            # Detailed status
            st.json(
                {
                    k: job_status.get(k)
                    for k in [
                        "job_id",
                        "status",
                        "progress",
                        "detections_found",
                        "created_at",
                        "completed_at",
                        "error_message",
                    ]
                }
            )

            if job_status.get("status") == "completed" and job_status.get("output_video"):
                out_path = Path(job_status["output_video"])
                st.success(f"✅ Processed video saved: `{out_path}`")
                if out_path.exists():
                    try:
                        with open(out_path, "rb") as vf:
                            st.download_button(
                                "⬇️ Download tracked video",
                                data=vf.read(),
                                file_name=out_path.name,
                                mime="video/mp4",
                                use_container_width=True,
                            )
                    except Exception as e:
                        st.error(f"Could not read output video: {e}")
                else:
                    st.warning("Output video path does not exist on disk yet.")
        else:
            st.warning(f"Job `{job_id_clean}` not found")

# ========================== TAB 1: OVERVIEW ================================
with tab_overview:
    st.markdown("### 📊 Detection Overview")

    try:
        time_data = get_detections_over_time(conn, bucket_minutes=5)
    except Exception as e:
        st.warning(f"Could not load time data: {e}")
        time_data = []

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
            line=dict(color="#111111", width=2),
            fillcolor="rgba(17,17,17,0.1)",
        )
        fig_time.update_layout(
            template="simple_white",
            font=dict(color="#111111"),
            title_font=dict(color="#111111", size=16),
            height=350,
        )
        st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.info(
            "No detection data yet. Run `process_videos.py` or `simulate_cameras.py` to populate."
        )

    try:
        heatmap_data = get_camera_heatmap_data(conn)
    except Exception as e:
        st.warning(f"Could not load camera data: {e}")
        heatmap_data = []

    if heatmap_data:
        df_cam = pd.DataFrame(heatmap_data)
        fig_cam = px.bar(
            df_cam,
            x="name",
            y="count",
            title="Detections per Camera",
            labels={"name": "Camera", "count": "Total Detections"},
            color="count",
            color_continuous_scale=["#CCCCCC", "#555555", "#111111"],
        )
        fig_cam.update_layout(
            template="simple_white",
            font=dict(color="#111111"),
            title_font=dict(color="#111111", size=16),
            showlegend=False,
            height=350,
        )
        st.plotly_chart(fig_cam, use_container_width=True)

# ========================= TAB 2: VEHICLE LOOKUP ==========================
with tab_lookup:
    st.markdown("### 🔍 Vehicle Trajectory Lookup")

    try:
        all_plates = get_all_plates(conn) or []
    except Exception as e:
        st.warning(f"Could not load plates: {e}")
        all_plates = []

    col_input, col_info = st.columns([2, 1])
    with col_input:
        plate_query = st.text_input(
            "Enter plate number",
            placeholder="e.g. MH12JC2813",
            help="Type a plate number to see its movement history",
        )
        if all_plates:
            st.caption(
                f"💡 {len(all_plates)} plates in database. Try: {', '.join(all_plates[:5])}"
            )

    if plate_query:
        plate_query = plate_query.strip().upper()
        try:
            history = query_plate_history(conn, plate_query)
        except Exception as e:
            st.warning(f"Lookup failed: {e}")
            history = []

        if not history:
            st.warning(f"No records found for plate **{plate_query}**")
        else:
            with col_info:
                try:
                    bl_reason = check_blacklist(conn, plate_query)
                except Exception:
                    bl_reason = None
                if bl_reason:
                    st.error(f"⚠️ BLACKLISTED: {bl_reason}")
                else:
                    st.success("✅ Not on blacklist")
                st.metric("Sightings", len(history))
                cameras_seen = len(set(h.get("camera_id") for h in history))
                st.metric("Cameras", cameras_seen)

            # Sightings table
            df_hist = pd.DataFrame(history)
            display_cols = ["timestamp", "camera_name", "detection_conf", "ocr_conf"]
            available_cols = [c for c in display_cols if c in df_hist.columns]
            st.dataframe(
                df_hist[available_cols] if available_cols else df_hist,
                use_container_width=True,
                hide_index=True,
            )

            # Trajectory map
            if cameras_seen >= 1 and _TRAJ_OK and generate_trajectory_map:
                try:
                    traj_map = generate_trajectory_map(conn, plate_query)
                    if traj_map:
                        try:
                            from streamlit_folium import st_folium
                            st_folium(traj_map, width=None, height=450, returned_objects=[])
                        except ImportError:
                            import tempfile
                            with tempfile.NamedTemporaryFile(
                                suffix=".html", delete=False, mode="w", encoding="utf-8"
                            ) as f:
                                traj_map.save(f.name)
                                with open(f.name, "r", encoding="utf-8") as hf:
                                    st.components.v1.html(hf.read(), height=450)
                                os.unlink(f.name)
                except Exception as e:
                    st.error(f"Map generation error: {e}")
            elif not _TRAJ_OK:
                st.info("Trajectory module not available. Maps are disabled.")

# ========================= TAB 3: TRAFFIC HEATMAP =========================
with tab_heatmap:
    st.markdown("### 🗺️ Camera Network Heatmap")

    try:
        heatmap_data = get_camera_heatmap_data(conn)
    except Exception as e:
        st.warning(f"Could not load heatmap data: {e}")
        heatmap_data = []

    if heatmap_data and any(h.get("latitude") for h in heatmap_data):
        if _TRAJ_OK and generate_overview_map:
            try:
                overview_map = generate_overview_map(conn)
                if overview_map:
                    try:
                        from streamlit_folium import st_folium
                        st_folium(overview_map, width=None, height=550, returned_objects=[])
                    except ImportError:
                        import tempfile
                        with tempfile.NamedTemporaryFile(
                            suffix=".html", delete=False, mode="w", encoding="utf-8"
                        ) as f:
                            overview_map.save(f.name)
                            with open(f.name, "r", encoding="utf-8") as hf:
                                st.components.v1.html(hf.read(), height=550)
                            os.unlink(f.name)
            except Exception as e:
                st.error(f"Map error: {e}")
        else:
            st.info("Overview map module not available.")

        # Also show the data table
        df_heat = pd.DataFrame(heatmap_data)
        st.dataframe(df_heat, use_container_width=True, hide_index=True)
    else:
        st.info(
            "No camera data available. Ensure `cameras.json` exists and data has been ingested."
        )

# ========================== TAB 4: TOP ROUTES ==============================
with tab_routes:
    st.markdown("### 🛣️ Most Common Vehicle Routes")

    try:
        routes = get_top_routes(conn, limit=15)
    except Exception as e:
        st.warning(f"Could not load routes: {e}")
        routes = []

    if routes:
        df_routes = pd.DataFrame(routes)

        # Sankey-style flow chart
        labels = list(
            set(df_routes["from_name"].tolist() + df_routes["to_name"].tolist())
        )
        label_idx = {name: i for i, name in enumerate(labels)}

        fig_sankey = go.Figure(
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=20,
                    thickness=25,
                    line=dict(color="#111111", width=2),
                    label=labels,
                    color=["#555555"] * len(labels),
                ),
                link=dict(
                    source=[label_idx[r["from_name"]] for r in routes],
                    target=[label_idx[r["to_name"]] for r in routes],
                    value=[r["count"] for r in routes],
                    color=["rgba(17,17,17,0.2)"] * len(routes),
                ),
            )
        )
        fig_sankey.update_layout(
            title="Vehicle Flow Between Cameras",
            template="simple_white",
            font=dict(color="#111111"),
            title_font=dict(color="#111111", size=16),
            height=400,
        )
        st.plotly_chart(fig_sankey, use_container_width=True)

        # Routes table with avg travel time
        st.markdown("#### Route Details")
        for r in routes:
            avg_secs = r.get("avg_travel_seconds")
            time_str = (
                f"{avg_secs / 60:.1f} min" if avg_secs and avg_secs > 0 else "N/A"
            )
            st.markdown(
                f"**{r['from_name']}** → **{r['to_name']}**: "
                f"`{r['count']}` vehicle(s), avg travel: `{time_str}`"
            )
    else:
        st.info("No route data yet. Need vehicles detected at multiple cameras.")

# ========================== TAB 5: ALERTS ==================================
with tab_alerts:
    st.markdown("### 🚨 Alert System")

    col_alerts, col_manage = st.columns([3, 1])

    with col_manage:
        st.markdown("#### Manage Blacklist")

        # Add to blacklist
        with st.form("add_blacklist", clear_on_submit=True):
            new_plate = st.text_input(
                "Plate to blacklist", placeholder="e.g. KA01XX9999"
            )
            new_reason = st.text_input("Reason", placeholder="e.g. Stolen vehicle")
            if st.form_submit_button("➕ Add to Blacklist"):
                plate_clean = new_plate.strip().upper()
                if plate_clean:
                    try:
                        add_to_blacklist(
                            conn, plate_clean, new_reason.strip() or "Manual entry"
                        )
                        st.success(f"Added {plate_clean} to blacklist")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to add: {e}")
                else:
                    st.warning("Please enter a plate number.")

        # Current blacklist
        try:
            bl = get_blacklist(conn)
        except Exception as e:
            st.warning(f"Could not load blacklist: {e}")
            bl = []

        if bl:
            st.markdown("#### Current Blacklist")
            for idx, entry in enumerate(bl):
                col_p, col_r = st.columns([2, 1])
                col_p.code(entry.get("plate_text", "N/A"))
                plate_key = entry.get("plate_text", f"unknown_{idx}")
                if col_r.button("🗑️", key=f"rm_{plate_key}_{idx}"):
                    try:
                        remove_from_blacklist(conn, entry["plate_text"])
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to remove: {e}")

    with col_alerts:
        # Alert history
        show_all = st.checkbox("Show acknowledged alerts", value=False)
        try:
            alerts = get_alerts(conn, only_unacknowledged=not show_all)
        except Exception as e:
            st.warning(f"Could not load alerts: {e}")
            alerts = []

        if alerts:
            for alert in alerts:
                severity = "🔴" if not alert.get("acknowledged") else "⚪"
                alert_id = alert.get("id", "unknown")
                alert_plate = alert.get("plate_text", "Unknown")
                alert_time = alert.get("timestamp", "Unknown time")
                with st.expander(
                    f"{severity} {alert_plate} @ {alert_time}",
                    expanded=not alert.get("acknowledged"),
                ):
                    st.markdown(f"**Plate:** `{alert_plate}`")
                    st.markdown(f"**Camera:** {alert.get('camera_id', 'Unknown')}")
                    st.markdown(f"**Time:** {alert_time}")
                    st.markdown(f"**Reason:** {alert.get('reason', 'Unknown')}")

                    if not alert.get("acknowledged"):
                        if st.button("✅ Acknowledge", key=f"ack_{alert_id}"):
                            try:
                                acknowledge_alert(conn, alert_id)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to acknowledge: {e}")
        else:
            if show_all:
                st.info("No alerts recorded.")
            else:
                st.success("✅ No pending alerts — all clear!")
