"""
GIS Map Integration & GeoJSON Adapter Module (Phase 7D).

Provides GIS representation and interactive map visualization of urban traffic networks:
- build_network_geojson(): Converts camera configuration, topological graph,
  and analytical NetworkCongestionReport into standard GeoJSON FeatureCollections.
  * Strictly presentation layer: consumes pre-computed metrics from Phase 7C without recalculation.
  * Preserves directed graph topology (only configured edges).
  * Direct color assignment based on los_proxy (get_los_color).
- generate_city_traffic_map(): Renders interactive Folium map with:
  * Positron, Dark Mode, and OpenStreetMap tile layers with LayerControl.
  * Styled camera nodes with throughput and temporal occupancy popups.
  * Directional corridor links colored by project LOS proxy.
  * Optional interactive vehicle trajectory overlay (Phase 7A GeoJSON integration).
  * Prominent Level of Service project proxy disclaimer.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import folium
    from folium import plugins
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

logger = logging.getLogger("alpr.gis")

# Centralized color mapping based directly on los_proxy string
LOS_COLOR_MAP: Dict[str, str] = {
    "A": "#10b981",  # Emerald Green - Free Flow
    "B": "#84cc16",  # Lime Green - Light
    "C": "#eab308",  # Amber Yellow - Moderate
    "D": "#f97316",  # Orange - Heavy
    "E": "#ef4444",  # Bright Red - Severe
    "F": "#991b1b",  # Crimson Red - Breakdown
}

COLOR_SPARSE = "#94a3b8"  # Slate Grey - Insufficient / No Data
COLOR_TRAJECTORY = "#0284c7"  # Sky Blue - Vehicle Route Overlay


def get_los_color(los_proxy: Optional[str]) -> str:
    """
    Return hexadecimal color code for a project LOS proxy.
    Direct presentation mapping; does NOT compute or alter traffic metrics.
    """
    if not los_proxy:
        return COLOR_SPARSE
    clean_proxy = str(los_proxy).strip().upper()
    return LOS_COLOR_MAP.get(clean_proxy, COLOR_SPARSE)


def build_network_geojson(
    cameras_path: Union[str, Path] = "configs/cameras.json",
    camera_graph_path: Optional[Union[str, Path]] = "configs/camera_graph.json",
    congestion_report: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Construct a GeoJSON FeatureCollection containing:
    1. Camera nodes as Point features with throughput and occupancy properties.
    2. Corridors as LineString features with congestion metrics from congestion_report.

    Strict architectural constraint: Does NOT independently calculate speed, TTI,
    SPI, density, or LOS. Strictly transforms input data and report into GeoJSON.
    """
    features: List[Dict[str, Any]] = []
    cameras_dict: Dict[str, dict] = {}

    c_path = Path(cameras_path)
    if c_path.exists():
        try:
            with open(c_path, "r", encoding="utf-8") as f:
                cam_list = json.load(f)
            for c in cam_list:
                cid = c.get("camera_id")
                if cid:
                    cameras_dict[cid] = c
        except Exception as e:
            logger.warning(f"Failed loading cameras from {c_path}: {e}")

    # Extract camera node metrics if report is provided
    node_metrics = getattr(congestion_report, "camera_node_metrics", {}) if congestion_report else {}
    corridor_metrics = getattr(congestion_report, "corridor_metrics", {}) if congestion_report else {}

    # 1. Build Camera Node Point Features
    for cid, cam in cameras_dict.items():
        lat = cam.get("latitude")
        lon = cam.get("longitude")
        if lat is None or lon is None:
            continue

        c_metric = node_metrics.get(cid)
        flow_hr = c_metric.camera_flow_rate_veh_hr if c_metric else 0.0
        occ_ratio = c_metric.estimated_temporal_occupancy_ratio if c_metric else 0.0
        uniq_veh = c_metric.unique_vehicles_observed if c_metric else 0

        features.append({
            "type": "Feature",
            "id": f"node-{cid}",
            "geometry": {
                "type": "Point",
                "coordinates": [float(lon), float(lat)],
            },
            "properties": {
                "feature_type": "camera_node",
                "camera_id": cid,
                "name": cam.get("name", cid),
                "latitude": float(lat),
                "longitude": float(lon),
                "description": cam.get("description", ""),
                "status": cam.get("status", "unknown"),
                "camera_flow_rate_veh_hr": round(float(flow_hr), 2),
                "estimated_temporal_occupancy_ratio": round(float(occ_ratio), 4),
                "estimated_temporal_occupancy_pct": round(float(occ_ratio) * 100.0, 2),
                "unique_vehicles_observed": int(uniq_veh),
            },
        })

    # 2. Build Directed Corridor LineString Features
    # Topology: strictly follow configured directed edges from camera_graph.json
    # or active corridors present in corridor_metrics.
    configured_edges: List[Tuple[str, str]] = []

    if camera_graph_path:
        g_path = Path(camera_graph_path)
        if g_path.exists():
            try:
                with open(g_path, "r", encoding="utf-8") as f:
                    graph_data = json.load(f)
                for src_id, node_info in graph_data.items():
                    for tgt_id in node_info.get("neighbors", []):
                        configured_edges.append((src_id, tgt_id))
            except Exception as e:
                logger.warning(f"Failed loading camera graph from {g_path}: {e}")

    # If no graph was loaded, fallback to corridors present in the congestion report
    if not configured_edges and corridor_metrics:
        for corr_key in corridor_metrics.keys():
            if isinstance(corr_key, tuple) and len(corr_key) == 2:
                configured_edges.append(corr_key)
            elif isinstance(corr_key, str) and " -> " in corr_key:
                parts = corr_key.split(" -> ")
                if len(parts) == 2:
                    configured_edges.append((parts[0].strip(), parts[1].strip()))

    seen_edges = set()
    for src_id, tgt_id in configured_edges:
        edge_key = (src_id, tgt_id)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)

        src_cam = cameras_dict.get(src_id)
        tgt_cam = cameras_dict.get(tgt_id)
        if not src_cam or not tgt_cam:
            continue

        s_lat, s_lon = src_cam.get("latitude"), src_cam.get("longitude")
        t_lat, t_lon = tgt_cam.get("latitude"), tgt_cam.get("longitude")
        if None in (s_lat, s_lon, t_lat, t_lon):
            continue

        # Look up pre-calculated metrics directly from report
        corr_metric = corridor_metrics.get(edge_key) or corridor_metrics.get(f"{src_id} -> {tgt_id}")

        if corr_metric:
            los_proxy = corr_metric.los_proxy
            color = get_los_color(los_proxy)
            corridor_props = {
                "feature_type": "corridor",
                "corridor_id": f"{src_id} -> {tgt_id}",
                "from_camera_id": src_id,
                "to_camera_id": tgt_id,
                "from_name": src_cam.get("name", src_id),
                "to_name": tgt_cam.get("name", tgt_id),
                "sample_size": corr_metric.observation_count,
                "transit_rate_veh_hr": round(float(corr_metric.corridor_transit_rate_veh_hr), 2),
                "speed_median_kmh": round(float(corr_metric.speed_median_kmh), 2) if corr_metric.speed_median_kmh is not None else None,
                "free_flow_speed_kmh": round(float(corr_metric.free_flow_speed_kmh), 2) if corr_metric.free_flow_speed_kmh is not None else None,
                "free_flow_source": corr_metric.free_flow_source,
                "travel_time_index": round(float(corr_metric.travel_time_index), 3) if corr_metric.travel_time_index is not None else None,
                "speed_performance_index": round(float(corr_metric.speed_performance_index), 2) if corr_metric.speed_performance_index is not None else None,
                "speed_degradation_pct": round(float(corr_metric.speed_degradation_pct), 2) if corr_metric.speed_degradation_pct is not None else None,
                "travel_time_increase_pct": round(float(corr_metric.travel_time_increase_pct), 2) if corr_metric.travel_time_increase_pct is not None else None,
                "los_proxy": los_proxy,
                "congestion_category": corr_metric.congestion_category,
                "sample_confidence_score": round(float(corr_metric.sample_confidence_score), 2),
                "color": color,
                "weight": 5 if corr_metric.travel_time_index and corr_metric.travel_time_index >= 1.5 else 4,
            }
        else:
            corridor_props = {
                "feature_type": "corridor",
                "corridor_id": f"{src_id} -> {tgt_id}",
                "from_camera_id": src_id,
                "to_camera_id": tgt_id,
                "from_name": src_cam.get("name", src_id),
                "to_name": tgt_cam.get("name", tgt_id),
                "sample_size": 0,
                "transit_rate_veh_hr": 0.0,
                "speed_median_kmh": None,
                "free_flow_speed_kmh": None,
                "free_flow_source": "NONE",
                "travel_time_index": None,
                "speed_performance_index": None,
                "speed_degradation_pct": None,
                "travel_time_increase_pct": None,
                "los_proxy": "UNKNOWN",
                "congestion_category": "NO_OBSERVATIONS",
                "sample_confidence_score": 0.0,
                "color": get_los_color("UNKNOWN"),
                "weight": 3,
            }

        features.append({
            "type": "Feature",
            "id": f"corridor-{src_id}-{tgt_id}",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [float(s_lon), float(s_lat)],
                    [float(t_lon), float(t_lat)],
                ],
            },
            "properties": corridor_props,
        })

    return {
        "type": "FeatureCollection",
        "metadata": {
            "title": "City-Wide ANPR Surveillance & Traffic Network",
            "los_proxy_disclaimer": "Level of Service (LOS) Proxy is a TTI-based project classification and is NOT claimed as facility-specific HCM LOS methodology.",
            "total_nodes": len([f for f in features if f["properties"]["feature_type"] == "camera_node"]),
            "total_corridors": len([f for f in features if f["properties"]["feature_type"] == "corridor"]),
        },
        "features": features,
    }


def generate_city_traffic_map(
    cameras_path: Union[str, Path] = "configs/cameras.json",
    camera_graph_path: Optional[Union[str, Path]] = "configs/camera_graph.json",
    congestion_report: Optional[Any] = None,
    active_trajectory_geojson: Optional[Dict[str, Any]] = None,
    default_center: Tuple[float, float] = (12.9716, 77.5946),
) -> Optional[Any]:
    """
    Generate an interactive Folium Map instance rendering camera nodes,
    color-coded corridors, and optional active vehicle trajectory overlay.
    """
    if not FOLIUM_AVAILABLE:
        logger.error("Folium is not installed. Cannot generate map.")
        return None

    network_geojson = build_network_geojson(
        cameras_path=cameras_path,
        camera_graph_path=camera_graph_path,
        congestion_report=congestion_report,
    )

    node_features = [f for f in network_geojson["features"] if f["properties"]["feature_type"] == "camera_node"]
    corridor_features = [f for f in network_geojson["features"] if f["properties"]["feature_type"] == "corridor"]

    # Determine center location
    if node_features:
        lats = [f["properties"]["latitude"] for f in node_features]
        lons = [f["properties"]["longitude"] for f in node_features]
        center = (sum(lats) / len(lats), sum(lons) / len(lons))
    else:
        center = default_center

    m = folium.Map(
        location=list(center),
        zoom_start=12,
        tiles="CartoDB positron",
        control_scale=True,
    )

    # Base Tile Layers
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Dark Mode").add_to(m)

    # Feature Groups
    corridors_group = folium.FeatureGroup(name="Traffic Corridors (LOS Proxy)", show=True)
    cameras_group = folium.FeatureGroup(name="Camera Nodes (Throughput & Occupancy)", show=True)
    hotspots_group = folium.FeatureGroup(name="Congestion Hotspots", show=True)

    # Render Corridors
    for cf in corridor_features:
        props = cf["properties"]
        coords = cf["geometry"]["coordinates"]
        latlon = [[pt[1], pt[0]] for pt in coords]
        color = props.get("color", COLOR_SPARSE)
        weight = props.get("weight", 4)
        is_dash = props.get("sample_size", 0) == 0

        tti_val = f"{props['travel_time_index']:.2f}" if props.get("travel_time_index") is not None else "N/A"
        speed_val = f"{props['speed_median_kmh']:.1f} km/h" if props.get("speed_median_kmh") is not None else "N/A"
        flow_val = f"{props['transit_rate_veh_hr']:.1f} veh/hr" if props.get("transit_rate_veh_hr") is not None else "N/A"

        popup_html = f"""
        <div style="font-family:Inter,sans-serif; min-width:220px; font-size:12px; color:#0f172a; padding:4px;">
            <div style="font-weight:700; color:#005577; font-size:14px; margin-bottom:4px; border-bottom:1px solid #e2e8f0; padding-bottom:4px;">
                {props['corridor_id']}
            </div>
            <div style="margin:3px 0;"><b>LOS Proxy:</b> <span style="font-weight:700; color:{color};">{props['los_proxy']}</span> ({props['congestion_category']})</div>
            <div style="margin:3px 0;"><b>Travel Time Index (TTI):</b> {tti_val}</div>
            <div style="margin:3px 0;"><b>Median Speed:</b> {speed_val}</div>
            <div style="margin:3px 0;"><b>Transit Rate:</b> {flow_val}</div>
            <div style="margin:3px 0;"><b>Sample Size (N):</b> {props['sample_size']}</div>
            <div style="margin-top:6px; font-size:10px; color:#64748b; font-style:italic; border-top:1px dashed #e2e8f0; padding-top:4px;">
                Project proxy classification; not HCM LOS.
            </div>
        </div>
        """

        folium.PolyLine(
            locations=latlon,
            color=color,
            weight=weight,
            opacity=0.85,
            dash_array="6, 6" if is_dash else None,
            tooltip=f"{props['corridor_id']} | LOS {props['los_proxy']} | TTI: {tti_val}",
            popup=folium.Popup(popup_html, max_width=300),
        ).add_to(corridors_group)

    # Render Hotspots
    if congestion_report and getattr(congestion_report, "hotspots", None):
        for h in congestion_report.hotspots:
            corr_id = h.get("corridor", "")
            for cf in corridor_features:
                if cf["properties"]["corridor_id"] == corr_id:
                    coords = cf["geometry"]["coordinates"]
                    mid_lat = (coords[0][1] + coords[1][1]) / 2.0
                    mid_lon = (coords[0][0] + coords[1][0]) / 2.0
                    folium.CircleMarker(
                        location=[mid_lat, mid_lon],
                        radius=8,
                        color="#dc2626",
                        fill=True,
                        fill_color="#ef4444",
                        fill_opacity=0.85,
                        tooltip=f"HOTSPOT: {corr_id} (TTI: {h.get('tti', 0.0):.2f})",
                    ).add_to(hotspots_group)

    # Render Camera Nodes
    for nf in node_features:
        props = nf["properties"]
        lat = props["latitude"]
        lon = props["longitude"]
        flow_rate = props.get("camera_flow_rate_veh_hr", 0.0)
        occ_pct = props.get("estimated_temporal_occupancy_pct", 0.0)

        node_popup_html = f"""
        <div style="font-family:Inter,sans-serif; min-width:210px; font-size:12px; color:#0f172a; padding:4px;">
            <div style="font-weight:700; color:#005577; font-size:14px; margin-bottom:4px; border-bottom:1px solid #e2e8f0; padding-bottom:4px;">
                {props['camera_id']} — {props['name']}
            </div>
            <div style="margin:3px 0;"><b>Coordinates:</b> {lat:.4f}, {lon:.4f}</div>
            <div style="margin:3px 0;"><b>Status:</b> <span style="text-transform:uppercase; font-weight:600; color:#16a34a;">{props['status']}</span></div>
            <div style="margin:3px 0;"><b>Flow Rate:</b> {flow_rate:.1f} veh/hr</div>
            <div style="margin:3px 0;"><b>Temporal Occupancy:</b> {occ_pct:.1f}%</div>
            <div style="margin:3px 0;"><b>Unique Vehicles:</b> {props['unique_vehicles_observed']}</div>
        </div>
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=7,
            color="#005577",
            weight=2,
            fill=True,
            fill_color="#38bdf8",
            fill_opacity=0.9,
            tooltip=f"{props['camera_id']} ({props['name']}): {flow_rate:.1f} veh/hr",
            popup=folium.Popup(node_popup_html, max_width=280),
        ).add_to(cameras_group)

    # Render Optional Active Vehicle Trajectory
    if active_trajectory_geojson and "features" in active_trajectory_geojson:
        traj_group = folium.FeatureGroup(name="Active Vehicle Trajectory Overlay", show=True)
        t_features = active_trajectory_geojson["features"]

        for idx, feat in enumerate(t_features, 1):
            geom_type = feat.get("geometry", {}).get("type")
            t_props = feat.get("properties", {})

            if geom_type == "Point":
                pt_lon, pt_lat = feat["geometry"]["coordinates"]
                node_idx = t_props.get("node_index", idx)
                cam_name = t_props.get("camera_name", t_props.get("camera_id", "Camera"))
                dwell = t_props.get("dwell_duration_seconds", 0.0)

                t_popup = f"""
                <div style="font-family:Inter,sans-serif; min-width:180px; font-size:12px; color:#0f172a; padding:4px;">
                    <div style="font-weight:700; color:#0284c7; margin-bottom:4px;">
                        Hop #{node_idx}: {cam_name}
                    </div>
                    <div><b>Camera ID:</b> {t_props.get('camera_id')}</div>
                    <div><b>Dwell Time:</b> {dwell:.1f}s</div>
                    <div><b>First Seen:</b> {t_props.get('first_seen_iso', 'N/A')}</div>
                </div>
                """

                folium.Marker(
                    location=[pt_lat, pt_lon],
                    tooltip=f"Hop #{node_idx}: {t_props.get('camera_id')}",
                    popup=folium.Popup(t_popup, max_width=250),
                    icon=folium.DivIcon(
                        html=f"""
                        <div style="background-color:#0284c7; color:#ffffff; border:2px solid #ffffff; border-radius:50%; width:24px; height:24px; display:flex; align-items:center; justify-content:center; font-weight:bold; font-size:11px; box-shadow:0 2px 5px rgba(0,0,0,0.3);">
                            {node_idx}
                        </div>
                        """,
                        icon_size=(24, 24),
                        icon_anchor=(12, 12),
                    ),
                ).add_to(traj_group)

            elif geom_type == "LineString":
                coords = feat["geometry"]["coordinates"]
                latlon = [[c[1], c[0]] for c in coords]
                v_speed = t_props.get("speed_kmh") if t_props.get("speed_kmh") is not None else t_props.get("transit_speed_kmh")
                speed_str = f"{v_speed:.1f} km/h" if v_speed is not None else "N/A"
                is_anom = t_props.get("is_velocity_anomaly", False)

                folium.PolyLine(
                    locations=latlon,
                    color="#dc2626" if is_anom else COLOR_TRAJECTORY,
                    weight=5,
                    opacity=0.9,
                    dash_array="8, 6" if is_anom else None,
                    tooltip=f"Transit: {speed_str} {'(Velocity Anomaly)' if is_anom else ''}",
                ).add_to(traj_group)

        traj_group.add_to(m)

    corridors_group.add_to(m)
    hotspots_group.add_to(m)
    cameras_group.add_to(m)

    # Layer Control
    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    # Disclaimer HTML Overlay
    disclaimer_html = """
    <div style="position: fixed; bottom: 12px; left: 12px; z-index: 9999; background: rgba(255, 255, 255, 0.92); backdrop-filter: blur(4px); padding: 8px 12px; border-radius: 6px; border: 1px solid #cbd5e1; font-family: Inter, sans-serif; font-size: 10px; color: #475569; max-width: 320px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
        <b>Traffic Level of Service (LOS):</b> Displayed LOS is a project proxy based on Travel Time Index (TTI) bounds (A: ≤1.10 to F: >2.50). It is <i>NOT</i> facility-specific HCM LOS methodology.
    </div>
    """
    m.get_root().html.add_child(folium.Element(disclaimer_html))

    return m
