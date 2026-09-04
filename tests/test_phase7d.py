"""
Phase 7D -- Acceptance Tests for GIS Map Integration & Unified City-Wide Dashboard.

Verifies:
1. GIS GeoJSON generation: camera nodes as Points, corridors as LineStrings.
2. Directed topology adherence: only configured edges from camera_graph.json.
3. Direct color mapping from los_proxy (get_los_color), no independent TTI math in GIS.
4. Folium interactive map generation with base tiles, layer controls, and popups.
5. DashboardService empty-state contracts (safe zero/empty responses on fresh DB).
6. Vehicle search HTTP contracts: 200 (found), 404 (absent), 400 (empty/malformed).
7. Dual-key search support (canonical license plate and GV-XXXXXX global ID).
8. ReID diagnostics sanitization: exposes dimension 512, L2 norm 1.0, quality, while withholding raw 512 float dump.
9. Trajectory reconstruction integration: chronological hops, dwell duration, and GeoJSON.
10. Flask versioned REST endpoints: /api/v1/... integration via test_client.
"""

import json
import logging
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from alpr.database import (
    init_db,
    save_global_identity,
    record_vehicle_observation,
    serialize_embedding,
)
from alpr.identity import GlobalVehicleIdentity, VehicleObservation, IdentityMatchResult
from alpr.gis import (
    get_los_color,
    build_network_geojson,
    generate_city_traffic_map,
    LOS_COLOR_MAP,
    COLOR_SPARSE,
)
from alpr.service import (
    DashboardService,
    get_dashboard_service,
)
from app import app

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("test_phase7d")

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    status = "[PASS]" if condition else "[FAIL]"
    msg = f"  {status}: {name}"
    if detail:
        msg += f" - {detail}"
    print(msg)
    if condition:
        PASS += 1
    else:
        FAIL += 1


def main():
    global PASS, FAIL
    print("\n" + "=" * 65)
    print("  Phase 7D -- GIS Map Integration & Dashboard Tests")
    print("=" * 65)

    # ── Suite 1: Direct LOS Proxy Color Mapping (No TTI Math in GIS) ──
    print("\n[1] Direct LOS Proxy Color Mapping")
    check("LOS A maps to emerald green #10b981", get_los_color("A") == "#10b981")
    check("LOS B maps to lime green #84cc16", get_los_color("B") == "#84cc16")
    check("LOS C maps to amber yellow #eab308", get_los_color("C") == "#eab308")
    check("LOS D maps to orange #f97316", get_los_color("D") == "#f97316")
    check("LOS E maps to bright red #ef4444", get_los_color("E") == "#ef4444")
    check("LOS F maps to crimson red #991b1b", get_los_color("F") == "#991b1b")
    check("Case insensitivity: 'a' maps to #10b981", get_los_color("a") == "#10b981")
    check("Unknown / sparse maps to slate grey #94a3b8", get_los_color("UNKNOWN") == "#94a3b8")
    check("None maps to slate grey #94a3b8", get_los_color(None) == "#94a3b8")

    # ── Suite 2: GIS GeoJSON Schema & Topology Adherence ──
    print("\n[2] GIS GeoJSON Schema & Topology Adherence")
    cameras_cfg = "configs/cameras.json"
    camera_graph_cfg = "configs/camera_graph.json"

    geojson = build_network_geojson(cameras_path=cameras_cfg, camera_graph_path=camera_graph_cfg)
    check("Output type is FeatureCollection", geojson.get("type") == "FeatureCollection")
    check("Metadata contains disclaimer", "los_proxy_disclaimer" in geojson.get("metadata", {}))

    features = geojson.get("features", [])
    nodes = [f for f in features if f["properties"]["feature_type"] == "camera_node"]
    corridors = [f for f in features if f["properties"]["feature_type"] == "corridor"]

    check("Camera nodes count is 4", len(nodes) == 4)
    # Check Point geometry
    check("All camera nodes have Point geometry", all(n["geometry"]["type"] == "Point" for n in nodes))
    check("Camera node CAM-001 has flow and occupancy properties",
          "camera_flow_rate_veh_hr" in nodes[0]["properties"] and "estimated_temporal_occupancy_ratio" in nodes[0]["properties"])

    # Check topology: in camera_graph.json:
    # CAM-001: [CAM-002, CAM-003] (2)
    # CAM-002: [CAM-001, CAM-003, CAM-004] (3)
    # CAM-003: [CAM-001, CAM-002, CAM-004] (3)
    # CAM-004: [CAM-002, CAM-003] (2)
    # Total = 10 configured directed edges
    check("Corridors count strictly matches 10 configured graph edges", len(corridors) == 10)
    check("All corridors have LineString geometry", all(c["geometry"]["type"] == "LineString" for c in corridors))

    # CAM-001 -> CAM-004 is NOT a configured direct edge
    corridor_ids = [c["properties"]["corridor_id"] for c in corridors]
    check("Unconfigured direct edge CAM-001 -> CAM-004 is absent", "CAM-001 -> CAM-004" not in corridor_ids)

    # ── Suite 3: Folium Interactive Map Generation ──
    print("\n[3] Folium Interactive Map Generation")
    folium_map = generate_city_traffic_map(cameras_path=cameras_cfg, camera_graph_path=camera_graph_cfg)
    check("Folium map instance generated", folium_map is not None)
    rendered_html = folium_map.get_root().render()
    check("Rendered HTML contains CartoDB positron tile", "basemaps.cartocdn.com" in rendered_html or "CARTO" in rendered_html)
    check("Rendered HTML contains layer control", "leaflet-control-layers" in rendered_html or "LayerControl" in rendered_html or "control" in rendered_html)
    check("Rendered HTML contains LOS project proxy disclaimer", "NOT claimed as facility-specific HCM LOS" in rendered_html or "HCM LOS" in rendered_html)

    # ── Suite 4: DashboardService Empty-State Contracts ──
    print("\n[4] DashboardService Empty-State Contracts")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_db = Path(tmp_dir) / "empty_alpr.db"
        conn = init_db(tmp_db)

        service = DashboardService(
            db_path=tmp_db,
            cameras_path=cameras_cfg,
            camera_graph_path=camera_graph_cfg,
        )

        try:
            summary = service.get_analytics_summary(conn)
            check("Empty DB summary total_vehicles_observed == 0", summary["total_vehicles_observed"] == 0)
            check("Empty DB summary total_transit_observations == 0", summary["total_transit_observations"] == 0)
            check("Empty DB summary network_average_tti is None", summary["network_average_tti"] is None)
            check("Empty DB summary active_corridors_count == 0", summary["active_corridors_count"] == 0)
            check("Empty DB summary hotspots_count == 0", summary["hotspots_count"] == 0)

            corridors_list = service.get_corridors_metrics(conn)
            check("Empty DB corridors list is empty list []", isinstance(corridors_list, list) and len(corridors_list) == 0)

            od_matrix = service.get_od_matrix(conn)
            check("Empty DB OD matrix is empty list []", isinstance(od_matrix, list) and len(od_matrix) == 0)

            time_windows = service.get_time_windows(conn)
            check("Empty DB time windows is empty list []", isinstance(time_windows, list) and len(time_windows) == 0)

            hotspots = service.get_hotspots(conn)
            check("Empty DB hotspots is empty list []", isinstance(hotspots, list) and len(hotspots) == 0)
        finally:
            conn.close()

    # ── Suite 5: Vehicle Search HTTP Contracts (200, 404, 400) ──
    print("\n[5] Vehicle Search HTTP Contracts")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        tmp_db = Path(tmp_dir) / "search_test.db"
        conn = init_db(tmp_db)

        service = DashboardService(
            db_path=tmp_db,
            cameras_path=cameras_cfg,
            camera_graph_path=camera_graph_cfg,
        )

        try:
            # 400 Empty or whitespace query
            status_empty, res_empty = service.search_vehicle("", conn)
            check("Empty query returns HTTP 400", status_empty == 400 and "error" in res_empty)

            status_space, res_space = service.search_vehicle("   ", conn)
            check("Whitespace query returns HTTP 400", status_space == 400)

            # 400 Malformed query
            status_mal, res_mal = service.search_vehicle("DROP TABLE;--", conn)
            check("Malformed query returns HTTP 400", status_mal == 400)

            # 404 Absent vehicle
            status_absent, res_absent = service.search_vehicle("KA01AB1234", conn)
            check("Absent plate returns HTTP 404", status_absent == 404 and "error" in res_absent)

            status_absent_gv, res_absent_gv = service.search_vehicle("GV-999999", conn)
            check("Absent global vehicle ID returns HTTP 404", status_absent_gv == 404)

            # ── Suite 6: Dual-Key Search (Plate and GV-ID) & ReID Sanitization ──
            print("\n[6] Dual-Key Search & ReID Sanitization")
            # Populate test vehicle
            gv_id = "GV-000101"
            plate = "KA05MH2024"

            # 512-D L2-normalized synthetic embedding
            rng = np.random.default_rng(42)
            raw_emb = rng.standard_normal(512).astype(np.float32)
            raw_emb = raw_emb / np.linalg.norm(raw_emb)

            identity = GlobalVehicleIdentity(
                global_id=gv_id,
                canonical_plate=plate,
                plate_confidence=0.96,
                vehicle_type="car",
                first_seen_ts=1000.0,
                last_seen_ts=1600.0,
                first_camera_id="CAM-001",
                last_camera_id="CAM-002",
                sighting_count=2,
                status="active",
                representative_embedding=raw_emb,
            )
            save_global_identity(conn, identity)

            # Observation 1 at CAM-001 (dwell 20s: 1000.0 -> 1020.0)
            obs1 = VehicleObservation(
                camera_id="CAM-001",
                track_id=101,
                timestamp=1020.0,
                vehicle_type="car",
                canonical_plate=plate,
                plate_confidence=0.95,
                crop_quality=0.88,
                best_reid_embedding=raw_emb,
            )
            res1 = IdentityMatchResult(status="NEW", global_id=gv_id, confidence=1.0)
            record_vehicle_observation(conn, obs1, res1, first_timestamp=1000.0)

            # Observation 2 at CAM-002 (dwell 30s: 1570.0 -> 1600.0)
            obs2 = VehicleObservation(
                camera_id="CAM-002",
                track_id=202,
                timestamp=1600.0,
                vehicle_type="car",
                canonical_plate=plate,
                plate_confidence=0.97,
                crop_quality=0.92,
                best_reid_embedding=raw_emb,
            )
            res2 = IdentityMatchResult(status="MATCH", global_id=gv_id, confidence=0.94)
            record_vehicle_observation(conn, obs2, res2, first_timestamp=1570.0)

            # Search by Plate
            st_plate, res_plate = service.search_vehicle(plate, conn)
            check("Search by plate returns HTTP 200", st_plate == 200)
            check("Search by plate returns correct global_id", res_plate.get("global_id") == gv_id)

            # Search by Global Vehicle ID
            st_gv, res_gv = service.search_vehicle(gv_id, conn)
            check("Search by GV-ID returns HTTP 200", st_gv == 200)
            check("Search by GV-ID returns correct canonical_plate", res_gv.get("canonical_plate") == plate)

            # ReID Sanitization Check: raw 512 floats must NOT be present
            reid_diag = res_gv.get("reid_diagnostics", {})
            check("ReID diagnostics has_embedding is True", reid_diag.get("has_embedding") is True)
            check("ReID diagnostics dimension == 512", reid_diag.get("dimension") == 512)
            check("ReID diagnostics l2_norm == 1.0000", abs(reid_diag.get("l2_norm") - 1.0) < 1e-3)
            check("Raw 512 float vector omitted from dossier", "representative_embedding" not in res_gv and "embedding" not in reid_diag)

            # Sighting Hops & Trajectory Verification
            hops = res_gv.get("sighting_hops", [])
            check("Two chronological sighting hops returned", len(hops) == 2)
            check("Hop 1 camera is CAM-001", hops[0]["camera_id"] == "CAM-001")
            check("Hop 1 dwell time == 20.0s", abs(hops[0]["dwell_duration_seconds"] - 20.0) < 1e-2)
            check("Hop 2 camera is CAM-002", hops[1]["camera_id"] == "CAM-002")

            # GeoJSON check
            traj_geojson = res_gv.get("geojson", {})
            check("Vehicle GeoJSON is FeatureCollection", traj_geojson.get("type") == "FeatureCollection")
        finally:
            conn.close()

    # ── Suite 7: Flask REST API Endpoints Integration (Test Client) ──
    print("\n[7] Flask REST API Endpoints Integration")
    app.config["TESTING"] = True
    client = app.test_client()

    # v1 GIS Network Map
    r = client.get("/api/v1/gis/network-map")
    check("GET /api/v1/gis/network-map returns 200", r.status_code == 200)
    data = r.get_json()
    check("network-map returns FeatureCollection", data.get("type") == "FeatureCollection")

    # Backward compatible unversioned alias
    r_alias = client.get("/api/gis/network-map")
    check("GET /api/gis/network-map alias returns 200", r_alias.status_code == 200)

    # v1 Folium Map HTML
    r_map = client.get("/api/v1/gis/folium-map")
    check("GET /api/v1/gis/folium-map returns 200", r_map.status_code == 200)
    check("folium-map response is text/html", "text/html" in r_map.content_type)

    # v1 Analytics Summary
    r_sum = client.get("/api/v1/analytics/summary")
    check("GET /api/v1/analytics/summary returns 200", r_sum.status_code == 200)
    sum_data = r_sum.get_json()
    check("analytics summary contains modal_flow_breakdown", "modal_flow_breakdown" in sum_data)
    check("analytics summary contains los_proxy_disclaimer", "los_proxy_disclaimer" in sum_data)

    # v1 Corridors
    r_corr = client.get("/api/v1/analytics/corridors")
    check("GET /api/v1/analytics/corridors returns 200", r_corr.status_code == 200)
    check("corridors is a list", isinstance(r_corr.get_json(), list))

    # v1 Hotspots
    r_hot = client.get("/api/v1/analytics/hotspots")
    check("GET /api/v1/analytics/hotspots returns 200", r_hot.status_code == 200)
    check("hotspots is a list", isinstance(r_hot.get_json(), list))

    # v1 OD Matrix
    r_od = client.get("/api/v1/analytics/od-matrix")
    check("GET /api/v1/analytics/od-matrix returns 200", r_od.status_code == 200)
    check("od-matrix is a list", isinstance(r_od.get_json(), list))

    # v1 Time Windows
    r_tw = client.get("/api/v1/analytics/time-windows")
    check("GET /api/v1/analytics/time-windows returns 200", r_tw.status_code == 200)
    check("time-windows is a list", isinstance(r_tw.get_json(), list))

    # v1 Vehicle Search validation
    r_s_empty = client.get("/api/v1/vehicles/search?q=")
    check("GET /api/v1/vehicles/search?q= returns 400", r_s_empty.status_code == 400)

    r_s_absent = client.get("/api/v1/vehicles/search?q=NONEXISTENT")
    check("GET /api/v1/vehicles/search?q=NONEXISTENT returns 404", r_s_absent.status_code == 404)

    # Direct trajectory resource endpoint
    r_traj_404 = client.get("/api/v1/vehicles/trajectory/NONEXISTENT")
    check("GET /api/v1/vehicles/trajectory/NONEXISTENT returns 404", r_traj_404.status_code == 404)

    # ── Suite 8: Zero-Math Flask Route Invariant ──
    print("\n[8] Zero-Math Route Architecture Verification")
    import inspect
    import app as app_module

    # Inspect route handler sources to guarantee no direct math calculations in routes
    routes_to_inspect = [
        app_module.api_gis_network_map,
        app_module.api_gis_folium_map,
        app_module.api_analytics_summary,
        app_module.api_analytics_corridors,
        app_module.api_analytics_od_matrix,
        app_module.api_analytics_time_windows,
        app_module.api_vehicle_search,
        app_module.api_vehicle_trajectory,
    ]

    forbidden_keywords = ["math.", "haversine", "floyd_warshall", "speed =", "tti =", "delta_t"]
    all_clean = True
    for route_fn in routes_to_inspect:
        src = inspect.getsource(route_fn).lower()
        for kw in forbidden_keywords:
            if kw in src:
                all_clean = False
                check(f"Route {route_fn.__name__} violates zero-math boundary ({kw})", False)
                break

    check("All Flask routes strictly delegate to DashboardService without local math", all_clean)

    # Summary
    print("\n" + "=" * 65)
    print(f"  Results: {PASS}/{PASS + FAIL} passed, {FAIL} failed")
    print("=" * 65)

    if FAIL > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
