"""
Phase 7A -- Acceptance Tests for Trajectory Reconstruction.

Verifies:
1. Single-camera sighting trajectory (boundary: 1 node, 0 segments).
2. Multi-camera sequential trajectory (CAM-001 -> CAM-002 -> CAM-004).
3. Exact transit interval definition: next.first_timestamp - prev.last_timestamp.
4. Dual distance preservation: network_distance_km and haversine_distance_km kept distinct.
5. Consecutive same-camera observations handling (is_same_camera=True, dist=0, speed=None).
6. Missing/unreachable camera graph edges (is_unreachable_network=True, dist=None).
7. Physical velocity anomaly detection (> 140 km/h plausibility bound).
8. Temporal inversion anomaly detection (delta_t <= 0).
9. GIS GeoJSON FeatureCollection serialization compliance.
10. Querying by canonical license plate via database.
11. End-to-end integration with Phase 6B persisted SQLite records.
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
    get_global_vehicle,
    get_global_vehicle_by_plate,
)
from alpr.identity import GlobalVehicleIdentity, VehicleObservation, IdentityMatchResult
from alpr.trajectory import (
    haversine_distance_km,
    TrajectoryNode,
    TrajectorySegment,
    VehicleTrajectory,
    TrajectoryReconstructor,
    reconstruct_trajectory,
    reconstruct_trajectory_by_plate,
    list_all_trajectories,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("test_phase7a")

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
    print("  Phase 7A -- Trajectory Reconstruction Acceptance Tests")
    print("=" * 65)

    reconstructor = TrajectoryReconstructor(
        cameras_path="configs/cameras.json",
        camera_graph_path="configs/camera_graph.json",
        max_plausible_speed_kmh=140.0,
    )

    # ── Suite 1: Single-Camera Sighting (Boundary Case) ──
    print("\n[1] Single-Camera Sighting (Boundary Case)")
    obs_single = [
        {
            "camera_id": "CAM-001",
            "first_timestamp": 1000.0,
            "last_timestamp": 1025.0,
            "local_track_id": 101,
            "canonical_plate": "KA01AB1234",
            "plate_confidence": 0.95,
            "crop_quality": 210.0,
            "match_status": "NEW",
            "match_method": "new_identity",
            "match_confidence": 1.0,
        }
    ]
    traj1 = reconstructor.reconstruct_from_observations("GV-000001", obs_single)
    check("Single observation produces trajectory", traj1 is not None)
    check("Nodes count == 1", len(traj1.nodes) == 1)
    check("Segments count == 0", len(traj1.segments) == 0)
    check("Total duration == 25.0s", abs(traj1.total_duration_seconds - 25.0) < 1e-4)
    check("Total network distance == 0.0 km", traj1.total_network_distance_km == 0.0)
    check("Total haversine distance == 0.0 km", traj1.total_haversine_distance_km == 0.0)
    check("Average speed is None for single camera", traj1.average_speed_kmh is None)
    check("Anomalies list is empty", len(traj1.anomalies) == 0)
    check("Canonical plate inferred from node", traj1.canonical_plate == "KA01AB1234")

    # ── Suite 2: Multi-Camera Sequential Trajectory ──
    print("\n[2] Multi-Camera Sequential Trajectory (CAM-001 -> CAM-002 -> CAM-004)")
    # CAM-001: [1000.0, 1025.0] (MG Road)
    # Transit CAM-001 -> CAM-002: 600s (10 min), road dist = 6.5 km => 39.0 km/h
    # CAM-002: [1625.0, 1650.0] (Silk Board)
    # Transit CAM-002 -> CAM-004: 1200s (20 min), road dist = 14.2 km => 42.6 km/h
    # CAM-004: [2850.0, 2875.0] (Hebbal Flyover)
    obs_multi = [
        {
            "camera_id": "CAM-001",
            "first_timestamp": 1000.0,
            "last_timestamp": 1025.0,
            "local_track_id": 17,
            "canonical_plate": "KA05NB5555",
            "plate_confidence": 0.94,
        },
        {
            "camera_id": "CAM-002",
            "first_timestamp": 1625.0,
            "last_timestamp": 1650.0,
            "local_track_id": 4,
            "canonical_plate": "KA05NB5555",
            "plate_confidence": 0.92,
        },
        {
            "camera_id": "CAM-004",
            "first_timestamp": 2850.0,
            "last_timestamp": 2875.0,
            "local_track_id": 88,
            "canonical_plate": "KA05NB5555",
            "plate_confidence": 0.89,
        },
    ]
    traj2 = reconstructor.reconstruct_from_observations("GV-000002", obs_multi)
    check("Trajectory has 3 nodes", len(traj2.nodes) == 3)
    check("Trajectory has 2 segments", len(traj2.segments) == 2)
    check("Total network distance == 6.5 + 14.2 = 20.7 km", abs(traj2.total_network_distance_km - 20.7) < 0.05)
    check("Total duration == 2875.0 - 1000.0 = 1875.0s", abs(traj2.total_duration_seconds - 1875.0) < 1e-4)

    # ── Suite 3: Transit Interval Definition ──
    print("\n[3] Transit Interval Definition (next.first - prev.last)")
    seg0 = traj2.segments[0]
    # Check transit time = 1625.0 - 1025.0 = 600.0s, NOT 1625.0 - 1000.0 (625.0s)
    check("Segment 0 from_cam == CAM-001", seg0.from_camera_id == "CAM-001")
    check("Segment 0 to_cam == CAM-002", seg0.to_camera_id == "CAM-002")
    check("Transit time is exactly 600.0s (next.first - prev.last)", abs(seg0.transit_time_seconds - 600.0) < 1e-4)
    check("Segment 0 speed == 39.0 km/h", abs(seg0.speed_kmh - 39.0) < 0.2)

    seg1 = traj2.segments[1]
    # Check transit time = 2850.0 - 1650.0 = 1200.0s
    check("Segment 1 transit time is exactly 1200.0s", abs(seg1.transit_time_seconds - 1200.0) < 1e-4)
    check("Segment 1 speed == 42.6 km/h", abs(seg1.speed_kmh - 42.6) < 0.2)

    # ── Suite 4: Dual Distance Preservation (Network vs. Haversine) ──
    print("\n[4] Dual Distance Preservation (Network vs. Haversine)")
    check("Network distance is present", seg0.network_distance_km is not None)
    check("Haversine distance is present", seg0.haversine_distance_km is not None)
    check("Network distance == 6.5 km", abs(seg0.network_distance_km - 6.5) < 1e-4)
    # Haversine between CAM-001 (12.9756, 77.6062) and CAM-002 (12.9177, 77.6233) is ~6.7 km
    check("Haversine distance is straight line (~6.7 km)", abs(seg0.haversine_distance_km - 6.70) < 0.2)
    check("Network distance != Haversine distance (not silently substituted)", abs(seg0.network_distance_km - seg0.haversine_distance_km) > 0.05)

    # ── Suite 5: Consecutive Same-Camera Observations ──
    print("\n[5] Consecutive Same-Camera Observations Test")
    # Same vehicle tracked twice at CAM-001 (tracks 17 then 18), then moving to CAM-002 (track 4)
    obs_same_cam = [
        {"camera_id": "CAM-001", "first_timestamp": 1000.0, "last_timestamp": 1020.0, "local_track_id": 17},
        {"camera_id": "CAM-001", "first_timestamp": 1030.0, "last_timestamp": 1050.0, "local_track_id": 18},
        {"camera_id": "CAM-002", "first_timestamp": 1650.0, "last_timestamp": 1675.0, "local_track_id": 4},
    ]
    traj_same = reconstructor.reconstruct_from_observations("GV-000003", obs_same_cam)
    check("Trajectory has 3 nodes", len(traj_same.nodes) == 3)
    check("Trajectory has 2 segments", len(traj_same.segments) == 2)

    seg_same = traj_same.segments[0]
    check("Segment 0 is_same_camera == True", seg_same.is_same_camera is True)
    check("Segment 0 network distance == 0.0 km", seg_same.network_distance_km == 0.0)
    check("Segment 0 haversine distance == 0.0 km", seg_same.haversine_distance_km == 0.0)
    check("Segment 0 speed is None (not treated as city transit)", seg_same.speed_kmh is None)
    check("Segment 0 transit_time == 10.0s", abs(seg_same.transit_time_seconds - 10.0) < 1e-4)
    check("No velocity anomaly on same-camera segment", seg_same.is_velocity_anomaly is False)

    seg_next = traj_same.segments[1]
    check("Segment 1 is_same_camera == False", seg_next.is_same_camera is False)
    check("Segment 1 network distance == 6.5 km", abs(seg_next.network_distance_km - 6.5) < 1e-4)
    check("Segment 1 speed == 39.0 km/h", abs(seg_next.speed_kmh - 39.0) < 0.2)

    # ── Suite 6: Missing / Unreachable Graph Edges ──
    print("\n[6] Missing / Unreachable Camera Graph Edges")
    # Observation at an unlinked camera ID
    obs_unreach = [
        {"camera_id": "CAM-001", "first_timestamp": 1000.0, "last_timestamp": 1025.0, "local_track_id": 1},
        {"camera_id": "CAM-ISOLATED", "first_timestamp": 1600.0, "last_timestamp": 1625.0, "local_track_id": 2},
    ]
    traj_unreach = reconstructor.reconstruct_from_observations("GV-000004", obs_unreach)
    check("Trajectory reconstructed with unreachable camera", len(traj_unreach.nodes) == 2)
    seg_unreach = traj_unreach.segments[0]
    check("Segment is_unreachable_network == True", seg_unreach.is_unreachable_network is True)
    check("Network distance is None", seg_unreach.network_distance_km is None)
    check("Speed is None", seg_unreach.speed_kmh is None)
    check("Anomaly recorded in segment", any("No route in network graph" in a for a in seg_unreach.anomalies))
    check("Anomaly recorded in trajectory", any("No route in network graph" in a for a in traj_unreach.anomalies))

    # ── Suite 7: Physical Velocity Anomaly (>140 km/h bound) ──
    print("\n[7] Physical Velocity Anomaly Detection")
    # CAM-001 -> CAM-002 (6.5 km) covered in only 60 seconds => 390 km/h!
    obs_speeding = [
        {"camera_id": "CAM-001", "first_timestamp": 1000.0, "last_timestamp": 1020.0, "local_track_id": 1},
        {"camera_id": "CAM-002", "first_timestamp": 1080.0, "last_timestamp": 1100.0, "local_track_id": 2},
    ]
    traj_speed = reconstructor.reconstruct_from_observations("GV-000005", obs_speeding)
    seg_speed = traj_speed.segments[0]
    check("Calculated speed == 390.0 km/h", abs(seg_speed.speed_kmh - 390.0) < 1.0)
    check("is_velocity_anomaly == True", seg_speed.is_velocity_anomaly is True)
    check("Anomaly message mentions physical plausibility bound", any("exceeds physical plausibility bound" in a for a in seg_speed.anomalies))
    check("Trajectory contains velocity anomaly", any("exceeds physical plausibility bound" in a for a in traj_speed.anomalies))

    # ── Suite 8: Temporal Inversion Anomaly ──
    print("\n[8] Temporal Inversion Anomaly Detection (delta_t <= 0)")
    obs_time_inv = [
        {"camera_id": "CAM-001", "first_timestamp": 1500.0, "last_timestamp": 1520.0, "local_track_id": 1},
        {"camera_id": "CAM-002", "first_timestamp": 1400.0, "last_timestamp": 1420.0, "local_track_id": 2},
    ]
    traj_inv = reconstructor.reconstruct_from_observations("GV-000006", obs_time_inv)
    # Sorted by first_timestamp automatically, so CAM-002 is node 0, CAM-001 is node 1.
    # What if timestamps are identical or overlap?
    obs_overlap = [
        {"camera_id": "CAM-001", "first_timestamp": 1000.0, "last_timestamp": 1050.0, "local_track_id": 1},
        {"camera_id": "CAM-002", "first_timestamp": 1020.0, "last_timestamp": 1060.0, "local_track_id": 2},
    ]
    # prev.last = 1050.0, next.first = 1020.0 => delta_t = -30.0s!
    traj_overlap = reconstructor.reconstruct_from_observations("GV-000007", obs_overlap)
    seg_overlap = traj_overlap.segments[0]
    check("Transit time is negative (-30.0s)", abs(seg_overlap.transit_time_seconds - (-30.0)) < 1e-4)
    check("is_temporal_anomaly == True", seg_overlap.is_temporal_anomaly is True)
    check("Anomaly message mentions negative or zero transit time", any("Negative or zero transit time" in a for a in seg_overlap.anomalies))

    # ── Suite 9: GIS GeoJSON FeatureCollection Serialization ──
    print("\n[9] GIS GeoJSON FeatureCollection Serialization Compliance")
    geojson = traj2.to_geojson()
    check("GeoJSON type == 'FeatureCollection'", geojson.get("type") == "FeatureCollection")
    check("GeoJSON has properties dict", isinstance(geojson.get("properties"), dict))
    check("GeoJSON properties has global_id", geojson["properties"]["global_id"] == "GV-000002")

    features = geojson.get("features", [])
    # 3 nodes (Points) + 2 segments (LineStrings) = 5 features
    check("Feature count == 5 (3 Points + 2 LineStrings)", len(features) == 5)

    point_features = [f for f in features if f["geometry"]["type"] == "Point"]
    linestring_features = [f for f in features if f["geometry"]["type"] == "LineString"]
    check("Point features count == 3", len(point_features) == 3)
    check("LineString features count == 2", len(linestring_features) == 2)

    # Verify Point coordinates [longitude, latitude]
    check("Point coordinate format is [lon, lat]", len(point_features[0]["geometry"]["coordinates"]) == 2)
    check("Point 0 has camera_id == 'CAM-001'", point_features[0]["properties"]["camera_id"] == "CAM-001")

    # Verify LineString coordinates [[lon1, lat1], [lon2, lat2]]
    coords0 = linestring_features[0]["geometry"]["coordinates"]
    check("LineString connects 2 coordinate pairs", len(coords0) == 2)
    check("LineString properties has transit_time_seconds", "transit_time_seconds" in linestring_features[0]["properties"])
    check("LineString properties has speed_kmh", linestring_features[0]["properties"]["speed_kmh"] == 39.0)

    # Verify same-camera segments are excluded from GeoJSON LineStrings
    geojson_same = traj_same.to_geojson()
    same_linestrings = [f for f in geojson_same["features"] if f["geometry"]["type"] == "LineString"]
    check("Zero-length LineString excluded on same-camera segment", len(same_linestrings) == 1)

    # ── Suite 10 & 11: Database Persistence & Query by Plate ──
    print("\n[10 & 11] Database Persistence & Query by Canonical Plate")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "traj_test.db"
        conn = init_db(db_path)

        # 1. Insert global identity
        gid = "GV-000088"
        plate_str = "DL8CAZ9592"
        identity = GlobalVehicleIdentity(
            global_id=gid,
            canonical_plate=plate_str,
            plate_confidence=0.96,
            vehicle_type="car",
            first_seen_ts=5000.0,
            last_seen_ts=6200.0,
            first_camera_id="CAM-001",
            last_camera_id="CAM-002",
            sighting_count=2,
            status="active",
        )
        save_global_identity(conn, identity)

        # 2. Insert observations
        obs_a = VehicleObservation(
            camera_id="CAM-001",
            track_id=12,
            timestamp=5030.0,
            vehicle_type="car",
            canonical_plate=plate_str,
            plate_confidence=0.96,
        )
        res_a = IdentityMatchResult(status="NEW", global_id=gid, confidence=1.0)
        record_vehicle_observation(conn, obs_a, res_a, first_timestamp=5000.0)

        obs_b = VehicleObservation(
            camera_id="CAM-002",
            track_id=5,
            timestamp=5660.0,
            vehicle_type="car",
            canonical_plate=plate_str,
            plate_confidence=0.95,
        )
        res_b = IdentityMatchResult(status="MATCH", global_id=gid, confidence=0.94)
        record_vehicle_observation(conn, obs_b, res_b, first_timestamp=5630.0)

        # Query by global_id
        traj_db = reconstructor.reconstruct(conn, gid)
        check("reconstruct(conn, global_id) returns valid trajectory", traj_db is not None)
        check("Trajectory global_id matches", traj_db.global_id == gid)
        check("Trajectory plate matches", traj_db.canonical_plate == plate_str)
        check("Trajectory has 2 nodes", len(traj_db.nodes) == 2)
        check("Transit time == 5630.0 - 5030.0 = 600.0s", abs(traj_db.segments[0].transit_time_seconds - 600.0) < 1e-4)

        # Query by canonical plate
        traj_by_plate = reconstructor.reconstruct_by_plate(conn, plate_str)
        check("reconstruct_by_plate(conn, plate) returns matching trajectory", traj_by_plate is not None)
        check("Resolved to same global_id GV-000088", traj_by_plate.global_id == gid)

        # Query nonexistent plate
        check("Query nonexistent plate returns None", reconstructor.reconstruct_by_plate(conn, "UP32XX9999") is None)

        # list_all_trajectories
        all_trajs = reconstructor.list_all_trajectories(conn)
        check("list_all_trajectories returns list with 1 trajectory", len(all_trajs) == 1)

        # Summary output check
        summary_txt = traj_db.summary()
        check("summary() generates formatted string", len(summary_txt) > 50)
        check("summary() contains global_id and plate", gid in summary_txt and plate_str in summary_txt)

        conn.close()

    # ── Final Summary ──
    print("\n" + "=" * 65)
    print(f"  Results: {PASS}/{PASS + FAIL} passed, {FAIL} failed")
    print("=" * 65 + "\n")

    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
