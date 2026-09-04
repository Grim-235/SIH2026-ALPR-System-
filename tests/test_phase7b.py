"""
Phase 7B -- Acceptance Tests for Network Speed & Travel-Time Analytics.

Verifies:
1. Corridor statistical computation: N, mean, median, P95 against known ground truth.
2. Speed distribution metrics: speed_p05_kmh, speed_median_kmh, speed_mean_kmh, speed_p95_kmh.
3. Edge cases: N=0, N=1, and N=2 behave safely without crashes or division-by-zero.
4. Same-camera segments are cleanly excluded from corridor transit analytics.
5. Dual distances (network_distance_km and haversine_distance_km) preserved distinctly.
6. Vehicle fleet mix reconciliation across corridors.
7. Departure-timestamp-based time-of-day bucketing (segment.from_timestamp).
8. Trajectory trip-level Origin-Destination (OD) matrix mapping.
9. Transparent anomaly metrics & mathematical bound on anomaly_rate in [0.0, 1.0].
10. Database direct analysis integration via SQLite (analyze_db).
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
)
from alpr.identity import GlobalVehicleIdentity, VehicleObservation, IdentityMatchResult
from alpr.trajectory import (
    TrajectoryNode,
    TrajectorySegment,
    VehicleTrajectory,
    TrajectoryReconstructor,
)
from alpr.analytics import (
    CorridorAnalytics,
    TripODRecord,
    NetworkAnalyticsReport,
    CorridorAnalyticsEngine,
    analyze_network_traffic,
    compute_distribution_stats,
    get_time_window_label,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("test_phase7b")

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


def create_synthetic_trajectory(
    global_id: str,
    hops: list,  # list of (cam_id, first_ts, last_ts, track_id)
    vehicle_type: str = "car",
    road_distances: dict = None,
) -> VehicleTrajectory:
    """Helper to build test trajectories with controlled timestamps and road distances."""
    if road_distances is None:
        road_distances = {
            ("CAM-001", "CAM-002"): 6.5,
            ("CAM-002", "CAM-004"): 14.2,
            ("CAM-001", "CAM-003"): 10.5,
        }

    nodes = []
    for cam_id, f_ts, l_ts, trk_id in hops:
        nodes.append(
            TrajectoryNode(
                camera_id=cam_id,
                camera_name=f"Camera {cam_id}",
                latitude=12.95 + (0.02 * int(cam_id[-1])),
                longitude=77.60 + (0.02 * int(cam_id[-1])),
                first_timestamp=float(f_ts),
                last_timestamp=float(l_ts),
                duration_seconds=float(l_ts - f_ts),
                local_track_id=trk_id,
                canonical_plate=f"KA05{global_id[-4:]}",
            )
        )

    segments = []
    tot_dist = 0.0
    for i in range(len(nodes) - 1):
        p = nodes[i]
        n = nodes[i + 1]
        dt = n.first_timestamp - p.last_timestamp
        is_same = (p.camera_id == n.camera_id)

        net_d = 0.0 if is_same else road_distances.get((p.camera_id, n.camera_id), 5.0)
        hav_d = 0.0 if is_same else (net_d * 0.95)  # approximate air distance

        speed = None
        is_vel_anom = False
        is_temp_anom = (dt <= 0)
        if not is_same and dt > 0 and net_d > 0:
            speed = net_d / (dt / 3600.0)
            if speed > 140.0:
                is_vel_anom = True
            tot_dist += net_d

        segments.append(
            TrajectorySegment(
                from_camera_id=p.camera_id,
                to_camera_id=n.camera_id,
                from_timestamp=p.last_timestamp,
                to_timestamp=n.first_timestamp,
                transit_time_seconds=dt,
                network_distance_km=net_d,
                haversine_distance_km=hav_d,
                speed_kmh=speed,
                is_same_camera=is_same,
                is_velocity_anomaly=is_vel_anom,
                is_temporal_anomaly=is_temp_anom,
            )
        )

    dur = nodes[-1].last_timestamp - nodes[0].first_timestamp
    return VehicleTrajectory(
        global_id=global_id,
        canonical_plate=nodes[0].canonical_plate,
        vehicle_type=vehicle_type,
        first_seen_ts=nodes[0].first_timestamp,
        last_seen_ts=nodes[-1].last_timestamp,
        total_duration_seconds=dur,
        total_network_distance_km=tot_dist,
        nodes=nodes,
        segments=segments,
    )


def main():
    global PASS, FAIL
    print("\n" + "=" * 65)
    print("  Phase 7B -- Network Speed & Travel-Time Analytics Tests")
    print("=" * 65)

    engine = CorridorAnalyticsEngine()

    # ── Suite 1: Robust Statistical Distribution Invariants (N=0, 1, 2, 10) ──
    print("\n[1] Distribution Statistics Invariants (N=0, N=1, N=2, N=10)")
    s0 = compute_distribution_stats([])
    check("N=0 returns None for all statistics", s0["mean"] is None and s0["median"] is None and s0["p05"] is None)

    s1 = compute_distribution_stats([600.0])
    check("N=1 returns value for mean and median", s1["mean"] == 600.0 and s1["median"] == 600.0)
    check("N=1 returns 0.0 for std", s1["std"] == 0.0)
    check("N=1 returns value for p05 and p95", s1["p05"] == 600.0 and s1["p95"] == 600.0)

    s2 = compute_distribution_stats([500.0, 700.0])
    check("N=2 mean is 600.0", s2["mean"] == 600.0)
    check("N=2 median is 600.0", s2["median"] == 600.0)
    check("N=2 min=500.0, max=700.0", s2["min"] == 500.0 and s2["max"] == 700.0)

    # N=10: [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    vals10 = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]
    s10 = compute_distribution_stats(vals10)
    check("N=10 mean is 550.0", abs(s10["mean"] - 550.0) < 1e-4)
    check("N=10 median is 550.0", abs(s10["median"] - 550.0) < 1e-4)
    check("N=10 p05 is approx 145.0", abs(s10["p05"] - float(np.percentile(vals10, 5))) < 1e-4)
    check("N=10 p95 is approx 955.0", abs(s10["p95"] - float(np.percentile(vals10, 95))) < 1e-4)

    # ── Suite 2: Corridor Analytics Aggregation (CAM-001 -> CAM-002) ──
    print("\n[2] Corridor Analytics & Sample Transparency (CAM-001 -> CAM-002)")
    # Generate 5 trajectories along CAM-001 -> CAM-002 with controlled travel times:
    # 6.5 km road distance.
    # Transit times: [500s, 600s, 600s, 650s, 750s]
    # Corresponding speeds: [46.8, 39.0, 39.0, 36.0, 31.2] km/h
    # All between 07:00 and 08:30 (Morning Peak window: 06:00-09:00)
    trajs = []
    tt_list = [500.0, 600.0, 600.0, 650.0, 750.0]
    vtypes = ["car", "car", "motorcycle", "bus", "truck"]
    base_t = 7 * 3600.0  # 07:00:00

    for i, (tt, vt) in enumerate(zip(tt_list, vtypes)):
        dep_t = base_t + (i * 300.0)
        arr_t = dep_t + tt
        t = create_synthetic_trajectory(
            global_id=f"GV-00010{i}",
            hops=[
                ("CAM-001", dep_t - 20.0, dep_t, 10 + i),
                ("CAM-002", arr_t, arr_t + 25.0, 20 + i),
            ],
            vehicle_type=vt,
        )
        trajs.append(t)

    report = engine.analyze_trajectories(trajs)
    check("Report processed 5 trajectories", report.total_trajectories_analyzed == 5)
    check("Total transit observations == 5", report.total_transit_observations == 5)
    check("Total valid observations == 5", report.total_valid_observations == 5)
    check("Total anomalous observations == 0", report.total_anomalous_observations == 0)

    corr = report.get_corridor("CAM-001", "CAM-002")
    check("Corridor CAM-001 -> CAM-002 exists", corr is not None)
    check("Sample size N == 5", corr.observation_count == 5)
    check("Valid observation count == 5", corr.valid_observation_count == 5)
    check("Anomalous observation count == 0", corr.anomalous_observation_count == 0)

    # Travel time checks
    # [500, 600, 600, 650, 750] => Mean = 3100 / 5 = 620.0s, Median = 600.0s
    check("Travel time mean == 620.0s", abs(corr.travel_time_mean_s - 620.0) < 1e-4)
    check("Travel time median == 600.0s", abs(corr.travel_time_median_s - 600.0) < 1e-4)
    check("Travel time min == 500.0s", abs(corr.travel_time_min_s - 500.0) < 1e-4)
    check("Travel time max == 750.0s", abs(corr.travel_time_max_s - 750.0) < 1e-4)
    check("Travel time P95 is calculated", corr.travel_time_p95_s is not None and corr.travel_time_p95_s > 650.0)

    # Speed metrics checks
    check("Speed median == 39.0 km/h", abs(corr.speed_median_kmh - 39.0) < 0.2)
    check("Speed p05 is calculated (low-speed tail)", corr.speed_p05_kmh is not None and corr.speed_p05_kmh < corr.speed_median_kmh)
    check("Speed p95 is calculated (high-speed tail)", corr.speed_p95_kmh is not None and corr.speed_p95_kmh > corr.speed_median_kmh)

    # Vehicle-type counts
    check("Fleet mix has 2 cars", corr.vehicle_type_counts.get("car") == 2)
    check("Fleet mix has 1 motorcycle", corr.vehicle_type_counts.get("motorcycle") == 1)
    check("Fleet mix has 1 bus", corr.vehicle_type_counts.get("bus") == 1)
    check("Fleet mix has 1 truck", corr.vehicle_type_counts.get("truck") == 1)
    check("Sum of vehicle types equals N", sum(corr.vehicle_type_counts.values()) == corr.observation_count)

    # ── Suite 3: Same-Camera Sightings Exclusion ──
    print("\n[3] Same-Camera Segments Exclusion")
    # Trajectory with same-camera consecutive sighting: CAM-001 -> CAM-001 -> CAM-002
    traj_same = create_synthetic_trajectory(
        global_id="GV-000200",
        hops=[
            ("CAM-001", 3000.0, 3020.0, 1),
            ("CAM-001", 3040.0, 3060.0, 2),  # Same camera!
            ("CAM-002", 3660.0, 3680.0, 3),  # 600s transit to CAM-002
        ],
        vehicle_type="car",
    )
    rep_same = engine.analyze_trajectories([traj_same])
    check("Total transit observations is 1 (excluding same-camera)", rep_same.total_transit_observations == 1)
    check("No corridor created for CAM-001 -> CAM-001", ("CAM-001", "CAM-001") not in rep_same.corridors)
    check("Corridor CAM-001 -> CAM-002 has N=1", rep_same.get_corridor("CAM-001", "CAM-002").observation_count == 1)

    # ── Suite 4: Distance Preservation ──
    print("\n[4] Dual Distance Preservation in Analytics")
    check("Corridor network distance == 6.5 km", abs(corr.network_distance_km - 6.5) < 1e-4)
    check("Corridor haversine distance is present", corr.haversine_distance_km is not None)
    check("Network distance != Haversine distance", abs(corr.network_distance_km - corr.haversine_distance_km) > 0.05)

    # ── Suite 5: Anomaly Rate & Transparency Invariants ──
    print("\n[5] Anomaly Rate & Transparency Invariants")
    # Add a normal transit and an anomalous speeding transit (390 km/h)
    traj_norm = create_synthetic_trajectory(
        global_id="GV-000301",
        hops=[("CAM-001", 1000.0, 1020.0, 1), ("CAM-002", 1620.0, 1640.0, 2)],  # 600s => 39 km/h
    )
    traj_speeding = create_synthetic_trajectory(
        global_id="GV-000302",
        hops=[("CAM-001", 1000.0, 1020.0, 1), ("CAM-002", 1080.0, 1100.0, 2)],  # 60s => 390 km/h!
    )
    rep_anom = engine.analyze_trajectories([traj_norm, traj_speeding])
    corr_anom = rep_anom.get_corridor("CAM-001", "CAM-002")

    check("Observation count N == 2", corr_anom.observation_count == 2)
    check("Valid observations count == 1", corr_anom.valid_observation_count == 1)
    check("Anomalous observations count == 1", corr_anom.anomalous_observation_count == 1)
    check("Velocity anomaly count == 1", corr_anom.velocity_anomaly_count == 1)
    check("Anomaly rate is exactly 0.5 (1/2)", abs(corr_anom.anomaly_rate - 0.5) < 1e-4)
    check("Anomaly rate <= 1.0 invariant holds", corr_anom.anomaly_rate <= 1.0)
    check("Valid + anomalous count == total N", corr_anom.valid_observation_count + corr_anom.anomalous_observation_count == corr_anom.observation_count)

    # ── Suite 6: Time-Window Bucketing by Departure Timestamp ──
    print("\n[6] Time-Window Bucketing by Departure Timestamp (from_timestamp)")
    # Dep 07:30 (Morning Peak: 06:00-09:00) vs Dep 13:00 (Early Afternoon: 12:00-15:00)
    traj_morning = create_synthetic_trajectory(
        global_id="GV-000401",
        hops=[("CAM-001", 7.5 * 3600, 7.5 * 3600 + 20, 1), ("CAM-002", 7.5 * 3600 + 620, 7.5 * 3600 + 640, 2)],
    )
    traj_afternoon = create_synthetic_trajectory(
        global_id="GV-000402",
        hops=[("CAM-001", 13.0 * 3600, 13.0 * 3600 + 20, 1), ("CAM-002", 13.0 * 3600 + 900, 13.0 * 3600 + 920, 2)],
    )
    rep_win = engine.analyze_trajectories([traj_morning, traj_afternoon])

    check("Morning peak bucket '06:00-09:00' exists", "06:00-09:00" in rep_win.time_windows)
    check("Afternoon bucket '12:00-15:00' exists", "12:00-15:00" in rep_win.time_windows)

    morn_corr = rep_win.time_windows["06:00-09:00"].get(("CAM-001", "CAM-002"))
    after_corr = rep_win.time_windows["12:00-15:00"].get(("CAM-001", "CAM-002"))

    check("Morning peak bucket has N=1", morn_corr is not None and morn_corr.observation_count == 1)
    check("Afternoon bucket has N=1", after_corr is not None and after_corr.observation_count == 1)
    check("Morning travel time == 600.0s", abs(morn_corr.travel_time_median_s - 600.0) < 1e-4)
    check("Afternoon travel time == 880.0s", abs(after_corr.travel_time_median_s - 880.0) < 1e-4)

    # ── Suite 7: Full Trip Origin-Destination (OD) Matrix ──
    print("\n[7] Full Trip Origin-Destination (OD) Matrix")
    # Multi-hop trajectory: CAM-001 -> CAM-002 -> CAM-004
    # Origin is CAM-001, Destination is CAM-004
    traj_od1 = create_synthetic_trajectory(
        global_id="GV-000501",
        hops=[
            ("CAM-001", 1000.0, 1020.0, 1),
            ("CAM-002", 1620.0, 1640.0, 2),
            ("CAM-004", 2840.0, 2860.0, 3),
        ],
        vehicle_type="bus",
    )
    traj_od2 = create_synthetic_trajectory(
        global_id="GV-000502",
        hops=[
            ("CAM-001", 2000.0, 2020.0, 1),
            ("CAM-002", 2620.0, 2640.0, 2),
            ("CAM-004", 3840.0, 3860.0, 3),
        ],
        vehicle_type="car",
    )
    rep_od = engine.analyze_trajectories([traj_od1, traj_od2])

    check("OD matrix has CAM-001 origin", "CAM-001" in rep_od.od_matrix)
    check("OD matrix CAM-001 -> CAM-004 trip count == 2", rep_od.od_matrix["CAM-001"].get("CAM-004") == 2)
    check("OD does NOT record intermediate segment as trip OD destination", "CAM-002" not in rep_od.od_matrix["CAM-001"])

    od_rec = rep_od.od_details.get(("CAM-001", "CAM-004"))
    check("TripODRecord exists for CAM-001 -> CAM-004", od_rec is not None)
    check("TripODRecord trip_count == 2", od_rec.trip_count == 2)
    check("TripODRecord has fleet mix (1 bus, 1 car)", od_rec.vehicle_type_counts.get("bus") == 1 and od_rec.vehicle_type_counts.get("car") == 1)

    # ── Suite 8: SQLite Database Integration (analyze_db) ──
    print("\n[8] SQLite Direct Database Analysis Integration (analyze_db)")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "analytics_test.db"
        conn = init_db(db_path)

        gid = "GV-000601"
        identity = GlobalVehicleIdentity(
            global_id=gid,
            canonical_plate="KA03MH1111",
            plate_confidence=0.95,
            vehicle_type="car",
            first_seen_ts=10000.0,
            last_seen_ts=10650.0,
            first_camera_id="CAM-001",
            last_camera_id="CAM-002",
            sighting_count=2,
            status="active",
        )
        save_global_identity(conn, identity)

        obs1 = VehicleObservation(camera_id="CAM-001", track_id=1, timestamp=10020.0, canonical_plate="KA03MH1111", vehicle_type="car")
        record_vehicle_observation(conn, obs1, IdentityMatchResult(status="NEW", global_id=gid, confidence=1.0), first_timestamp=10000.0)

        obs2 = VehicleObservation(camera_id="CAM-002", track_id=2, timestamp=10650.0, canonical_plate="KA03MH1111", vehicle_type="car")
        record_vehicle_observation(conn, obs2, IdentityMatchResult(status="MATCH", global_id=gid, confidence=0.95), first_timestamp=10620.0)

        recon = TrajectoryReconstructor(cameras_path="configs/cameras.json", camera_graph_path="configs/camera_graph.json")
        db_report = engine.analyze_db(conn, reconstructor=recon)

        check("analyze_db produces NetworkAnalyticsReport", db_report is not None)
        check("DB report analyzed 1 trajectory", db_report.total_trajectories_analyzed == 1)
        check("DB report has 1 transit observation", db_report.total_transit_observations == 1)
        db_corr = db_report.get_corridor("CAM-001", "CAM-002")
        check("DB corridor CAM-001 -> CAM-002 N=1", db_corr is not None and db_corr.observation_count == 1)
        check("DB corridor travel time == 600.0s", abs(db_corr.travel_time_median_s - 600.0) < 1e-4)
        check("DB corridor speed == 39.0 km/h", abs(db_corr.speed_median_kmh - 39.0) < 0.2)

        # Dictionary export and summary check
        rep_dict = db_report.to_dict()
        check("to_dict() contains 'totals'", "totals" in rep_dict)
        check("to_dict() contains 'corridors'", "corridors" in rep_dict)
        check("summary() contains report title", "ANALYTICS REPORT" in db_report.summary())

        conn.close()

    # ── Final Summary ──
    print("\n" + "=" * 65)
    print(f"  Results: {PASS}/{PASS + FAIL} passed, {FAIL} failed")
    print("=" * 65 + "\n")

    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
