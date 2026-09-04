"""
Phase 7C -- Acceptance Tests for Traffic Flow, Density & Congestion Modeling.

Verifies:
1. Flow rate primitives: camera_flow_rate_veh_hr vs corridor_transit_rate_veh_hr.
2. Fundamental density relation consistency: k = q / v (veh/km).
3. Free-flow baseline sourcing (CORRIDOR_CONFIG vs DEFAULT_ASSUMPTION).
4. Zero or invalid free-flow speed (safe handling, no division by zero).
5. Kinematic degradation metrics: SPI, speed_degradation_pct, TTI, travel_time_increase_pct.
6. Project Level of Service (LOS) Proxy classification (A through F, NOT claimed as HCM LOS).
7. Sample-size confidence scoring (sample_confidence_score = min(1.0, N / 10)).
8. Camera node temporal occupancy ratio via interval union (guaranteed in [0.0, 1.0]).
9. Congestion hotspot detection and severity ranking.
10. Direct SQLite database analysis integration (analyze_db).
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
from alpr.analytics import CorridorAnalytics
from alpr.congestion import (
    CameraNodeFlowMetrics,
    CorridorCongestionMetrics,
    NetworkCongestionReport,
    TrafficCongestionEngine,
    analyze_traffic_congestion,
    classify_los_proxy,
    compute_interval_union_duration,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("test_phase7c")

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
    print("  Phase 7C -- Traffic Flow, Density & Congestion Modeling Tests")
    print("=" * 65)

    engine = TrafficCongestionEngine(
        default_free_flow_speed_kmh=50.0,
        corridor_free_flow_speeds={
            ("CAM-001", "CAM-002"): 60.0,  # Express corridor config
        },
    )

    # ── Suite 1: Temporal Occupancy via Interval Union ──
    print("\n[1] Camera Temporal Occupancy via Interval Union")
    # Window: [1000.0, 1100.0] (100 seconds)
    # Track 1: [1010.0, 1030.0] (20s)
    # Track 2: [1020.0, 1040.0] (20s, overlapping with Track 1 between 1020-1030)
    # Union is [1010.0, 1040.0] = 30 seconds (NOT 40 seconds)
    intervals = [(1010.0, 1030.0), (1020.0, 1040.0)]
    occ_dur = compute_interval_union_duration(intervals, 1000.0, 1100.0)
    check("Interval union correctly resolves overlapping tracks to 30.0s", abs(occ_dur - 30.0) < 1e-4)

    # Empty intervals
    check("Empty intervals yield 0.0s", compute_interval_union_duration([], 1000.0, 1100.0) == 0.0)

    # Out of window intervals clipped
    out_intervals = [(900.0, 950.0), (1150.0, 1200.0)]
    check("Out-of-window intervals clipped to 0.0s", compute_interval_union_duration(out_intervals, 1000.0, 1100.0) == 0.0)

    # ── Suite 2: LOS Proxy Classification (TTI-Based Proxy, NOT HCM LOS) ──
    print("\n[2] Project LOS Proxy Classification (TTI-Based)")
    los_a, cat_a = classify_los_proxy(1.05)
    check("TTI=1.05 maps to LOS Proxy A (FREE_FLOW)", los_a == "A" and cat_a == "FREE_FLOW")

    los_b, cat_b = classify_los_proxy(1.18)
    check("TTI=1.18 maps to LOS Proxy B (LIGHT)", los_b == "B" and cat_b == "LIGHT")

    los_c, cat_c = classify_los_proxy(1.35)
    check("TTI=1.35 maps to LOS Proxy C (MODERATE)", los_c == "C" and cat_c == "MODERATE")

    los_d, cat_d = classify_los_proxy(1.80)
    check("TTI=1.80 maps to LOS Proxy D (HEAVY)", los_d == "D" and cat_d == "HEAVY")

    los_e, cat_e = classify_los_proxy(2.20)
    check("TTI=2.20 maps to LOS Proxy E (SEVERE)", los_e == "E" and cat_e == "SEVERE")

    los_f, cat_f = classify_los_proxy(2.80)
    check("TTI=2.80 maps to LOS Proxy F (BREAKDOWN)", los_f == "F" and cat_f == "BREAKDOWN")

    los_u, cat_u = classify_los_proxy(None)
    check("TTI=None maps to UNKNOWN / INSUFFICIENT_DATA", los_u == "UNKNOWN" and cat_u == "INSUFFICIENT_DATA")

    # ── Suite 3: Free-Flow Speed Sourcing & Fallbacks ──
    print("\n[3] Free-Flow Speed Sourcing & Fallbacks")
    ff_speed_conf, ff_src_conf = engine.get_free_flow_speed("CAM-001", "CAM-002")
    check("Configured corridor uses CORRIDOR_CONFIG (60 km/h)", ff_speed_conf == 60.0 and ff_src_conf == "CORRIDOR_CONFIG")

    ff_speed_def, ff_src_def = engine.get_free_flow_speed("CAM-002", "CAM-004")
    check("Unconfigured corridor uses DEFAULT_ASSUMPTION (50 km/h)", ff_speed_def == 50.0 and ff_src_def == "DEFAULT_ASSUMPTION")

    # ── Suite 4: Density Consistency (k = q / v) ──
    print("\n[4] Fundamental Density Consistency (k = q / v in veh/km)")
    # Case: Q = 600 veh/hr, v = 30 km/h => k = 20 veh/km
    # Setup CorridorAnalytics with N=600 transits in 1 hour (3600s), median speed = 30 km/h
    c_stat_dens = CorridorAnalytics(
        from_camera="CAM-TEST-A",
        to_camera="CAM-TEST-B",
        observation_count=600,
        valid_observation_count=600,
        anomalous_observation_count=0,
        network_distance_km=10.0,
        haversine_distance_km=9.5,
        travel_time_median_s=1200.0,  # 20 min = 1200s => 30 km/h
        speed_median_kmh=30.0,
    )
    cm_dens = engine.analyze_corridor(c_stat_dens, window_duration_seconds=3600.0)
    check("Corridor transit rate == 600.0 veh/hr", abs(cm_dens.corridor_transit_rate_veh_hr - 600.0) < 1e-4)
    check("Corridor transit rate == 10.0 veh/min", abs(cm_dens.corridor_transit_rate_veh_min - 10.0) < 1e-4)
    check("Estimated density k == 20.0 veh/km (600 / 30)", abs(cm_dens.estimated_density_veh_km - 20.0) < 1e-4)

    # ── Suite 5: Kinematic Baseline & Degradation Metrics ──
    print("\n[5] Kinematic Baseline & Degradation Metrics (SPI, Degradation %, TTI)")
    # Test case from user prompt:
    # d = 10 km, v_ff = 50 km/h => t_ff = 720s
    # observed median speed = 30 km/h, observed travel time = 1200s
    # SPI = (30 / 50) * 100 = 60.0%
    # Speed degradation % = (1 - 30/50) * 100 = 40.0%
    # TTI = 1200 / 720 = 1.6667
    # Travel time increase % = (1.6667 - 1) * 100 = 66.67%
    c_stat_deg = CorridorAnalytics(
        from_camera="CAM-002",
        to_camera="CAM-004",  # Uses default v_ff = 50.0 km/h
        observation_count=15,
        valid_observation_count=15,
        anomalous_observation_count=0,
        network_distance_km=10.0,
        haversine_distance_km=9.0,
        travel_time_median_s=1200.0,
        speed_median_kmh=30.0,
    )
    cm_deg = engine.analyze_corridor(c_stat_deg, window_duration_seconds=3600.0)
    check("Free-flow travel time == 720.0s", abs(cm_deg.free_flow_travel_time_s - 720.0) < 1e-4)
    check("Speed Performance Index (SPI) == 60.0%", abs(cm_deg.speed_performance_index - 60.0) < 1e-4)
    check("Speed degradation == 40.0%", abs(cm_deg.speed_degradation_pct - 40.0) < 1e-4)
    check("Travel Time Index (TTI) == 1.6667", abs(cm_deg.travel_time_index - 1.6667) < 0.001)
    check("Travel time increase == 66.67%", abs(cm_deg.travel_time_increase_pct - 66.67) < 0.05)
    check("LOS Proxy is D (approaching unstable)", cm_deg.los_proxy == "D")
    check("Congestion category is HEAVY", cm_deg.congestion_category == "HEAVY")

    # ── Suite 6: Sample Size Confidence Penalty ──
    print("\n[6] Sample Size Confidence Penalty (N < 3 vs N >= 10)")
    # High sample size (N=12)
    c_stat_high = CorridorAnalytics(
        from_camera="CAM-002", to_camera="CAM-004", observation_count=12, valid_observation_count=12,
        anomalous_observation_count=0, network_distance_km=10.0, travel_time_median_s=1200.0, speed_median_kmh=30.0
    )
    cm_high = engine.analyze_corridor(c_stat_high, 3600.0)
    check("N=12 yields sample_confidence_score == 1.0", cm_high.sample_confidence_score == 1.0)
    check("N=12 category is 'HEAVY'", cm_high.congestion_category == "HEAVY")

    # Sparse sample size (N=2)
    c_stat_sparse = CorridorAnalytics(
        from_camera="CAM-002", to_camera="CAM-004", observation_count=2, valid_observation_count=2,
        anomalous_observation_count=0, network_distance_km=10.0, travel_time_median_s=1200.0, speed_median_kmh=30.0
    )
    cm_sparse = engine.analyze_corridor(c_stat_sparse, 3600.0)
    check("N=2 yields sample_confidence_score == 0.20", abs(cm_sparse.sample_confidence_score - 0.20) < 1e-4)
    check("N=2 category flagged with LOW_SAMPLE", "LOW_SAMPLE" in cm_sparse.congestion_category)

    # Empty sample size (N=0)
    c_stat_empty = CorridorAnalytics(
        from_camera="CAM-002", to_camera="CAM-004", observation_count=0, valid_observation_count=0,
        anomalous_observation_count=0, network_distance_km=10.0, travel_time_median_s=None, speed_median_kmh=None
    )
    cm_empty = engine.analyze_corridor(c_stat_empty, 3600.0)
    check("N=0 category is INSUFFICIENT_DATA", cm_empty.congestion_category == "INSUFFICIENT_DATA")
    check("N=0 confidence score == 0.0", cm_empty.sample_confidence_score == 0.0)

    # ── Suite 7: Zero / Invalid Free-Flow Speed Resilience ──
    print("\n[7] Zero / Invalid Free-Flow Speed Resilience")
    engine_zero = TrafficCongestionEngine(default_free_flow_speed_kmh=0.0)
    cm_zero = engine_zero.analyze_corridor(c_stat_deg, 3600.0)
    check("Zero free-flow speed handles gracefully without crash", cm_zero is not None)
    check("TTI is None when v_ff <= 0", cm_zero.travel_time_index is None)
    check("SPI is None when v_ff <= 0", cm_zero.speed_performance_index is None)
    check("Speed degradation is None when v_ff <= 0", cm_zero.speed_degradation_pct is None)
    check("Free-flow source is INVALID_BASELINE", cm_zero.free_flow_speed_source == "INVALID_BASELINE")

    # ── Suite 8: Network Multi-Camera Trajectory Congestion Analysis ──
    print("\n[8] Full Network Trajectory Congestion & Hotspot Analysis")
    # Build 4 trajectories with heavy congestion on CAM-001 -> CAM-002 (TTI = 1.8)
    # and free-flow on CAM-002 -> CAM-004 (TTI = 1.02)
    trajs = []
    for i in range(4):
        n1 = TrajectoryNode("CAM-001", "MG Road", 12.97, 77.60, 1000.0 + i*10, 1020.0 + i*10, 20.0, 1)
        # 6.5 km road dist, free flow is 60 km/h => t_ff = 390s.
        # Transit time = 702s => TTI = 702 / 390 = 1.80 (HEAVY congestion)
        n2 = TrajectoryNode("CAM-002", "Silk Board", 12.91, 77.62, 1020.0 + i*10 + 702.0, 1020.0 + i*10 + 722.0, 20.0, 2)
        # 14.2 km road dist, free flow is 50 km/h => t_ff = 1022.4s.
        # Transit time = 1042s => TTI = 1042 / 1022.4 = 1.019 (FREE_FLOW)
        n3 = TrajectoryNode("CAM-004", "Hebbal Flyover", 13.03, 77.59, 1020.0 + i*10 + 722.0 + 1042.0, 1020.0 + i*10 + 722.0 + 1062.0, 20.0, 3)

        s1 = TrajectorySegment("CAM-001", "CAM-002", n1.last_timestamp, n2.first_timestamp, 702.0, 6.5, 6.2, 33.33)
        s2 = TrajectorySegment("CAM-002", "CAM-004", n2.last_timestamp, n3.first_timestamp, 1042.0, 14.2, 13.5, 49.06)

        trajs.append(VehicleTrajectory(
            global_id=f"GV-00070{i}",
            canonical_plate=f"KA01AB000{i}",
            vehicle_type="car",
            first_seen_ts=n1.first_timestamp,
            last_seen_ts=n3.last_timestamp,
            total_duration_seconds=n3.last_timestamp - n1.first_timestamp,
            total_network_distance_km=20.7,
            nodes=[n1, n2, n3],
            segments=[s1, s2],
        ))

    net_rep = engine.analyze(trajs, window_duration_seconds=3600.0)
    check("Network report generated", net_rep is not None)
    check("Total transit observations == 8 (4 per corridor)", net_rep.total_transit_observations == 8)
    check("Camera node CAM-001 observed 4 unique vehicles", net_rep.get_camera("CAM-001").unique_vehicles_observed == 4)
    check("Camera node CAM-001 flow rate == 4.0 veh/hr", net_rep.get_camera("CAM-001").camera_flow_rate_veh_hr == 4.0)

    cm12 = net_rep.get_corridor("CAM-001", "CAM-002")
    check("Corridor CAM-001 -> CAM-002 has TTI approx 1.80", abs(cm12.travel_time_index - 1.80) < 0.02)
    check("Corridor CAM-001 -> CAM-002 has LOS Proxy D", cm12.los_proxy == "D")

    cm24 = net_rep.get_corridor("CAM-002", "CAM-004")
    check("Corridor CAM-002 -> CAM-004 has TTI approx 1.02", abs(cm24.travel_time_index - 1.02) < 0.02)
    check("Corridor CAM-002 -> CAM-004 has LOS Proxy A", cm24.los_proxy == "A")

    # Hotspot ranking check
    check("Exactly 1 congested hotspot identified (TTI >= 1.25)", len(net_rep.hotspots) == 1)
    check("Top hotspot is CAM-001 -> CAM-002", net_rep.hotspots[0]["corridor"] == "CAM-001 -> CAM-002")

    # ── Suite 9: SQLite Direct Database Analysis ──
    print("\n[9] SQLite Direct Database Analysis (analyze_db)")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "congestion_test.db"
        conn = init_db(db_path)

        gid = "GV-000801"
        save_global_identity(conn, GlobalVehicleIdentity(
            global_id=gid, canonical_plate="KA04AB9999", vehicle_type="car",
            first_seen_ts=1000.0, last_seen_ts=1750.0, first_camera_id="CAM-001", last_camera_id="CAM-002"
        ))
        record_vehicle_observation(conn, VehicleObservation("CAM-001", 1, 1020.0, canonical_plate="KA04AB9999", vehicle_type="car"),
                                   IdentityMatchResult(status="NEW", global_id=gid, confidence=1.0), first_timestamp=1000.0)
        record_vehicle_observation(conn, VehicleObservation("CAM-002", 2, 1750.0, canonical_plate="KA04AB9999", vehicle_type="car"),
                                   IdentityMatchResult(status="MATCH", global_id=gid, confidence=0.95), first_timestamp=1720.0)

        recon = TrajectoryReconstructor(cameras_path="configs/cameras.json", camera_graph_path="configs/camera_graph.json")
        db_cong_rep = engine.analyze_db(conn, reconstructor=recon)

        check("analyze_db produces NetworkCongestionReport", db_cong_rep is not None)
        check("DB report contains CAM-001 -> CAM-002 corridor", db_cong_rep.get_corridor("CAM-001", "CAM-002") is not None)

        # Verify summary output
        sum_txt = db_cong_rep.summary()
        check("summary() contains report header", "CONGESTION & TRAFFIC FLOW REPORT" in sum_txt)

        conn.close()

    # ── Summary ──
    print("\n" + "=" * 65)
    print(f"  Results: {PASS}/{PASS + FAIL} passed, {FAIL} failed")
    print("=" * 65 + "\n")

    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
