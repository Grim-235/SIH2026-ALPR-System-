"""
Phase 7E -- Acceptance Tests for Security Alerts, Blacklist Enforcement, Suspicious Route Diagnostics & Anomaly Filtering.

Verifies:
1. Exact Blacklist Matching: Watchlist match generates BLACKLIST_EXACT with CRITICAL or HIGH severity.
2. Fuzzy Blacklist Matching: Indian character visual confusion generates BLACKLIST_FUZZY with similarity score.
3. Kinematic Velocity Anomaly: Speeds > 140 km/h bound generate VELOCITY_ANOMALY diagnostic warnings.
4. Temporal Inversion Anomaly: Transit intervals <= 0s generate TEMPORAL_INVERSION warnings.
5. Topological Disconnection: Sighting hops across unlinked nodes generate TOPOLOGY_VIOLATION alerts.
6. Identity Uncertainty Diagnostics: Observations with match_status == 'UNCERTAIN' generate IDENTITY_UNCERTAIN alerts.
7. Behavioral Loitering & Rapid Looping: Excessive nodal dwell (> 180s) and rapid corridor traversals generate behavioral alerts.
8. Database Persistence & Idempotency: Duplicate alert scans do not duplicate records (ON CONFLICT update).
9. Alert Acknowledgment Lifecycle: Acknowledgment updates status, timestamp, and operator without data corruption.
10. Summary & Metric Counters: Aggregations across total, unacknowledged, severity, and alert type.
11. REST API /api/v1 Endpoints: Verification of alert list, summary, acknowledgment, scan, and blacklist CRUD.
12. Architectural Boundary Verification: Zero alert or kinematic formulas inside Flask app.py handlers.
"""

import inspect
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
    record_security_alert,
    get_security_alerts,
    get_security_alert_by_id,
    acknowledge_security_alert,
    get_security_alerts_summary,
    add_enriched_blacklist_entry,
    get_enriched_blacklist,
)
from alpr.identity import (
    GlobalVehicleIdentity,
    VehicleObservation,
    IdentityMatchResult,
    compute_plate_similarity,
)
from alpr.trajectory import (
    TrajectoryNode,
    TrajectorySegment,
    VehicleTrajectory,
    TrajectoryReconstructor,
)
from alpr.alerts import (
    AlertRecord,
    AlertEngine,
    evaluate_blacklist_match,
    evaluate_kinematic_anomalies,
    evaluate_topological_anomalies,
    evaluate_identity_uncertainty,
    evaluate_behavioral_anomalies,
    ALERT_BLACKLIST_EXACT,
    ALERT_BLACKLIST_FUZZY,
    ALERT_VELOCITY_ANOMALY,
    ALERT_TEMPORAL_INVERSION,
    ALERT_TOPOLOGY_VIOLATION,
    ALERT_IDENTITY_UNCERTAIN,
    ALERT_EXCESSIVE_DWELL,
    ALERT_RAPID_LOOPING,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_MEDIUM,
    SEVERITY_LOW,
)
from alpr.service import (
    DashboardService,
    get_dashboard_service,
)
from app import app

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("test_phase7e")

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    status = "[PASS]" if condition else "[FAIL]"
    if condition:
        PASS += 1
        log.info(f"{status} {name}")
    else:
        FAIL += 1
        msg = f"{status} {name}"
        if detail:
            msg += f" -- {detail}"
        log.error(msg)


# ============================================================================
# SUITE 1: BLACKLIST MATCHING (EXACT & FUZZY WITH INDIAN CHARACTER CONFUSION)
# ============================================================================
def test_suite_1_blacklist_matching():
    log.info("\n--- SUITE 1: Blacklist Matching (Exact & Fuzzy) ---")

    blacklist = [
        {"plate_text": "DL8CAZ9592", "category": "STOLEN", "severity": "CRITICAL", "reason": "FIR #102/2026 Vehicle Theft"},
        {"plate_text": "MH12DE1433", "category": "WANTED", "severity": "CRITICAL", "reason": "Homicide suspect"},
        {"plate_text": "KA01AB5000", "category": "EXPIRED", "severity": "MEDIUM", "reason": "Commercial tax evasion"},
    ]

    # 1. Exact match stolen
    alerts1 = evaluate_blacklist_match("DL8CAZ9592", blacklist, "CAM-001", 1725450000.0)
    check("Exact match triggers 1 alert", len(alerts1) == 1)
    check("Alert type is BLACKLIST_EXACT", alerts1[0].alert_type == ALERT_BLACKLIST_EXACT)
    check("Stolen vehicle severity is CRITICAL", alerts1[0].severity == SEVERITY_CRITICAL)
    check("Details contain exact match flag", alerts1[0].details.get("match_type") == "EXACT")

    # 2. Case and whitespace insensitivity
    alerts2 = evaluate_blacklist_match("  mh12de1433  ", blacklist, "CAM-002", 1725450010.0)
    check("Exact match with whitespace/lowercase", len(alerts2) == 1)
    check("Matched canonical plate is uppercase", alerts2[0].canonical_plate == "MH12DE1433")

    # 3. Fuzzy match: Indian 8 vs B confusion (DL8CAZ9592 vs DLBCAZ9592)
    alerts3 = evaluate_blacklist_match("DLBCAZ9592", blacklist, "CAM-001", 1725450020.0, fuzzy_threshold=0.85)
    check("Fuzzy match triggers 1 alert", len(alerts3) == 1)
    check("Alert type is BLACKLIST_FUZZY", alerts3[0].alert_type == ALERT_BLACKLIST_FUZZY)
    check("Fuzzy alert severity is MEDIUM", alerts3[0].severity == SEVERITY_MEDIUM)
    check("Fuzzy similarity >= 0.85 recorded", alerts3[0].details.get("similarity", 0.0) >= 0.85)

    # 4. Unrelated plate does not match
    alerts4 = evaluate_blacklist_match("UP16BT9999", blacklist, "CAM-001", 1725450030.0)
    check("Unrelated plate produces zero alerts", len(alerts4) == 0)

    # 5. Empty or None plate produces zero alerts
    check("None plate produces zero alerts", len(evaluate_blacklist_match(None, blacklist, "CAM-001", 100.0)) == 0)
    check("Empty string plate produces zero alerts", len(evaluate_blacklist_match("", blacklist, "CAM-001", 100.0)) == 0)


# ============================================================================
# SUITE 2: KINEMATIC PLAUSIBILITY & VELOCITY ANOMALY DIAGNOSTICS
# ============================================================================
def test_suite_2_kinematic_anomalies():
    log.info("\n--- SUITE 2: Kinematic Plausibility & Velocity Anomalies ---")

    # Normal plausible segment (6.5 km in 390s = 60 km/h)
    normal_seg = TrajectorySegment(
        from_camera_id="CAM-001",
        to_camera_id="CAM-002",
        from_timestamp=100.0,
        to_timestamp=490.0,
        transit_time_seconds=390.0,
        network_distance_km=6.5,
        speed_kmh=60.0,
        is_velocity_anomaly=False,
    )
    alerts_norm = evaluate_kinematic_anomalies(normal_seg, global_id="GV-000001", canonical_plate="MH12DE1433")
    check("Normal speed produces zero kinematic alerts", len(alerts_norm) == 0)

    # Velocity anomaly: 6.5 km in 50s = 468 km/h (> 140 km/h bound)
    speed_seg = TrajectorySegment(
        from_camera_id="CAM-001",
        to_camera_id="CAM-002",
        from_timestamp=100.0,
        to_timestamp=150.0,
        transit_time_seconds=50.0,
        network_distance_km=6.5,
        speed_kmh=468.0,
        is_velocity_anomaly=True,
    )
    alerts_speed = evaluate_kinematic_anomalies(speed_seg, global_id="GV-000001", canonical_plate="MH12DE1433")
    check("Excessive speed produces 1 alert", len(alerts_speed) == 1)
    check("Alert type is VELOCITY_ANOMALY", alerts_speed[0].alert_type == ALERT_VELOCITY_ANOMALY)
    check("Velocity anomaly labeled as diagnostic", alerts_speed[0].details.get("is_diagnostic") is True)
    check("Reported speed matches segment speed", alerts_speed[0].details.get("speed_kmh") == 468.0)
    check("Destination camera is target camera", alerts_speed[0].camera_id == "CAM-002")

    # Temporal inversion: transit_time_seconds <= 0
    inversion_seg = TrajectorySegment(
        from_camera_id="CAM-002",
        to_camera_id="CAM-001",
        from_timestamp=500.0,
        to_timestamp=480.0,
        transit_time_seconds=-20.0,
        network_distance_km=6.5,
        speed_kmh=None,
        is_temporal_anomaly=True,
    )
    alerts_inv = evaluate_kinematic_anomalies(inversion_seg, global_id="GV-000002")
    check("Temporal inversion produces 1 alert", len(alerts_inv) == 1)
    check("Alert type is TEMPORAL_INVERSION", alerts_inv[0].alert_type == ALERT_TEMPORAL_INVERSION)
    check("Temporal inversion severity is HIGH", alerts_inv[0].severity == SEVERITY_HIGH)
    check("Transit interval is negative", alerts_inv[0].details.get("transit_time_seconds") == -20.0)


# ============================================================================
# SUITE 3: TOPOLOGICAL VIOLATIONS & NETWORK REACHABILITY
# ============================================================================
def test_suite_3_topological_violations():
    log.info("\n--- SUITE 3: Topological Violations ---")

    # Segment between disconnected cameras
    unreach_seg = TrajectorySegment(
        from_camera_id="CAM-001",
        to_camera_id="CAM-999",
        from_timestamp=100.0,
        to_timestamp=300.0,
        transit_time_seconds=200.0,
        network_distance_km=None,
        is_unreachable_network=True,
    )
    alerts_top = evaluate_topological_anomalies(unreach_seg, global_id="GV-000003")
    check("Unreachable network segment produces 1 alert", len(alerts_top) == 1)
    check("Alert type is TOPOLOGY_VIOLATION", alerts_top[0].alert_type == ALERT_TOPOLOGY_VIOLATION)
    check("Alert contains corridor details", alerts_top[0].details.get("from_camera_id") == "CAM-001")

    # Same-camera sighting is not a topology violation
    same_cam_seg = TrajectorySegment(
        from_camera_id="CAM-001",
        to_camera_id="CAM-001",
        from_timestamp=100.0,
        to_timestamp=150.0,
        transit_time_seconds=50.0,
        is_same_camera=True,
        is_unreachable_network=True,
    )
    alerts_same = evaluate_topological_anomalies(same_cam_seg)
    check("Same camera transit is excluded from topology alerts", len(alerts_same) == 0)


# ============================================================================
# SUITE 4: IDENTITY RESOLVER UNCERTAINTY DIAGNOSTICS
# ============================================================================
def test_suite_4_identity_uncertainty():
    log.info("\n--- SUITE 4: Identity Uncertainty Diagnostics ---")

    # Sighting node with UNCERTAIN match status
    uncertain_node = TrajectoryNode(
        camera_id="CAM-003",
        camera_name="KR Puram Bridge",
        latitude=13.0073,
        longitude=77.6964,
        first_timestamp=1725451000.0,
        last_timestamp=1725451015.0,
        duration_seconds=15.0,
        local_track_id=42,
        canonical_plate="KA03MJ1122",
        plate_confidence=0.62,
        match_status="UNCERTAIN",
        match_method="fused_multimodal",
        match_confidence=0.64,
    )
    alerts = evaluate_identity_uncertainty(uncertain_node, global_id="GV-000004")
    check("Uncertain node produces 1 alert", len(alerts) == 1)
    check("Alert type is IDENTITY_UNCERTAIN", alerts[0].alert_type == ALERT_IDENTITY_UNCERTAIN)
    check("Alert severity is LOW", alerts[0].severity == SEVERITY_LOW)
    check("Local track ID is preserved in details", alerts[0].details.get("local_track_id") == 42)

    # MATCH node produces no uncertainty alert
    match_node = TrajectoryNode(
        camera_id="CAM-001",
        camera_name="MG Road",
        latitude=12.9756,
        longitude=77.6062,
        first_timestamp=100.0,
        last_timestamp=110.0,
        duration_seconds=10.0,
        local_track_id=1,
        match_status="MATCH",
    )
    check("MATCH node produces zero uncertainty alerts", len(evaluate_identity_uncertainty(match_node)) == 0)


# ============================================================================
# SUITE 5: BEHAVIORAL SURVEILLANCE RULES (LOITERING & RAPID LOOPING)
# ============================================================================
def test_suite_5_behavioral_rules():
    log.info("\n--- SUITE 5: Behavioral Surveillance Rules ---")

    # 1. Excessive Dwell: node duration = 240s (> 180s default)
    long_dwell_node = TrajectoryNode(
        camera_id="CAM-001",
        camera_name="MG Road",
        latitude=12.9756,
        longitude=77.6062,
        first_timestamp=1000.0,
        last_timestamp=1240.0,
        duration_seconds=240.0,
        local_track_id=10,
    )
    traj_dwell = VehicleTrajectory(
        global_id="GV-000010",
        nodes=[long_dwell_node],
        first_seen_ts=1000.0,
        last_seen_ts=1240.0,
    )
    alerts_dwell = evaluate_behavioral_anomalies(traj_dwell, max_dwell_seconds=180.0)
    check("Dwell > 180s produces EXCESSIVE_DWELL alert", len(alerts_dwell) == 1)
    check("Alert type is EXCESSIVE_DWELL", alerts_dwell[0].alert_type == ALERT_EXCESSIVE_DWELL)
    check("Recorded dwell duration is 240s", alerts_dwell[0].details.get("dwell_duration_seconds") == 240.0)

    # 2. Rapid Looping: 3 passes across CAM-001 -> CAM-002 within 250s (threshold 300s)
    seg1 = TrajectorySegment("CAM-001", "CAM-002", 100.0, 130.0, 30.0)
    seg2 = TrajectorySegment("CAM-002", "CAM-001", 140.0, 170.0, 30.0)
    seg3 = TrajectorySegment("CAM-001", "CAM-002", 180.0, 210.0, 30.0)
    seg4 = TrajectorySegment("CAM-002", "CAM-001", 220.0, 250.0, 30.0)
    seg5 = TrajectorySegment("CAM-001", "CAM-002", 260.0, 290.0, 30.0)

    traj_loop = VehicleTrajectory(
        global_id="GV-000011",
        nodes=[long_dwell_node],  # dummy
        segments=[seg1, seg2, seg3, seg4, seg5],
    )
    alerts_loop = evaluate_behavioral_anomalies(traj_loop, max_dwell_seconds=999.0, rapid_loop_window_seconds=300.0)
    check("Looping corridor 3 times in 160s produces RAPID_LOOPING alert", len(alerts_loop) >= 1)
    loop_alert = [a for a in alerts_loop if a.alert_type == ALERT_RAPID_LOOPING][0]
    check("Loop alert indicates correct corridor", loop_alert.details.get("from_camera_id") == "CAM-001")


# ============================================================================
# SUITE 6: FULL TRAJECTORY EVALUATION WITH ALERT ENGINE
# ============================================================================
def test_suite_6_alert_engine_evaluation():
    log.info("\n--- SUITE 6: Full Trajectory AlertEngine Evaluation ---")

    engine = AlertEngine(velocity_bound_kmh=140.0)
    blacklist = [
        {"plate_text": "KA04MH7777", "category": "STOLEN", "severity": "CRITICAL", "reason": "Carjacked"},
    ]

    node1 = TrajectoryNode(
        camera_id="CAM-001",
        camera_name="MG Road",
        latitude=12.9756,
        longitude=77.6062,
        first_timestamp=100.0,
        last_timestamp=110.0,
        duration_seconds=10.0,
        local_track_id=1,
        canonical_plate="KA04MH7777",
    )
    node2 = TrajectoryNode(
        camera_id="CAM-002",
        camera_name="Silk Board",
        latitude=12.9177,
        longitude=77.6233,
        first_timestamp=130.0,
        last_timestamp=140.0,
        duration_seconds=10.0,
        local_track_id=2,
        canonical_plate="KA04MH7777",
    )
    # Transit from 110s to 130s = 20s across 6.5 km -> 1170 km/h (speed anomaly)
    seg = TrajectorySegment(
        from_camera_id="CAM-001",
        to_camera_id="CAM-002",
        from_timestamp=110.0,
        to_timestamp=130.0,
        transit_time_seconds=20.0,
        network_distance_km=6.5,
        speed_kmh=1170.0,
        is_velocity_anomaly=True,
    )

    trajectory = VehicleTrajectory(
        global_id="GV-000099",
        canonical_plate="KA04MH7777",
        first_seen_ts=100.0,
        last_seen_ts=140.0,
        nodes=[node1, node2],
        segments=[seg],
    )

    alerts = engine.evaluate_trajectory(trajectory, blacklist_records=blacklist)
    types = {a.alert_type for a in alerts}
    check("Discovered multiple multi-modal alerts on trajectory", len(alerts) >= 2)
    check("Discovered BLACKLIST_EXACT alert", ALERT_BLACKLIST_EXACT in types)
    check("Discovered VELOCITY_ANOMALY alert", ALERT_VELOCITY_ANOMALY in types)


# ============================================================================
# SUITE 7: DATABASE PERSISTENCE, IDEMPOTENCY & ACKNOWLEDGMENT
# ============================================================================
def test_suite_7_database_persistence():
    log.info("\n--- SUITE 7: Database Persistence & Idempotency ---")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        conn = init_db(db_path)

        # 1. Record alert
        aid = record_security_alert(
            conn=conn,
            alert_id="ALT-TEST-001",
            alert_type=ALERT_BLACKLIST_EXACT,
            severity=SEVERITY_CRITICAL,
            title="Blacklist: KA01AB1111",
            description="Wanted vehicle",
            camera_id="CAM-001",
            timestamp=1725450000.0,
            iso_timestamp="2024-09-04T12:00:00Z",
            canonical_plate="KA01AB1111",
            global_id="GV-0001",
            details={"reason": "Stolen"},
        )
        check("record_security_alert returned alert_id", aid == "ALT-TEST-001")

        # 2. Idempotency test (insert same alert_id again)
        record_security_alert(
            conn=conn,
            alert_id="ALT-TEST-001",
            alert_type=ALERT_BLACKLIST_EXACT,
            severity=SEVERITY_CRITICAL,
            title="Blacklist: KA01AB1111 (Updated)",
            description="Wanted vehicle confirmed",
            camera_id="CAM-001",
            timestamp=1725450000.0,
            iso_timestamp="2024-09-04T12:00:00Z",
            canonical_plate="KA01AB1111",
            global_id="GV-0001",
            details={"reason": "Stolen updated"},
        )
        all_alerts = get_security_alerts(conn)
        check("Idempotent update preserves single row", len(all_alerts) == 1)
        check("Title was updated in place", all_alerts[0]["title"] == "Blacklist: KA01AB1111 (Updated)")

        # 3. Filtering
        crit_alerts = get_security_alerts(conn, severity="CRITICAL")
        check("Filter by severity CRITICAL works", len(crit_alerts) == 1)
        med_alerts = get_security_alerts(conn, severity="MEDIUM")
        check("Filter by non-existent severity returns empty list", len(med_alerts) == 0)

        # 4. Summary counts
        summary = get_security_alerts_summary(conn)
        check("Summary reports total_alerts = 1", summary["total_alerts"] == 1)
        check("Summary reports unacknowledged_count = 1", summary["unacknowledged_count"] == 1)
        check("Summary breakdown by severity matches", summary["unack_by_severity"]["CRITICAL"] == 1)

        # 5. Acknowledgment
        ack_success = acknowledge_security_alert(conn, "ALT-TEST-001", acknowledged_by="sergeant_davis")
        check("Acknowledge operation returns True", ack_success is True)
        updated_alert = get_security_alert_by_id(conn, "ALT-TEST-001")
        check("Alert acknowledged flag is 1", updated_alert["acknowledged"] == 1)
        check("Acknowledged by matches operator", updated_alert["acknowledged_by"] == "sergeant_davis")

        # 6. Unacknowledged query after ack
        unack = get_security_alerts(conn, only_unacknowledged=True)
        check("only_unacknowledged query returns 0", len(unack) == 0)

        conn.close()
    finally:
        try:
            os.remove(db_path)
        except Exception:
            pass


# ============================================================================
# SUITE 8: DASHBOARD SERVICE LAYER & TRAJECTORY SCAN
# ============================================================================
def test_suite_8_service_layer():
    log.info("\n--- SUITE 8: DashboardService Layer & Trajectory Scan ---")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        conn = init_db(db_path)
        # Register a blacklisted plate
        add_enriched_blacklist_entry(
            conn, "KA05MK9999", category="STOLEN", severity="CRITICAL", reason="Armed vehicle theft"
        )

        # Insert sightings for this plate
        gid = save_global_identity(
            conn,
            GlobalVehicleIdentity(
                global_id="GV-999999",
                canonical_plate="KA05MK9999",
                plate_confidence=0.95,
                vehicle_type="car",
                first_seen_ts=100.0,
                last_seen_ts=200.0,
                first_camera_id="CAM-001",
                last_camera_id="CAM-002",
                sighting_count=2,
            )
        )
        obs1 = VehicleObservation(
            camera_id="CAM-001",
            track_id=1,
            timestamp=110.0,
            vehicle_type="car",
            canonical_plate="KA05MK9999",
            plate_confidence=0.95,
            crop_quality=0.85,
        )
        res1 = IdentityMatchResult(
            status="MATCH",
            global_id="GV-999999",
            confidence=1.0,
            match_method="new_identity",
        )
        record_vehicle_observation(conn, obs1, res1, first_timestamp=100.0)

        obs2 = VehicleObservation(
            camera_id="CAM-002",
            track_id=2,
            timestamp=130.0,
            vehicle_type="car",
            canonical_plate="KA05MK9999",
            plate_confidence=0.95,
            crop_quality=0.88,
        )
        res2 = IdentityMatchResult(
            status="MATCH",
            global_id="GV-999999",
            confidence=0.99,
            match_method="plate_exact",
        )
        record_vehicle_observation(conn, obs2, res2, first_timestamp=120.0)

        service = DashboardService(
            db_path=db_path,
            cameras_path="configs/cameras.json",
            camera_graph_path="configs/camera_graph.json",
        )

        # Trigger scan
        scan_res = service.scan_and_sync_alerts(conn=conn)
        check("Scan completed successfully", scan_res["status"] == "success")
        check("Scanned 1 trajectory", scan_res["trajectories_scanned"] == 1)
        check("Evaluated >= 1 alert", scan_res["alerts_evaluated"] >= 1)

        # Query alerts through service
        alerts = service.get_alerts(conn=conn)
        check("Service returns populated alerts list", len(alerts) >= 1)
        bl_alerts = [a for a in alerts if a["alert_type"] == ALERT_BLACKLIST_EXACT]
        check("Blacklist exact alert discovered via scan", len(bl_alerts) >= 1)

        conn.close()
    finally:
        try:
            os.remove(db_path)
        except Exception:
            pass


# ============================================================================
# SUITE 9: REST API ENDPOINTS & HTTP CONTRACTS
# ============================================================================
def test_suite_9_rest_api_endpoints():
    log.info("\n--- SUITE 9: REST API Endpoints & HTTP Contracts ---")

    client = app.test_client()

    # 1. GET /api/v1/alerts/summary
    res_sum = client.get("/api/v1/alerts/summary")
    check("GET /api/v1/alerts/summary returns 200", res_sum.status_code == 200)
    data_sum = res_sum.get_json()
    check("Summary contains unacknowledged_count", "unacknowledged_count" in data_sum)

    # 2. GET /api/v1/alerts
    res_alerts = client.get("/api/v1/alerts")
    check("GET /api/v1/alerts returns 200", res_alerts.status_code == 200)
    check("Alerts response is a list", isinstance(res_alerts.get_json(), list))

    # 3. POST /api/v1/blacklist (create)
    payload_bl = {
        "plate": "KA03NB8888",
        "category": "STOLEN",
        "severity": "CRITICAL",
        "reason": "Test armed robbery",
    }
    res_add = client.post("/api/v1/blacklist", json=payload_bl)
    check("POST /api/v1/blacklist returns 200", res_add.status_code == 200)

    # 4. GET /api/v1/blacklist (verify listing)
    res_bl = client.get("/api/v1/blacklist")
    check("GET /api/v1/blacklist returns 200", res_bl.status_code == 200)
    plates = [b["plate"] for b in res_bl.get_json()]
    check("Registered plate present in watchlist", "KA03NB8888" in plates)

    # 5. POST /api/v1/alerts/scan
    res_scan = client.post("/api/v1/alerts/scan")
    check("POST /api/v1/alerts/scan returns 200", res_scan.status_code == 200)

    # 6. DELETE /api/v1/blacklist/<plate>
    res_del = client.delete("/api/v1/blacklist/KA03NB8888")
    check("DELETE /api/v1/blacklist/<plate> returns 200", res_del.status_code == 200)


# ============================================================================
# SUITE 10: ZERO-MATH ARCHITECTURAL GUARDRAIL INSPECTION
# ============================================================================
def test_suite_10_architectural_separation():
    log.info("\n--- SUITE 10: Zero-Math Architectural Boundary ---")

    route_funcs = [
        app.view_functions.get("api_alerts_list"),
        app.view_functions.get("api_alerts_summary"),
        app.view_functions.get("api_ack_alert"),
        app.view_functions.get("api_alerts_scan"),
        app.view_functions.get("api_blacklist"),
        app.view_functions.get("api_add_blacklist"),
        app.view_functions.get("api_remove_blacklist"),
    ]

    prohibited_tokens = [
        "speed_kmh >",
        "140",
        "haversine",
        "dwell_duration",
        "levenshtein",
        "fuzzy",
        "compute_plate_similarity",
    ]

    for fn in route_funcs:
        if not fn:
            continue
        source = inspect.getsource(fn)
        for token in prohibited_tokens:
            check(
                f"Route handler {fn.__name__} does not contain '{token}'",
                token not in source,
                detail=f"Found token '{token}' in handler {fn.__name__}",
            )


def main():
    log.info("================================================================")
    log.info("STARTING PHASE 7E ACCEPTANCE TEST SUITE")
    log.info("================================================================")

    test_suite_1_blacklist_matching()
    test_suite_2_kinematic_anomalies()
    test_suite_3_topological_violations()
    test_suite_4_identity_uncertainty()
    test_suite_5_behavioral_rules()
    test_suite_6_alert_engine_evaluation()
    test_suite_7_database_persistence()
    test_suite_8_service_layer()
    test_suite_9_rest_api_endpoints()
    test_suite_10_architectural_separation()

    log.info("================================================================")
    log.info(f"PHASE 7E TEST SUMMARY: PASS: {PASS} | FAIL: {FAIL}")
    log.info("================================================================")

    if FAIL > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
