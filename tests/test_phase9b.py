"""
Phase 9B Test Suite: Evidence & e-Challan Dossier Generation, SHA-256 Manifest,
PDF/JSON/CSV Export, and Zero-Math REST API Integration.
"""

import hashlib
import io
import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path
import pytest
import numpy as np
import cv2

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from alpr.database import (
    init_db,
    save_global_identity,
    record_vehicle_observation,
    record_security_alert,
)
from alpr.identity import (
    GlobalVehicleIdentity,
    VehicleObservation,
    IdentityMatchResult,
)
from alpr.evidence import (
    EvidenceRecord,
    EvidenceCollector,
    DossierExporter,
    compute_manifest_sha256,
    hash_image_bytes,
    verify_evidence_manifest,
    get_legal_disclaimer,
    DISCLAIMER_VELOCITY_ANOMALY,
    DISCLAIMER_BLACKLIST_FUZZY,
    DISCLAIMER_BLACKLIST_EXACT,
    DISCLAIMER_TEMPORAL_INVERSION,
    DISCLAIMER_TOPOLOGY_VIOLATION,
    DISCLAIMER_IDENTITY_UNCERTAIN,
    DISCLAIMER_GENERAL,
)
from alpr.service import DashboardService


@pytest.fixture
def test_env():
    """Set up temporary database, config files, crop images, and service instance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        db_path = tmp_path / "test_alpr.db"
        crops_dir = tmp_path / "data" / "evidence" / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy cameras config
        cameras_json = tmp_path / "cameras.json"
        cameras_data = {
            "cameras": [
                {
                    "camera_id": "CAM-001",
                    "name": "North Toll Plaza",
                    "latitude": 28.6139,
                    "longitude": 77.2090,
                    "description": "Delhi Entry",
                },
                {
                    "camera_id": "CAM-002",
                    "name": "Ring Road Junction",
                    "latitude": 28.6250,
                    "longitude": 77.2180,
                    "description": "Ring Road Overpass",
                },
            ]
        }
        with open(cameras_json, "w") as f:
            json.dump(cameras_data, f)

        # Create dummy graph config
        graph_json = tmp_path / "camera_graph.json"
        graph_data = {
            "corridors": [
                {
                    "corridor_id": "CORR-01",
                    "name": "North Expressway",
                    "from_camera": "CAM-001",
                    "to_camera": "CAM-002",
                    "distance_km": 1.5,
                    "free_flow_speed_kmh": 60.0,
                }
            ]
        }
        with open(graph_json, "w") as f:
            json.dump(graph_data, f)

        # Initialize SQLite database
        conn = init_db(str(db_path))

        # Create sample dummy crop images
        v_crop_img = np.zeros((120, 200, 3), dtype=np.uint8)
        v_crop_img[:] = (30, 40, 150)
        p_crop_img = np.zeros((40, 100, 3), dtype=np.uint8)
        p_crop_img[:] = (200, 200, 200)

        v_crop_file = crops_dir / "CAM-001_101_veh.jpg"
        p_crop_file = crops_dir / "CAM-001_101_plate.jpg"
        cv2.imwrite(str(v_crop_file), v_crop_img)
        cv2.imwrite(str(p_crop_file), p_crop_img)

        # Relative paths as stored in database
        rel_v_crop = "data/evidence/crops/CAM-001_101_veh.jpg"
        rel_p_crop = "data/evidence/crops/CAM-001_101_plate.jpg"

        # Insert Global Identity
        ident = GlobalVehicleIdentity(
            global_id="GV-999001",
            canonical_plate="DL8CAZ9592",
            plate_confidence=0.96,
            vehicle_type="car",
            first_seen_ts=1700000000.0,
            last_seen_ts=1700000045.0,
            first_camera_id="CAM-001",
            last_camera_id="CAM-002",
            sighting_count=2,
            camera_trajectory=["CAM-001", "CAM-002"],
            track_refs=[("CAM-001", 101), ("CAM-002", 202)],
        )
        save_global_identity(conn, ident)

        # Insert Observations
        obs1 = VehicleObservation(
            camera_id="CAM-001",
            track_id=101,
            timestamp=1700000000.0,
            vehicle_type="car",
            canonical_plate="DL8CAZ9592",
            plate_confidence=0.96,
            crop_quality=0.88,
            bbox=(10, 20, 210, 140),
            vehicle_crop_path=rel_v_crop,
            plate_crop_path=rel_p_crop,
        )
        res1 = IdentityMatchResult(
            global_id="GV-999001",
            status="CONFIRMED_MATCH",
            confidence=0.96,
            match_method="EXACT_PLATE",
            plate_similarity=1.0,
        )
        record_vehicle_observation(conn, obs1, res1, first_timestamp=1700000000.0)

        obs2 = VehicleObservation(
            camera_id="CAM-002",
            track_id=202,
            timestamp=1700000045.0,
            vehicle_type="car",
            canonical_plate="DL8CAZ9592",
            plate_confidence=0.95,
            crop_quality=0.85,
            bbox=(15, 25, 215, 145),
            vehicle_crop_path=rel_v_crop,
            plate_crop_path=rel_p_crop,
        )
        res2 = IdentityMatchResult(
            global_id="GV-999001",
            status="CONFIRMED_MATCH",
            confidence=0.95,
            match_method="EXACT_PLATE",
            plate_similarity=1.0,
            transit_speed_kmh=120.0,
            distance_km=1.5,
        )
        record_vehicle_observation(conn, obs2, res2, first_timestamp=1700000040.0)

        # Insert Security Alert
        record_security_alert(
            conn=conn,
            alert_id="ALT-2026-TEST-001",
            alert_type="VELOCITY_ANOMALY",
            severity="HIGH",
            title="Diagnostic: Physical Velocity Bound Exceeded (148.5 km/h)",
            description="Observed speed 148.5 km/h exceeds plausibility bound (140.0 km/h). Diagnostic flag.",
            camera_id="CAM-002",
            timestamp=1700000045.0,
            iso_timestamp="2023-11-14T22:14:05+00:00",
            global_id="GV-999001",
            canonical_plate="DL8CAZ9592",
            details={
                "transit_speed_kmh": 148.5,
                "velocity_bound_kmh": 140.0,
                "distance_km": 1.5,
                "transit_time_seconds": 36.3,
            },
        )
        conn.close()

        service = DashboardService(
            db_path=db_path,
            cameras_path=cameras_json,
            camera_graph_path=graph_json,
            velocity_bound_kmh=140.0,
        )

        yield {
            "tmp_path": tmp_path,
            "db_path": db_path,
            "cameras_json": cameras_json,
            "graph_json": graph_json,
            "crops_dir": crops_dir,
            "service": service,
        }


# ============================================================================
# TEST 1: Evidence collection for Alert
# ============================================================================
def test_evidence_collection_for_alert(test_env):
    collector = EvidenceCollector(
        base_dir=test_env["tmp_path"],
        cameras_path=test_env["cameras_json"],
        camera_graph_path=test_env["graph_json"],
    )
    conn = sqlite3.connect(str(test_env["db_path"]))
    record = collector.collect_for_alert(conn, "ALT-2026-TEST-001")
    conn.close()

    assert record is not None
    assert record.incident_id == "INC-ALT-2026-TEST-001"
    assert record.incident_type == "VELOCITY_ANOMALY"
    assert record.severity == "HIGH"
    assert record.canonical_plate == "DL8CAZ9592"
    assert record.global_id == "GV-999001"
    assert record.camera_id == "CAM-002"
    assert record.camera_name == "Ring Road Junction"
    assert record.transit_speed_kmh == 148.5
    assert record.plausibility_bound_kmh == 140.0
    assert len(record.trajectory_hops) == 2
    assert record.vehicle_crop_sha256 is not None
    assert record.plate_crop_sha256 is not None
    assert len(record.manifest_sha256) == 64  # Valid SHA-256 hex string


# ============================================================================
# TEST 2: Evidence collection for Vehicle
# ============================================================================
def test_evidence_collection_for_vehicle(test_env):
    collector = EvidenceCollector(
        base_dir=test_env["tmp_path"],
        cameras_path=test_env["cameras_json"],
        camera_graph_path=test_env["graph_json"],
    )
    conn = sqlite3.connect(str(test_env["db_path"]))
    record = collector.collect_for_vehicle(conn, "GV-999001")
    conn.close()

    assert record is not None
    assert record.incident_id == "VEH-GV-999001"
    assert record.incident_type == "VEHICLE_TRAJECTORY_DOSSIER"
    assert record.canonical_plate == "DL8CAZ9592"
    assert record.global_id == "GV-999001"
    assert len(record.trajectory_hops) == 2
    assert record.manifest_sha256 != ""


# ============================================================================
# TEST 3: Cryptographic manifest integrity & tamper verification
# ============================================================================
def test_cryptographic_manifest_and_tamper_detection(test_env):
    collector = EvidenceCollector(
        base_dir=test_env["tmp_path"],
        cameras_path=test_env["cameras_json"],
        camera_graph_path=test_env["graph_json"],
    )
    conn = sqlite3.connect(str(test_env["db_path"]))
    record = collector.collect_for_alert(conn, "ALT-2026-TEST-001")
    conn.close()

    # 1. Verification of untouched record passes
    is_valid, msg = verify_evidence_manifest(record, base_dir=test_env["tmp_path"])
    assert is_valid is True, f"Integrity check failed on genuine record: {msg}"

    # 2. Tampering with incident metadata must fail verification
    tampered_record = EvidenceRecord(**record.to_dict())
    tampered_record.transit_speed_kmh = 95.0  # altered speed
    is_valid_t, msg_t = verify_evidence_manifest(tampered_record, base_dir=test_env["tmp_path"])
    assert is_valid_t is False
    assert "Manifest signature mismatch" in msg_t

    # 3. Tampering with image bytes on disk must fail verification
    crop_file = test_env["tmp_path"] / record.vehicle_crop_path
    original_bytes = crop_file.read_bytes()
    try:
        # Overwrite with modified image bytes
        crop_file.write_bytes(original_bytes + b"MODIFIED_EXTRA_BYTES")
        is_valid_img, msg_img = verify_evidence_manifest(record, base_dir=test_env["tmp_path"])
        assert is_valid_img is False
        assert "Vehicle crop image tampered" in msg_img
    finally:
        # Restore original image bytes
        crop_file.write_bytes(original_bytes)


# ============================================================================
# TEST 4: ReportLab PDF generation validity and byte emission
# ============================================================================
def test_reportlab_pdf_export_generation(test_env):
    collector = EvidenceCollector(
        base_dir=test_env["tmp_path"],
        cameras_path=test_env["cameras_json"],
        camera_graph_path=test_env["graph_json"],
    )
    conn = sqlite3.connect(str(test_env["db_path"]))
    record = collector.collect_for_alert(conn, "ALT-2026-TEST-001")
    conn.close()

    pdf_bytes = DossierExporter.export_pdf(record, base_dir=test_env["tmp_path"])
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000  # Non-trivial PDF size
    assert pdf_bytes.startswith(b"%PDF-")  # Standard PDF signature


# ============================================================================
# TEST 5: JSON and CSV serialization formatting
# ============================================================================
def test_json_and_csv_dossier_exports(test_env):
    collector = EvidenceCollector(
        base_dir=test_env["tmp_path"],
        cameras_path=test_env["cameras_json"],
        camera_graph_path=test_env["graph_json"],
    )
    conn = sqlite3.connect(str(test_env["db_path"]))
    record = collector.collect_for_alert(conn, "ALT-2026-TEST-001")
    conn.close()

    # JSON export
    json_str = DossierExporter.export_json(record)
    parsed = json.loads(json_str)
    assert parsed["incident_id"] == "INC-ALT-2026-TEST-001"
    assert parsed["manifest_sha256"] == record.manifest_sha256
    assert parsed["canonical_plate"] == "DL8CAZ9592"

    # CSV export
    csv_str = DossierExporter.export_csv(record)
    assert "incident_id,generated_at_iso" in csv_str
    assert "INC-ALT-2026-TEST-001" in csv_str
    assert "DL8CAZ9592" in csv_str


# ============================================================================
# TEST 6: Legal disclaimer boundary rule enforcement across alert types
# ============================================================================
def test_legal_disclaimer_rule_enforcement():
    # Diagnostic Kinematic Plausibility Bound
    d_kine = get_legal_disclaimer("VELOCITY_ANOMALY")
    assert "KINEMATIC PLAUSIBILITY ANOMALY" in d_kine
    assert "140.0 km/h" in d_kine
    assert "NOT constitute an authoritative or legally binding traffic speeding conviction" in d_kine

    # Temporal Inversion
    d_temp = get_legal_disclaimer("TEMPORAL_INVERSION")
    assert "TEMPORAL INVERSION SENSOR ANOMALY" in d_temp
    assert "clocks" in d_temp

    # Watchlist Fuzzy
    d_fuzz = get_legal_disclaimer("BLACKLIST_FUZZY")
    assert "MANDATORY HUMAN OPERATOR VERIFICATION" in d_fuzz

    # Watchlist Exact
    d_exact = get_legal_disclaimer("BLACKLIST_EXACT")
    assert "AUTOMATED WATCHLIST MATCH" in d_exact
    assert "dispatch verification" in d_exact

    # General / Fallback
    d_gen = get_legal_disclaimer("UNKNOWN_ANOMALY")
    assert "AUTOMATED MACHINE-GENERATED SURVEILLANCE RECORD" in d_gen


# ============================================================================
# TEST 7: Service methods & zero-math REST API integration
# ============================================================================
def test_service_evidence_and_flask_routes(test_env):
    service = test_env["service"]

    # 1. Service direct query
    rec = service.get_alert_evidence("ALT-2026-TEST-001")
    assert rec is not None
    assert rec.canonical_plate == "DL8CAZ9592"

    # 2. Service export helpers
    pdf_data, pdf_mime, pdf_fname = service.export_dossier(rec, "pdf")
    assert pdf_mime == "application/pdf"
    assert pdf_fname.endswith(".pdf")
    assert pdf_data.startswith(b"%PDF-")

    json_data, json_mime, json_fname = service.export_dossier(rec, "json")
    assert json_mime == "application/json"
    assert json_fname.endswith(".json")
    assert "DL8CAZ9592" in json_data

    csv_data, csv_mime, csv_fname = service.export_dossier(rec, "csv")
    assert csv_mime == "text/csv"
    assert csv_fname.endswith(".csv")
    assert "DL8CAZ9592" in csv_data

    # 3. Flask Test Client Integration
    from app import app
    client = app.test_client()

    # Verify 404 for nonexistent alert
    res_404 = client.get("/api/v1/evidence/alerts/NON_EXISTENT_ALERT")
    assert res_404.status_code == 404
    data_404 = res_404.get_json()
    assert "error" in data_404

    # Verify 404 for nonexistent vehicle
    res_v404 = client.get("/api/v1/evidence/vehicles/GV-NONEXISTENT")
    assert res_v404.status_code == 404
    data_v404 = res_v404.get_json()
    assert "error" in data_v404


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-xvs", __file__]))
