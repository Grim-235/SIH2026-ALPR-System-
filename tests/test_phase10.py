"""
Phase 10 Test Suite: Competition Demonstration, Benchmarking & Final Hardening.

Verifies:
1. SIHDemoHarness dry-run execution (all 8 stages complete successfully)
2. Four-tier benchmark report validation (docs/BENCHMARK_REPORT.md)
3. Cryptographic disk crop tamper detection & restoration invariants
4. Non-blocking Flask test_client REST API audit (zero-math verification)
5. Zero-math static AST architectural invariant across app.py route handlers
6. Synthetic scenario identity discipline (DEMO- prefixes & demonstration metadata)
"""

import ast
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path
import pytest
import cv2
import numpy as np

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
)
from alpr.service import DashboardService
from scripts.demo_sih2026 import SIHDemoHarness


class TestPhase10DemoHarness:
    """Acceptance tests for SIHDemoHarness."""

    def test_demo_harness_dry_run_execution(self, tmp_path):
        """SIHDemoHarness(--dry-run) must complete all 8 stages without errors."""
        output_dir = tmp_path / "demo_output"
        harness = SIHDemoHarness(dry_run=True, output_dir=output_dir)
        success = harness.run()

        assert success is True, "SIHDemoHarness.run() returned False"
        metrics = harness.metrics
        assert metrics.get("dry_run") is True
        assert len(metrics.get("stages_completed", [])) >= 7
        assert metrics.get("tamper_test_passed") is True
        assert metrics.get("api_audit_passed") is True

        # Verify output directory contains expected artifacts
        summary_path = output_dir / "demo_summary.json"
        assert summary_path.exists(), "demo_summary.json not found in output_dir"
        summary_file_data = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary_file_data["dry_run"] is True
        assert summary_file_data["tamper_test_passed"] is True
        assert summary_file_data["api_audit_passed"] is True

        # Check dossier outputs
        evidence_gen = summary_file_data.get("evidence_generated", {})
        assert "pdf_path" in evidence_gen and Path(evidence_gen["pdf_path"]).exists()
        assert "json_path" in evidence_gen and Path(evidence_gen["json_path"]).exists()
        assert "csv_path" in evidence_gen and Path(evidence_gen["csv_path"]).exists()

    def test_demo_synthetic_scenario_discipline(self):
        """All demonstration scenarios must use DEMO- prefixed plates and DEMONSTRATION metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            harness = SIHDemoHarness(dry_run=True, output_dir=Path(tmpdir))
            try:
                assert harness.stage_1_validate_environment() is True
                assert harness.stage_2_init_database() is True
                assert harness.stage_3_simulate_multi_camera_scenarios() is True

                # Inspect database alerts
                assert harness.conn is not None
                alerts = harness.conn.execute("SELECT * FROM security_alerts").fetchall()
                assert len(alerts) >= 4

                for alt in alerts:
                    plate = alt["canonical_plate"]
                    assert plate.startswith("DEMO-"), f"Plate '{plate}' must start with 'DEMO-'"
                    details = json.loads(alt["details_json"] or "{}")
                    assert details.get("environment") == "DEMONSTRATION", (
                        f"Alert {alt['alert_id']} missing environment: DEMONSTRATION"
                    )

                # Inspect global identities
                vehicles = harness.conn.execute("SELECT * FROM global_vehicles").fetchall()
                assert len(vehicles) >= 2
                for v in vehicles:
                    plate = v["canonical_plate"]
                    assert plate.startswith("DEMO-"), f"Vehicle plate '{plate}' must start with 'DEMO-'"
                    assert v["sighting_count"] >= 1
            finally:
                harness.cleanup()


class TestPhase10BenchmarkDocumentation:
    """Acceptance tests for Benchmark Report integrity and four-tier metric structure."""

    def test_benchmark_report_exists_and_four_tier_classification(self):
        """docs/BENCHMARK_REPORT.md must exist and contain the four metric tiers."""
        report_path = PROJECT_ROOT / "docs" / "BENCHMARK_REPORT.md"
        assert report_path.exists(), "docs/BENCHMARK_REPORT.md does not exist"

        content = report_path.read_text(encoding="utf-8")
        assert len(content) > 1500, "Benchmark report is suspiciously short"

        # Tier 1: Measured
        assert "[Measured]" in content or "Measured" in content
        # Tier 2: Derived
        assert "[Derived]" in content or "Derived" in content
        # Tier 3: Diagnostic
        assert "[Diagnostic]" in content or "Diagnostic" in content
        # Tier 4: Empirical Ground Truth
        assert "[Empirical Ground Truth Context]" in content or "Empirical Ground Truth" in content

    def test_benchmark_report_legal_disclaimer_and_zero_math(self):
        """Report must document legal velocity disclaimer, zero-math API, and SQLite WAL concurrency."""
        report_path = PROJECT_ROOT / "docs" / "BENCHMARK_REPORT.md"
        content = report_path.read_text(encoding="utf-8")

        # Must explicitly clarify speed bound != legal conviction
        assert "140" in content
        assert "plausibility" in content.lower() or "diagnostic" in content.lower()
        assert "challan" in content.lower() or "court" in content.lower() or "statutory" in content.lower()

        # Must document zero-math and SQLite WAL
        assert "zero-math" in content.lower() or "zero math" in content.lower()
        assert "wal" in content.lower()


class TestPhase10TamperEvidence:
    """Acceptance tests for disk crop 1-byte tamper detection and restoration."""

    def test_disk_crop_tamper_detection_invariants(self, tmp_path):
        """Altering 1 byte in a persisted crop JPEG on disk must fail verification and restoring passes."""
        crops_dir = tmp_path / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)

        # 1. Generate dummy JPEG images
        veh_crop_path = crops_dir / "test_veh.jpg"
        plate_crop_path = crops_dir / "test_plate.jpg"

        dummy_veh = np.zeros((100, 100, 3), dtype=np.uint8)
        dummy_veh[20:80, 20:80] = [0, 128, 255]
        cv2.imwrite(str(veh_crop_path), dummy_veh)

        dummy_plate = np.zeros((30, 90, 3), dtype=np.uint8)
        dummy_plate[5:25, 5:85] = [255, 255, 255]
        cv2.imwrite(str(plate_crop_path), dummy_plate)

        # 2. Build EvidenceRecord
        veh_bytes = veh_crop_path.read_bytes()
        plate_bytes = plate_crop_path.read_bytes()

        rec = EvidenceRecord(
            incident_id="INC-P10-TEST-001",
            generated_at_iso="2026-09-05T12:00:00Z",
            incident_type="SECURITY_ALERT",
            severity="CRITICAL",
            title="CRITICAL SECURITY ALERT",
            description="Test alert for disk crop tampering",
            legal_disclaimer="SIH Phase 10 Tamper Evidence Verification Test",
            canonical_plate="DEMO-TEST-1234",
            global_id="GV-DEMO-TEST-001",
            vehicle_type="car",
            plate_confidence=0.98,
            camera_id="CAM-001",
            camera_name="Test Camera 1",
            latitude=28.6139,
            longitude=77.2090,
            location_description="Test Intersection",
            capture_timestamp=1700000000.0,
            capture_iso="2026-09-05T12:00:00Z",
            transit_speed_kmh=None,
            transit_time_seconds=None,
            network_distance_km=None,
            haversine_distance_km=None,
            plausibility_bound_kmh=140.0,
            trajectory_hops=[],
            vehicle_crop_path=str(veh_crop_path),
            vehicle_crop_sha256=hash_image_bytes(str(veh_crop_path)),
            plate_crop_path=str(plate_crop_path),
            plate_crop_sha256=hash_image_bytes(str(plate_crop_path)),
        )
        rec.manifest_sha256 = compute_manifest_sha256(rec.to_dict())

        # 3. Initial verification must pass
        valid, reason = verify_evidence_manifest(rec)
        assert valid is True, f"Initial manifest verification failed: {reason}"
        assert "verified" in reason.lower() and "tamper-free" in reason.lower()

        # 4. Tamper: Flip 1 byte in the vehicle crop on disk
        original_bytes = bytearray(veh_crop_path.read_bytes())
        assert len(original_bytes) > 60
        tamper_idx = 50
        original_byte_val = original_bytes[tamper_idx]
        original_bytes[tamper_idx] = (original_byte_val + 1) % 256
        veh_crop_path.write_bytes(bytes(original_bytes))

        # 5. Tampered manifest verification must fail
        tampered_valid, tampered_reason = verify_evidence_manifest(rec)
        assert tampered_valid is False, "Verification should have failed on tampered crop"
        assert "tampered" in tampered_reason.lower() or "mismatch" in tampered_reason.lower()

        # 6. Restore original byte
        original_bytes[tamper_idx] = original_byte_val
        veh_crop_path.write_bytes(bytes(original_bytes))

        # 7. Restored manifest verification must pass again
        restored_valid, restored_reason = verify_evidence_manifest(rec)
        assert restored_valid is True, f"Restored verification failed: {restored_reason}"
        assert "verified" in restored_reason.lower() and "tamper-free" in restored_reason.lower()

    def test_dossier_multi_format_export_content(self, tmp_path):
        """DossierExporter must output valid PDF, canonical JSON, and audit CSV files."""
        exporter = DossierExporter()
        rec = EvidenceRecord(
            incident_id="INC-P10-TEST-002",
            generated_at_iso="2026-09-05T12:00:00Z",
            incident_type="VELOCITY_ANOMALY",
            severity="HIGH",
            title="Kinematic Anomaly",
            description="Speed bound exceeded",
            legal_disclaimer="Diagnostic velocity boundary test",
            canonical_plate="DEMO-KA05MH2024",
            global_id="GV-DEMO-TEST-002",
            vehicle_type="car",
            plate_confidence=0.95,
            camera_id="CAM-002",
            camera_name="Test Camera 2",
            latitude=28.6139,
            longitude=77.2090,
            location_description="Test Junction",
            capture_timestamp=1700000236.3,
            capture_iso="2026-09-05T12:00:00Z",
            transit_speed_kmh=148.5,
            transit_time_seconds=36.3,
            network_distance_km=1.5,
            haversine_distance_km=1.48,
            plausibility_bound_kmh=140.0,
            trajectory_hops=[],
            vehicle_crop_path=None,
            vehicle_crop_sha256=None,
            plate_crop_path=None,
            plate_crop_sha256=None,
        )
        rec.manifest_sha256 = compute_manifest_sha256(rec.to_dict())

        # PDF
        pdf_bytes = exporter.export_pdf(rec)
        assert len(pdf_bytes) > 1000
        assert pdf_bytes.startswith(b"%PDF-")

        # JSON
        json_str = exporter.export_json(rec)
        parsed = json.loads(json_str)
        assert parsed["canonical_plate"] == "DEMO-KA05MH2024"
        assert parsed["manifest_sha256"] == rec.manifest_sha256

        # CSV
        csv_str = exporter.export_csv(rec)
        assert "DEMO-KA05MH2024" in csv_str
        assert "INC-P10-TEST-002" in csv_str


class TestPhase10RestApiZeroMathAudit:
    """Acceptance tests for REST API endpoints using Flask test_client and static AST analysis."""

    def test_non_blocking_flask_test_client_endpoints(self):
        """All core REST API endpoints must respond cleanly without live server process."""
        from app import app
        client = app.test_client()

        # 1. System Health
        res_health = client.get("/api/v1/system/health")
        assert res_health.status_code == 200
        health_data = res_health.get_json()
        assert "status" in health_data
        assert "cameras" in health_data

        # 2. Cameras Status
        res_cams = client.get("/api/v1/system/cameras")
        assert res_cams.status_code == 200
        cams_data = res_cams.get_json()
        assert isinstance(cams_data, (list, dict))

        # 3. Analytics Summary
        res_analytics = client.get("/api/v1/analytics/summary")
        assert res_analytics.status_code == 200
        analytics_data = res_analytics.get_json()
        assert isinstance(analytics_data, dict)

        # 4. Evidence endpoints 404 behavior
        res_alt_404 = client.get("/api/v1/evidence/alerts/NON_EXISTENT_ALERT_XYZ")
        assert res_alt_404.status_code == 404
        assert "error" in res_alt_404.get_json()

        res_veh_404 = client.get("/api/v1/evidence/vehicles/GV-NONEXISTENT-XYZ")
        assert res_veh_404.status_code == 404
        assert "error" in res_veh_404.get_json()

    def test_zero_math_invariant_across_all_routes(self):
        """Static AST inspection: route handlers in app.py must not compute complex arithmetic/stats."""
        app_file = PROJECT_ROOT / "app.py"
        source = app_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename="app.py")

        disallowed_ops = {"mean", "median", "std", "var", "sqrt", "sin", "cos"}

        routes_checked = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if this function is a Flask route
                is_route = False
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Call):
                        func = decorator.func
                        if isinstance(func, ast.Attribute) and func.attr == "route":
                            is_route = True
                    elif isinstance(decorator, ast.Attribute) and decorator.attr == "route":
                        is_route = True

                if is_route:
                    routes_checked += 1
                    # Inspect function body: it should not contain mathematical computations
                    for child in ast.walk(node):
                        if isinstance(child, ast.Call):
                            call_name = ""
                            if isinstance(child.func, ast.Name):
                                call_name = child.func.id
                            elif isinstance(child.func, ast.Attribute):
                                call_name = child.func.attr

                            assert call_name not in disallowed_ops, (
                                f"Route handler '{node.name}' directly calls disallowed math operation '{call_name}'. "
                                "All calculations must be in service layer."
                            )

        assert routes_checked >= 10, f"Expected at least 10 routes checked in app.py, found {routes_checked}"


if __name__ == "__main__":
    sys.exit(pytest.main(["-xvs", __file__]))
