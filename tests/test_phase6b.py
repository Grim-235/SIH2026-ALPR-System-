"""
Phase 6B -- Acceptance tests for Database Persistence & Pipeline Wiring.

Tests:
1. SQLite schema extension (global_vehicles & vehicle_observations tables).
2. Binary BLOB embedding serialization / deserialization.
3. Database indexes verification.
4. Strict separation of local vs. global IDs.
5. Idempotent observation recording (UNIQUE constraint on camera_id, local_track_id).
6. Preservation of resolver decision metadata and UNCERTAIN match states.
7. Multi-camera cross-sighting trajectory query test (CAM-001 -> CAM-002).
8. CameraWorker pipeline wiring with GlobalIdentityResolver and database.
"""

import logging
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from alpr.database import (
    init_db,
    serialize_embedding,
    deserialize_embedding,
    save_global_identity,
    record_vehicle_observation,
    get_global_vehicle,
    get_vehicle_trajectory,
    get_all_global_vehicles,
)
from alpr.identity import (
    GlobalVehicleIdentity,
    VehicleObservation,
    IdentityMatchResult,
    GlobalIdentityResolver,
)
from alpr.tracker import VehicleTracker, VehicleTrackState
from workers.camera_worker import CameraWorker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger("test_phase6b")

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


def make_unit_embedding(seed: int = 42) -> np.ndarray:
    """Generate a reproducible, unit-normalized 512-D embedding."""
    rng = np.random.RandomState(seed)
    v = rng.randn(512).astype(np.float32)
    return v / np.linalg.norm(v)


def main():
    print("\n" + "=" * 65)
    print("  Phase 6B -- Database Persistence & Pipeline Wiring Tests")
    print("=" * 65)

    # Use a temporary database for isolation
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "test_phase6b.db"

        # ── Suite 1: Database Schema & Index Verification ──
        print("\n[1] SQLite Schema & Indexes Verification")
        conn = init_db(db_path)
        check("init_db initializes without error", conn is not None)

        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        check("global_vehicles table exists", "global_vehicles" in tables)
        check("vehicle_observations table exists", "vehicle_observations" in tables)

        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = {row[0] for row in cursor.fetchall()}
        check("Index idx_global_vehicles_plate exists", "idx_global_vehicles_plate" in indexes)
        check("Index idx_obs_global_id exists", "idx_obs_global_id" in indexes)
        check("Index idx_obs_cam_ts exists", "idx_obs_cam_ts" in indexes)

        # ── Suite 2: Embedding Binary BLOB Serialization ──
        print("\n[2] Embedding Binary BLOB Serialization & Safeguards")
        emb_original = make_unit_embedding(777)
        blob = serialize_embedding(emb_original)
        check("Embedding serializes to bytes", isinstance(blob, bytes))
        check("Serialized size == 512 * 4 = 2048 bytes", len(blob) == 2048)

        emb_restored = deserialize_embedding(blob)
        check("Deserialized embedding shape is (512,)", emb_restored.shape == (512,))
        check("Deserialized embedding dtype is float32", emb_restored.dtype == np.float32)
        check("Embedding values preserved exactly", np.allclose(emb_original, emb_restored, atol=1e-6))
        check("Embedding norm preserved", abs(float(np.linalg.norm(emb_restored)) - 1.0) < 1e-5)
        check("None embedding serializes to None", serialize_embedding(None) is None)
        check("None blob deserializes to None", deserialize_embedding(None) is None)

        # ── Suite 3: Global Vehicle & Observation CRUD ──
        print("\n[3] Global Identity Persistence & Separation of IDs")
        t0 = 1700000000.0
        identity = GlobalVehicleIdentity(
            global_id="GV-000001",
            canonical_plate="MH12DE1432",
            plate_confidence=0.94,
            vehicle_type="car",
            representative_embedding=emb_original,
            first_seen_ts=t0,
            last_seen_ts=t0,
            first_camera_id="CAM-001",
            last_camera_id="CAM-001",
            sighting_count=1,
            status="active",
        )
        save_global_identity(conn, identity)

        rec = get_global_vehicle(conn, "GV-000001")
        check("get_global_vehicle returns saved record", rec is not None)
        check("Canonical plate matches", rec["canonical_plate"] == "MH12DE1432")
        check("Representative embedding restored as ndarray", isinstance(rec["representative_embedding"], np.ndarray))

        # Record observation 1 at CAM-001 / local_track_id 17
        obs1 = VehicleObservation(
            camera_id="CAM-001",
            track_id=17,
            timestamp=t0 + 12.0,  # last_timestamp
            vehicle_type="car",
            canonical_plate="MH12DE1432",
            plate_confidence=0.94,
            best_reid_embedding=emb_original,
            crop_quality=180.0,
            bbox=(100, 150, 300, 400),
        )
        res1 = IdentityMatchResult(
            status="NEW",
            global_id="GV-000001",
            matched_candidate=identity,
            confidence=1.0,
            match_method="new_identity",
            reason="Initial observation",
        )
        ok1 = record_vehicle_observation(conn, obs1, res1, first_timestamp=t0)
        check("record_vehicle_observation succeeds", ok1 is True)

        # Verify separation of IDs in stored observation
        traj = get_vehicle_trajectory(conn, "GV-000001")
        check("Trajectory has 1 observation", len(traj) == 1)
        o1 = traj[0]
        check("Separate camera_id == 'CAM-001'", o1["camera_id"] == "CAM-001")
        check("Separate local_track_id == 17", o1["local_track_id"] == 17)
        check("Separate global_id == 'GV-000001'", o1["global_id"] == "GV-000001")
        check("Bbox coordinates preserved", (o1["bbox_x1"], o1["bbox_y1"], o1["bbox_x2"], o1["bbox_y2"]) == (100, 150, 300, 400))
        check("Capture timestamps preserved", o1["first_timestamp"] == t0 and o1["last_timestamp"] == t0 + 12.0)

        # ── Suite 4: Idempotent Writes & Constraint Enforcement ──
        print("\n[4] Idempotent Writes & Constraint Enforcement")
        # Record identical observation again (e.g. reprocessing on shutdown)
        ok_dup = record_vehicle_observation(conn, obs1, res1, first_timestamp=t0)
        check("Duplicate observation write succeeds via upsert", ok_dup is True)

        traj_after_dup = get_vehicle_trajectory(conn, "GV-000001")
        check("Observation count remains 1 (no duplicates inserted)", len(traj_after_dup) == 1)

        # ── Suite 5: Preservation of Decision Metadata & UNCERTAIN State ──
        print("\n[5] Decision Metadata & UNCERTAIN State Preservation")
        obs_unc = VehicleObservation(
            camera_id="CAM-003",
            track_id=55,
            timestamp=t0 + 200.0,
            vehicle_type="car",
            canonical_plate=None,
        )
        res_unc = IdentityMatchResult(
            status="UNCERTAIN",
            global_id="GV-000001",
            confidence=0.62,
            match_method="uncertain_borderline",
            plate_similarity=None,
            reid_similarity=0.62,
            transit_speed_kmh=41.5,
            distance_km=10.5,
            reason="Borderline score 0.62 requiring operator review",
        )
        record_vehicle_observation(conn, obs_unc, res_unc, first_timestamp=t0 + 190.0)

        cursor.execute("SELECT match_status, match_method, match_confidence, transit_speed_kmh, match_reason FROM vehicle_observations WHERE camera_id='CAM-003' AND local_track_id=55")
        unc_row = cursor.fetchone()
        check("UNCERTAIN status preserved in database (not converted to MATCH)", unc_row[0] == "UNCERTAIN")
        check("Match method preserved", unc_row[1] == "uncertain_borderline")
        check("Confidence preserved", abs(unc_row[2] - 0.62) < 1e-4)
        check("Transit speed preserved", abs(unc_row[3] - 41.5) < 1e-4)
        check("Match reason preserved", "Borderline score" in unc_row[4])

        # ── Suite 6: Multi-Camera Cross-Sighting Integration Test ──
        print("\n[6] Multi-Camera Cross-Sighting Trajectory Test (CAM-001 -> CAM-002)")
        # Fresh resolver and isolated DB
        cross_db_path = Path(tmpdir) / "cross_cam.db"
        cross_conn = init_db(cross_db_path)
        resolver = GlobalIdentityResolver(camera_graph_path="configs/camera_graph.json")

        t_base = 20000.0
        emb_target = make_unit_embedding(888)

        # Step 1: CAM-001 Track 17 observes vehicle
        obs_cam1 = VehicleObservation(
            camera_id="CAM-001",
            track_id=17,
            timestamp=t_base + 30.0,
            vehicle_type="car",
            canonical_plate="KA05NB5555",
            plate_confidence=0.93,
            best_reid_embedding=emb_target,
            crop_quality=210.0,
        )
        ident1, res_step1 = resolver.resolve_observation(obs_cam1)
        save_global_identity(cross_conn, ident1)
        record_vehicle_observation(cross_conn, obs_cam1, res_step1, first_timestamp=t_base)

        check("CAM-001 observation creates new identity", res_step1.status == "NEW")
        gv_id = ident1.global_id
        check("Assigned global ID is GV-000001", gv_id == "GV-000001")

        # Step 2: CAM-002 Track 4 observes same vehicle 10 mins later (6.5 km at 39.0 km/h)
        t_transit = t_base + 30.0 + 600.0  # 10 mins later (delta_t = 600s = 10min)
        obs_cam2 = VehicleObservation(
            camera_id="CAM-002",
            track_id=4,
            timestamp=t_transit,
            vehicle_type="car",
            canonical_plate="KA05NB5555",
            plate_confidence=0.91,
            best_reid_embedding=emb_target,
            crop_quality=195.0,
        )
        ident2, res_step2 = resolver.resolve_observation(obs_cam2)
        save_global_identity(cross_conn, ident2)
        record_vehicle_observation(cross_conn, obs_cam2, res_step2, first_timestamp=t_transit - 20.0)

        check("CAM-002 observation matches existing global identity", res_step2.status == "MATCH")
        check("Resolved to same global ID", ident2.global_id == gv_id)

        # Step 3: Query Database for Trajectory & State
        gv_rec = get_global_vehicle(cross_conn, gv_id)
        check("Global vehicle record updated sighting count to 2", gv_rec["sighting_count"] == 2)
        check("First camera is CAM-001", gv_rec["first_camera_id"] == "CAM-001")
        check("Last camera is CAM-002", gv_rec["last_camera_id"] == "CAM-002")

        traj_result = get_vehicle_trajectory(cross_conn, gv_id)
        check("Trajectory contains exactly 2 observations", len(traj_result) == 2)
        check("First sighting is CAM-001 (local track 17)", traj_result[0]["camera_id"] == "CAM-001" and traj_result[0]["local_track_id"] == 17)
        check("Second sighting is CAM-002 (local track 4)", traj_result[1]["camera_id"] == "CAM-002" and traj_result[1]["local_track_id"] == 4)
        check("Transit speed calculated and recorded", traj_result[1]["transit_speed_kmh"] is not None and abs(traj_result[1]["transit_speed_kmh"] - 39.0) < 0.2)
        check("Distance recorded as 6.5 km", abs(traj_result[1]["distance_km"] - 6.5) < 0.01)

        # ── Suite 7: CameraWorker Pipeline Integration ──
        print("\n[7] CameraWorker Pipeline Integration with Identity Resolution")
        worker_db_path = Path(tmpdir) / "worker_pipe.db"
        worker_conn = init_db(worker_db_path)
        worker_resolver = GlobalIdentityResolver(camera_graph_path="configs/camera_graph.json")

        tracker = VehicleTracker(camera_id="TEST-CAM-PIPE")
        worker = CameraWorker(
            camera_id="TEST-CAM-PIPE",
            source="inputs/1.jpg",
            tracker=tracker,
            identity_resolver=worker_resolver,
            db_path=worker_db_path,
        )

        resolved_records = []
        def on_resolved(cam_id, identity, result):
            resolved_records.append((cam_id, identity.global_id, result.status))

        worker.on_global_identity_resolved = on_resolved
        worker.start()

        check("Worker runs tracking loop and finalizes tracks", worker.frames_processed >= 1)
        check("Worker resolved identities", worker.identities_resolved >= 1, f"count={worker.identities_resolved}")
        check("Callback fired with global identity", len(resolved_records) >= 1)

        # Verify records written to SQLite database
        all_gvs = get_all_global_vehicles(worker_conn)
        check("Global vehicles persisted in DB from worker", len(all_gvs) >= 1, f"gvs_found={len(all_gvs)}")

        cursor_w = worker_conn.cursor()
        cursor_w.execute("SELECT COUNT(*) FROM vehicle_observations")
        obs_count = cursor_w.fetchone()[0]
        check("Vehicle observations persisted in DB from worker", obs_count >= 1, f"obs_count={obs_count}")

        conn.close()
        cross_conn.close()
        worker_conn.close()

    # ── Summary ──
    print("\n" + "=" * 65)
    print(f"  Results: {PASS}/{PASS + FAIL} passed, {FAIL} failed")
    print("=" * 65 + "\n")

    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
