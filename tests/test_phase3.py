"""
Phase 3 — Acceptance tests for Single-Camera Vehicle Tracking (ByteTrack).

Functional criteria:
- ByteTrack loads successfully.
- Vehicle detections enter tracker.
- Track IDs are integers.
- IDs persist across consecutive frames.
- Different vehicles receive different IDs.
- Track state is maintained independently per camera.
- Tracks finalize after disappearance.
- Vehicle type remains associated with track.
- Best crop is preserved.
- Tracking metrics compute cleanly.
"""

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from alpr.tracker import (
    VehicleTracker,
    VehicleTrackState,
    ActiveVehicleTrack,
)
from workers.camera_worker import CameraWorker

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
    print("\n" + "=" * 60)
    print("  Phase 3 — Vehicle Tracking Acceptance Tests (ByteTrack)")
    print("=" * 60 + "\n")

    # ── Test 1: Tracker Initialization ──
    print("[1] VehicleTracker initialization")
    tracker = VehicleTracker(
        model_path="data/models/yolov8n.pt",
        tracker_type="bytetrack.yaml",
        camera_id="CAM-001",
        conf=0.25,
        max_age=5,
    )
    check("VehicleTracker instantiates", tracker is not None)
    check("Tracker backend set to ByteTrack", tracker.tracker_type == "bytetrack.yaml")
    check("Camera ID correctly bound", tracker.camera_id == "CAM-001")
    check("Active tracks initially empty", len(tracker.active_tracks) == 0)
    check("Finalized tracks initially empty", len(tracker.finalized_tracks) == 0)

    # ── Test 2: Frame Ingestion & ID Assignment ──
    print("\n[2] Track generation & ID properties")
    sample_img = cv2.imread("inputs/1.jpg")
    check("Sample image exists", sample_img is not None)

    active_tracks = tracker.update(sample_img, frame_number=1, timestamp=100.0)
    check("Tracker produces active tracks", len(active_tracks) > 0, f"found {len(active_tracks)} tracks")

    all_int_ids = True
    all_valid_vtypes = True
    all_valid_bboxes = True
    ids_seen = set()

    for track in active_tracks:
        if not isinstance(track.track_id, int):
            all_int_ids = False
        ids_seen.add(track.track_id)
        if track.vehicle_type not in ("car", "motorcycle", "bus", "truck"):
            all_valid_vtypes = False
        x1, y1, x2, y2 = track.bbox
        h, w = sample_img.shape[:2]
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            all_valid_bboxes = False

    check("Track IDs are integers", all_int_ids)
    check("Vehicle types are valid (car/bus/truck/motorcycle)", all_valid_vtypes)
    check("Track bounding boxes within frame", all_valid_bboxes)
    check("Different vehicles receive distinct IDs", len(ids_seen) == len(active_tracks))

    # ── Test 3: ID Persistence Across Consecutive Frames ──
    print("\n[3] ID persistence across consecutive frames")
    # Feed the same frame again (frame 2) to verify ByteTrack associates the tracks
    active_frame2 = tracker.update(sample_img, frame_number=2, timestamp=100.1)
    ids_frame2 = {t.track_id for t in active_frame2}

    # At least some tracks should persist across identical or near-identical frames
    persisted_ids = ids_seen.intersection(ids_frame2)
    check(
        "Track IDs persist across consecutive frames",
        len(persisted_ids) > 0,
        f"persisted IDs: {persisted_ids}",
    )

    # Check track state accumulation
    sample_tid = next(iter(persisted_ids))
    state = tracker.active_tracks[sample_tid]
    check("Track history has 2 frames", state.frame_count == 2)
    check("Bbox history length matches frame count", len(state.bbox_history) == 2)
    check("Best vehicle crop captured", state.best_vehicle_crop is not None)
    check("Crop quality computed", state.best_crop_quality > 0.0)

    # ── Test 4: Independent Camera State ──
    print("\n[4] Independent camera tracking state")
    tracker_cam2 = VehicleTracker(
        model_path="data/models/yolov8n.pt",
        tracker_type="bytetrack.yaml",
        camera_id="CAM-002",
        conf=0.25,
    )
    cam2_tracks = tracker_cam2.update(sample_img, frame_number=1)
    check("CAM-002 tracks independently", tracker_cam2.camera_id == "CAM-002")
    check("CAM-001 active tracks count unaffected", len(tracker.active_tracks) > 0)
    check("CAM-002 has its own track storage", len(tracker_cam2.active_tracks) > 0)

    # ── Test 5: Track Finalization on Disappearance ──
    print("\n[5] Track finalization on disappearance")
    # Feed blank frames until max_age (5 frames) is exceeded
    blank_frame = np.zeros_like(sample_img)
    for fnum in range(3, 10):
        tracker.update(blank_frame, frame_number=fnum, timestamp=100.0 + fnum * 0.1)

    check(
        "Active tracks emptied after disappearance (>max_age)",
        len(tracker.active_tracks) == 0,
    )
    check(
        "Stale tracks moved to finalized_tracks",
        len(tracker.finalized_tracks) > 0,
        f"finalized {len(tracker.finalized_tracks)} tracks",
    )

    # ── Test 6: Tracking Metrics Calculation ──
    print("\n[6] Tracking metrics computation")
    metrics = tracker.get_metrics()
    check("tracks_created >= finalized_tracks", metrics["tracks_created"] >= metrics["finalized_tracks"])
    check("avg_track_length > 0", metrics["avg_track_length"] > 0.0)
    check("median_track_length >= 1", metrics["median_track_length"] >= 1.0)
    check("track_fragmentation is percentage [0, 100]", 0.0 <= metrics["track_fragmentation"] <= 100.0)

    # ── Test 7: Motion Trail Annotation ──
    print("\n[7] Visual trail & annotation")
    # Reset tracker and ingest 2 frames
    tracker.reset()
    tracks_drawn = tracker.update(sample_img, frame_number=1)
    annotated = tracker.draw_tracks(sample_img, tracks_drawn, draw_trail=True, copy=True)
    check("draw_tracks outputs valid numpy image", isinstance(annotated, np.ndarray))
    check("draw_tracks preserves image shape", annotated.shape == sample_img.shape)

    # ── Test 8: Integration with CameraWorker ──
    print("\n[8] CameraWorker orchestration with VehicleTracker")
    tracked_callbacks = []

    def on_track_cb(cam_id, frame, tracks, lat, ts):
        tracked_callbacks.append((cam_id, len(tracks), lat, ts))

    worker_tracker = VehicleTracker(
        model_path="data/models/yolov8n.pt",
        camera_id="TEST-TRACK-WORKER",
        conf=0.25,
    )

    worker = CameraWorker(
        camera_id="TEST-TRACK-WORKER",
        source="inputs/1.jpg",
        fps_target=0,
        tracker=worker_tracker,
        on_tracks=on_track_cb,
    )

    worker.start()
    check("Worker runs tracking loop", worker.frames_processed >= 1)
    check("Worker records track callback", len(tracked_callbacks) >= 1)
    check("Worker finalizes tracks on shutdown", worker_tracker.finalized_track_count >= 1)

    # ── Summary ──
    print(f"\n{'=' * 60}")
    total = PASS + FAIL
    print(f"  Results: {PASS}/{total} passed, {FAIL} failed")
    print(f"{'=' * 60}\n")

    return FAIL == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
