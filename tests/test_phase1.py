"""Phase 1 — Acceptance tests for streaming infrastructure."""

import json
import logging
import os
import sys
import time
from pathlib import Path

# Ensure project root is on path when running from tests/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger("test_phase1")

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
    print("  Phase 1 — Acceptance Tests")
    print("=" * 60 + "\n")

    # ── Test 1: CameraSource import and API ──
    print("[1] CameraSource abstraction")
    try:
        from alpr.camera import CameraSource

        check("CameraSource imports", True)
    except ImportError as e:
        check("CameraSource imports", False, str(e))
        sys.exit(1)

    # Test with image file (MP4-like behavior)
    cs = CameraSource("inputs/1.jpg", "TEST-001")
    check("is_stream=False for file", not cs.is_stream)
    check("connected=False before connect", not cs.connected)
    check("status='unknown' initially", cs.status == "unknown")

    ok = cs.connect()
    check("connect() succeeds for file", ok)
    check("connected=True after connect", cs.connected)
    check("status='online' when connected", cs.status == "online")

    success, frame, ts = cs.read_frame()
    check("read_frame() returns frame", success and frame is not None)
    check("capture_timestamp is populated", ts > 0, f"ts={ts:.3f}")
    check("frames_read increments", cs.frames_read == 1)
    check("get_resolution() works", cs.get_resolution()[0] > 0)
    check("get_native_fps() works", cs.get_native_fps() >= 0)

    cs.release()
    check("release() disconnects", not cs.connected)
    check("status='offline' when released", cs.status == "offline")

    # Context manager
    with CameraSource("inputs/1.jpg", "TEST-CM") as cs2:
        ok2, f2, t2 = cs2.read_frame()
        check("context manager works", ok2)
    check("context manager releases", not cs2.connected)

    # RTSP detection
    cs_rtsp = CameraSource("rtsp://localhost:8554/cam01", "TEST-RTSP")
    check("is_stream=True for RTSP", cs_rtsp.is_stream)

    cs_http = CameraSource("http://example.com/stream", "TEST-HTTP")
    check("is_stream=True for HTTP", cs_http.is_stream)

    cs_webcam = CameraSource(0, "TEST-WEBCAM")
    check("is_stream=False for webcam index", not cs_webcam.is_stream)

    # ── Test 2: Camera config ──
    print("\n[2] Camera configuration")
    config_path = Path("configs/cameras.json")
    check("configs/cameras.json exists", config_path.exists())

    with open(config_path) as f:
        cams = json.load(f)
    check("Config has 4 cameras", len(cams) == 4, f"found {len(cams)}")

    required_fields = ["camera_id", "stream_url", "video", "protocol", "fps", "resolution", "status"]
    for cam in cams:
        missing = [f for f in required_fields if f not in cam]
        cam_id = cam.get("camera_id", "?")
        check(f"{cam_id} has all required fields", len(missing) == 0, f"missing: {missing}" if missing else "")

    # ── Test 3: Camera graph ──
    print("\n[3] Camera topology graph")
    graph_path = Path("configs/camera_graph.json")
    check("camera_graph.json exists", graph_path.exists())

    with open(graph_path) as f:
        graph = json.load(f)
    check("Graph has 4 cameras", len(graph) == 4)

    for cam_id, info in graph.items():
        check(f"{cam_id} has neighbors", "neighbors" in info and len(info["neighbors"]) > 0)
        check(f"{cam_id} has distances", "distances_km" in info)

    # ── Test 4: Streaming modules ──
    print("\n[4] Streaming modules")
    try:
        from streaming.publish_cameras import StreamPublisher

        check("StreamPublisher imports", True)
    except ImportError as e:
        check("StreamPublisher imports", False, str(e))

    check("mediamtx.yml exists", Path("streaming/mediamtx.yml").exists())

    # ── Test 5: Camera worker ──
    print("\n[5] Camera worker")
    try:
        from workers.camera_worker import CameraWorker, MultiCameraManager

        check("CameraWorker imports", True)
        check("MultiCameraManager imports", True)
    except ImportError as e:
        check("CameraWorker imports", False, str(e))

    mgr = MultiCameraManager()
    loaded = mgr.load_cameras("configs/cameras.json")
    check("MultiCameraManager loads config", len(loaded) == 4)

    # Test single camera worker with file (quick test)
    worker = CameraWorker(
        camera_id="TEST-WORKER",
        source="inputs/1.jpg",
        fps_target=0,
    )
    check("CameraWorker instantiates", worker.camera_id == "TEST-WORKER")
    check("Worker not running before start", not worker.running)

    # ── Test 6: Legacy files ──
    print("\n[6] Legacy files")
    legacy_files = [
        "legacy/simulate_cameras.py",
        "legacy/worker.py",
        "legacy/process_videos.py",
        "legacy/dashboard_old.py",
    ]
    for f in legacy_files:
        check(f"{f} exists", Path(f).exists())

    # Verify originals are gone from root
    removed_files = [
        "simulate_cameras.py",
        "worker.py",
        "process_videos.py",
        "dashboard_old.py",
    ]
    for f in removed_files:
        check(f"{f} removed from root", not Path(f).exists())

    # ── Test 7: No synthetic data ──
    print("\n[7] No synthetic data generation")
    check("simulate_cameras.py not in root", not Path("simulate_cameras.py").exists())

    # ── Summary ──
    print(f"\n{'=' * 60}")
    total = PASS + FAIL
    print(f"  Results: {PASS}/{total} passed, {FAIL} failed")
    print(f"{'=' * 60}\n")

    return FAIL == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
