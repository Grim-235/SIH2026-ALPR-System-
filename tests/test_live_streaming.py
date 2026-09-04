"""
Live RTSP Integration Test for Phase 1.

Tests:
1. MultiCameraManager concurrently reading from all 4 live RTSP streams.
2. Verified frame counts, FPS, timestamps, status transitions.
3. Stream interruption and automatic reconnection.
"""

import os
import sys
import time
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from alpr.camera import CameraSource
from workers.camera_worker import CameraWorker, MultiCameraManager

def test_live_multi_camera_workers():
    print("\n" + "="*70)
    print("  TEST 1: 4 Live Concurrent RTSP Camera Workers")
    print("="*70)
    
    manager = MultiCameraManager()
    cameras = manager.load_cameras("configs/cameras.json")
    print(f"Loaded {len(cameras)} cameras from configs/cameras.json")
    
    workers = {}
    threads = {}
    for cam in cameras:
        cam_id = cam["camera_id"]
        stream_url = cam["stream_url"]
        worker = CameraWorker(camera_id=cam_id, source=stream_url, fps_target=10.0)
        workers[cam_id] = worker
        t = threading.Thread(target=worker.start, daemon=True)
        threads[cam_id] = t
        t.start()
        
    print("All 4 worker threads started. Ingesting for 8 seconds...")
    time.sleep(8)
    
    print("\n--- Live Camera Status & Metrics ---")
    all_ok = True
    for cam_id, worker in workers.items():
        cam = worker._camera
        frames = cam.frames_read if cam else 0
        fps = cam.get_fps() if cam else 0.0
        status = worker.status
        print(f"[{cam_id}] status={status:<10} | frames_read={frames:<5} | fps={fps:.1f}")
        if status != "online" or frames < 5:
            all_ok = False
            
    print("\nStopping workers...")
    for worker in workers.values():
        worker.stop()
    for t in threads.values():
        t.join(timeout=3.0)
        
    assert all_ok, "Not all cameras successfully ingested frames!"
    print("[PASS] Test 1: All 4 RTSP cameras successfully ingested frames concurrently!")

def test_reconnection_lifecycle():
    print("\n" + "="*70)
    print("  TEST 2: Stream Interruption and Reconnection")
    print("="*70)
    
    # Test with cam01
    source = "rtsp://localhost:8554/cam01"
    cam = CameraSource(source=source, camera_id="CAM-001", fps_target=15.0, reconnect_base_delay=0.5)
    
    print("Connecting to CAM-001...")
    assert cam.connect(), "Initial connection failed"
    assert cam.status == "online", f"Expected online, got {cam.status}"
    
    # Read some frames
    for _ in range(5):
        ok, frame, ts = cam.read_frame()
        assert ok, "Frame read failed"
    print(f"Successfully read 5 frames. Status: {cam.status}")
    
    # Simulate stream drop by releasing underlying cap
    print("Simulating stream drop...")
    with cam._lock:
        if cam._cap:
            cam._cap.release()
            cam._cap = None
            
    # read_frame should now fail
    ok, frame, ts = cam.read_frame()
    assert not ok, "Expected read to fail after cap drop"
    print(f"Detected dropped frame. Running reconnect()...")
    
    reconnected = cam.reconnect()
    assert reconnected, "Reconnection failed"
    assert cam.status == "online", f"Expected status online after reconnect, got {cam.status}"
    
    # Read frame after reconnect
    ok, frame, ts = cam.read_frame()
    assert ok, "Frame read failed after reconnection"
    print(f"Successfully read frame after reconnect! Status: {cam.status}")
    
    cam.release()
    assert cam.status == "offline", f"Expected offline after release, got {cam.status}"
    print("[PASS] Test 2: Stream interruption and reconnection lifecycle verified!")

if __name__ == "__main__":
    test_live_multi_camera_workers()
    test_reconnection_lifecycle()
    print("\n" + "="*70)
    print("  ALL LIVE INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("="*70 + "\n")
