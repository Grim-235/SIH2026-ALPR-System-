"""
Multi-Camera Vehicle Tracking Benchmark for Phase 3.

Measures runtime tracking metrics across cameras:
- Tracking Latency (ms)
- Input FPS & Tracking FPS
- Active Tracks
- Total Tracks Created & Finalized
- Average & Median Track Length
- Track Fragmentation (%)

Usage:
    python tests/benchmark_tracking.py --direct --frames 50
    python tests/benchmark_tracking.py --duration 10
"""

import argparse
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from alpr.tracker import VehicleTracker
from workers.camera_worker import CameraWorker, MultiCameraManager


def run_tracking_benchmark(
    config_path: str = "configs/cameras.json",
    tracker_type: str = "bytetrack.yaml",
    max_frames: int = 50,
    use_direct: bool = True,
):
    manager = MultiCameraManager()
    cameras = manager.load_cameras(config_path)
    if not cameras:
        print(f"Error: No cameras loaded from {config_path}")
        return

    print("\n" + "=" * 65)
    print("           INITIALIZING VEHICLE TRACKING BENCHMARK")
    print("=" * 65)
    print(f"Cameras to benchmark: {[c['camera_id'] for c in cameras]}")
    print(f"Tracker Backend:      {tracker_type}")
    print(f"Max Frames / Camera:  {max_frames}")
    print(f"Source Mode:          {'Direct MP4' if use_direct else 'RTSP Stream'}")

    workers: Dict[str, CameraWorker] = {}
    trackers: Dict[str, VehicleTracker] = {}
    threads: Dict[str, threading.Thread] = {}

    for cam in cameras:
        cam_id = cam["camera_id"]
        source = cam["video"] if use_direct else cam["stream_url"]
        fps_target = float(cam.get("fps", 10.0))

        tracker = VehicleTracker(
            model_path="data/models/yolov8n.pt",
            tracker_type=tracker_type,
            camera_id=cam_id,
            conf=0.35,
            iou=0.5,
            device="auto",
        )
        trackers[cam_id] = tracker

        worker = CameraWorker(
            camera_id=cam_id,
            source=source,
            fps_target=fps_target,
            tracker=tracker,
        )
        worker._stats_interval = 999.0  # Keep logs quiet during benchmark
        workers[cam_id] = worker

        t = threading.Thread(target=worker.start, name=f"track-bench-{cam_id}", daemon=True)
        threads[cam_id] = t

    sample_tracker = next(iter(trackers.values()))
    device_name = sample_tracker.device.upper()
    model_name = sample_tracker.model_name

    print(f"Model:                {model_name}")
    print(f"Device:               {device_name}")
    print("\nRunning multi-camera tracking threads...")

    bench_start = time.time()
    for t in threads.values():
        t.start()

    # Poll until all workers have processed max_frames or completed
    try:
        while True:
            all_done = True
            for w in workers.values():
                if w.running and w.frames_processed < max_frames:
                    all_done = False
                    break
            if all_done:
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Benchmark interrupted by user.")
    finally:
        bench_elapsed = time.time() - bench_start
        print("Stopping camera workers and compiling metrics...")
        for w in workers.values():
            w.stop()
        for t in threads.values():
            t.join(timeout=3.0)

    # ── Display Benchmark Table ──
    print("\n" + "=" * 74)
    print("                 VEHICLE TRACKING BENCHMARK (ByteTrack)")
    print("=" * 74)
    print(
        f"{'Camera':<10} {'Latency':<9} {'Proc FPS':<10} {'Tracks':<8} {'Active':<8} "
        f"{'Avg Len':<10} {'Frag (%)':<8}"
    )
    print("-" * 74)

    for cam_id in sorted(trackers.keys()):
        tr = trackers[cam_id]
        worker = workers[cam_id]
        m = tr.get_metrics()

        lat_str = f"{worker.avg_latency_ms:.0f}ms"
        fps_str = f"{worker.inference_fps:.1f}"
        tracks_str = f"{m['tracks_created']}"
        active_str = f"{m['active_tracks']}"
        avg_len_str = f"{m['avg_track_length']:.1f}f"
        frag_str = f"{m['track_fragmentation']:.1f}%"

        print(
            f"{cam_id:<10} {lat_str:<9} {fps_str:<10} {tracks_str:<8} {active_str:<8} "
            f"{avg_len_str:<10} {frag_str:<8}"
        )

    print("=" * 74)
    print(f"Device:   {device_name}")
    print(f"Model:    {model_name}")
    print(f"Backend:  {tracker_type}")
    print(f"Elapsed:  {bench_elapsed:.1f}s")
    print("=" * 74 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3 Vehicle Tracking Benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cameras.json",
        help="Path to cameras.json",
    )
    parser.add_argument(
        "--tracker-type",
        type=str,
        default="bytetrack.yaml",
        choices=["bytetrack.yaml", "botsort.yaml"],
        help="Tracker backend (default: bytetrack.yaml)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=30,
        help="Number of frames to benchmark per camera (default: 30)",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        default=True,
        help="Run directly on MP4 files (default: True)",
    )
    parser.add_argument(
        "--rtsp",
        action="store_true",
        help="Run on live RTSP streams instead of direct MP4 files",
    )

    args = parser.parse_args()
    use_direct = not args.rtsp

    run_tracking_benchmark(
        config_path=args.config,
        tracker_type=args.tracker_type,
        max_frames=args.frames,
        use_direct=use_direct,
    )
