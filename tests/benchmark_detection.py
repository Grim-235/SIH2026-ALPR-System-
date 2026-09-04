"""
Multi-Camera Vehicle Detection Benchmark for Phase 2.

Measures real runtime performance across all four live RTSP camera streams:
- Input FPS
- Inference FPS
- Latency (ms)
- Average vehicles detected per frame
- Device and model architecture

Usage:
    python tests/benchmark_detection.py --duration 10
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

from alpr.detector import VehicleDetector
from workers.camera_worker import CameraWorker, MultiCameraManager


def run_benchmark(duration: float = 10.0, config_path: str = "configs/cameras.json"):
    manager = MultiCameraManager()
    cameras = manager.load_cameras(config_path)
    if not cameras:
        print(f"Error: No cameras loaded from {config_path}")
        return

    print("\n" + "=" * 60)
    print("        INITIALIZING VEHICLE DETECTION BENCHMARK")
    print("=" * 60)
    print(f"Cameras to benchmark: {[c['camera_id'] for c in cameras]}")
    print(f"Benchmark duration:   {duration:.1f} seconds")

    # Initialize a detector instance per camera to allow concurrent thread execution
    workers: Dict[str, CameraWorker] = {}
    threads: Dict[str, threading.Thread] = {}
    detectors: Dict[str, VehicleDetector] = {}

    for cam in cameras:
        cam_id = cam["camera_id"]
        source = cam["stream_url"]
        fps_target = float(cam.get("fps", 10.0))

        detector = VehicleDetector(
            model_path="data/models/yolov8n.pt",
            conf=0.35,
            iou=0.5,
            imgsz=640,
            device="auto",
        )
        detectors[cam_id] = detector

        worker = CameraWorker(
            camera_id=cam_id,
            source=source,
            fps_target=fps_target,
            detector=detector,
        )
        # Keep internal stats interval quiet during benchmark
        worker._stats_interval = 999.0
        workers[cam_id] = worker

        t = threading.Thread(target=worker.start, name=f"bench-{cam_id}", daemon=True)
        threads[cam_id] = t

    sample_detector = next(iter(detectors.values()))
    device_name = sample_detector.device.upper()
    model_name = sample_detector.model_name

    print(f"Detector Model:       {model_name}")
    print(f"Inference Device:     {device_name}")
    print("\nStarting camera worker threads and warming up pipelines...")

    for t in threads.values():
        t.start()

    # Warmup for 2 seconds to let RTSP handshakes settle
    time.sleep(2.0)
    print(f"Warmed up. Running measurement window ({duration:.1f}s)...")

    # Record baseline frame counts
    baseline_frames = {cid: w.frames_processed for cid, w in workers.items()}
    bench_start = time.time()

    time.sleep(duration)
    bench_elapsed = time.time() - bench_start

    # Collect stats before stopping
    results = {}
    for cam_id, worker in workers.items():
        cam = worker._camera
        in_fps = cam.get_fps() if cam else 0.0
        infer_fps = worker.inference_fps
        latency_ms = worker.avg_latency_ms
        vehicles_per_frame = worker.avg_vehicles_per_frame
        frames_in_window = worker.frames_processed - baseline_frames.get(cam_id, 0)
        window_infer_fps = frames_in_window / bench_elapsed if bench_elapsed > 0 else 0.0

        results[cam_id] = {
            "status": worker.status,
            "input_fps": in_fps,
            "infer_fps": window_infer_fps if window_infer_fps > 0 else infer_fps,
            "latency_ms": latency_ms,
            "vehicles_per_frame": vehicles_per_frame,
            "total_vehicles": worker.vehicles_detected,
            "frames_processed": worker.frames_processed,
        }

    print("Stopping camera workers...")
    for worker in workers.values():
        worker.stop()
    for t in threads.values():
        t.join(timeout=3.0)

    # ── Display Benchmark Table ──
    print("\n" + "=" * 62)
    print("                 VEHICLE DETECTION BENCHMARK")
    print("=" * 62)
    print(f"{'Camera':<12} {'Input FPS':<11} {'Inference FPS':<15} {'Latency':<9} {'Vehicles/Frame'}")
    print("-" * 62)

    for cam_id in sorted(results.keys()):
        r = results[cam_id]
        in_fps_str = f"{r['input_fps']:.1f}"
        infer_fps_str = f"{r['infer_fps']:.1f}"
        lat_str = f"{r['latency_ms']:.0f}ms"
        veh_str = f"{r['vehicles_per_frame']:.1f}"
        print(f"{cam_id:<12} {in_fps_str:<11} {infer_fps_str:<15} {lat_str:<9} {veh_str}")

    print("=" * 62)
    print(f"Device: {device_name}")
    print(f"Model:  {model_name}")
    print(f"Window: {bench_elapsed:.1f}s")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 Vehicle Detection Benchmark")
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Duration of benchmark measurement in seconds (default: 10)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cameras.json",
        help="Path to camera config file",
    )
    args = parser.parse_args()
    run_benchmark(duration=args.duration, config_path=args.config)
