"""
Multi-Camera ANPR Pipeline Benchmark for Phase 4.

Pipeline:
Vehicle Track -> Vehicle Crop -> YOLO Plate Detector -> Quality Gate -> EasyOCR -> Consensus.

Reports:
- Total Tracks Created
- Plate Detections in Vehicle Crops
- Quality-Passing OCR Reads
- Valid Indian Plates Identified
- Canonical Plate Summaries

Usage:
    python tests/benchmark_anpr.py --direct --frames 25
    python tests/benchmark_anpr.py --rtsp --frames 50
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

from alpr.anpr import VehicleANPR
from alpr.ocr import is_probable_indian_plate
from alpr.tracker import VehicleTracker
from workers.camera_worker import CameraWorker, MultiCameraManager


def run_anpr_benchmark(
    config_path: str = "configs/cameras.json",
    plate_model_path: str = "data/models/license_plate_yolov8_best.pt",
    vehicle_model_path: str = "data/models/yolov8n.pt",
    max_frames: int = 25,
    ocr_every_n: int = 3,
    use_direct: bool = True,
):
    manager = MultiCameraManager()
    cameras = manager.load_cameras(config_path)
    if not cameras:
        print(f"Error: No cameras loaded from {config_path}")
        return

    print("\n" + "=" * 65)
    print("             INITIALIZING FULL ANPR BENCHMARK")
    print("=" * 65)
    print(f"Cameras to benchmark: {[c['camera_id'] for c in cameras]}")
    print(f"Max Frames / Camera:  {max_frames}")
    print(f"OCR Cadence:          Every {ocr_every_n} frames per track")
    print(f"Source Mode:          {'Direct MP4' if use_direct else 'Live RTSP'}")

    # Initialize shared ANPR engine and per-camera tracker
    print("\nInitializing shared VehicleANPR engine (YOLO plate detector + EasyOCR)...")
    anpr = VehicleANPR(
        plate_model_path=plate_model_path,
        device="auto",
        ocr_every_n=ocr_every_n,
        enable_ocr=True,
    )

    workers: Dict[str, CameraWorker] = {}
    trackers: Dict[str, VehicleTracker] = {}
    threads: Dict[str, threading.Thread] = {}

    # Stats accumulators per camera
    plate_detection_counts: Dict[str, int] = {c["camera_id"]: 0 for c in cameras}
    ocr_read_counts: Dict[str, int] = {c["camera_id"]: 0 for c in cameras}
    valid_plate_counts: Dict[str, int] = {c["camera_id"]: 0 for c in cameras}

    def on_plate_read_cb(cam_id: str, track_id: int, read):
        plate_detection_counts[cam_id] = plate_detection_counts.get(cam_id, 0) + 1
        ocr_read_counts[cam_id] = ocr_read_counts.get(cam_id, 0) + 1
        if read.is_valid_indian or is_probable_indian_plate(read.text):
            valid_plate_counts[cam_id] = valid_plate_counts.get(cam_id, 0) + 1

    for cam in cameras:
        cam_id = cam["camera_id"]
        source = cam["video"] if use_direct else cam["stream_url"]
        fps_target = float(cam.get("fps", 10.0))

        tracker = VehicleTracker(
            model_path=vehicle_model_path,
            tracker_type="bytetrack.yaml",
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
            anpr=anpr,
            on_plate_read=on_plate_read_cb,
        )
        worker._stats_interval = 999.0  # Keep logs quiet during benchmark
        workers[cam_id] = worker

        t = threading.Thread(target=worker.start, name=f"anpr-bench-{cam_id}", daemon=True)
        threads[cam_id] = t

    print("Starting multi-camera ANPR processing threads...")
    bench_start = time.time()
    for t in threads.values():
        t.start()

    # Poll until all workers reach max_frames or finish
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
        print("Stopping camera workers and finalizing tracks...")
        for w in workers.values():
            w.stop()
        for t in threads.values():
            t.join(timeout=3.0)

    # ── Display Benchmark Table ──
    print("\n" + "=" * 72)
    print("                         ANPR BENCHMARK")
    print("=" * 72)
    print(f"{'Camera':<10} {'Tracks':<9} {'Plate Dets':<13} {'OCR Reads':<12} {'Valid Plates'}")
    print("-" * 72)

    for cam_id in sorted(trackers.keys()):
        tr = trackers[cam_id]
        total_tracks = tr.total_tracks_created
        dets = plate_detection_counts.get(cam_id, 0)
        reads = ocr_read_counts.get(cam_id, 0)
        valid = valid_plate_counts.get(cam_id, 0)

        print(f"{cam_id:<10} {total_tracks:<9} {dets:<13} {reads:<12} {valid}")

    print("=" * 72)
    print(f"Elapsed:      {bench_elapsed:.1f}s")
    print(f"Plate Model:  {anpr.plate_model_name}")
    print(f"OCR Engine:   EasyOCR (en)")
    print("=" * 72)

    # Print sample canonical plate consensus outcomes
    print("\n--- Canonical Plate Consensus Samples ---")
    found_any = False
    for cam_id in sorted(trackers.keys()):
        tr = trackers[cam_id]
        all_trks = tr.finalized_tracks + list(tr.active_tracks.values())
        canonical_plates = [
            f"Track #{t.track_id} ({t.vehicle_type}): '{t.canonical_plate}' (conf={t.plate_confidence:.2f}, {len(t.plate_reads)} reads)"
            for t in all_trks
            if t.canonical_plate
        ]
        if canonical_plates:
            found_any = True
            print(f"[{cam_id}]:")
            for cp in canonical_plates[:5]:
                print(f"  • {cp}")
    if not found_any:
        print("  (No high-confidence plate consensus finalized in benchmark window)")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4 ANPR Benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cameras.json",
        help="Path to cameras.json",
    )
    parser.add_argument(
        "--plate-model",
        type=str,
        default="data/models/license_plate_yolov8_best.pt",
        help="Path to plate detector model weights",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=25,
        help="Number of frames to benchmark per camera (default: 25)",
    )
    parser.add_argument(
        "--ocr-every-n",
        type=int,
        default=3,
        help="Process OCR every N frames per track (default: 3)",
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
        help="Run on live RTSP streams",
    )

    args = parser.parse_args()
    use_direct = not args.rtsp

    run_anpr_benchmark(
        config_path=args.config,
        plate_model_path=args.plate_model,
        max_frames=args.frames,
        ocr_every_n=args.ocr_every_n,
        use_direct=use_direct,
    )
