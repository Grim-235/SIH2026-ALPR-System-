"""
Camera Worker — per-camera stream processing and pipeline orchestrator.

Connects to a video source via CameraSource, reads frames in a loop,
runs single-camera vehicle tracking (Phase 3) or detection (Phase 2),
and logs performance and tracking metrics.

Usage:
    # Single camera with ByteTrack vehicle tracking
    python -m workers.camera_worker --camera-id CAM-001 --source rtsp://localhost:8554/cam01 --track

    # All cameras with vehicle tracking
    python -m workers.camera_worker --config configs/cameras.json --track

    # Detection-only mode (Phase 2)
    python -m workers.camera_worker --config configs/cameras.json --detect

    # Ingestion-only mode (Phase 1)
    python -m workers.camera_worker --config configs/cameras.json
"""

import argparse
import json
import logging
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from alpr.camera import CameraSource
from alpr.detector import VehicleDetector, VehicleDetection
from alpr.tracker import VehicleTracker, ActiveVehicleTrack, VehicleTrackState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("workers.camera")


class CameraWorker:
    """
    Per-camera frame reader and pipeline orchestrator.

    Connects to a source, reads frames continuously, handles
    reconnection on failure, runs tracking/detection if configured,
    and reports performance and tracking metrics.
    """

    def __init__(
        self,
        camera_id: str,
        source: str,
        fps_target: float = 0.0,
        reconnect_max_retries: int = 10,
        detector: Optional[VehicleDetector] = None,
        tracker: Optional[VehicleTracker] = None,
        on_detections: Optional[Callable[[str, np.ndarray, List[VehicleDetection], float, float], None]] = None,
        on_tracks: Optional[Callable[[str, np.ndarray, List[ActiveVehicleTrack], float, float], None]] = None,
    ):
        """
        Initialize a camera worker.

        Args:
            camera_id: Logical camera identifier (e.g., 'CAM-001').
            source: Video source — RTSP URL, file path, or webcam index.
            fps_target: Target FPS for throttling (0 = no throttle).
            reconnect_max_retries: Max reconnection attempts.
            detector: Optional VehicleDetector instance (for detection-only mode).
            tracker: Optional VehicleTracker instance (for tracking mode).
            on_detections: Optional callback(camera_id, frame, detections, latency_ms, capture_ts).
            on_tracks: Optional callback(camera_id, frame, active_tracks, latency_ms, capture_ts).
        """
        self.camera_id = camera_id
        self.source = source
        self.fps_target = fps_target
        self.reconnect_max_retries = reconnect_max_retries
        self.detector = detector
        self.tracker = tracker
        self.on_detections = on_detections
        self.on_tracks = on_tracks

        self._running = False
        self._camera: Optional[CameraSource] = None
        self._stats_interval = 10.0  # Log stats every N seconds
        self._last_stats_time = 0.0

        # Performance & detection metrics
        self.frames_processed = 0
        self.vehicles_detected = 0
        self._latencies: deque = deque(maxlen=30)
        self._proc_times: deque = deque(maxlen=30)
        self._vehicle_counts: deque = deque(maxlen=30)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    @property
    def running(self) -> bool:
        """Whether the worker is actively running."""
        return self._running

    @property
    def status(self) -> str:
        """Dynamic runtime status of the camera: unknown, connecting, online, reconnecting, offline."""
        if self._camera is not None:
            return self._camera.status
        return "unknown"

    @property
    def input_fps(self) -> float:
        """Measured rate of frames read from the video source."""
        if self._camera:
            return self._camera.get_fps()
        return 0.0

    @property
    def inference_fps(self) -> float:
        """Rate of inference frame processing (rolling average)."""
        if len(self._proc_times) < 2:
            return 0.0
        times = list(self._proc_times)
        time_diff = times[-1] - times[0]
        return (len(times) - 1) / time_diff if time_diff > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        """Average inference/tracking latency in milliseconds."""
        if not self._latencies:
            return 0.0
        return sum(self._latencies) / len(self._latencies)

    @property
    def avg_vehicles_per_frame(self) -> float:
        """Average number of vehicles detected per frame."""
        if not self._vehicle_counts:
            return 0.0
        return sum(self._vehicle_counts) / len(self._vehicle_counts)

    def start(self) -> None:
        """
        Main processing loop.

        Connects to the source, reads frames continuously, runs tracking
        or detection if configured, and handles reconnection on failure.
        """
        self._running = True
        if self.tracker:
            mode_desc = f"with vehicle tracking ({self.tracker.tracker_type})"
        elif self.detector:
            mode_desc = "with vehicle detection"
        else:
            mode_desc = "frame ingestion only"

        logger.info(f"[{self.camera_id}] Worker starting ({mode_desc}) — source: {self.source}")

        self._camera = CameraSource(
            source=self.source,
            camera_id=self.camera_id,
            fps_target=self.fps_target,
            reconnect_max_retries=self.reconnect_max_retries,
        )

        if not self._camera.connect():
            logger.error(f"[{self.camera_id}] Initial connection failed.")
            if self._camera.is_stream:
                logger.info(f"[{self.camera_id}] Source is a stream, attempting reconnection...")
                if not self._camera.reconnect():
                    logger.error(f"[{self.camera_id}] Reconnection failed. Worker exiting.")
                    self._running = False
                    return
            else:
                logger.error(f"[{self.camera_id}] Source is not a stream. Worker exiting.")
                self._running = False
                return

        self._last_stats_time = time.time()
        consecutive_failures = 0
        max_consecutive_failures = 30  # After this many failures, attempt reconnect

        try:
            while self._running:
                cam = self._camera
                if cam is None or not self._running:
                    break

                success, frame, capture_ts = cam.read_frame()

                if success:
                    consecutive_failures = 0

                    # ── Stage: Tracking (Phase 3) ──
                    if self.tracker is not None and frame is not None:
                        t0 = time.perf_counter()
                        active_tracks = self.tracker.update(
                            frame,
                            frame_number=cam.frames_read,
                            timestamp=capture_ts,
                        )
                        latency_ms = (time.perf_counter() - t0) * 1000.0

                        self.frames_processed += 1
                        self.vehicles_detected += len(active_tracks)
                        self._latencies.append(latency_ms)
                        self._proc_times.append(time.time())
                        self._vehicle_counts.append(len(active_tracks))

                        if self.on_tracks:
                            try:
                                self.on_tracks(
                                    self.camera_id,
                                    frame,
                                    active_tracks,
                                    latency_ms,
                                    capture_ts,
                                )
                            except Exception as cb_err:
                                logger.error(f"[{self.camera_id}] Track callback error: {cb_err}")

                    # ── Stage: Detection-only (Phase 2 fallback) ──
                    elif self.detector is not None and frame is not None:
                        detections, latency_ms = self.detector.detect(frame)
                        self.frames_processed += 1
                        self.vehicles_detected += len(detections)
                        self._latencies.append(latency_ms)
                        self._proc_times.append(time.time())
                        self._vehicle_counts.append(len(detections))

                        if self.on_detections:
                            try:
                                self.on_detections(
                                    self.camera_id,
                                    frame,
                                    detections,
                                    latency_ms,
                                    capture_ts,
                                )
                            except Exception as cb_err:
                                logger.error(f"[{self.camera_id}] Callback error: {cb_err}")

                    self._log_stats_if_due()
                else:
                    if not self._running or self._camera is None:
                        break
                    consecutive_failures += 1

                    if not cam.is_stream:
                        # File ended (MP4 finished)
                        logger.info(
                            f"[{self.camera_id}] Source exhausted (file ended). "
                            f"Total frames read: {cam.frames_read}"
                        )
                        break

                    if consecutive_failures >= max_consecutive_failures:
                        logger.warning(
                            f"[{self.camera_id}] {consecutive_failures} consecutive failures. "
                            f"Attempting reconnection..."
                        )
                        if cam.reconnect():
                            consecutive_failures = 0
                        else:
                            logger.error(f"[{self.camera_id}] Reconnection failed. Worker exiting.")
                            break

        except Exception as e:
            logger.error(f"[{self.camera_id}] Unexpected error: {e}", exc_info=True)
        finally:
            self.stop()

    def stop(self) -> None:
        """Gracefully stop the worker and release resources."""
        if not self._running and self._camera is None:
            return

        self._running = False
        cam = self._camera
        read_cnt = cam.frames_read if cam else 0
        drop_cnt = cam.frames_dropped if cam else 0
        in_fps = cam.get_fps() if cam else 0.0

        if self.tracker:
            finalized = self.tracker.finalize_all()
            metrics = self.tracker.get_metrics()
            logger.info(
                f"[{self.camera_id}] Worker stopped. "
                f"read={read_cnt}, proc={self.frames_processed}, "
                f"tracks_created={metrics['tracks_created']}, "
                f"tracks_finalized={len(finalized)}, "
                f"avg_track_len={metrics['avg_track_length']:.1f}f, "
                f"in_fps={in_fps:.1f}, infer_fps={self.inference_fps:.1f}, "
                f"avg_latency={self.avg_latency_ms:.1f}ms"
            )
        elif self.detector:
            logger.info(
                f"[{self.camera_id}] Worker stopped. "
                f"read={read_cnt}, proc={self.frames_processed}, "
                f"vehicles={self.vehicles_detected}, "
                f"in_fps={in_fps:.1f}, infer_fps={self.inference_fps:.1f}, "
                f"avg_latency={self.avg_latency_ms:.1f}ms"
            )
        else:
            logger.info(
                f"[{self.camera_id}] Worker stopped. "
                f"read={read_cnt}, dropped={drop_cnt}, in_fps={in_fps:.1f}"
            )

        if self._camera:
            self._camera.release()
            self._camera = None

    def _log_stats_if_due(self) -> None:
        """Log performance and tracking statistics periodically."""
        now = time.time()
        if now - self._last_stats_time >= self._stats_interval:
            self._last_stats_time = now
            cam = self._camera
            if not cam:
                return

            if self.tracker is not None:
                metrics = self.tracker.get_metrics()
                logger.info(
                    f"[{self.camera_id}] "
                    f"status={self.status:<7} | "
                    f"input_fps={self.input_fps:4.1f} | "
                    f"infer_fps={self.inference_fps:4.1f} | "
                    f"latency={self.avg_latency_ms:5.1f}ms | "
                    f"active_tracks={metrics['active_tracks']:<2} | "
                    f"total_tracks={metrics['tracks_created']:<3} | "
                    f"avg_len={metrics['avg_track_length']:4.1f}f"
                )
            elif self.detector is not None:
                logger.info(
                    f"[{self.camera_id}] "
                    f"status={self.status:<7} | "
                    f"input_fps={self.input_fps:4.1f} | "
                    f"infer_fps={self.inference_fps:4.1f} | "
                    f"latency={self.avg_latency_ms:5.1f}ms | "
                    f"vehicles/frame={self.avg_vehicles_per_frame:3.1f} | "
                    f"total_vehicles={self.vehicles_detected}"
                )
            else:
                logger.info(
                    f"[{self.camera_id}] "
                    f"status={self.status:<7} | "
                    f"frames={cam.frames_read} | "
                    f"dropped={cam.frames_dropped} | "
                    f"fps={cam.get_fps():.1f} | "
                    f"native_fps={cam.get_native_fps():.1f} | "
                    f"resolution={cam.get_resolution()} | "
                    f"uptime={cam.connection_uptime:.0f}s"
                )


class MultiCameraManager:
    """
    Manages multiple CameraWorker instances, one per camera.

    Loads camera definitions from a JSON config and spawns a thread
    per camera. Handles graceful shutdown of all workers.
    """

    def __init__(self):
        self.workers: Dict[str, CameraWorker] = {}
        self._threads: Dict[str, threading.Thread] = {}
        self._shutdown = False

    def load_cameras(self, config_path: str) -> List[dict]:
        """Load camera definitions from a JSON file."""
        path = Path(config_path)
        if not path.exists():
            logger.error(f"Config file not found: {config_path}")
            return []

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "cameras" in data:
            cameras = data["cameras"]
        elif isinstance(data, list):
            cameras = data
        else:
            logger.error("Invalid config format. Expected list or dict with 'cameras' key.")
            return []

        return cameras

    def start_all(
        self,
        config_path: str,
        use_stream: bool = True,
        detector: Optional[VehicleDetector] = None,
        tracker_type: Optional[str] = None,
        model_path: str = "data/models/yolov8n.pt",
        conf: float = 0.35,
        iou: float = 0.5,
        device: str = "auto",
        on_detections: Optional[Callable] = None,
        on_tracks: Optional[Callable] = None,
    ) -> None:
        """
        Start workers for all cameras in the config.

        Args:
            config_path: Path to cameras.json.
            use_stream: If True, connect via stream_url (RTSP).
                        If False, connect directly to the video file.
            detector: Optional VehicleDetector instance (detection-only mode).
            tracker_type: If provided, initializes a VehicleTracker per camera.
            model_path: Path to YOLO weights for trackers.
            conf: Confidence threshold.
            iou: NMS IoU threshold.
            device: Compute device.
            on_detections: Optional detection callback.
            on_tracks: Optional tracking callback.
        """
        cameras = self.load_cameras(config_path)
        if not cameras:
            logger.error("No cameras found in config.")
            return

        for cam in cameras:
            cam_id = cam.get("camera_id")
            if not cam_id:
                continue

            if use_stream:
                source = cam.get("stream_url")
            else:
                source = cam.get("video")

            if not source:
                logger.warning(f"[{cam_id}] No source configured, skipping.")
                continue

            fps_target = cam.get("fps", 0)

            # Initialize tracker for this camera if requested
            worker_tracker = None
            if tracker_type:
                worker_tracker = VehicleTracker(
                    model_path=model_path,
                    tracker_type=tracker_type,
                    conf=conf,
                    iou=iou,
                    device=device,
                    camera_id=cam_id,
                )

            worker = CameraWorker(
                camera_id=cam_id,
                source=source,
                fps_target=float(fps_target),
                detector=detector if not worker_tracker else None,
                tracker=worker_tracker,
                on_detections=on_detections,
                on_tracks=on_tracks,
            )
            self.workers[cam_id] = worker

            thread = threading.Thread(
                target=worker.start,
                name=f"worker-{cam_id}",
                daemon=True,
            )
            self._threads[cam_id] = thread

        # Print startup summary
        print(f"\n{'='*75}")
        if tracker_type:
            mode_str = f"Vehicle Tracking ({tracker_type})"
        elif detector:
            mode_str = "Vehicle Detection Only"
        else:
            mode_str = "Frame Ingestion Only"

        print(f"  Multi-Camera Pipeline Mode: {mode_str}")
        print(f"{'='*75}")
        print(f"{'Camera ID':<12} | {'Source':<45} | {'FPS Target'}")
        print(f"{'-'*75}")
        for cam_id, worker in self.workers.items():
            src = worker.source
            if len(src) > 42:
                src = "..." + src[-39:]
            print(f"{cam_id:<12} | {src:<45} | {worker.fps_target}")
        print(f"{'='*75}\n")

        # Start all threads
        for cam_id, thread in self._threads.items():
            logger.info(f"Starting worker thread for {cam_id}")
            thread.start()

        # Wait for all threads
        try:
            while not self._shutdown:
                alive = any(t.is_alive() for t in self._threads.values())
                if not alive:
                    logger.info("All workers have finished.")
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received.")
        finally:
            self.stop_all()

    def stop_all(self) -> None:
        """Stop all camera workers gracefully."""
        self._shutdown = True
        logger.info("Stopping all camera workers...")

        for cam_id, worker in self.workers.items():
            worker.stop()

        for cam_id, thread in self._threads.items():
            thread.join(timeout=5.0)
            if thread.is_alive():
                logger.warning(f"[{cam_id}] Worker thread did not stop in time.")

        logger.info("All workers stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Camera Worker — multi-camera ingestion, vehicle detection & tracking"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--source",
        type=str,
        help="Video source: RTSP URL, file path, or webcam index",
    )
    group.add_argument(
        "--config",
        type=str,
        help="Path to cameras.json config file",
    )

    parser.add_argument(
        "--camera-id",
        type=str,
        default="CAM-001",
        help="Camera ID (used with --source)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=0,
        help="Target FPS for throttling (0 = no throttle)",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="When using --config, connect directly to video files instead of RTSP streams",
    )
    parser.add_argument(
        "--stats-interval",
        type=float,
        default=10.0,
        help="Seconds between stats log messages (default: 10)",
    )

    # Phase 2: Vehicle Detection flag
    parser.add_argument(
        "--detect",
        action="store_true",
        help="Enable YOLO vehicle detection (detection-only mode)",
    )

    # Phase 3: Single-Camera Tracking flags
    parser.add_argument(
        "--track",
        action="store_true",
        help="Enable single-camera vehicle tracking (ByteTrack)",
    )
    parser.add_argument(
        "--tracker-type",
        type=str,
        default="bytetrack.yaml",
        choices=["bytetrack.yaml", "botsort.yaml"],
        help="Tracker backend: bytetrack.yaml or botsort.yaml (default: bytetrack.yaml)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="data/models/yolov8n.pt",
        help="Path to YOLO vehicle model weights",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold for vehicle detection (default: 0.35)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="NMS IoU threshold (default: 0.5)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (default: 640)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device: auto, cpu, cuda:0",
    )

    args = parser.parse_args()

    # Handle shutdown signals
    def signal_handler(sig, frame):
        logger.info("Shutdown signal received.")
        if hasattr(main, "_manager"):
            main._manager.stop_all()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, signal_handler)

    tracker = None
    detector = None

    if args.track:
        logger.info(f"Initializing VehicleTracker with {args.tracker_type}...")
        if args.source:
            tracker = VehicleTracker(
                model_path=args.model,
                tracker_type=args.tracker_type,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                camera_id=args.camera_id,
            )
    elif args.detect:
        logger.info(f"Loading vehicle detector from {args.model}...")
        detector = VehicleDetector(
            model_path=args.model,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
        )

    if args.source:
        worker = CameraWorker(
            camera_id=args.camera_id,
            source=args.source,
            fps_target=args.fps,
            detector=detector,
            tracker=tracker,
        )
        worker._stats_interval = args.stats_interval
        worker.start()
    else:
        manager = MultiCameraManager()
        main._manager = manager  # type: ignore
        manager.start_all(
            args.config,
            use_stream=not args.direct,
            detector=detector,
            tracker_type=args.tracker_type if args.track else None,
            model_path=args.model,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
        )


if __name__ == "__main__":
    main()
