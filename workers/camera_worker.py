"""
Camera Worker — per-camera stream processing worker.

Connects to a video source via CameraSource, reads frames in a loop,
and logs performance metrics. This is the Phase 1 skeleton: no AI
processing, just validates streaming infrastructure works.

Usage:
    # Single camera from RTSP
    python -m workers.camera_worker --camera-id CAM-001 --source rtsp://localhost:8554/cam01

    # Single camera from MP4 (same CameraSource API)
    python -m workers.camera_worker --camera-id CAM-001 --source inputs/cam01.mp4

    # All cameras from config
    python -m workers.camera_worker --config configs/cameras.json --all
"""

import argparse
import json
import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from alpr.camera import CameraSource

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("workers.camera")


class CameraWorker:
    """
    Per-camera frame reader with lifecycle management.

    Connects to a source, reads frames continuously, handles
    reconnection on failure, and reports performance metrics.
    No AI processing — skeleton for Phase 1 validation.
    """

    def __init__(
        self,
        camera_id: str,
        source: str,
        fps_target: float = 0.0,
        reconnect_max_retries: int = 10,
    ):
        """
        Initialize a camera worker.

        Args:
            camera_id: Logical camera identifier (e.g., 'CAM-001').
            source: Video source — RTSP URL, file path, or webcam index.
            fps_target: Target FPS for throttling (0 = no throttle).
            reconnect_max_retries: Max reconnection attempts.
        """
        self.camera_id = camera_id
        self.source = source
        self.fps_target = fps_target
        self.reconnect_max_retries = reconnect_max_retries

        self._running = False
        self._camera: Optional[CameraSource] = None
        self._stats_interval = 10.0  # Log stats every N seconds
        self._last_stats_time = 0.0

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

    def start(self) -> None:
        """
        Main processing loop.

        Connects to the source and reads frames continuously.
        On stream failure, attempts reconnection if the source is a stream.
        Logs FPS and frame count statistics periodically.
        """
        self._running = True
        logger.info(f"[{self.camera_id}] Worker starting — source: {self.source}")

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
        logger.info(
            f"[{self.camera_id}] Worker stopping. "
            f"Frames read: {self._camera.frames_read if self._camera else 0}, "
            f"Frames dropped: {self._camera.frames_dropped if self._camera else 0}, "
            f"Measured FPS: {self._camera.get_fps():.1f}" if self._camera else ""
        )
        if self._camera:
            self._camera.release()
            self._camera = None

    def _log_stats_if_due(self) -> None:
        """Log performance statistics periodically."""
        now = time.time()
        if now - self._last_stats_time >= self._stats_interval:
            self._last_stats_time = now
            cam = self._camera
            if cam:
                logger.info(
                    f"[{self.camera_id}] "
                    f"status={self.status} | "
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

        # Support both list and dict-with-cameras-key formats
        if isinstance(data, dict) and "cameras" in data:
            cameras = data["cameras"]
        elif isinstance(data, list):
            cameras = data
        else:
            logger.error("Invalid config format. Expected list or dict with 'cameras' key.")
            return []

        return cameras

    def start_all(self, config_path: str, use_stream: bool = True) -> None:
        """
        Start workers for all cameras in the config.

        Args:
            config_path: Path to cameras.json.
            use_stream: If True, connect via stream_url (RTSP).
                        If False, connect directly to the video file.
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

            worker = CameraWorker(
                camera_id=cam_id,
                source=source,
                fps_target=float(fps_target),
            )
            self.workers[cam_id] = worker

            thread = threading.Thread(
                target=worker.start,
                name=f"worker-{cam_id}",
                daemon=True,
            )
            self._threads[cam_id] = thread

        # Print startup summary
        print(f"\n{'='*70}")
        print(f"{'Camera ID':<12} | {'Source':<45} | {'FPS Target'}")
        print(f"{'-'*70}")
        for cam_id, worker in self.workers.items():
            src = worker.source
            if len(src) > 42:
                src = "..." + src[-39:]
            print(f"{cam_id:<12} | {src:<45} | {worker.fps_target}")
        print(f"{'='*70}\n")

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
        description="Camera Worker — per-camera stream reader (Phase 1 skeleton)"
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

    if args.source:
        # Single camera mode
        worker = CameraWorker(
            camera_id=args.camera_id,
            source=args.source,
            fps_target=args.fps,
        )
        worker._stats_interval = args.stats_interval
        worker.start()
    else:
        # Multi-camera mode from config
        manager = MultiCameraManager()
        main._manager = manager  # type: ignore
        manager.start_all(args.config, use_stream=not args.direct)


if __name__ == "__main__":
    main()
