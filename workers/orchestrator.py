"""
Pipeline Orchestrator (Phase 8).

Coordinates multi-camera video ingestion, worker concurrency, supervisor heartbeat monitoring,
worker failure auto-recovery, online threat alerting, and graceful shutdown for production deployment.

Architectural Guarantees:
1. Failure Isolation: Detector, tracker, ANPR, and ReID are worker-owned runtime instances.
2. Shared Analytics: GlobalIdentityResolver and AlertEngine are shared and thread-safe.
3. Concurrent Persistence: Database operations use per-thread WAL connections with retry logic.
4. Production Supervisor: Heartbeat monitor detects dropped workers and executes auto-restart.
5. Clean Termination: Graceful signal trapping (SIGINT, SIGTERM) flushes pending tracks and releases hardware.
"""

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import psutil

from alpr.detector import VehicleDetector
from alpr.tracker import VehicleTracker
from alpr.anpr import VehicleANPR
from alpr.reid import VehicleReID
from alpr.identity import GlobalIdentityResolver, GlobalVehicleIdentity, IdentityMatchResult
from alpr.alerts import AlertEngine, AlertRecord
from alpr.database import (
    init_db,
    get_thread_connection,
    update_camera_status,
    get_camera_statuses as db_get_camera_statuses,
    get_enriched_blacklist,
    get_db_concurrency_metrics,
)
from workers.camera_worker import CameraWorker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("workers.orchestrator")


@dataclass
class CameraTelemetry:
    """Snapshot of a single camera worker's runtime state."""
    camera_id: str
    status: str = "offline"
    input_fps: float = 0.0
    inference_fps: float = 0.0
    latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    frames_processed: int = 0
    vehicles_detected: int = 0
    plates_detected: int = 0
    identities_resolved: int = 0
    alerts_triggered: int = 0
    restarts: int = 0
    loop_count: int = 0
    last_heartbeat: float = field(default_factory=time.time)
    thread_alive: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "status": self.status,
            "input_fps": round(self.input_fps, 2),
            "inference_fps": round(self.inference_fps, 2),
            "latency_ms": round(self.latency_ms, 1),
            "p50_latency_ms": round(self.p50_latency_ms, 1),
            "p95_latency_ms": round(self.p95_latency_ms, 1),
            "frames_processed": self.frames_processed,
            "vehicles_detected": self.vehicles_detected,
            "plates_detected": self.plates_detected,
            "identities_resolved": self.identities_resolved,
            "alerts_triggered": self.alerts_triggered,
            "restarts": self.restarts,
            "loop_count": self.loop_count,
            "thread_alive": self.thread_alive,
            "last_heartbeat": self.last_heartbeat,
        }


class PipelineOrchestrator:
    """
    Supervisor coordinating concurrent camera workers, health heartbeat,
    auto-recovery, and production lifecycle management.
    """

    def __init__(
        self,
        config_path: Union[str, Path] = "configs/cameras.json",
        camera_graph_path: Union[str, Path] = "configs/camera_graph.json",
        db_path: Union[str, Path] = "data/alpr.db",
        use_stream: bool = True,
        mode: str = "full-pipeline",  # 'full-pipeline', 'track', 'detect', 'ingest'
        loop_video: bool = True,
        model_path: str = "data/models/yolov8n.pt",
        plate_model_path: str = "data/models/license_plate_yolov8_best.pt",
        tracker_type: str = "bytetrack.yaml",
        reid_weights: Optional[str] = None,
        conf: float = 0.35,
        iou: float = 0.5,
        device: str = "auto",
        ocr_every_n: int = 3,
        reid_every_n: int = 15,
        heartbeat_interval: float = 2.0,
        max_worker_restarts: int = 5,
        on_alert_triggered: Optional[Callable[[str, AlertRecord], None]] = None,
        on_global_identity_resolved: Optional[Callable[[str, GlobalVehicleIdentity, IdentityMatchResult], None]] = None,
    ):
        self.config_path = Path(config_path)
        self.camera_graph_path = Path(camera_graph_path)
        self.db_path = Path(db_path)
        self.use_stream = use_stream
        self.mode = mode
        self.loop_video = loop_video
        self.model_path = model_path
        self.plate_model_path = plate_model_path
        self.tracker_type = tracker_type
        self.reid_weights = reid_weights
        self.conf = conf
        self.iou = iou
        self.device = device
        self.ocr_every_n = ocr_every_n
        self.reid_every_n = reid_every_n
        self.heartbeat_interval = heartbeat_interval
        self.max_worker_restarts = max_worker_restarts
        self.on_alert_triggered = on_alert_triggered
        self.on_global_identity_resolved = on_global_identity_resolved

        self.start_time: float = 0.0
        self._running: bool = False
        self._shutdown_initiated: bool = False

        # Shared analytical instances (Thread-Safe)
        self.identity_resolver = GlobalIdentityResolver(
            camera_graph_path=self.camera_graph_path if self.camera_graph_path.exists() else None
        )
        self.alert_engine = AlertEngine()

        # Camera configurations & workers
        self.camera_configs: Dict[str, Dict[str, Any]] = {}
        self.workers: Dict[str, CameraWorker] = {}
        self.worker_threads: Dict[str, threading.Thread] = {}
        self.telemetry: Dict[str, CameraTelemetry] = {}
        self.worker_restarts: Dict[str, int] = {}
        self.restart_backoff: Dict[str, float] = {}

        # Supervisor thread
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Initialize database schema
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        init_db(self.db_path)

        self._load_camera_configs()

    def _load_camera_configs(self) -> None:
        """Load and parse cameras.json configuration."""
        if not self.config_path.exists():
            logger.error(f"Cameras config file not found: {self.config_path}")
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cams = data.get("cameras", data) if isinstance(data, dict) else data
            conn = get_thread_connection(self.db_path)
            from alpr.database import upsert_camera
            for c in cams:
                cid = c.get("camera_id")
                if cid:
                    self.camera_configs[cid] = c
                    self.telemetry[cid] = CameraTelemetry(camera_id=cid)
                    self.worker_restarts[cid] = 0
                    try:
                        upsert_camera(
                            conn=conn,
                            camera_id=cid,
                            name=c.get("name", cid),
                            lat=float(c.get("latitude", 0.0)),
                            lon=float(c.get("longitude", 0.0)),
                            description=c.get("description"),
                        )
                    except Exception as upsert_err:
                        logger.debug(f"Camera upsert notice for [{cid}]: {upsert_err}")
            logger.info(f"Loaded configuration for {len(self.camera_configs)} camera(s).")
        except Exception as e:
            logger.error(f"Error loading cameras config: {e}")

    def _create_worker_for_camera(self, camera_id: str) -> CameraWorker:
        """
        Factory method to instantiate a CameraWorker with worker-owned runtime inference models.
        Detector, Tracker, ANPR, and ReID are instantiated per worker for failure isolation.
        """
        cam_cfg = self.camera_configs[camera_id]
        source = cam_cfg.get("stream_url") if self.use_stream else cam_cfg.get("video")
        if not source:
            source = cam_cfg.get("video") or cam_cfg.get("stream_url")

        fps_target = float(cam_cfg.get("fps", 0.0))

        # Build worker-owned inference instances based on mode
        detector: Optional[VehicleDetector] = None
        tracker: Optional[VehicleTracker] = None
        anpr: Optional[VehicleANPR] = None
        reid: Optional[VehicleReID] = None

        if self.mode in ("full-pipeline", "track"):
            tracker = VehicleTracker(
                model_path=self.model_path,
                tracker_type=self.tracker_type,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                camera_id=camera_id,
            )

        if self.mode == "full-pipeline":
            anpr = VehicleANPR(
                plate_model_path=self.plate_model_path,
                device=self.device,
                ocr_every_n=self.ocr_every_n,
            )
            reid = VehicleReID(
                weights_path=self.reid_weights,
                device=self.device,
            )
        elif self.mode == "detect":
            detector = VehicleDetector(
                model_path=self.model_path,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
            )

        # Cache blacklist records for worker
        bl_records = None
        try:
            conn = get_thread_connection(self.db_path)
            bl_records = get_enriched_blacklist(conn, active_only=True)
        except Exception:
            bl_records = []

        worker = CameraWorker(
            camera_id=camera_id,
            source=str(source),
            fps_target=fps_target,
            detector=detector,
            tracker=tracker,
            anpr=anpr,
            reid=reid,
            reid_every_n=self.reid_every_n,
            identity_resolver=self.identity_resolver if self.mode in ("full-pipeline", "track") else None,
            alert_engine=self.alert_engine if self.mode == "full-pipeline" else None,
            blacklist_records=bl_records,
            db_path=self.db_path,
            loop=self.loop_video,
            on_alert_triggered=self.on_alert_triggered,
            on_global_identity_resolved=self.on_global_identity_resolved,
        )
        return worker

    def start(self) -> None:
        """Start all camera workers and supervisor heartbeat thread."""
        with self._lock:
            if self._running:
                logger.warning("PipelineOrchestrator is already running.")
                return

            self._running = True
            self._shutdown_initiated = False
            self.start_time = time.time()

        logger.info(f"Starting PipelineOrchestrator in mode: '{self.mode}'...")

        # Spawn per-camera worker threads
        for camera_id in self.camera_configs:
            self._spawn_worker_thread(camera_id)

        # Start supervisor heartbeat thread
        self._heartbeat_thread = threading.Thread(
            target=self._supervisor_heartbeat_loop,
            name="OrchestratorSupervisor",
            daemon=True,
        )
        self._heartbeat_thread.start()
        logger.info("Supervisor heartbeat thread started.")

    def _spawn_worker_thread(self, camera_id: str) -> bool:
        """Instantiate and launch worker thread for a specific camera."""
        try:
            worker = self._create_worker_for_camera(camera_id)
            thread = threading.Thread(
                target=worker.start,
                name=f"Worker-{camera_id}",
                daemon=True,
            )
            with self._lock:
                self.workers[camera_id] = worker
                self.worker_threads[camera_id] = thread
            thread.start()
            logger.info(f"Worker thread for [{camera_id}] launched.")
            return True
        except Exception as e:
            logger.error(f"Failed to spawn worker thread for [{camera_id}]: {e}", exc_info=True)
            return False

    def _supervisor_heartbeat_loop(self) -> None:
        """
        Background loop: gathers telemetry, updates SQLite camera status,
        and automatically restarts dropped or crashed workers.
        """
        while self._running and not self._shutdown_initiated:
            try:
                now = time.time()
                for camera_id, cfg in list(self.camera_configs.items()):
                    worker = self.workers.get(camera_id)
                    thread = self.worker_threads.get(camera_id)
                    alive = thread.is_alive() if thread else False

                    # Read telemetry
                    status = worker.status if worker else ("running" if alive else "offline")
                    in_fps = worker.input_fps if worker else 0.0
                    infer_fps = worker.inference_fps if worker else 0.0
                    lat_ms = worker.avg_latency_ms if worker else 0.0
                    proc_frames = worker.frames_processed if worker else 0
                    veh_det = worker.vehicles_detected if worker else 0
                    plt_det = worker.plates_detected if worker else 0
                    id_res = worker.identities_resolved if worker else 0
                    alt_trig = getattr(worker, "alerts_triggered", 0) if worker else 0

                    p50_lat = 0.0
                    p95_lat = 0.0
                    if worker and hasattr(worker, "_latencies") and len(worker._latencies) > 0:
                        lats = list(worker._latencies)
                        p50_lat = float(np.percentile(lats, 50))
                        p95_lat = float(np.percentile(lats, 95))

                    loop_cnt = 0
                    if worker and worker._camera and hasattr(worker._camera, "loop_count"):
                        loop_cnt = worker._camera.loop_count

                    telem = self.telemetry.get(camera_id)
                    if telem:
                        telem.status = status
                        telem.input_fps = in_fps
                        telem.inference_fps = infer_fps
                        telem.latency_ms = lat_ms
                        telem.p50_latency_ms = p50_lat
                        telem.p95_latency_ms = p95_lat
                        telem.frames_processed = proc_frames
                        telem.vehicles_detected = veh_det
                        telem.plates_detected = plt_det
                        telem.identities_resolved = id_res
                        telem.alerts_triggered = alt_trig
                        telem.restarts = self.worker_restarts.get(camera_id, 0)
                        telem.loop_count = loop_cnt
                        telem.thread_alive = alive
                        telem.last_heartbeat = now

                    # Persist telemetry to SQLite
                    try:
                        conn = get_thread_connection(self.db_path)
                        update_camera_status(
                            conn=conn,
                            camera_id=camera_id,
                            status=status,
                            fps=in_fps,
                            latency_ms=lat_ms,
                            last_seen_ts=now if alive else None,
                            total_frames=proc_frames,
                            total_detections=veh_det,
                        )
                    except Exception as db_err:
                        logger.debug(f"Supervisor DB update error for [{camera_id}]: {db_err}")

                    # Worker failure auto-recovery
                    if not alive and self._running and not self._shutdown_initiated:
                        restarts = self.worker_restarts.get(camera_id, 0)
                        if restarts < self.max_worker_restarts:
                            backoff = min(15.0, 1.5 ** restarts)
                            last_restart = self.restart_backoff.get(camera_id, 0.0)
                            if now - last_restart >= backoff:
                                self.worker_restarts[camera_id] = restarts + 1
                                self.restart_backoff[camera_id] = now
                                if telem:
                                    telem.restarts += 1
                                logger.warning(
                                    f"Worker [{camera_id}] stopped unexpectedly. "
                                    f"Triggering auto-restart {restarts + 1}/{self.max_worker_restarts} (backoff {backoff:.1f}s)..."
                                )
                                self._spawn_worker_thread(camera_id)
                        else:
                            logger.error(
                                f"Worker [{camera_id}] exceeded maximum restarts ({self.max_worker_restarts}). Marked as offline."
                            )

            except Exception as e:
                logger.error(f"Error in supervisor heartbeat loop: {e}", exc_info=True)

            time.sleep(self.heartbeat_interval)

    def stop(self, timeout: float = 5.0) -> None:
        """Gracefully stop all camera workers, flush pending tracks, and release resources."""
        with self._lock:
            if not self._running:
                return
            self._shutdown_initiated = True
            self._running = False

        logger.info("PipelineOrchestrator shutdown initiated. Stopping all workers...")

        # 1. Stop all workers (this finalizes tracks and records to DB)
        for cid, worker in list(self.workers.items()):
            try:
                worker.stop()
            except Exception as e:
                logger.error(f"Error stopping worker [{cid}]: {e}")

        # 2. Join worker threads with timeout
        for cid, thread in list(self.worker_threads.items()):
            if thread.is_alive():
                thread.join(timeout=timeout)
                if thread.is_alive():
                    logger.warning(f"Worker thread [{cid}] did not terminate within {timeout}s.")

        # 3. Mark all cameras as offline in database
        try:
            conn = get_thread_connection(self.db_path)
            for cid in self.camera_configs:
                update_camera_status(conn=conn, camera_id=cid, status="offline", fps=0.0)
        except Exception as e:
            logger.debug(f"Error marking cameras offline on shutdown: {e}")

        # 4. Join supervisor thread
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=timeout)

        logger.info("PipelineOrchestrator successfully terminated.")

    def get_health(self) -> Dict[str, Any]:
        """
        Aggregate system health and camera telemetry snapshot for REST consumption.
        """
        now = time.time()
        uptime = round(now - self.start_time, 1) if self.start_time > 0 else 0.0

        total_cameras = len(self.camera_configs)
        active_cameras = 0
        reconnecting_cameras = 0
        offline_cameras = 0
        total_fps = 0.0
        latencies = []
        total_proc_frames = 0
        total_veh_detected = 0
        total_plates = 0
        total_identities = 0
        total_alerts = 0

        cameras_snapshot = {}
        for cid, telem in self.telemetry.items():
            cdict = telem.to_dict()
            cameras_snapshot[cid] = cdict
            if telem.thread_alive and telem.status in ("online", "running"):
                active_cameras += 1
            elif telem.status == "reconnecting":
                reconnecting_cameras += 1
            else:
                offline_cameras += 1

            total_fps += telem.input_fps
            if telem.latency_ms > 0:
                latencies.append(telem.latency_ms)
            total_proc_frames += telem.frames_processed
            total_veh_detected += telem.vehicles_detected
            total_plates += telem.plates_detected
            total_identities += telem.identities_resolved
            total_alerts += telem.alerts_triggered

        avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else 0.0
        p50_latency = round(float(np.percentile(latencies, 50)), 1) if latencies else 0.0
        p95_latency = round(float(np.percentile(latencies, 95)), 1) if latencies else 0.0

        cpu_pct = 0.0
        mem_mb = 0.0
        try:
            cpu_pct = round(psutil.cpu_percent(interval=None), 1)
            mem_mb = round(psutil.Process().memory_info().rss / (1024 * 1024), 1)
        except Exception:
            pass

        db_metrics = get_db_concurrency_metrics()

        if not self._running:
            system_status = "offline"
        elif active_cameras == total_cameras and total_cameras > 0:
            system_status = "healthy"
        elif active_cameras > 0 or reconnecting_cameras > 0:
            system_status = "degraded"
        else:
            system_status = "offline"

        return {
            "status": system_status,
            "orchestrator_running": self._running,
            "uptime_seconds": uptime,
            "total_cameras": total_cameras,
            "active_cameras": active_cameras,
            "reconnecting_cameras": reconnecting_cameras,
            "offline_cameras": offline_cameras,
            "total_fps": round(total_fps, 2),
            "avg_latency_ms": avg_latency,
            "p50_latency_ms": p50_latency,
            "p95_latency_ms": p95_latency,
            "cpu_percent": cpu_pct,
            "memory_mb": mem_mb,
            "db_metrics": db_metrics,
            "total_frames_processed": total_proc_frames,
            "total_vehicles_detected": total_veh_detected,
            "total_plates_detected": total_plates,
            "total_identities_resolved": total_identities,
            "total_alerts_triggered": total_alerts,
            "cameras": cameras_snapshot,
        }

    def get_camera_statuses(self) -> List[Dict[str, Any]]:
        """Return list of camera statuses combining memory telemetry and SQLite."""
        statuses = []
        for cid in sorted(self.camera_configs.keys()):
            telem = self.telemetry.get(cid)
            cfg = self.camera_configs.get(cid, {})
            statuses.append({
                "camera_id": cid,
                "name": cfg.get("name", cid),
                "latitude": cfg.get("latitude", 0.0),
                "longitude": cfg.get("longitude", 0.0),
                "status": telem.status if telem else "offline",
                "fps": telem.input_fps if telem else 0.0,
                "latency_ms": telem.latency_ms if telem else 0.0,
                "p50_latency_ms": telem.p50_latency_ms if telem else 0.0,
                "p95_latency_ms": telem.p95_latency_ms if telem else 0.0,
                "total_frames": telem.frames_processed if telem else 0,
                "total_detections": telem.vehicles_detected if telem else 0,
                "restarts": telem.restarts if telem else 0,
                "loop_count": telem.loop_count if telem else 0,
                "thread_alive": telem.thread_alive if telem else False,
            })
        return statuses


# ============================================================================
# CLI ENTRYPOINT & SIGNAL TRAPPING
# ============================================================================

def setup_signal_handlers(orchestrator: PipelineOrchestrator) -> None:
    """Register graceful termination hooks for SIGINT and SIGTERM."""
    def _sig_handler(sig, frame):
        sig_name = "SIGINT" if sig == signal.SIGINT else "SIGTERM"
        logger.info(f"Received {sig_name}. Initiating graceful shutdown...")
        orchestrator.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sig_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _sig_handler)


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Orchestrator: Multi-Camera Live Stream Supervisor & Alert Engine (Phase 8)",
    )
    parser.add_argument("--config", default="configs/cameras.json", help="Path to cameras.json config")
    parser.add_argument("--graph-path", default="configs/camera_graph.json", help="Path to camera_graph.json")
    parser.add_argument("--db-path", default="data/alpr.db", help="Path to SQLite database")
    parser.add_argument("--mode", default="full-pipeline", choices=["full-pipeline", "track", "detect", "ingest"], help="Execution mode")
    parser.add_argument("--direct", action="store_true", help="Connect directly to video files instead of RTSP streams")
    parser.add_argument("--model", default="data/models/yolov8n.pt", help="Vehicle detector weights")
    parser.add_argument("--plate-model", default="data/models/license_plate_yolov8_best.pt", help="Plate detector weights")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Tracker type (bytetrack.yaml or botsort.yaml)")
    parser.add_argument("--reid-weights", default=None, help="Optional ReID model weights (.pth)")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--device", default="auto", help="Inference device (auto, cpu, cuda)")
    parser.add_argument("--ocr-every-n", type=int, default=3, help="OCR evaluation throttle interval")
    parser.add_argument("--reid-every-n", type=int, default=15, help="ReID extraction throttle interval")
    parser.add_argument("--heartbeat-interval", type=float, default=2.0, help="Supervisor heartbeat frequency in seconds")
    parser.add_argument("--max-restarts", type=int, default=5, help="Maximum automatic worker restarts on drop")
    parser.add_argument("--loop-video", action=argparse.BooleanOptionalAction, default=True, help="Loop video files upon EOF in direct mode")

    args = parser.parse_args()

    orchestrator = PipelineOrchestrator(
        config_path=args.config,
        camera_graph_path=args.graph_path,
        db_path=args.db_path,
        use_stream=not args.direct,
        mode=args.mode,
        loop_video=args.loop_video,
        model_path=args.model,
        plate_model_path=args.plate_model,
        tracker_type=args.tracker,
        reid_weights=args.reid_weights,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        ocr_every_n=args.ocr_every_n,
        reid_every_n=args.reid_every_n,
        heartbeat_interval=args.heartbeat_interval,
        max_worker_restarts=args.max_restarts,
    )

    setup_signal_handlers(orchestrator)
    orchestrator.start()

    logger.info("PipelineOrchestrator running. Press Ctrl+C to terminate.")
    try:
        while orchestrator._running:
            time.sleep(1.0)
    except KeyboardInterrupt:
        orchestrator.stop()


if __name__ == "__main__":
    main()
