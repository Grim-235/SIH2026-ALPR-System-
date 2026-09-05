"""
Camera Worker -- per-camera stream processing and pipeline orchestrator.

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
from typing import Callable, Dict, List, Optional, Union

import numpy as np

from alpr.camera import CameraSource
from alpr.detector import VehicleDetector, VehicleDetection
from alpr.tracker import VehicleTracker, ActiveVehicleTrack, VehicleTrackState, PlateRead
from alpr.anpr import VehicleANPR
from alpr.reid import VehicleReID
from alpr.identity import (
    GlobalIdentityResolver,
    GlobalVehicleIdentity,
    VehicleObservation,
    IdentityMatchResult,
)
from alpr.alerts import (
    AlertEngine,
    AlertRecord,
    ALERT_VELOCITY_ANOMALY,
    SEVERITY_HIGH,
    generate_alert_id,
    format_iso_timestamp,
)
from alpr.database import (
    init_db,
    save_global_identity,
    record_vehicle_observation,
    get_thread_connection,
    update_camera_status,
    record_security_alert_obj,
    get_enriched_blacklist,
)

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
        anpr: Optional[VehicleANPR] = None,
        reid: Optional[VehicleReID] = None,
        reid_every_n: int = 15,
        identity_resolver: Optional[GlobalIdentityResolver] = None,
        alert_engine: Optional[AlertEngine] = None,
        blacklist_records: Optional[List[Dict[str, Any]]] = None,
        db_path: Optional[Union[str, Path]] = None,
        loop: bool = False,
        on_detections: Optional[Callable[[str, np.ndarray, List[VehicleDetection], float, float], None]] = None,
        on_tracks: Optional[Callable[[str, np.ndarray, List[ActiveVehicleTrack], float, float], None]] = None,
        on_plate_read: Optional[Callable[[str, int, PlateRead], None]] = None,
        on_reid_extracted: Optional[Callable[[str, int, np.ndarray], None]] = None,
        on_global_identity_resolved: Optional[Callable[[str, GlobalVehicleIdentity, IdentityMatchResult], None]] = None,
        on_alert_triggered: Optional[Callable[[str, AlertRecord], None]] = None,
    ):
        """
        Initialize a camera worker.

        Args:
            camera_id: Logical camera identifier (e.g., 'CAM-001').
            source: Video source -- RTSP URL, file path, or webcam index.
            fps_target: Target FPS for throttling (0 = no throttle).
            reconnect_max_retries: Max reconnection attempts.
            detector: Optional VehicleDetector instance (for detection-only mode).
            tracker: Optional VehicleTracker instance (for tracking mode).
            anpr: Optional VehicleANPR instance (for ANPR plate recognition mode).
            reid: Optional VehicleReID instance (for visual appearance ReID).
            reid_every_n: Throttle interval for representative ReID feature extraction.
            identity_resolver: Optional GlobalIdentityResolver instance for multi-camera tracking.
            alert_engine: Optional AlertEngine instance for online threat & anomaly diagnostics.
            blacklist_records: Optional pre-loaded watchlist records.
            db_path: Optional path to SQLite database for persisting global identities.
            loop: If True and source is a video file, continuous loop playback.
            on_detections: Optional callback(camera_id, frame, detections, latency_ms, capture_ts).
            on_tracks: Optional callback(camera_id, frame, active_tracks, latency_ms, capture_ts).
            on_plate_read: Optional callback(camera_id, track_id, plate_read).
            on_reid_extracted: Optional callback(camera_id, track_id, embedding).
            on_global_identity_resolved: Optional callback(camera_id, identity, result).
            on_alert_triggered: Optional callback(camera_id, alert_record).
        """
        self.camera_id = camera_id
        self.source = source
        self.fps_target = fps_target
        self.reconnect_max_retries = reconnect_max_retries
        self.detector = detector
        self.tracker = tracker
        self.anpr = anpr
        self.reid = reid
        self.reid_every_n = reid_every_n
        self.identity_resolver = identity_resolver
        self.alert_engine = alert_engine
        self.blacklist_records = blacklist_records
        self.db_path = db_path
        self.loop = loop
        self.on_detections = on_detections
        self.on_tracks = on_tracks
        self.on_plate_read = on_plate_read
        self.on_reid_extracted = on_reid_extracted
        self.on_global_identity_resolved = on_global_identity_resolved
        self.on_alert_triggered = on_alert_triggered

        self._running = False
        self._camera: Optional[CameraSource] = None
        self._stats_interval = 10.0  # Log stats every N seconds
        self._last_stats_time = 0.0

        # Performance & detection metrics
        self.frames_processed = 0
        self.vehicles_detected = 0
        self.plates_detected = 0
        self.reid_extractions = 0
        self.identities_resolved = 0
        self.alerts_triggered = 0
        self._latencies: deque = deque(maxlen=30)
        self._proc_times: deque = deque(maxlen=30)
        self._vehicle_counts: deque = deque(maxlen=30)
        self._reid_latencies: deque = deque(maxlen=30)

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

    def _process_finalized_track(self, trk: VehicleTrackState) -> None:
        """Process a finalized vehicle track through identity resolution, online alerts, and database persistence."""
        if self.identity_resolver is None:
            return

        # Phase 9B: Persist best vehicle and plate crops as evidence snapshots
        vehicle_crop_path = None
        plate_crop_path = None
        try:
            crops_dir = Path("data/evidence/crops")
            crops_dir.mkdir(parents=True, exist_ok=True)
            import cv2

            if trk.best_vehicle_crop is not None and isinstance(trk.best_vehicle_crop, np.ndarray) and trk.best_vehicle_crop.size > 0:
                rel_vpath = f"data/evidence/crops/{self.camera_id}_{trk.track_id}_veh.jpg"
                full_vpath = crops_dir / f"{self.camera_id}_{trk.track_id}_veh.jpg"
                if cv2.imwrite(str(full_vpath), trk.best_vehicle_crop):
                    vehicle_crop_path = rel_vpath

            if trk.best_plate_crop is not None and isinstance(trk.best_plate_crop, np.ndarray) and trk.best_plate_crop.size > 0:
                rel_ppath = f"data/evidence/crops/{self.camera_id}_{trk.track_id}_plate.jpg"
                full_ppath = crops_dir / f"{self.camera_id}_{trk.track_id}_plate.jpg"
                if cv2.imwrite(str(full_ppath), trk.best_plate_crop):
                    plate_crop_path = rel_ppath
        except Exception as crop_err:
            logger.debug(f"[{self.camera_id}] Error saving evidence crops: {crop_err}")

        obs = VehicleObservation(
            camera_id=self.camera_id,
            track_id=trk.track_id,
            timestamp=trk.last_timestamp,  # Capture timestamp of vehicle track exit
            vehicle_type=trk.vehicle_type,
            canonical_plate=trk.canonical_plate,
            plate_confidence=trk.plate_confidence,
            best_reid_embedding=trk.best_reid_embedding,
            crop_quality=trk.best_crop_quality,
            bbox=trk.latest_bbox,
            vehicle_crop_path=vehicle_crop_path,
            plate_crop_path=plate_crop_path,
        )

        identity, result = self.identity_resolver.resolve_observation(obs)
        self.identities_resolved += 1

        # Online alert evaluation if AlertEngine is provided
        generated_alerts: List[AlertRecord] = []
        if self.alert_engine is not None:
            try:
                bl_records = self.blacklist_records
                if bl_records is None and self.db_path:
                    try:
                        conn_bl = get_thread_connection(self.db_path)
                        bl_records = get_enriched_blacklist(conn_bl, active_only=True)
                        self.blacklist_records = bl_records
                    except Exception:
                        bl_records = []

                obs_alerts = self.alert_engine.evaluate_observation(
                    camera_id=self.camera_id,
                    timestamp=trk.last_timestamp,
                    plate_text=trk.canonical_plate,
                    global_id=identity.global_id,
                    match_status=result.status,
                    match_confidence=result.confidence,
                    match_method=result.match_method,
                    blacklist_records=bl_records,
                )
                generated_alerts.extend(obs_alerts)

                # Kinematic plausibility diagnostic
                kine_speed = result.transit_speed_kmh
                kine_dist = result.distance_km
                kine_gid = identity.global_id

                # If resolver rejected transit feasibility due to impossible speed for same plate:
                if kine_speed is None and trk.canonical_plate and self.identity_resolver is not None:
                    for prev_ident in list(self.identity_resolver.identities.values()):
                        if prev_ident.global_id != identity.global_id and prev_ident.canonical_plate == trk.canonical_plate:
                            dist = self.identity_resolver.get_distance_km(prev_ident.last_camera_id, self.camera_id)
                            delta_t = trk.last_timestamp - prev_ident.last_seen_ts
                            if dist is not None and delta_t > 0:
                                calc_speed = (dist / delta_t) * 3600.0
                                if calc_speed > self.alert_engine.velocity_bound_kmh:
                                    kine_speed = calc_speed
                                    kine_dist = dist
                                    kine_gid = prev_ident.global_id
                                    break

                if (
                    kine_speed is not None
                    and kine_speed > self.alert_engine.velocity_bound_kmh
                ):
                    aid = generate_alert_id(
                        ALERT_VELOCITY_ANOMALY,
                        self.camera_id,
                        trk.last_timestamp,
                        f"KINE:{kine_gid}:{kine_speed:.1f}",
                    )
                    kine_alert = AlertRecord(
                        alert_id=aid,
                        alert_type=ALERT_VELOCITY_ANOMALY,
                        severity=SEVERITY_HIGH,
                        title=f"Diagnostic: Physical Velocity Bound Exceeded ({kine_speed:.1f} km/h)",
                        description=(
                            f"Observed speed {kine_speed:.1f} km/h exceeds plausibility bound "
                            f"({self.alert_engine.velocity_bound_kmh:.1f} km/h). Diagnostic flag."
                        ),
                        camera_id=self.camera_id,
                        timestamp=trk.last_timestamp,
                        iso_timestamp=format_iso_timestamp(trk.last_timestamp),
                        global_id=kine_gid,
                        canonical_plate=trk.canonical_plate,
                        details={
                            "transit_speed_kmh": round(kine_speed, 2),
                            "velocity_bound_kmh": self.alert_engine.velocity_bound_kmh,
                            "distance_km": kine_dist,
                        },
                    )
                    generated_alerts.append(kine_alert)

                self.alerts_triggered += len(generated_alerts)
            except Exception as alert_err:
                logger.error(f"[{self.camera_id}] Error in online alert evaluation: {alert_err}")

        if self.db_path:
            try:
                conn = get_thread_connection(self.db_path)
                save_global_identity(conn, identity)
                record_vehicle_observation(conn, obs, result, first_timestamp=trk.first_timestamp)
                for alert in generated_alerts:
                    record_security_alert_obj(conn, alert)
            except Exception as e:
                logger.error(f"[{self.camera_id}] Error saving identity/alerts to DB: {e}")

        # Dispatch callbacks
        if generated_alerts and self.on_alert_triggered:
            for alert in generated_alerts:
                try:
                    self.on_alert_triggered(self.camera_id, alert)
                except Exception as cb_err:
                    logger.error(f"[{self.camera_id}] Alert callback error: {cb_err}")

        if self.on_global_identity_resolved:
            try:
                self.on_global_identity_resolved(self.camera_id, identity, result)
            except Exception as cb_err:
                logger.error(f"[{self.camera_id}] Global identity callback error: {cb_err}")

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

        logger.info(f"[{self.camera_id}] Worker starting ({mode_desc}) -- source: {self.source}")

        self._camera = CameraSource(
            source=self.source,
            camera_id=self.camera_id,
            fps_target=self.fps_target,
            reconnect_max_retries=self.reconnect_max_retries,
            loop=self.loop,
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

                        # ── Stage: ANPR (Phase 4) ──
                        if self.anpr is not None:
                            for trk in active_tracks:
                                state = self.tracker.active_tracks.get(trk.track_id)
                                if state:
                                    read = self.anpr.process_track(
                                        frame,
                                        state,
                                        frame_number=cam.frames_read,
                                        timestamp=capture_ts,
                                    )
                                    if read:
                                        self.plates_detected += 1
                                        if self.on_plate_read:
                                            try:
                                                self.on_plate_read(self.camera_id, trk.track_id, read)
                                            except Exception as cb_err:
                                                logger.error(f"[{self.camera_id}] ANPR callback error: {cb_err}")

                        # ── Stage: ReID Feature Extraction (Phase 5) ──
                        if self.reid is not None:
                            for trk in active_tracks:
                                state = self.tracker.active_tracks.get(trk.track_id)
                                if state and state.best_vehicle_crop is not None:
                                    should_extract = False
                                    if len(state.reid_embeddings) == 0:
                                        should_extract = True
                                    elif (
                                        len(state.reid_embeddings) < 5
                                        and (
                                            state.best_crop_quality > (state.reid_qualities[-1] if state.reid_qualities else 0) * 1.15
                                            or (self.reid_every_n > 0 and cam.frames_read % self.reid_every_n == 0)
                                        )
                                    ):
                                        should_extract = True

                                    if should_extract:
                                        t_reid0 = time.perf_counter()
                                        emb = self.reid.extract_embedding(state.best_vehicle_crop)
                                        reid_lat = (time.perf_counter() - t_reid0) * 1000.0
                                        if emb is not None:
                                            state.update_reid(emb, state.best_vehicle_crop, state.best_crop_quality)
                                            self.reid_extractions += 1
                                            self._reid_latencies.append(reid_lat)
                                            if self.on_reid_extracted:
                                                try:
                                                    self.on_reid_extracted(self.camera_id, trk.track_id, emb)
                                                except Exception as cb_err:
                                                    logger.error(f"[{self.camera_id}] ReID callback error: {cb_err}")

                        # ── Stage: Finalized Track Identity Resolution (Phase 6B) ──
                        finalized_in_step = self.tracker.pop_finalized_tracks()
                        for trk_state in finalized_in_step:
                            self._process_finalized_track(trk_state)

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

                    else:
                        # Frame ingestion only mode
                        self.frames_processed += 1
                        self._proc_times.append(time.time())

                    self._log_stats_if_due()
                else:
                    if not self._running or self._camera is None:
                        break
                    consecutive_failures += 1

                    if not cam.is_stream:
                        if getattr(cam, "loop", False):
                            time.sleep(0.01)
                            continue
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
            # Process remaining finalized tracks through identity resolution
            rem_finalized = self.tracker.pop_finalized_tracks()
            for trk_state in rem_finalized:
                self._process_finalized_track(trk_state)

            metrics = self.tracker.get_metrics()
            reid_info = f", reid_extracted={self.reid_extractions}" if self.reid else ""
            id_info = f", identities_resolved={self.identities_resolved}" if self.identity_resolver else ""
            logger.info(
                f"[{self.camera_id}] Worker stopped. "
                f"read={read_cnt}, proc={self.frames_processed}, "
                f"tracks_created={metrics['tracks_created']}, "
                f"tracks_finalized={len(finalized)}, "
                f"avg_track_len={metrics['avg_track_length']:.1f}f, "
                f"in_fps={in_fps:.1f}, infer_fps={self.inference_fps:.1f}, "
                f"avg_latency={self.avg_latency_ms:.1f}ms{reid_info}{id_info}"
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

        if self.db_path:
            try:
                conn = get_thread_connection(self.db_path)
                update_camera_status(
                    conn=conn,
                    camera_id=self.camera_id,
                    status="offline",
                    fps=0.0,
                    latency_ms=self.avg_latency_ms,
                    last_seen_ts=time.time(),
                    total_frames=self.frames_processed,
                    total_detections=self.vehicles_detected,
                )
            except Exception as e:
                logger.debug(f"[{self.camera_id}] Error writing offline status to DB: {e}")

    def _log_stats_if_due(self) -> None:
        """Log performance and tracking statistics periodically."""
        now = time.time()
        if now - self._last_stats_time >= self._stats_interval:
            self._last_stats_time = now
            cam = self._camera
            if not cam:
                return

            if self.db_path:
                try:
                    conn = get_thread_connection(self.db_path)
                    update_camera_status(
                        conn=conn,
                        camera_id=self.camera_id,
                        status=self.status,
                        fps=self.input_fps,
                        latency_ms=self.avg_latency_ms,
                        last_seen_ts=now,
                        total_frames=self.frames_processed,
                        total_detections=self.vehicles_detected,
                    )
                except Exception as e:
                    logger.debug(f"[{self.camera_id}] Error writing telemetry to DB: {e}")

            if self.tracker is not None:
                metrics = self.tracker.get_metrics()
                plate_str = ""
                if self.anpr is not None:
                    canonical_plates = [
                        f"#{s.track_id}:{s.canonical_plate}"
                        for s in self.tracker.active_tracks.values()
                        if s.canonical_plate
                    ]
                    plate_str = f" | plates_read={self.plates_detected} | canonical={canonical_plates}"

                reid_str = ""
                if self.reid is not None:
                    reid_tracks = sum(1 for s in self.tracker.active_tracks.values() if s.best_reid_embedding is not None)
                    reid_str = f" | reid_extracted={self.reid_extractions} (embedded={reid_tracks})"

                logger.info(
                    f"[{self.camera_id}] "
                    f"status={self.status:<7} | "
                    f"input_fps={self.input_fps:4.1f} | "
                    f"infer_fps={self.inference_fps:4.1f} | "
                    f"latency={self.avg_latency_ms:5.1f}ms | "
                    f"active_tracks={metrics['active_tracks']:<2} | "
                    f"total_tracks={metrics['tracks_created']:<3} | "
                    f"avg_len={metrics['avg_track_length']:4.1f}f"
                    f"{plate_str}"
                    f"{reid_str}"
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
        anpr_enabled: bool = False,
        plate_model_path: str = "data/models/license_plate_yolov8_best.pt",
        ocr_every_n: int = 3,
        reid_enabled: bool = False,
        reid_weights: Optional[str] = None,
        reid_every_n: int = 15,
        identity_resolver: Optional[GlobalIdentityResolver] = None,
        resolve_identity: bool = False,
        db_path: Optional[str] = None,
        on_detections: Optional[Callable] = None,
        on_tracks: Optional[Callable] = None,
        on_plate_read: Optional[Callable] = None,
        on_reid_extracted: Optional[Callable] = None,
        on_global_identity_resolved: Optional[Callable] = None,
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
            anpr_enabled: Whether to attach VehicleANPR to tracked vehicles.
            plate_model_path: Path to plate detector weights.
            ocr_every_n: Process OCR every N frames per vehicle track.
            reid_enabled: Whether to attach VehicleReID feature extractor.
            reid_weights: Optional path to custom ReID weights (.pth / .pt).
            reid_every_n: Process ReID every N frames per vehicle track.
            identity_resolver: Shared GlobalIdentityResolver instance.
            resolve_identity: If True, resolves global identity for finalized tracks.
            db_path: Optional path to SQLite database.
            on_detections: Optional detection callback.
            on_tracks: Optional tracking callback.
            on_plate_read: Optional ANPR plate read callback.
            on_reid_extracted: Optional ReID extraction callback.
            on_global_identity_resolved: Optional global identity callback.
        """
        cameras = self.load_cameras(config_path)
        if not cameras:
            logger.error("No cameras found in config.")
            return

        shared_anpr = None
        if anpr_enabled:
            logger.info(f"Initializing VehicleANPR with plate model: {plate_model_path}...")
            shared_anpr = VehicleANPR(
                plate_model_path=plate_model_path,
                device=device,
                ocr_every_n=ocr_every_n,
            )

        shared_reid = None
        if reid_enabled:
            logger.info("Initializing VehicleReID feature extractor...")
            shared_reid = VehicleReID(weights_path=reid_weights, device=device)

        shared_resolver = identity_resolver
        if resolve_identity and shared_resolver is None:
            logger.info("Initializing GlobalIdentityResolver for multi-camera tracking...")
            shared_resolver = GlobalIdentityResolver()

        if db_path and resolve_identity:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            init_db(db_path)

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
            if tracker_type or anpr_enabled or reid_enabled or resolve_identity:
                t_type = tracker_type or "bytetrack.yaml"
                worker_tracker = VehicleTracker(
                    model_path=model_path,
                    tracker_type=t_type,
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
                anpr=shared_anpr,
                reid=shared_reid,
                reid_every_n=reid_every_n,
                identity_resolver=shared_resolver,
                db_path=db_path,
                on_detections=on_detections,
                on_tracks=on_tracks,
                on_plate_read=on_plate_read,
                on_reid_extracted=on_reid_extracted,
                on_global_identity_resolved=on_global_identity_resolved,
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
        if anpr_enabled:
            mode_str = f"Full ANPR Pipeline (Tracking + Plate Detector + OCR)"
        elif tracker_type:
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
        description="Camera Worker -- multi-camera ingestion, vehicle detection & tracking"
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

    # Phase 4: ANPR Pipeline flags
    parser.add_argument(
        "--anpr",
        action="store_true",
        help="Enable full ANPR pipeline (Vehicle Tracking + Plate Detection + OCR)",
    )
    parser.add_argument(
        "--plate-model",
        type=str,
        default="data/models/license_plate_yolov8_best.pt",
        help="Path to plate detector model weights",
    )
    parser.add_argument(
        "--ocr-every-n",
        type=int,
        default=3,
        help="Process OCR every N frames per vehicle track (default: 3)",
    )

    # Phase 5: ReID Feature Extraction flags
    parser.add_argument(
        "--reid",
        action="store_true",
        help="Enable vehicle appearance ReID feature extraction (Phase 5)",
    )
    parser.add_argument(
        "--reid-weights",
        type=str,
        default=None,
        help="Path to custom ReID model weights (.pth / .pt). Defaults to ImageNet ResNet-18 baseline",
    )
    parser.add_argument(
        "--reid-every-n",
        type=int,
        default=15,
        help="Extract ReID embedding at most every N frames per track on crop improvement (default: 15)",
    )

    # Phase 6B: Global Identity Resolution flags
    parser.add_argument(
        "--resolve-identity",
        action="store_true",
        help="Enable multi-camera global vehicle identity resolution and database persistence (Phase 6B)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/alpr.db",
        help="Path to SQLite database for persisting global vehicle identities (default: data/alpr.db)",
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
    anpr = None
    reid = None
    identity_resolver = None

    if args.resolve_identity:
        logger.info("Initializing GlobalIdentityResolver for multi-camera tracking...")
        identity_resolver = GlobalIdentityResolver()
        Path(args.db_path).parent.mkdir(parents=True, exist_ok=True)
        init_db(args.db_path)

    if args.reid:
        logger.info("Initializing VehicleReID feature extractor...")
        reid = VehicleReID(
            weights_path=args.reid_weights,
            device=args.device,
        )

    if args.anpr:
        logger.info(f"Initializing full ANPR Pipeline (Tracking + Plate Detection + OCR)...")
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
            anpr = VehicleANPR(
                plate_model_path=args.plate_model,
                device=args.device,
                ocr_every_n=args.ocr_every_n,
            )
    elif args.track or args.reid or args.resolve_identity:
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
            anpr=anpr,
            reid=reid,
            reid_every_n=args.reid_every_n,
            identity_resolver=identity_resolver,
            db_path=args.db_path if args.resolve_identity else None,
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
            tracker_type=args.tracker_type if (args.track or args.anpr or args.reid or args.resolve_identity) else None,
            model_path=args.model,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            anpr_enabled=args.anpr,
            plate_model_path=args.plate_model,
            ocr_every_n=args.ocr_every_n,
            reid_enabled=args.reid,
            reid_weights=args.reid_weights,
            reid_every_n=args.reid_every_n,
            identity_resolver=identity_resolver,
            resolve_identity=args.resolve_identity,
            db_path=args.db_path if args.resolve_identity else None,
        )




if __name__ == "__main__":
    main()
