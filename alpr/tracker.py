"""
Single-Camera Vehicle Tracking with ByteTrack / BoT-SORT.

Tracks vehicle bounding boxes (car, motorcycle, bus, truck) across consecutive
video frames, maintaining persistent local_track_ids per camera and accumulating
vehicle track state (bbox history, confidence history, best vehicle crops).
"""

import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from alpr.detector import (
    VEHICLE_CLASS_MAP,
    VEHICLE_COLORS,
    resolve_device,
    DEFAULT_VEHICLE_MODEL_PATH,
)
from alpr.ocr import is_probable_indian_plate


@dataclass
class PlateRead:
    """Individual OCR observation for a tracked vehicle."""
    text: str
    ocr_confidence: float
    detector_confidence: float
    quality_score: float
    frame_number: int
    timestamp: float
    plate_bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in full-frame coordinates
    is_valid_indian: bool = False


@dataclass
class VehicleTrackState:
    """
    Accumulated state for one tracked vehicle within a single camera.
    """
    track_id: int
    camera_id: str
    vehicle_type: str

    first_frame: int
    last_frame: int
    first_timestamp: float
    last_timestamp: float

    bbox_history: List[Tuple[int, int, int, int]] = field(default_factory=list)
    confidence_history: List[float] = field(default_factory=list)

    frame_count: int = 0

    best_vehicle_crop: Optional[np.ndarray] = None
    best_crop_quality: float = 0.0

    # Phase 4: License plate recognition & consensus
    plate_reads: List[PlateRead] = field(default_factory=list)
    best_plate_crop: Optional[np.ndarray] = None
    best_plate_quality: float = 0.0
    canonical_plate: Optional[str] = None
    plate_confidence: float = 0.0

    def add_plate_read(
        self,
        read: PlateRead,
        plate_crop: Optional[np.ndarray] = None,
    ) -> None:
        """Add an OCR observation and update canonical plate consensus."""
        self.plate_reads.append(read)
        if plate_crop is not None and read.quality_score > self.best_plate_quality:
            self.best_plate_crop = plate_crop.copy()
            self.best_plate_quality = read.quality_score
        self.compute_plate_consensus()

    def compute_plate_consensus(self) -> Optional[Tuple[str, float]]:
        """
        Compute consensus plate text from multiple OCR reads using
        confidence-weighted majority voting with Indian plate preference.
        """
        if not self.plate_reads:
            return None

        # Prefer valid Indian plates if any exist
        valid_reads = [r for r in self.plate_reads if r.is_valid_indian or is_probable_indian_plate(r.text)]
        reads_to_use = valid_reads if valid_reads else self.plate_reads

        # Score text candidates: sum of (ocr_conf * quality_score)
        scores: Dict[str, float] = {}
        for r in reads_to_use:
            weight = r.ocr_confidence * max(0.2, r.quality_score)
            scores[r.text] = scores.get(r.text, 0.0) + weight

        best_text = max(scores, key=scores.get)
        matching_confs = [r.ocr_confidence for r in reads_to_use if r.text == best_text]
        best_conf = max(matching_confs) if matching_confs else 0.0

        self.canonical_plate = best_text
        self.plate_confidence = float(best_conf)
        return best_text, self.plate_confidence

    @property
    def latest_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """Most recent bounding box (x1, y1, x2, y2)."""
        return self.bbox_history[-1] if self.bbox_history else None

    @property
    def latest_confidence(self) -> float:
        """Most recent detection confidence."""
        return self.confidence_history[-1] if self.confidence_history else 0.0

    @property
    def avg_confidence(self) -> float:
        """Average confidence over all observations."""
        if not self.confidence_history:
            return 0.0
        return sum(self.confidence_history) / len(self.confidence_history)

    @property
    def track_duration_seconds(self) -> float:
        """Duration in seconds between first and last sighting."""
        return max(0.0, self.last_timestamp - self.first_timestamp)


@dataclass
class ActiveVehicleTrack:
    """Represents a vehicle track in the current frame."""
    track_id: int
    camera_id: str
    vehicle_type: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    frame_number: int
    timestamp: float


class VehicleTracker:
    """
    Single-camera vehicle tracker using ByteTrack or BoT-SORT via Ultralytics YOLO.
    Maintains track persistence, ID assignment, and crop accumulation.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_VEHICLE_MODEL_PATH,
        tracker_type: str = "bytetrack.yaml",
        conf: float = 0.35,
        iou: float = 0.5,
        imgsz: int = 640,
        device: str = "auto",
        camera_id: str = "CAM-001",
        max_age: int = 30,
        min_crop_size: Tuple[int, int] = (40, 40),
    ):
        """
        Initialize the VehicleTracker.

        Args:
            model_path: Path to YOLO weights (e.g. data/models/yolov8n.pt).
            tracker_type: Tracker config ('bytetrack.yaml' or 'botsort.yaml').
            conf: Minimum detection confidence threshold.
            iou: NMS IoU threshold.
            imgsz: Inference image size.
            device: Compute device ('auto', 'cpu', 'cuda:0').
            camera_id: Logical identifier for this camera feed.
            max_age: Number of consecutive unseen frames before track finalization.
            min_crop_size: Minimum (width, height) to consider vehicle crop.
        """
        self.camera_id = camera_id
        self.tracker_type = tracker_type
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = resolve_device(device)
        self.max_age = max_age
        self.min_crop_size = min_crop_size

        # Target only COCO vehicle classes: car(2), motorcycle(3), bus(5), truck(7)
        self.classes = list(VEHICLE_CLASS_MAP.keys())

        # Load Ultralytics YOLO model
        from ultralytics import YOLO

        model_str = str(model_path)
        if not Path(model_path).exists() and Path("yolov8n.pt").exists():
            model_str = "yolov8n.pt"
        self.model = YOLO(model_str)
        self.model_name = Path(model_str).name

        # State management
        self.active_tracks: Dict[int, VehicleTrackState] = {}
        self.finalized_tracks: List[VehicleTrackState] = []
        self._total_tracks_created = 0
        self.current_frame = 0

    def update(
        self,
        frame: np.ndarray,
        frame_number: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> List[ActiveVehicleTrack]:
        """
        Process a new video frame with tracking.

        Args:
            frame: Video frame (BGR numpy array).
            frame_number: Optional frame sequence counter (auto-increments if None).
            timestamp: Capture timestamp (defaults to current time if None).

        Returns:
            List[ActiveVehicleTrack]: Vehicles currently active in this frame.
        """
        if frame is None or frame.size == 0:
            return []

        if frame_number is not None:
            self.current_frame = frame_number
        else:
            self.current_frame += 1

        curr_ts = timestamp if timestamp is not None else time.time()
        h, w = frame.shape[:2]

        # Run Ultralytics tracking
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_type,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )

        active_in_frame: List[ActiveVehicleTrack] = []

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()
            cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, track_id, conf, cls_id in zip(boxes, track_ids, confs, cls_ids):
                track_id = int(track_id)
                vtype = VEHICLE_CLASS_MAP.get(int(cls_id), "car")

                x1, y1, x2, y2 = (int(v) for v in box.astype(int))
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                bbox = (x1, y1, x2, y2)
                conf_val = float(conf)

                # Compute crop quality heuristic: area * confidence * aspect ratio penalty
                bw = x2 - x1
                bh = y2 - y1
                crop_quality = (bw * bh) * conf_val

                # Extract vehicle crop if it meets minimum resolution
                crop = None
                if bw >= self.min_crop_size[0] and bh >= self.min_crop_size[1]:
                    crop = frame[y1:y2, x1:x2].copy()

                # Update or initialize track state
                if track_id not in self.active_tracks:
                    self._total_tracks_created += 1
                    self.active_tracks[track_id] = VehicleTrackState(
                        track_id=track_id,
                        camera_id=self.camera_id,
                        vehicle_type=vtype,
                        first_frame=self.current_frame,
                        last_frame=self.current_frame,
                        first_timestamp=curr_ts,
                        last_timestamp=curr_ts,
                        bbox_history=[bbox],
                        confidence_history=[conf_val],
                        frame_count=1,
                        best_vehicle_crop=crop,
                        best_crop_quality=crop_quality,
                    )
                else:
                    state = self.active_tracks[track_id]
                    state.last_frame = self.current_frame
                    state.last_timestamp = curr_ts
                    state.frame_count += 1
                    state.bbox_history.append(bbox)
                    state.confidence_history.append(conf_val)

                    # Update best crop if higher quality
                    if crop is not None and crop_quality > state.best_crop_quality:
                        state.best_vehicle_crop = crop
                        state.best_crop_quality = crop_quality

                active_in_frame.append(
                    ActiveVehicleTrack(
                        track_id=track_id,
                        camera_id=self.camera_id,
                        vehicle_type=vtype,
                        bbox=bbox,
                        confidence=conf_val,
                        frame_number=self.current_frame,
                        timestamp=curr_ts,
                    )
                )

        # Check and finalize stale tracks
        self.finalize_stale_tracks()

        return active_in_frame

    def finalize_stale_tracks(self) -> List[VehicleTrackState]:
        """
        Check for tracks that have disappeared for more than max_age frames
        and move them to finalized_tracks.

        Returns:
            List[VehicleTrackState]: Newly finalized tracks in this step.
        """
        stale_ids = [
            tid
            for tid, state in self.active_tracks.items()
            if (self.current_frame - state.last_frame) > self.max_age
        ]

        just_finalized = []
        for tid in stale_ids:
            state = self.active_tracks.pop(tid)
            self.finalized_tracks.append(state)
            just_finalized.append(state)

        return just_finalized

    def finalize_all(self) -> List[VehicleTrackState]:
        """Finalize all remaining active tracks (e.g. at end of stream/video)."""
        all_remaining = list(self.active_tracks.values())
        self.finalized_tracks.extend(all_remaining)
        self.active_tracks.clear()
        return all_remaining

    def reset(self) -> None:
        """Reset internal tracker state."""
        self.active_tracks.clear()
        self.finalized_tracks.clear()
        self._total_tracks_created = 0
        self.current_frame = 0

    # ── Performance & Tracking Metrics ──

    @property
    def total_tracks_created(self) -> int:
        """Total distinct vehicle track IDs generated."""
        return self._total_tracks_created

    @property
    def active_track_count(self) -> int:
        """Number of vehicles currently being actively tracked."""
        return len(self.active_tracks)

    @property
    def finalized_track_count(self) -> int:
        """Number of vehicle tracks completed/finalized."""
        return len(self.finalized_tracks)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Compute summary tracking metrics.

        Returns:
            dict containing:
            - tracks_created: total tracks assigned
            - active_tracks: currently active tracks
            - finalized_tracks: finalized tracks
            - avg_track_length: average frame count per track
            - median_track_length: median frame count per track
            - track_fragmentation: percentage of tracks lasting <= 2 frames
        """
        all_tracks = list(self.active_tracks.values()) + self.finalized_tracks
        if not all_tracks:
            return {
                "tracks_created": 0,
                "active_tracks": 0,
                "finalized_tracks": 0,
                "avg_track_length": 0.0,
                "median_track_length": 0.0,
                "track_fragmentation": 0.0,
            }

        lengths = [t.frame_count for t in all_tracks]
        short_tracks = sum(1 for l in lengths if l <= 2)

        return {
            "tracks_created": self._total_tracks_created,
            "active_tracks": len(self.active_tracks),
            "finalized_tracks": len(self.finalized_tracks),
            "avg_track_length": float(np.mean(lengths)),
            "median_track_length": float(np.median(lengths)),
            "track_fragmentation": (short_tracks / len(all_tracks)) * 100.0,
        }

    def draw_tracks(
        self,
        frame: np.ndarray,
        active_tracks: List[ActiveVehicleTrack],
        draw_trail: bool = True,
        trail_length: int = 15,
        copy: bool = False,
    ) -> np.ndarray:
        """
        Draw color-coded bounding boxes, persistent track IDs, class names,
        and centroid motion trails on the frame.
        """
        out = frame.copy() if copy else frame

        for track in active_tracks:
            x1, y1, x2, y2 = track.bbox
            color = VEHICLE_COLORS.get(track.vehicle_type, (0, 255, 0))

            # Bounding box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # Header label: VEHICLE_TYPE #ID | PLATE (confidence)
            state = self.active_tracks.get(track.track_id)
            if state and state.canonical_plate:
                label = f"{track.vehicle_type.upper()} #{track.track_id} | {state.canonical_plate} ({state.plate_confidence:.2f})"
            else:
                label = f"{track.vehicle_type.upper()} #{track.track_id} ({track.confidence:.2f})"
            (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                out,
                (x1, max(0, y1 - lh - baseline - 4)),
                (x1 + lw + 4, y1),
                color,
                -1,
            )
            cv2.putText(
                out,
                label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

            # Draw centroid trajectory trail
            if draw_trail and track.track_id in self.active_tracks:
                state = self.active_tracks[track.track_id]
                history = state.bbox_history[-trail_length:]
                if len(history) > 1:
                    points = [
                        (int((bx1 + bx2) / 2), int((by1 + by2) / 2))
                        for bx1, by1, bx2, by2 in history
                    ]
                    for i in range(1, len(points)):
                        thickness = int(np.sqrt(trail_length / float(i + 1)) * 2)
                        cv2.line(out, points[i - 1], points[i], color, max(1, thickness))

        return out


# ── Backward Compatibility Aliases & Helpers ──

@dataclass
class TrackState:
    """Legacy compatibility TrackState."""
    track_id: int
    ocr_reads: List[Tuple[str, float]] = field(default_factory=list)
    best_bbox: Optional[Tuple[int, int, int, int]] = None
    best_det_conf: float = 0.0
    first_frame: int = 0
    last_frame: int = 0
    frame_count: int = 0


@dataclass
class FinalizedDetection:
    """Legacy compatibility FinalizedDetection."""
    track_id: int
    plate_text: str
    plate_confidence: float
    detection_confidence: float
    bbox: Tuple[int, int, int, int]
    first_frame: int
    last_frame: int
    total_frames: int
