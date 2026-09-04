"""
Vehicle ANPR Pipeline (Phase 4).

Connects tracked vehicles to license plate detection and OCR:
Vehicle Track -> Vehicle Crop -> YOLO Plate Detector -> Quality Gate -> EasyOCR -> Consensus.
"""

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from alpr.detector import (
    DEFAULT_PLATE_MODEL_PATH,
    resolve_device,
)
from alpr.ocr import (
    PlateQualityGate,
    assess_plate_quality,
    load_ocr,
    recognize_plate,
    is_probable_indian_plate,
)
from alpr.tracker import PlateRead, VehicleTrackState


class VehicleANPR:
    """
    ANPR Engine that operates strictly on vehicle crops.

    1. Extracts vehicle crop from full frame.
    2. Runs fine-tuned YOLO plate detector on vehicle crop.
    3. Maps plate bbox back to full-frame coordinates.
    4. Evaluates plate crop quality (dimensions, sharpness, blur).
    5. Runs EasyOCR on quality-passing crops.
    6. Accumulates observations onto VehicleTrackState for consensus.
    """

    def __init__(
        self,
        plate_model_path: str | Path = DEFAULT_PLATE_MODEL_PATH,
        device: str = "auto",
        conf: float = 0.45,
        iou: float = 0.5,
        imgsz: int = 320,
        quality_gate: Optional[PlateQualityGate] = None,
        ocr_every_n: int = 3,
        enable_ocr: bool = True,
    ):
        """
        Initialize the VehicleANPR engine.

        Args:
            plate_model_path: Path to fine-tuned plate YOLO detector weights.
            device: Compute device ('auto', 'cpu', 'cuda:0').
            conf: Minimum confidence for plate detector.
            iou: NMS IoU threshold for plate detector.
            imgsz: Inference size for plate detector (typically 320 or 640).
            quality_gate: PlateQualityGate configuration.
            ocr_every_n: Process OCR every N frames per track to conserve CPU.
            enable_ocr: Whether to run OCR (can be disabled for detector-only benchmarking).
        """
        self.plate_model_path = Path(plate_model_path)
        self.device = resolve_device(device)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.quality_gate = quality_gate or PlateQualityGate()
        self.ocr_every_n = ocr_every_n
        self.enable_ocr = enable_ocr
        self._lock = threading.Lock()

        # Frame cadence counters per track: track_id -> frame_count
        self._track_cadence: Dict[int, int] = {}

        # Load Plate YOLO detector
        from ultralytics import YOLO

        model_str = str(self.plate_model_path)
        if not self.plate_model_path.exists():
            if Path("data/models/license_plate_yolov8_best.pt").exists():
                model_str = "data/models/license_plate_yolov8_best.pt"
            elif Path("data/models/yolov8n.pt").exists():
                model_str = "data/models/yolov8n.pt"

        self.plate_model = YOLO(model_str)
        self.plate_model_name = Path(model_str).name

        # Load EasyOCR reader if enabled
        self.ocr_reader = None
        if self.enable_ocr:
            self.ocr_reader = load_ocr("easyocr", self.device)

    def should_process_track(self, track_id: int) -> bool:
        """Determines whether OCR should be attempted for this track on this frame."""
        count = self._track_cadence.get(track_id, 0)
        self._track_cadence[track_id] = count + 1
        return (count % self.ocr_every_n) == 0

    def process_track(
        self,
        frame: np.ndarray,
        track_state: VehicleTrackState,
        frame_number: int,
        timestamp: float,
        force: bool = False,
    ) -> Optional[PlateRead]:
        """
        Run ANPR on a tracked vehicle.

        Args:
            frame: Full video frame.
            track_state: VehicleTrackState for the target vehicle.
            frame_number: Current frame sequence counter.
            timestamp: Capture timestamp.
            force: If True, bypasses ocr_every_n cadence check.

        Returns:
            Optional[PlateRead]: Newly created PlateRead if successful, else None.
        """
        if frame is None or frame.size == 0:
            return None

        # Check cadence unless forced
        if not force and not self.should_process_track(track_state.track_id):
            return None

        bbox = track_state.latest_bbox
        if bbox is None:
            return None

        vx1, vy1, vx2, vy2 = bbox
        vh, vw = frame.shape[:2]

        # Clamp vehicle bbox to frame
        vx1, vy1 = max(0, vx1), max(0, vy1)
        vx2, vy2 = min(vw - 1, vx2), min(vh - 1, vy2)

        if (vx2 - vx1) < 30 or (vy2 - vy1) < 20:
            return None

        # 1. Extract vehicle crop
        vehicle_crop = frame[vy1:vy2, vx1:vx2]
        if vehicle_crop.size == 0:
            return None

        # 2. Run plate detector strictly on vehicle crop
        with self._lock:
            results = self.plate_model.predict(
                source=vehicle_crop,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return None

        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        best_read: Optional[PlateRead] = None

        for box, det_conf in zip(xyxy, confs):
            det_conf_val = float(det_conf)
            px1, py1, px2, py2 = (int(v) for v in box.astype(int))

            # Clamp plate box inside vehicle crop
            crop_h, crop_w = vehicle_crop.shape[:2]
            px1, py1 = max(0, px1), max(0, py1)
            px2, py2 = min(crop_w - 1, px2), min(crop_h - 1, py2)

            if px2 <= px1 or py2 <= py1:
                continue

            # 3. Map plate bbox back to full-frame coordinates
            full_plate_bbox = (vx1 + px1, vy1 + py1, vx1 + px2, vy1 + py2)

            # 4. Extract plate crop
            plate_crop = vehicle_crop[py1:py2, px1:px2]
            if plate_crop.size == 0:
                continue

            # 5. Quality Gate
            quality = assess_plate_quality(plate_crop)
            if not self.quality_gate.passes(quality, det_conf_val):
                continue

            # 6. Run OCR if enabled
            if self.ocr_reader is not None:
                text, ocr_conf = recognize_plate(self.ocr_reader, plate_crop)
                if text:
                    is_valid = is_probable_indian_plate(text)
                    read = PlateRead(
                        text=text,
                        ocr_confidence=float(ocr_conf),
                        detector_confidence=det_conf_val,
                        quality_score=quality["quality_score"],
                        frame_number=frame_number,
                        timestamp=timestamp,
                        plate_bbox=full_plate_bbox,
                        is_valid_indian=is_valid,
                    )
                    # 7. Accumulate onto VehicleTrackState
                    track_state.add_plate_read(read, plate_crop=plate_crop)
                    best_read = read
                    break  # Found best plate for this frame

        return best_read
