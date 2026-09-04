"""
Phase 2 — Acceptance tests for Vehicle Detection.

Functional criteria:
- YOLO vehicle model loads successfully.
- Only car/motorcycle/bus/truck classes are detected.
- Bounding boxes are valid and inside frame boundaries.
- Confidence filtering works.
- Annotated frames can be produced.
- Latency in milliseconds is measured.
- Integration with CameraWorker works.
"""

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from alpr.detector import (
    VehicleDetector,
    VehicleDetection,
    VEHICLE_CLASS_MAP,
    VEHICLE_COLORS,
)
from workers.camera_worker import CameraWorker

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    status = "[PASS]" if condition else "[FAIL]"
    msg = f"  {status}: {name}"
    if detail:
        msg += f" - {detail}"
    print(msg)
    if condition:
        PASS += 1
    else:
        FAIL += 1


def main():
    print("\n" + "=" * 60)
    print("  Phase 2 — Vehicle Detection Acceptance Tests")
    print("=" * 60 + "\n")

    # ── Test 1: Detector Initialization ──
    print("[1] VehicleDetector initialization")
    detector = VehicleDetector(
        model_path="data/models/yolov8n.pt",
        conf=0.25,
        iou=0.45,
        device="auto",
    )
    check("VehicleDetector instantiates", detector is not None)
    check("Model loaded", detector.model is not None)
    check("Model name recorded", detector.model_name == "yolov8n.pt")
    check("Device recorded", detector.device in ("cpu", "cuda:0"))
    check("Target classes filter to 4 vehicle types", set(detector.classes) == {2, 3, 5, 7})

    # ── Test 2: Detection on Sample Image ──
    print("\n[2] Vehicle Detection on image")
    sample_img = cv2.imread("inputs/1.jpg")
    check("Sample image exists", sample_img is not None)

    detections, latency_ms = detector.detect(sample_img)
    check("Inference executes without error", True)
    check("Latency measured in ms", latency_ms > 0.0, f"latency={latency_ms:.1f}ms")
    check("Returns detection list", isinstance(detections, list))

    # Validate detection properties
    h, w = sample_img.shape[:2]
    all_valid_boxes = True
    all_valid_classes = True
    all_valid_conf = True

    for det in detections:
        # Check classes
        if det.class_id not in VEHICLE_CLASS_MAP or det.class_name not in VEHICLE_CLASS_MAP.values():
            all_valid_classes = False
        # Check coordinates inside bounds
        if not (0 <= det.x1 < det.x2 <= w and 0 <= det.y1 < det.y2 <= h):
            all_valid_boxes = False
        # Check confidence
        if det.confidence < 0.25:
            all_valid_conf = False

    check("Only car/motorcycle/bus/truck returned", all_valid_classes)
    check("Bounding boxes within image bounds", all_valid_boxes)
    check("Confidence above threshold", all_valid_conf)

    # ── Test 3: Confidence Filtering ──
    print("\n[3] Confidence threshold filtering")
    high_conf_detector = VehicleDetector(
        model_path="data/models/yolov8n.pt",
        conf=0.85,
        device="auto",
    )
    high_conf_dets, _ = high_conf_detector.detect(sample_img)
    check("High-conf detector returns <= dets than low-conf", len(high_conf_dets) <= len(detections))
    check(
        "All high-conf detections meet threshold",
        all(d.confidence >= 0.85 for d in high_conf_dets),
    )

    # ── Test 4: Frame Annotation ──
    print("\n[4] Frame annotation (draw_detections)")
    annotated = detector.draw_detections(sample_img, detections, copy=True)
    check("Annotated frame has same shape", annotated.shape == sample_img.shape)
    check("Annotated frame is numpy array", isinstance(annotated, np.ndarray))

    # ── Test 5: Synthetic/Edge Cases ──
    print("\n[5] Edge cases handling")
    empty_dets, empty_lat = detector.detect(None)
    check("None frame returns empty list", empty_dets == [] and empty_lat == 0.0)

    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    blank_dets, blank_lat = detector.detect(blank_frame)
    check("Blank frame processed without error", isinstance(blank_dets, list))
    check("Blank frame latency measured", blank_lat > 0.0)

    # ── Test 6: Worker Orchestration Integration ──
    print("\n[6] CameraWorker orchestration with detector")
    received_callbacks = []

    def on_det_cb(cam_id, frame, dets, lat, ts):
        received_callbacks.append((cam_id, len(dets), lat, ts))

    worker = CameraWorker(
        camera_id="TEST-VEHICLE-WORKER",
        source="inputs/1.jpg",
        fps_target=0,
        detector=detector,
        on_detections=on_det_cb,
    )
    check("Worker initialized with detector", worker.detector is not None)

    # Run worker on image file (processes 1 frame and completes)
    worker.start()

    check("Worker processed frame", worker.frames_processed >= 1)
    check("Worker recorded vehicles", worker.vehicles_detected >= 0)
    check("Worker recorded inference latency", worker.avg_latency_ms > 0.0)
    check("Detection callback fired", len(received_callbacks) >= 1)
    check("Callback received valid timestamp", received_callbacks[0][3] > 0.0)

    # ── Summary ──
    print(f"\n{'=' * 60}")
    total = PASS + FAIL
    print(f"  Results: {PASS}/{total} passed, {FAIL} failed")
    print(f"{'=' * 60}\n")

    return FAIL == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
