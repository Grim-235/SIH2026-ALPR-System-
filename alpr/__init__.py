from alpr.detector import (
    load_detector,
    detect_plates,
    resolve_device,
    ensure_model,
    VehicleDetector,
    VehicleDetection,
    VEHICLE_CLASS_MAP,
)
from alpr.ocr import load_ocr, recognize_plate, is_probable_indian_plate
