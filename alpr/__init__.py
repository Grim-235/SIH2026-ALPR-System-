from alpr.detector import (
    load_detector,
    detect_plates,
    resolve_device,
    ensure_model,
    VehicleDetector,
    VehicleDetection,
    VEHICLE_CLASS_MAP,
)
from alpr.ocr import (
    load_ocr,
    recognize_plate,
    is_probable_indian_plate,
    PlateQualityGate,
    assess_plate_quality,
)
from alpr.tracker import (
    VehicleTracker,
    VehicleTrackState,
    ActiveVehicleTrack,
    PlateRead,
)
from alpr.anpr import VehicleANPR
from alpr.reid import (
    VehicleReID,
    extract_embedding,
    compute_similarity,
    aggregate_embeddings,
)
