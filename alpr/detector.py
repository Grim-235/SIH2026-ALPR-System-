import argparse
import os
import threading
import time
import urllib.request
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import cv2
import numpy as np


# License plate detector model URL and defaults
MODEL_URL = (
    "https://huggingface.co/yasirfaizahmed/license-plate-object-detection/"
    "resolve/main/best.pt?download=true"
)
DEFAULT_PLATE_MODEL_PATH = Path("data/models/license_plate_yolov8_best.pt")
DEFAULT_MODEL_PATH = DEFAULT_PLATE_MODEL_PATH  # Backward compatibility
DEFAULT_VEHICLE_MODEL_PATH = Path("data/models/yolov8n.pt")

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm", ".wmv"}


PROJECT_CACHE_DIR = Path.cwd() / ".cache"
for cache_name in ("ultralytics", "matplotlib", "torch"):
    (PROJECT_CACHE_DIR / cache_name).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_CACHE_DIR / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_CACHE_DIR / "matplotlib"))
os.environ.setdefault("TORCH_HOME", str(PROJECT_CACHE_DIR / "torch"))


# COCO vehicle classes of interest
VEHICLE_CLASS_MAP = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Distinct BGR colors for visualization
VEHICLE_COLORS = {
    "car": (0, 255, 0),          # Green
    "motorcycle": (255, 255, 0),  # Cyan
    "bus": (0, 215, 255),        # Amber
    "truck": (255, 0, 255),      # Magenta
}


class Detection(NamedTuple):
    """Plate detection bounding box and confidence."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


class VehicleDetection(NamedTuple):
    """Vehicle detection with class label and coordinates."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str


def ensure_model(model_path: Path, download: bool) -> None:
    """Ensure a model file exists, optionally downloading it."""
    if model_path.exists():
        return
    if not download:
        raise FileNotFoundError(
            f"Model not found: {model_path}. Run again with --download-model or pass --model path/to/best.pt"
        )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading model to {model_path} ...")
    urllib.request.urlretrieve(MODEL_URL, model_path)


def resolve_device(device: str) -> str | None:
    """Resolve compute device string to 'cuda:0' or 'cpu'."""
    if device != "auto":
        return device
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def load_detector(model_path: Path):
    """Load an Ultralytics YOLO model from a weights file."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install ultralytics first.") from exc
    return YOLO(str(model_path))


class VehicleDetector:
    """
    Dedicated vehicle object detector wrapping YOLOv8 COCO models.
    Filters specifically for car, motorcycle, bus, and truck classes.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_VEHICLE_MODEL_PATH,
        conf: float = 0.35,
        iou: float = 0.5,
        imgsz: int = 640,
        device: str = "auto",
        classes: Optional[List[int]] = None,
    ):
        self.model_path = Path(model_path)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = resolve_device(device)
        self.classes = classes if classes is not None else list(VEHICLE_CLASS_MAP.keys())
        self._lock = threading.Lock()

        # Load YOLO model
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Missing dependency: install ultralytics first.") from exc

        # If model doesn't exist locally, check fallback in data/models or let ultralytics load
        model_str = str(self.model_path)
        if not self.model_path.exists():
            if Path("yolov8n.pt").exists():
                model_str = "yolov8n.pt"
            else:
                model_str = "yolov8n.pt"  # Ultralytics will auto-download

        self.model = YOLO(model_str)
        self.model_name = Path(model_str).name

    def detect(self, frame: np.ndarray) -> Tuple[List[VehicleDetection], float]:
        """
        Run vehicle detection on a single frame.

        Args:
            frame: Input image (BGR numpy array).

        Returns:
            Tuple[List[VehicleDetection], float]: List of detected vehicles
            and inference latency in milliseconds.
        """
        if frame is None or frame.size == 0:
            return [], 0.0

        t0 = time.perf_counter()
        with self._lock:
            results = self.model.predict(
                source=frame,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                classes=self.classes,
                device=self.device,
                verbose=False,
            )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if not results or results[0].boxes is None:
            return [], latency_ms

        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)

        h, w = frame.shape[:2]
        detections: List[VehicleDetection] = []

        for box, conf, cls_id in zip(xyxy, confs, cls_ids):
            if cls_id not in VEHICLE_CLASS_MAP:
                continue

            x1, y1, x2, y2 = (int(v) for v in box.astype(int))
            # Clamp to frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            if x2 > x1 and y2 > y1:
                detections.append(
                    VehicleDetection(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        confidence=float(conf),
                        class_id=int(cls_id),
                        class_name=VEHICLE_CLASS_MAP[cls_id],
                    )
                )

        return detections, latency_ms

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[VehicleDetection],
        copy: bool = False,
    ) -> np.ndarray:
        """
        Annotate frame with vehicle bounding boxes, class labels, and confidence.
        """
        out = frame.copy() if copy else frame
        for det in detections:
            color = VEHICLE_COLORS.get(det.class_name, (0, 255, 0))
            # Bounding box
            cv2.rectangle(out, (det.x1, det.y1), (det.x2, det.y2), color, 2)
            # Label
            label = f"{det.class_name.upper()} {det.confidence:.2f}"
            (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                out,
                (det.x1, max(0, det.y1 - lh - baseline - 4)),
                (det.x1 + lw + 4, det.y1),
                color,
                -1,
            )
            cv2.putText(
                out,
                label,
                (det.x1 + 2, det.y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        return out


def detect_plates(
    model,
    frame: np.ndarray,
    args: argparse.Namespace,
    device: str | None,
) -> list[Detection]:
    """Run plate detection on a frame using a fine-tuned plate YOLO model."""
    results = model.predict(
        source=frame,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=device,
        verbose=False,
    )
    if not results or results[0].boxes is None:
        return []
    boxes = results[0].boxes
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    h, w = frame.shape[:2]
    detections = []
    for box, conf in zip(xyxy, confs):
        x1, y1, x2, y2 = (int(value) for value in box.astype(int))
        x1, y1 = int(max(0, x1)), int(max(0, y1))
        x2, y2 = int(min(w - 1, x2)), int(min(h - 1, y2))
        if x2 > x1 and y2 > y1:
            detections.append(Detection(x1, y1, x2, y2, float(conf)))
    return detections
