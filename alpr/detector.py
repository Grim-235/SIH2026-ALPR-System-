import argparse
import os
import urllib.request
from pathlib import Path
from typing import NamedTuple

import numpy as np


MODEL_URL = (
    "https://huggingface.co/yasirfaizahmed/license-plate-object-detection/"
    "resolve/main/best.pt?download=true"
)
DEFAULT_MODEL_PATH = Path("data/models/license_plate_yolov8_best.pt")
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm", ".wmv"}


PROJECT_CACHE_DIR = Path.cwd() / ".cache"
for cache_name in ("ultralytics", "matplotlib", "torch"):
    (PROJECT_CACHE_DIR / cache_name).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_CACHE_DIR / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_CACHE_DIR / "matplotlib"))
os.environ.setdefault("TORCH_HOME", str(PROJECT_CACHE_DIR / "torch"))


class Detection(NamedTuple):
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


def ensure_model(model_path: Path, download: bool) -> None:
    if model_path.exists():
        return
    if not download:
        raise FileNotFoundError(
            f"Model not found: {model_path}. Run again with --download-model or pass --model path/to/best.pt"
        )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading plate detector to {model_path} ...")
    urllib.request.urlretrieve(MODEL_URL, model_path)


def resolve_device(device: str) -> str | None:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return None


def load_detector(model_path: Path):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install ultralytics first.") from exc
    return YOLO(str(model_path))


def detect_plates(model, frame: np.ndarray, args: argparse.Namespace, device: str | None) -> list[Detection]:
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
