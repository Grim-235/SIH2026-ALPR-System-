from __future__ import annotations

import argparse
import os
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


from alpr.detector import DEFAULT_MODEL_PATH, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, load_detector, detect_plates, resolve_device, ensure_model, Detection
from alpr.ocr import load_ocr, recognize_plate, is_probable_indian_plate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Modern license plate detection and OCR for images, videos, and webcam."
    )
    parser.add_argument("--source", default="inputs/1.jpg", help="Image/video path, folder, or webcam index.")
    parser.add_argument("--output", default="results/modern_output.jpg", help="Output file or folder.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="YOLO plate detector .pt file.")
    parser.add_argument("--download-model", action="store_true", help="Download the default detector if missing.")
    parser.add_argument("--conf", type=float, default=0.35, help="Minimum plate detection confidence.")
    parser.add_argument("--iou", type=float, default=0.5, help="YOLO NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or another torch device.")
    parser.add_argument("--ocr", choices=["easyocr", "none"], default="easyocr", help="OCR backend.")
    parser.add_argument("--show", action="store_true", help="Show live preview window.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop video/webcam after N frames; 0 means no limit.")
    return parser.parse_args()



def draw_detection(frame: np.ndarray, box: tuple[int, int, int, int, float], text: str, ocr_conf: float) -> None:
    x1, y1, x2, y2, det_conf = box
    valid = is_probable_indian_plate(text) if text else False
    color = (36, 186, 77) if valid else (0, 170, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{text or 'plate'} det:{det_conf:.2f}"
    if text:
        label += f" ocr:{ocr_conf:.2f}"
    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    label_y = max(22, y1 - 8)
    cv2.rectangle(frame, (x1, label_y - text_size[1] - 8), (x1 + text_size[0] + 8, label_y + 4), color, -1)
    cv2.putText(frame, label, (x1 + 4, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)


def process_frame(model, reader, frame: np.ndarray, args: argparse.Namespace, device: str | None) -> list[dict]:
    detections = []
    for box in detect_plates(model, frame, args, device):
        x1, y1, x2, y2, det_conf = box
        crop = frame[y1:y2, x1:x2]
        text, ocr_conf = recognize_plate(reader, crop)
        draw_detection(frame, box, text, ocr_conf)
        detections.append(
            {
                "box": [x1, y1, x2, y2],
                "detection_confidence": round(det_conf, 4),
                "text": text,
                "ocr_confidence": round(ocr_conf, 4),
                "matches_indian_plate_pattern": is_probable_indian_plate(text) if text else False,
            }
        )
    return detections


def is_webcam_source(source: str) -> bool:
    return source.isdigit()


def is_image_source(source: str) -> bool:
    return Path(source).suffix.lower() in IMAGE_EXTENSIONS


def image_output_path(source: Path, output: Path, multiple: bool) -> Path:
    if output.suffix and not multiple:
        return output
    output.mkdir(parents=True, exist_ok=True)
    return output / f"{source.stem}_modern{source.suffix}"


def process_image(model, reader, source: Path, output: Path, args: argparse.Namespace, device: str | None, multiple: bool = False) -> None:
    frame = cv2.imread(str(source))
    if frame is None:
        raise RuntimeError(f"Could not read image: {source}")
    detections = process_frame(model, reader, frame, args, device)
    destination = image_output_path(source, output, multiple)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), frame)
    print(f"{source}: {len(detections)} plate(s) -> {destination}")
    for detection in detections:
        print(f"  {detection}")
    if args.show:
        cv2.imshow("Modern ALPR", frame)
        cv2.waitKey(0)


def process_video(model, reader, source: str, output: Path, args: argparse.Namespace, device: str | None) -> None:
    capture_source = int(source) if is_webcam_source(source) else source
    cap = cv2.VideoCapture(capture_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 1 else 20.0

    if output.suffix:
        destination = output
    else:
        output.mkdir(parents=True, exist_ok=True)
        suffix = "webcam" if is_webcam_source(source) else Path(source).stem
        destination = output / f"{suffix}_modern.mp4"
    destination.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    frame_count = 0
    started = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1
        process_frame(model, reader, frame, args, device)
        writer.write(frame)
        if args.show:
            cv2.imshow("Modern ALPR", frame)
            if cv2.waitKey(1) == 27:
                break
        if args.max_frames and frame_count >= args.max_frames:
            break

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    elapsed = max(0.001, time.time() - started)
    print(f"Processed {frame_count} frame(s) at {frame_count / elapsed:.2f} FPS -> {destination}")


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)
    ensure_model(model_path, args.download_model)
    device = resolve_device(args.device)
    print(f"Using device: {device or 'default'}")
    detector = load_detector(model_path)
    reader = load_ocr(args.ocr, device)

    source = args.source
    output = Path(args.output)
    source_path = Path(source)

    if source_path.is_dir():
        images = sorted(path for path in source_path.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            raise RuntimeError(f"No supported images found in {source_path}")
        for image in images:
            process_image(detector, reader, image, output, args, device, multiple=True)
    elif is_webcam_source(source) or source_path.suffix.lower() in VIDEO_EXTENSIONS:
        process_video(detector, reader, source, output, args, device)
    elif is_image_source(source):
        process_image(detector, reader, source_path, output, args, device)
    else:
        raise RuntimeError(f"Unsupported source: {source}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
