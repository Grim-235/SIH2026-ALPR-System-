import re

import cv2
import numpy as np

from dataclasses import dataclass
from alpr.detector import PROJECT_CACHE_DIR


@dataclass
class PlateQualityGate:
    """Configurable quality thresholds for accepting a plate crop for OCR."""
    min_width: int = 80
    min_height: int = 20
    min_confidence: float = 0.45
    min_sharpness: float = 20.0
    min_quality_score: float = 0.20

    def passes(self, quality: dict, detector_confidence: float) -> bool:
        """Check if a plate detection passes quality criteria."""
        if detector_confidence < self.min_confidence:
            return False
        if quality["width"] < self.min_width or quality["height"] < self.min_height:
            return False
        if quality["sharpness"] < self.min_sharpness:
            return False
        if quality["quality_score"] < self.min_quality_score:
            return False
        return True


def assess_plate_quality(crop: np.ndarray) -> dict:
    """
    Assess the visual quality of a license plate crop image.

    Returns:
        dict: width, height, aspect_ratio, sharpness, brightness,
              contrast, is_blurry, quality_score
    """
    if crop is None or crop.size == 0:
        return {
            "width": 0,
            "height": 0,
            "aspect_ratio": 0.0,
            "sharpness": 0.0,
            "brightness": 0.0,
            "contrast": 0.0,
            "is_blurry": True,
            "quality_score": 0.0,
        }

    h, w = crop.shape[:2]
    aspect_ratio = float(w) / float(max(1, h))

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop

    # Sharpness via Laplacian variance
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Brightness & contrast
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    # Normalized component scores (0.0 to 1.0)
    size_score = min(1.0, (w * h) / (160.0 * 40.0))
    sharp_score = min(1.0, sharpness / 100.0)
    contrast_score = min(1.0, contrast / 50.0)

    quality_score = 0.40 * size_score + 0.40 * sharp_score + 0.20 * contrast_score
    is_blurry = sharpness < 25.0

    return {
        "width": w,
        "height": h,
        "aspect_ratio": round(aspect_ratio, 2),
        "sharpness": round(sharpness, 2),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "is_blurry": is_blurry,
        "quality_score": round(quality_score, 3),
    }


INDIAN_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DL", "DN", "GA", "GJ", "HP", "HR",
    "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "PB",
    "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB",
}
STATE_CODE_OCR_FIXES = {
    "HH": "MH",
}
DIGIT_FIXES = str.maketrans({"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8"})
LETTER_FIXES = str.maketrans({"0": "D", "1": "I", "2": "Z", "4": "A", "5": "S", "8": "B"})


def load_ocr(name: str, device: str | None):
    if name == "none":
        return None
    try:
        import easyocr
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install easyocr first, or run with --ocr none.") from exc
    gpu = bool(device and device.startswith("cuda"))
    model_dir = PROJECT_CACHE_DIR / "easyocr"
    model_dir.mkdir(parents=True, exist_ok=True)
    return easyocr.Reader(["en"], gpu=gpu, model_storage_directory=str(model_dir))


def clean_plate_text(text: str) -> str:
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    if len(text) >= 4:
        text = text[:-4] + text[-4:].translate(DIGIT_FIXES)
    return text


def is_probable_indian_plate(text: str) -> bool:
    if re.match(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$", text):
        return True
    match = re.match(r"^([A-Z]{2})[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$", text)
    return bool(match and match.group(1) in INDIAN_STATE_CODES)


def normalize_state_code(state_code: str) -> str:
    if state_code in INDIAN_STATE_CODES:
        return state_code
    return STATE_CODE_OCR_FIXES.get(state_code, state_code)


def normalize_plate_layout(text: str) -> list[str]:
    text = clean_plate_text(text)
    candidates = [text]

    if len(text) == 10:
        normalized = (
            normalize_state_code(text[:2].translate(LETTER_FIXES))
            + text[2:4].translate(DIGIT_FIXES)
            + text[4:6].translate(LETTER_FIXES)
            + text[6:10].translate(DIGIT_FIXES)
        )
        candidates.append(normalized)
    elif len(text) == 9:
        normalized = (
            normalize_state_code(text[:2].translate(LETTER_FIXES))
            + text[2:3].translate(DIGIT_FIXES)
            + text[3:5].translate(LETTER_FIXES)
            + text[5:9].translate(DIGIT_FIXES)
        )
        candidates.append(normalized)
    elif len(text) == 11 and text[2:4].upper() == "BH":
        candidates.append(text[:2].translate(DIGIT_FIXES) + "BH" + text[4:8].translate(DIGIT_FIXES) + text[8:].translate(LETTER_FIXES))

    return list(dict.fromkeys(candidates))


def enhance_plate_crop(crop: np.ndarray) -> np.ndarray:
    if crop.size == 0:
        return crop
    scale = max(2, int(180 / max(1, crop.shape[0])))
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 45, 45)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def recognize_plate(reader, crop: np.ndarray) -> tuple[str, float]:
    if reader is None:
        return "", 0.0
    candidates: list[tuple[str, float, int]] = []
    for source_index, image in enumerate((crop, enhance_plate_crop(crop))):
        results = reader.readtext(
            image,
            detail=1,
            paragraph=False,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        )
        for _, text, confidence in results:
            for cleaned in normalize_plate_layout(text):
                if cleaned:
                    candidates.append((cleaned, float(confidence), source_index))
    if not candidates:
        return "", 0.0
    candidates.sort(
        key=lambda item: (
            is_probable_indian_plate(item[0]),
            item[2] == 0,
            item[1],
            len(item[0]),
        ),
        reverse=True,
    )
    return candidates[0][0], candidates[0][1]
