import cv2
import numpy as np
from dataclasses import dataclass, field
from collections import Counter
from typing import List, Tuple, Dict, Optional, Any

from alpr.ocr import recognize_plate, is_probable_indian_plate

@dataclass
class TrackState:
    """Accumulated state for one tracked vehicle."""
    track_id: int
    ocr_reads: List[Tuple[str, float]]  # (plate_text, ocr_confidence)
    best_bbox: Optional[Tuple[int, int, int, int]] = None  # bbox with highest det conf
    best_det_conf: float = 0.0
    first_frame: int = 0
    last_frame: int = 0
    frame_count: int = 0

@dataclass 
class FinalizedDetection:
    """One unique vehicle passage with consensus plate text."""
    track_id: int
    plate_text: str
    plate_confidence: float
    detection_confidence: float
    bbox: Tuple[int, int, int, int]
    first_frame: int
    last_frame: int
    total_frames: int

class TrackAggregator:
    """Accumulates detections per track ID, finalizes with best plate text."""
    
    def __init__(self, ocr_every_n: int = 3, min_ocr_conf: float = 0.3):
        self.tracks: Dict[int, TrackState] = {}
        self.ocr_every_n = ocr_every_n  # run OCR every N frames per track
        self.min_ocr_conf = min_ocr_conf
        self._ocr_counters: Dict[int, int] = {}  # track_id -> frames since last OCR
    
    def should_run_ocr(self, track_id: int) -> bool:
        """Returns True if we should run OCR for this track on this frame."""
        count = self._ocr_counters.get(track_id, 0)
        self._ocr_counters[track_id] = count + 1
        return count % self.ocr_every_n == 0
    
    def update(self, track_id: int, bbox: Tuple[int,int,int,int], det_conf: float, 
               plate_text: Optional[str], ocr_conf: float, frame_idx: int):
        """Update track with new detection. plate_text can be None if OCR was skipped."""
        if track_id not in self.tracks:
            self.tracks[track_id] = TrackState(
                track_id=track_id, ocr_reads=[], first_frame=frame_idx
            )
        state = self.tracks[track_id]
        state.last_frame = frame_idx
        state.frame_count += 1
        
        if det_conf > state.best_det_conf:
            state.best_det_conf = det_conf
            state.best_bbox = bbox
            
        if plate_text and ocr_conf >= self.min_ocr_conf:
            state.ocr_reads.append((plate_text, ocr_conf))
    
    def finalize(self, track_id: int) -> Optional[FinalizedDetection]:
        """Finalize a track and return the best plate text via consensus."""
        state = self.tracks.get(track_id)
        if not state or not state.ocr_reads:
            return None
        
        # Prefer valid Indian plates
        valid_reads = [(t, c) for t, c in state.ocr_reads if is_probable_indian_plate(t)]
        reads_to_use = valid_reads if valid_reads else state.ocr_reads
        
        # Majority vote weighted by confidence
        text_scores: Dict[str, float] = {}
        for text, conf in reads_to_use:
            text_scores[text] = text_scores.get(text, 0) + conf
        
        best_text = max(text_scores, key=text_scores.get)
        best_conf = max(c for t, c in reads_to_use if t == best_text)
        
        return FinalizedDetection(
            track_id=state.track_id,
            plate_text=best_text,
            plate_confidence=best_conf,
            detection_confidence=state.best_det_conf,
            bbox=state.best_bbox, # type: ignore
            first_frame=state.first_frame,
            last_frame=state.last_frame,
            total_frames=state.frame_count,
        )
    
    def finalize_all(self) -> List[FinalizedDetection]:
        """Finalize all tracks."""
        results = []
        for track_id in list(self.tracks):
            det = self.finalize(track_id)
            if det:
                results.append(det)
        return results
    
    def get_stale_tracks(self, current_frame: int, max_age: int = 30) -> List[int]:
        """Return track IDs that haven't been seen for max_age frames."""
        return [
            tid for tid, state in self.tracks.items()
            if current_frame - state.last_frame > max_age
        ]

def draw_tracked_detection(frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                           track_id: int, plate_text: Optional[str], 
                           det_conf: float, ocr_conf: float):
    """Draw bounding box with track ID and plate text on frame."""
    x1, y1, x2, y2 = bbox
    
    # Generate a consistent color based on track_id
    np.random.seed(track_id % 10000) 
    color = tuple(int(c) for c in np.random.randint(0, 255, 3))
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Draw label
    label = f"ID: {track_id}"
    if plate_text:
        label += f" | {plate_text} ({ocr_conf:.2f})"
        
    (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.rectangle(frame, (x1, y1 - label_h - baseline - 10), (x1 + label_w, y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


def process_video_with_tracking(
    model: Any,           # ultralytics YOLO model
    reader: Any,          # EasyOCR reader (or None)
    video_path: str,
    conf: float = 0.35,
    iou: float = 0.5,
    imgsz: int = 640,
    device: Optional[str] = None,
    ocr_every_n: int = 3,
    max_frames: int = 0,
    show: bool = False,
    output_video: Optional[str] = None,
) -> List[FinalizedDetection]:
    """
    Process a video file with tracking enabled.
    Returns list of FinalizedDetections (one per unique vehicle).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video at {video_path}")
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30.0
        
    video_writer = None
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
        video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
    aggregator = TrackAggregator(ocr_every_n=ocr_every_n, min_ocr_conf=0.3)
    finalized_results: List[FinalizedDetection] = []
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        if max_frames > 0 and frame_idx > max_frames:
            break
            
        # Run tracking
        results = model.track(
            frame, 
            persist=True, 
            conf=conf, 
            iou=iou, 
            imgsz=imgsz, 
            device=device, 
            tracker='bytetrack.yaml', 
            verbose=False
        )
        
        display_frame = frame.copy() if (show or output_video) else None
        
        if len(results) > 0:
            result = results[0]
            if result.boxes and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()
                
                for bbox, track_id, det_conf in zip(boxes, track_ids, confs):
                    x1, y1, x2, y2 = bbox
                    
                    plate_text = None
                    ocr_conf = 0.0
                    
                    if aggregator.should_run_ocr(track_id):
                        if reader is not None:
                            # Clamp bbox
                            x1_c = max(0, x1)
                            y1_c = max(0, y1)
                            x2_c = min(width, x2)
                            y2_c = min(height, y2)
                            
                            if x2_c > x1_c and y2_c > y1_c:
                                crop = frame[y1_c:y2_c, x1_c:x2_c]
                                try:
                                    plate_text, ocr_conf = recognize_plate(reader, crop)
                                except Exception as e:
                                    print(f"OCR error on frame {frame_idx}: {e}")
                                    plate_text, ocr_conf = None, 0.0
                    
                    aggregator.update(
                        track_id=track_id,
                        bbox=(x1, y1, x2, y2),
                        det_conf=float(det_conf),
                        plate_text=plate_text,
                        ocr_conf=ocr_conf,
                        frame_idx=frame_idx
                    )
                    
                    if display_frame is not None:
                        # get best known plate text from aggregator state to display
                        state = aggregator.tracks[track_id]
                        disp_text = plate_text
                        disp_conf = ocr_conf
                        if disp_text is None and state.ocr_reads:
                            # Just show the most recent OCR read for display
                            disp_text, disp_conf = state.ocr_reads[-1]
                            
                        draw_tracked_detection(display_frame, (x1, y1, x2, y2), track_id, disp_text, det_conf, disp_conf)
        
        # Check for stale tracks
        stale_ids = aggregator.get_stale_tracks(frame_idx, max_age=30)
        for tid in stale_ids:
            det = aggregator.finalize(tid)
            if det:
                finalized_results.append(det)
            del aggregator.tracks[tid]
            
        if display_frame is not None:
            if output_video and video_writer:
                video_writer.write(display_frame)
            if show:
                cv2.imshow("Tracking", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames...")
            
    # Finalize remaining tracks
    finalized_results.extend(aggregator.finalize_all())
    
    cap.release()
    if video_writer:
        video_writer.release()
    if show:
        cv2.destroyAllWindows()
        
    return finalized_results
