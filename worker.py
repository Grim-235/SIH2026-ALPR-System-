#!/usr/bin/env python
"""Background worker for processing videos asynchronously."""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import cv2
from alpr.database import (
    check_blacklist,
    get_job,
    init_db,
    insert_alert,
    insert_detection,
    mark_job_completed,
    mark_job_failed,
    mark_job_started,
    update_job_progress,
)
from alpr.detector import (
    DEFAULT_MODEL_PATH,
    ensure_model,
    load_detector,
    resolve_device,
)
from alpr.ocr import load_ocr
from alpr.tracker import process_video_with_tracking


def process_video_job(
    job_id: str,
    db_path: str,
    video_path: str,
    camera_id: str,
    conf: float = 0.35,
    iou: float = 0.5,
    imgsz: int = 640,
    device: str = "auto",
    ocr_every_n: int = 3,
    max_frames: int = 0,
    output_dir: str = "results/tracked",
):
    """
    Process a video file and store results in the database.
    Updates job progress as it goes.
    """
    try:
        print(f"[{job_id}] Starting video processing...")

        # Connect to database
        conn = init_db(db_path)
        job = None
        for _ in range(5):
            job = get_job(conn, job_id)
            if job:
                break
            import time
            time.sleep(0.5)

        if not job:
            raise ValueError(f"Job {job_id} not found in database")

        # Mark as started
        mark_job_started(conn, job_id)
        print(f"[{job_id}] Marked as started")

        # Verify video exists
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Get video info
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        frames_to_process = max_frames if max_frames > 0 else total_frames

        print(f"[{job_id}] Video info: {total_frames} frames, {fps:.1f} FPS")
        update_job_progress(conn, job_id, 5, 0)

        # Load models
        print(f"[{job_id}] Loading models...")
        ensure_model(DEFAULT_MODEL_PATH, download=True)
        resolved_device = resolve_device(device)
        model = load_detector(DEFAULT_MODEL_PATH)
        reader = load_ocr("easyocr", resolved_device)
        print(f"[{job_id}] Models loaded on device: {resolved_device}")
        update_job_progress(conn, job_id, 15, 0)

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        out_video = str(
            output_path / f"{camera_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        )

        # Process video
        print(f"[{job_id}] Processing video with tracking...")
        detections = process_video_with_tracking(
            model=model,
            reader=reader,
            video_path=video_path,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=resolved_device,
            ocr_every_n=ocr_every_n,
            max_frames=max(0, max_frames),
            show=False,
            output_video=out_video,
        )

        print(f"[{job_id}] Found {len(detections)} unique vehicles")
        update_job_progress(conn, job_id, 85, len(detections))

        # Store detections in database
        print(f"[{job_id}] Storing detections in database...")
        start_time = datetime.now()
        alert_count = 0

        for det in detections:
            ts = (start_time + timedelta(seconds=det.first_frame / fps)).strftime('%Y-%m-%d %H:%M:%S')
            insert_detection(
                conn,
                det.plate_text,
                camera_id,
                ts,
                det.detection_confidence,
                det.plate_confidence,
                det.bbox,
                det.track_id,
                det.first_frame,
            )
            reason = check_blacklist(conn, det.plate_text)
            if reason:
                insert_alert(conn, det.plate_text, camera_id, ts, reason)
                alert_count += 1

        print(f"[{job_id}] {alert_count} blacklisted vehicles detected")
        update_job_progress(conn, job_id, 95, len(detections))

        # Mark as complete
        mark_job_completed(conn, job_id, out_video, len(detections))
        print(f"[{job_id}] Job completed successfully!")
        update_job_progress(conn, job_id, 100, len(detections))

        conn.close()
        return True

    except Exception as e:
        print(f"[{job_id}] ERROR: {e!s}")
        try:
            conn = init_db(db_path)
            mark_job_failed(conn, job_id, str(e))
            conn.close()
        except:
            pass
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Background video processing worker")
    parser.add_argument("--job-id", required=True, help="Job ID to process")
    parser.add_argument("--db", default="data/alpr.db", help="Database path")
    parser.add_argument("--video", required=True, help="Video file path")
    parser.add_argument("--camera", required=True, help="Camera ID")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence")
    parser.add_argument("--iou", type=float, default=0.5, help="IOU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", default="auto", help="Device")
    parser.add_argument("--ocr-every-n", type=int, default=3, help="OCR frequency")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames")
    parser.add_argument(
        "--output-dir", default="results/tracked", help="Output directory"
    )

    args = parser.parse_args()

    success = process_video_job(
        job_id=args.job_id,
        db_path=args.db,
        video_path=args.video,
        camera_id=args.camera,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        ocr_every_n=args.ocr_every_n,
        max_frames=args.max_frames,
        output_dir=args.output_dir,
    )

    sys.exit(0 if success else 1)
