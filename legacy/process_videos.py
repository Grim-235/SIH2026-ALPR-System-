#!/usr/bin/env python
"""Process multiple camera video feeds and store detections in the database."""
import argparse
import json
from pathlib import Path
from datetime import datetime, timedelta
import cv2

from alpr.detector import load_detector, resolve_device, ensure_model, DEFAULT_MODEL_PATH
from alpr.ocr import load_ocr
from alpr.tracker import process_video_with_tracking
from alpr.database import (
    init_db, load_cameras_from_json, load_blacklist_from_file,
    insert_detection, check_blacklist, insert_alert
)

def main():
    parser = argparse.ArgumentParser(description="Multi-camera batch processor.")
    parser.add_argument("--cameras", default="cameras.json", help="path to cameras.json")
    parser.add_argument("--db", default="data/alpr.db", help="path to SQLite DB")
    parser.add_argument("--blacklist", default="blacklist.txt", help="path to blacklist file")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="model path")
    parser.add_argument("--download-model", action="store_true", help="download model if missing")
    parser.add_argument("--device", default="auto", help="device to use")
    parser.add_argument("--conf", type=float, default=0.35, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IOU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="image size")
    parser.add_argument("--ocr-every-n", type=int, default=3, help="OCR frequency")
    parser.add_argument("--max-frames", type=int, default=0, help="max frames to process")
    parser.add_argument("--output-dir", default="results/tracked", help="output directory for tracked videos")
    args = parser.parse_args()

    # Load cameras config
    with open(args.cameras, 'r') as f:
        cameras = json.load(f)

    # Init DB
    conn = init_db(args.db)
    load_cameras_from_json(conn, args.cameras)
    load_blacklist_from_file(conn, args.blacklist)

    # Load models
    model_path = Path(args.model)
    ensure_model(model_path, args.download_model)
    device = resolve_device(args.device)
    model = load_detector(model_path)
    reader = load_ocr("easyocr", device)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    total_detections = 0
    total_alerts = 0
    
    for cam in cameras:
        cam_id = cam.get("camera_id")
        video_path = cam.get("video")
        
        if not video_path:
            continue
            
        print(f"Processing camera {cam_id} with video {video_path}")
        
        start_time_str = cam.get("start_time")
        if start_time_str:
            start_time = datetime.fromisoformat(start_time_str)
        else:
            start_time = datetime.now()
            
        # Get FPS
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video {video_path}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30.0 # fallback
        cap.release()
        
        out_video_path = str(Path(args.output_dir) / f"{cam_id}_tracked.mp4")
        
        detections = process_video_with_tracking(
            model=model,
            reader=reader,
            video_path=video_path,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=device,
            ocr_every_n=args.ocr_every_n,
            max_frames=args.max_frames,
            show=False,
            output_video=out_video_path
        )
        
        cam_det_count = 0
        cam_alert_count = 0
        
        for det in detections:
            timestamp = start_time + timedelta(seconds=det.first_frame / fps)
            timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            
            insert_detection(
                conn, 
                det.plate_text, 
                cam_id, 
                timestamp_str, 
                det.detection_confidence, 
                det.plate_confidence, 
                det.bbox, 
                det.track_id, 
                det.first_frame
            )
            cam_det_count += 1
            
            reason = check_blacklist(conn, det.plate_text)
            if reason:
                insert_alert(conn, det.plate_text, cam_id, timestamp_str, reason)
                print(f"WARNING: Blacklisted plate {det.plate_text} detected at camera {cam_id} - {reason}")
                cam_alert_count += 1
                
        print(f"Camera {cam_id} summary: {cam_det_count} detections, {cam_alert_count} alerts.")
        total_detections += cam_det_count
        total_alerts += cam_alert_count
        
    print(f"Overall summary: {total_detections} total detections, {total_alerts} total alerts.")
    conn.commit()
    conn.close()

if __name__ == "__main__":
    main()
