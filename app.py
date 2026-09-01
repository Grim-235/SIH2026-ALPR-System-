#!/usr/bin/env python
"""Flask backend for ANPR Intelligence Dashboard"""
from flask import Flask, render_template, request, jsonify, send_file, Response
from flask_cors import CORS
import json
import tempfile
import subprocess
import uuid
import sys
from pathlib import Path
from datetime import datetime, timedelta
import cv2

from alpr.detector import load_detector, resolve_device, ensure_model, DEFAULT_MODEL_PATH
from alpr.ocr import load_ocr
from alpr.tracker import process_video_with_tracking
from alpr.database import (
    init_db,
    load_cameras_from_json,
    load_blacklist_from_file,
    upsert_camera,
    get_detection_stats,
    get_recent_detections,
    get_detections_over_time,
    get_camera_heatmap_data,
    get_all_plates,
    query_plate_history,
    get_top_routes,
    get_blacklist,
    get_alerts,
    acknowledge_alert,
    add_to_blacklist,
    remove_from_blacklist,
    check_blacklist,
    insert_alert,
    insert_detection,
    create_job,
    get_job_status,
)

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Database
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "data" / "alpr.db"
CAMERAS_JSON = BASE_DIR / "cameras.json"
BLACKLIST_FILE = BASE_DIR / "blacklist.txt"

def get_db():
    """Get database connection"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = init_db(str(DB_PATH))
    load_cameras_from_json(conn, str(CAMERAS_JSON))
    load_blacklist_from_file(conn, str(BLACKLIST_FILE))
    return conn

conn = get_db()

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/stats', methods=['GET'])
def api_stats():
    """Get dashboard statistics"""
    stats = get_detection_stats(conn)
    alerts = get_alerts(conn, only_unacknowledged=True)
    return jsonify({
        'total_detections': stats.get('total_detections', 0),
        'unique_plates': stats.get('unique_plates', 0),
        'unique_cameras': stats.get('unique_cameras', 0),
        'unacknowledged_alerts': len(alerts),
        'min_time': stats.get('min_time'),
        'max_time': stats.get('max_time'),
    })

@app.route('/api/detections-timeline', methods=['GET'])
def api_detections_timeline():
    """Get detections over time"""
    time_data = get_detections_over_time(conn, bucket_minutes=5)
    return jsonify(time_data or [])

@app.route('/api/camera-heatmap', methods=['GET'])
def api_camera_heatmap():
    """Get camera detection heatmap data"""
    heatmap_data = get_camera_heatmap_data(conn)
    return jsonify(heatmap_data or [])

@app.route('/api/plates', methods=['GET'])
def api_plates():
    """Get all plates in database"""
    all_plates = get_all_plates(conn)
    return jsonify(all_plates or [])

@app.route('/api/plate/<plate>', methods=['GET'])
def api_plate_history(plate):
    """Get history for a specific plate"""
    history = query_plate_history(conn, plate.upper())
    bl_status = check_blacklist(conn, plate.upper())
    return jsonify({
        'plate': plate.upper(),
        'history': history,
        'sightings': len(history),
        'cameras': len(set(h['camera_id'] for h in history)) if history else 0,
        'blacklist_reason': bl_status,
    })

@app.route('/api/routes', methods=['GET'])
def api_routes():
    """Get top routes"""
    routes = get_top_routes(conn)
    return jsonify(routes or [])

@app.route('/api/alerts', methods=['GET'])
def api_alerts_list():
    """Get all alerts"""
    alerts = get_alerts(conn)
    return jsonify(alerts or [])

@app.route('/api/alerts/<int:alert_id>/acknowledge', methods=['POST'])
def api_ack_alert(alert_id):
    """Acknowledge an alert"""
    acknowledge_alert(conn, alert_id)
    return jsonify({'success': True})

@app.route('/api/blacklist', methods=['GET'])
def api_blacklist():
    """Get blacklist"""
    blacklist = get_blacklist(conn)
    return jsonify([{'plate': b['plate_text'], 'reason': b.get('reason', 'Flagged')} for b in blacklist] if blacklist else [])

@app.route('/api/blacklist', methods=['POST'])
def api_add_blacklist():
    """Add to blacklist"""
    data = request.json
    add_to_blacklist(conn, data['plate'].upper(), data.get('reason', 'Flagged'))
    return jsonify({'success': True})

@app.route('/api/blacklist/<plate>', methods=['DELETE'])
def api_remove_blacklist(plate):
    """Remove from blacklist"""
    remove_from_blacklist(conn, plate.upper())
    return jsonify({'success': True})

@app.route('/api/upload-video', methods=['POST'])
def api_upload_video():
    """Upload and process video"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400
    
    video_file = request.files['video']
    camera_id = request.form.get('camera_id')
    conf = float(request.form.get('conf', 0.35))
    ocr_n = int(request.form.get('ocr_n', 3))
    max_frames = int(request.form.get('max_frames', 0))
    
    # Save uploaded file
    uploads_dir = Path("data/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = str(uploads_dir / f"{str(uuid.uuid4())[:8]}.mp4")
    video_file.save(tmp_path)
    
    # Get video info
    cap = cv2.VideoCapture(tmp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    
    # Create job
    job_id = str(uuid.uuid4())[:8]
    create_job(conn, job_id, camera_id, tmp_path, total_frames)
    
    # Spawn worker using available python interpreter (preferring virtualenv if present)
    py_exec = sys.executable
    possible_venvs = [
        Path(__file__).parent / ".venv" / "Scripts" / "python.exe",
        Path(__file__).parent.parent / "Modern-Indian-ALPR" / ".venv" / "Scripts" / "python.exe",
        Path("d:/SIH2026/Modern-Indian-ALPR/.venv/Scripts/python.exe"),
    ]
    for venv_py in possible_venvs:
        if venv_py.exists():
            py_exec = str(venv_py.resolve())
            break

    worker_script = str((Path(__file__).parent / "worker.py").resolve())
    db_abs_path = str(DB_PATH.resolve())
    video_abs_path = str(Path(tmp_path).resolve())
    logs_dir = Path("data/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = logs_dir / f"{job_id}.log"

    cmd = [
        py_exec, worker_script,
        "--job-id", job_id,
        "--db", db_abs_path,
        "--video", video_abs_path,
        "--camera", camera_id,
        "--conf", str(conf),
        "--ocr-every-n", str(ocr_n),
        "--max-frames", str(max_frames),
    ]
    log_f = open(log_file_path, "a")
    subprocess.Popen(cmd, stdout=log_f, stderr=log_f, close_fds=False)
    
    return jsonify({'job_id': job_id, 'status': 'pending'})

@app.route('/api/job/<job_id>', methods=['GET'])
def api_job_status(job_id):
    """Get job status"""
    job_status = get_job_status(conn, job_id)
    if not job_status:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(job_status)

@app.route('/api/recent-activity', methods=['GET'])
def api_recent_activity():
    """Get most recent detections"""
    recent = get_recent_detections(conn, limit=15)
    return jsonify(recent or [])

@app.route('/api/job/<job_id>/download', methods=['GET'])
def api_download_job_video(job_id):
    """Download output tracked video for a completed job"""
    job = get_job_status(conn, job_id)
    if not job or not job.get('output_video'):
        return jsonify({'error': 'Video output not ready or job not found'}), 404
    video_path = Path(job['output_video'])
    if not video_path.exists():
        return jsonify({'error': 'Video file not found on server'}), 404
    return send_file(video_path, as_attachment=True, download_name=video_path.name)

@app.route('/api/map/overview', methods=['GET'])
def api_map_overview():
    """Generate and serve interactive Folium overview map"""
    try:
        from trajectory import generate_overview_map
        overview_map = generate_overview_map(conn)
        if overview_map:
            html = overview_map.get_root().render()
            return Response(html, mimetype='text/html')
        return "<h3>Map unavailable — No camera location data</h3>", 404
    except ImportError:
        return "<h3>Folium library not installed (pip install folium)</h3>", 404
    except Exception as e:
        return f"<h3>Map generator error: {e}</h3>", 404

@app.route('/api/map/trajectory/<plate>', methods=['GET'])
def api_map_trajectory(plate):
    """Generate and serve interactive Folium plate trajectory map"""
    try:
        from trajectory import generate_trajectory_map
        traj_map = generate_trajectory_map(conn, plate.upper())
        if traj_map:
            html = traj_map.get_root().render()
            return Response(html, mimetype='text/html')
        return f"<h3>No trajectory data for plate {plate}</h3>", 404
    except ImportError:
        return "<h3>Folium library not installed (pip install folium)</h3>", 404
    except Exception as e:
        return f"<h3>Map generator error: {e}</h3>", 404

@app.route('/api/cameras', methods=['GET'])
def api_cameras():
    """Get all cameras"""
    try:
        with open("cameras.json", "r") as f:
            cameras = json.load(f)
        return jsonify(cameras)
    except:
        return jsonify([])

@app.route('/api/cameras', methods=['POST'])
def api_create_camera():
    """Create new camera"""
    data = request.json
    cam_id = f"CAM_{data['name'].upper().replace(' ', '_')[:10]}"
    upsert_camera(conn, cam_id, data['name'], data['lat'], data['lon'], data.get('description'))
    return jsonify({'camera_id': cam_id, 'success': True})

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False, host='127.0.0.1', port=5000)
