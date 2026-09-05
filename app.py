#!/usr/bin/env python
"""Flask backend for ANPR Intelligence Dashboard"""
from flask import Flask, render_template, request, jsonify, send_file, Response
from flask_cors import CORS
import io
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
from alpr.service import get_dashboard_service

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Database
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "data" / "alpr.db"
CAMERAS_JSON = BASE_DIR / "configs" / "cameras.json" if (BASE_DIR / "configs" / "cameras.json").exists() else BASE_DIR / "cameras.json"
CAMERA_GRAPH_JSON = BASE_DIR / "configs" / "camera_graph.json"
BLACKLIST_FILE = BASE_DIR / "blacklist.txt"

def get_db():
    """Get database connection"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = init_db(str(DB_PATH))
    load_cameras_from_json(conn, str(CAMERAS_JSON))
    load_blacklist_from_file(conn, str(BLACKLIST_FILE))
    # Auto-cleanup orphaned jobs from previous server restarts
    cur = conn.cursor()
    cur.execute("UPDATE processing_jobs SET status = 'failed', error_message = 'Interrupted by server restart' WHERE status IN ('processing', 'pending')")
    conn.commit()
    return conn

conn = get_db()
dashboard_service = get_dashboard_service(
    db_path=DB_PATH,
    cameras_path=CAMERAS_JSON,
    camera_graph_path=CAMERA_GRAPH_JSON,
)

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

# ============================================================================
# PHASE 7E: SECURITY ALERTS & BLACKLIST ENFORCEMENT API
# ============================================================================

@app.route('/api/v1/alerts', methods=['GET'])
@app.route('/api/alerts', methods=['GET'])
def api_alerts_list():
    """Get filtered security alerts."""
    alert_type = request.args.get('type')
    severity = request.args.get('severity')
    only_unack = request.args.get('unacknowledged', 'false').lower() in ('true', '1')
    limit = int(request.args.get('limit', 100))
    alerts = dashboard_service.get_alerts(
        alert_type=alert_type,
        severity=severity,
        only_unacknowledged=only_unack,
        limit=limit,
        conn=conn,
    )
    return jsonify(alerts or [])

@app.route('/api/v1/alerts/summary', methods=['GET'])
@app.route('/api/alerts/summary', methods=['GET'])
def api_alerts_summary():
    """Get security alerts overview counts and breakdowns."""
    summary = dashboard_service.get_alerts_summary(conn)
    return jsonify(summary)

@app.route('/api/v1/alerts/<alert_id>/acknowledge', methods=['POST'])
@app.route('/api/alerts/<alert_id>/acknowledge', methods=['POST'])
def api_ack_alert(alert_id):
    """Acknowledge a security alert."""
    data = request.get_json(silent=True) or {}
    operator = data.get('operator', 'operator')
    success = dashboard_service.acknowledge_alert(alert_id, operator=operator, conn=conn)
    if not success and alert_id.isdigit():
        # Fallback for legacy numeric IDs
        acknowledge_alert(conn, int(alert_id))
        success = True
    return jsonify({'success': success, 'alert_id': alert_id})

@app.route('/api/v1/alerts/scan', methods=['POST'])
@app.route('/api/alerts/scan', methods=['POST'])
def api_alerts_scan():
    """Trigger on-demand scan of stored vehicle trajectories to evaluate alerts."""
    result = dashboard_service.scan_and_sync_alerts(conn)
    return jsonify(result)

@app.route('/api/v1/blacklist', methods=['GET'])
@app.route('/api/blacklist', methods=['GET'])
def api_blacklist():
    """Get watchlist / blacklist."""
    category = request.args.get('category')
    blacklist = dashboard_service.get_blacklist(category=category, conn=conn)
    # Return formatted payload supporting both v1 and legacy schema
    out = []
    for b in (blacklist or []):
        out.append({
            'plate': b['plate_text'],
            'plate_text': b['plate_text'],
            'category': b.get('category', 'CUSTOM'),
            'reason': b.get('reason', 'Flagged'),
            'severity': b.get('severity', 'HIGH'),
            'notes': b.get('notes'),
            'added_at': b.get('added_at'),
            'is_active': bool(b.get('is_active', 1)),
        })
    return jsonify(out)

@app.route('/api/v1/blacklist', methods=['POST'])
@app.route('/api/blacklist', methods=['POST'])
def api_add_blacklist():
    """Add or update an entry on the blacklist."""
    data = request.json or {}
    plate = data.get('plate') or data.get('plate_text')
    if not plate:
        return jsonify({'error': 'Plate number is required'}), 400
    category = data.get('category', 'CUSTOM')
    reason = data.get('reason', 'Flagged vehicle')
    severity = data.get('severity', 'HIGH')
    notes = data.get('notes')
    dashboard_service.add_to_blacklist(
        plate=plate,
        category=category,
        reason=reason,
        severity=severity,
        notes=notes,
        conn=conn,
    )
    return jsonify({'success': True, 'plate': plate.upper()})

@app.route('/api/v1/blacklist/<plate>', methods=['DELETE'])
@app.route('/api/blacklist/<plate>', methods=['DELETE'])
def api_remove_blacklist(plate):
    """Remove a plate from the watchlist / blacklist."""
    dashboard_service.remove_from_blacklist(plate, conn=conn)
    return jsonify({'success': True, 'plate': plate.upper()})


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
    
    # Spawn worker using active python interpreter (or local .venv if present)
    py_exec = sys.executable
    local_venv = Path(__file__).parent / ".venv" / "Scripts" / "python.exe"
    if local_venv.exists():
        py_exec = str(local_venv.resolve())

    worker_script_path = Path(__file__).parent / "worker.py"
    if not worker_script_path.exists():
        worker_script_path = Path(__file__).parent / "legacy" / "worker.py"
    worker_script = str(worker_script_path.resolve())
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

@app.route('/api/job/<job_id>/download', methods=['GET'])
def api_download_job_video(job_id):
    """Download output video for a completed job"""
    job = get_job_status(conn, job_id)
    if not job or not job.get('output_video'):
        return jsonify({'error': 'Job or output video not found'}), 404
    
    out_path = Path(job['output_video'])
    if not out_path.exists():
        out_path = BASE_DIR / job['output_video']
        if not out_path.exists():
            return jsonify({'error': 'Output video file not found on disk'}), 404
            
    return send_file(str(out_path.resolve()), as_attachment=True, mimetype='video/mp4')

@app.route('/api/job/<job_id>', methods=['DELETE'])
def api_delete_job(job_id):
    """Delete or cancel a specific job"""
    cursor = conn.cursor()
    cursor.execute("DELETE FROM processing_jobs WHERE job_id = ?", (job_id,))
    conn.commit()
    return jsonify({'status': 'deleted', 'job_id': job_id})

@app.route('/api/jobs/clear-all', methods=['DELETE'])
def api_clear_all_jobs():
    """Clear all jobs from queue"""
    cursor = conn.cursor()
    cursor.execute("DELETE FROM processing_jobs")
    conn.commit()
    return jsonify({'status': 'cleared'})

@app.route('/api/reset-data', methods=['POST'])
def api_reset_all_data():
    """Reset all database tables back to zero"""
    cursor = conn.cursor()
    cursor.execute("DELETE FROM detections")
    cursor.execute("DELETE FROM alerts")
    cursor.execute("DELETE FROM blacklist")
    cursor.execute("DELETE FROM processing_jobs")
    cursor.execute("DELETE FROM cameras")
    conn.commit()
    cursor.execute("VACUUM")
    load_cameras_from_json(conn, str(CAMERAS_JSON))
    return jsonify({'status': 'reset', 'message': 'All system data reset to zero'})

def _sanitize_dict(d: dict) -> dict:
    sanitized = {}
    for k, v in d.items():
        if isinstance(v, bytes):
            try:
                sanitized[k] = v.decode('utf-8', errors='ignore').strip('\x00')
            except Exception:
                sanitized[k] = str(v)
        else:
            sanitized[k] = v
    return sanitized

@app.route('/api/jobs', methods=['GET'])
def api_all_jobs():
    """Get all processing jobs"""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM processing_jobs ORDER BY created_at DESC LIMIT 20")
    rows = cursor.fetchall()
    return jsonify([_sanitize_dict(dict(r)) for r in rows])

@app.route('/api/recent-activity', methods=['GET'])
def api_recent_activity():
    """Get most recent detections"""
    recent = get_recent_detections(conn, limit=15)
    return jsonify(recent or [])

# ============================================================================
# PHASE 7D: GIS MAP, NETWORK ANALYTICS & VEHICLE SEARCH API (v1)
# ============================================================================

@app.route('/api/v1/gis/network-map', methods=['GET'])
@app.route('/api/gis/network-map', methods=['GET'])
def api_gis_network_map():
    """GeoJSON FeatureCollection representing camera nodes and corridors with congestion metrics."""
    geojson_data = dashboard_service.get_network_geojson(conn)
    return jsonify(geojson_data)

@app.route('/api/v1/gis/folium-map', methods=['GET'])
@app.route('/api/gis/folium-map', methods=['GET'])
@app.route('/api/map/overview', methods=['GET'])
def api_gis_folium_map():
    """Serve interactive Folium map with optional vehicle trajectory overlay."""
    vehicle_query = request.args.get('q') or request.args.get('vehicle_id') or request.args.get('plate')
    active_traj_geojson = None
    if vehicle_query:
        status_code, result = dashboard_service.search_vehicle(vehicle_query, conn)
        if status_code == 200:
            active_traj_geojson = result.get('geojson')
    html = dashboard_service.get_folium_map_html(conn, active_trajectory_geojson=active_traj_geojson)
    return Response(html, mimetype='text/html')

@app.route('/api/v1/analytics/summary', methods=['GET'])
@app.route('/api/analytics/summary', methods=['GET'])
def api_analytics_summary():
    """Network throughput, average TTI, analysis window, and modal flow breakdown."""
    summary_data = dashboard_service.get_analytics_summary(conn)
    return jsonify(summary_data)

@app.route('/api/v1/analytics/corridors', methods=['GET'])
@app.route('/api/analytics/corridors', methods=['GET'])
def api_analytics_corridors():
    """Corridor transit metrics, speeds (median, P05, P95), TTI, degradation, and LOS proxy."""
    corridors = dashboard_service.get_corridors_metrics(conn)
    return jsonify(corridors)

@app.route('/api/v1/analytics/hotspots', methods=['GET'])
@app.route('/api/analytics/hotspots', methods=['GET'])
def api_analytics_hotspots():
    """Congestion hotspots ordered by severity."""
    hotspots = dashboard_service.get_hotspots(conn)
    return jsonify(hotspots)

@app.route('/api/v1/analytics/od-matrix', methods=['GET'])
@app.route('/api/analytics/od-matrix', methods=['GET'])
def api_analytics_od_matrix():
    """Trip-level origin-destination (first camera -> last camera) flow records."""
    od_records = dashboard_service.get_od_matrix(conn)
    return jsonify(od_records)

@app.route('/api/v1/analytics/time-windows', methods=['GET'])
@app.route('/api/analytics/time-windows', methods=['GET'])
def api_analytics_time_windows():
    """Departure-time traffic performance profiles."""
    time_windows = dashboard_service.get_time_windows(conn)
    return jsonify(time_windows)

@app.route('/api/v1/vehicles/search', methods=['GET'])
@app.route('/api/vehicles/search', methods=['GET'])
def api_vehicle_search():
    """
    Search for vehicle by canonical license plate OR global vehicle ID (GV-XXXXXX).
    Returns 200 (found), 404 (absent), or 400 (empty/invalid).
    """
    q = request.args.get('q', '').strip()
    status_code, result = dashboard_service.search_vehicle(q, conn)
    return jsonify(result), status_code

@app.route('/api/v1/vehicles/trajectory/<identifier>', methods=['GET'])
@app.route('/api/vehicles/trajectory/<identifier>', methods=['GET'])
@app.route('/api/map/trajectory/<identifier>', methods=['GET'])
def api_vehicle_trajectory(identifier):
    """Retrieve trajectory details, sighting hops, and GeoJSON for a specific vehicle."""
    if request.path.startswith('/api/map/trajectory/'):
        status_code, result = dashboard_service.search_vehicle(identifier, conn)
        active_traj_geojson = result.get('geojson') if status_code == 200 else None
        html = dashboard_service.get_folium_map_html(conn, active_trajectory_geojson=active_traj_geojson)
        return Response(html, mimetype='text/html')

    status_code, result = dashboard_service.search_vehicle(identifier, conn)
    return jsonify(result), status_code

# ============================================================================
# PHASE 8: SYSTEM HEALTH & LIVE TELEMETRY REST API
# ============================================================================

@app.route('/api/v1/system/health', methods=['GET'])
@app.route('/api/system/health', methods=['GET'])
def api_system_health():
    """Overall system health status, active worker counts, and throughput metrics."""
    health = dashboard_service.get_system_health(conn=conn)
    return jsonify(health)

@app.route('/api/v1/system/cameras', methods=['GET'])
@app.route('/api/system/cameras', methods=['GET'])
def api_system_cameras():
    """Live camera node statuses, FPS, latency, and detection counts."""
    cameras = dashboard_service.get_camera_statuses(conn=conn)
    return jsonify(cameras)

@app.route('/api/cameras', methods=['GET'])
def api_cameras():
    """Get all cameras with live status and telemetry."""
    cameras = dashboard_service.get_camera_statuses(conn=conn)
    return jsonify(cameras)

@app.route('/api/cameras', methods=['POST'])
def api_create_camera():
    """Create new camera"""
    data = request.json
    cam_id = f"CAM_{data['name'].upper().replace(' ', '_')[:10]}"
    upsert_camera(conn, cam_id, data['name'], data['lat'], data['lon'], data.get('description'))
    return jsonify({'camera_id': cam_id, 'success': True})

# ============================================================================
# PHASE 9B: EVIDENCE DOSSIER & CRYPTOGRAPHIC MANIFEST REST API
# ============================================================================

@app.route('/api/v1/evidence/alerts/<alert_id>', methods=['GET'])
@app.route('/api/evidence/alerts/<alert_id>', methods=['GET'])
def api_evidence_alert(alert_id):
    """Retrieve structured evidence record and SHA-256 manifest for a security alert."""
    record = dashboard_service.get_alert_evidence(alert_id, conn=conn)
    if not record:
        return jsonify({'error': f'Alert not found or no evidence available for: {alert_id}'}), 404
    return jsonify(record.to_dict())

@app.route('/api/v1/evidence/alerts/<alert_id>/download', methods=['GET'])
@app.route('/api/evidence/alerts/<alert_id>/download', methods=['GET'])
def api_evidence_alert_download(alert_id):
    """Download evidence dossier in PDF, JSON, or CSV format."""
    record = dashboard_service.get_alert_evidence(alert_id, conn=conn)
    if not record:
        return jsonify({'error': f'Alert not found or no evidence available for: {alert_id}'}), 404
    fmt = request.args.get('format', 'pdf').lower().strip()
    content, mimetype, filename = dashboard_service.export_dossier(record, export_format=fmt)
    if isinstance(content, bytes):
        return send_file(io.BytesIO(content), mimetype=mimetype, as_attachment=True, download_name=filename)
    return Response(content, mimetype=mimetype, headers={'Content-Disposition': f'attachment; filename="{filename}"'})

@app.route('/api/v1/evidence/vehicles/<identifier>', methods=['GET'])
@app.route('/api/evidence/vehicles/<identifier>', methods=['GET'])
def api_evidence_vehicle(identifier):
    """Retrieve multi-camera trajectory evidence record and SHA-256 manifest for a vehicle."""
    # Resolve plate or global ID
    target_gid = identifier
    if not identifier.startswith('GV-'):
        veh = get_global_vehicle_by_plate(conn, identifier)
        if veh:
            target_gid = veh['global_id']
        else:
            return jsonify({'error': f'Vehicle not found: {identifier}'}), 404

    record = dashboard_service.get_vehicle_evidence(target_gid, conn=conn)
    if not record:
        return jsonify({'error': f'No trajectory evidence found for vehicle: {identifier}'}), 404
    return jsonify(record.to_dict())

@app.route('/api/v1/evidence/vehicles/<identifier>/download', methods=['GET'])
@app.route('/api/evidence/vehicles/<identifier>/download', methods=['GET'])
def api_evidence_vehicle_download(identifier):
    """Download vehicle trajectory dossier in PDF, JSON, or CSV format."""
    target_gid = identifier
    if not identifier.startswith('GV-'):
        veh = get_global_vehicle_by_plate(conn, identifier)
        if veh:
            target_gid = veh['global_id']
        else:
            return jsonify({'error': f'Vehicle not found: {identifier}'}), 404

    record = dashboard_service.get_vehicle_evidence(target_gid, conn=conn)
    if not record:
        return jsonify({'error': f'No trajectory evidence found for vehicle: {identifier}'}), 404
    fmt = request.args.get('format', 'pdf').lower().strip()
    content, mimetype, filename = dashboard_service.export_dossier(record, export_format=fmt)
    if isinstance(content, bytes):
        return send_file(io.BytesIO(content), mimetype=mimetype, as_attachment=True, download_name=filename)
    return Response(content, mimetype=mimetype, headers={'Content-Disposition': f'attachment; filename="{filename}"'})

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False, host='127.0.0.1', port=5000)
