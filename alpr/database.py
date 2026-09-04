import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


def init_db(db_path: str | Path) -> sqlite3.Connection:
    """Initialize the database schema and return a connection."""
    conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=30.0)
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA busy_timeout=30000;")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cameras (
            camera_id   TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            latitude    REAL NOT NULL,
            longitude   REAL NOT NULL,
            description TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_text      TEXT NOT NULL,
            camera_id       TEXT NOT NULL,
            timestamp       TEXT NOT NULL,
            detection_conf  REAL,
            ocr_conf        REAL,
            bbox_x1         INTEGER,
            bbox_y1         INTEGER,
            bbox_x2         INTEGER,
            bbox_y2         INTEGER,
            track_id        INTEGER,
            frame_number    INTEGER,
            FOREIGN KEY (camera_id) REFERENCES cameras(camera_id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS blacklist (
            plate_text  TEXT PRIMARY KEY,
            reason      TEXT,
            added_at    TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_text  TEXT NOT NULL,
            camera_id   TEXT NOT NULL,
            timestamp   TEXT NOT NULL,
            reason      TEXT,
            acknowledged INTEGER NOT NULL DEFAULT 0,
            created_at  TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS processing_jobs (
            job_id          TEXT PRIMARY KEY,
            camera_id       TEXT NOT NULL,
            video_path      TEXT NOT NULL,
            status          TEXT NOT NULL DEFAULT 'pending',
            progress        INTEGER DEFAULT 0,
            total_frames    INTEGER,
            detections_found INTEGER DEFAULT 0,
            error_message   TEXT,
            created_at      TEXT NOT NULL DEFAULT (datetime('now')),
            started_at      TEXT,
            completed_at    TEXT,
            output_video    TEXT,
            FOREIGN KEY (camera_id) REFERENCES cameras(camera_id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS global_vehicles (
            global_id               TEXT PRIMARY KEY,
            canonical_plate         TEXT,
            plate_confidence        REAL,
            vehicle_type            TEXT NOT NULL,
            first_seen_ts           REAL NOT NULL,
            last_seen_ts            REAL NOT NULL,
            first_camera_id         TEXT NOT NULL,
            last_camera_id          TEXT NOT NULL,
            sighting_count          INTEGER NOT NULL DEFAULT 1,
            status                  TEXT NOT NULL DEFAULT 'active',
            representative_embedding BLOB,
            created_at              TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at              TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vehicle_observations (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            global_id               TEXT NOT NULL,
            camera_id               TEXT NOT NULL,
            local_track_id          INTEGER NOT NULL,
            first_timestamp         REAL NOT NULL,
            last_timestamp          REAL NOT NULL,
            vehicle_type            TEXT NOT NULL,
            canonical_plate         TEXT,
            plate_confidence        REAL,
            crop_quality            REAL,
            reid_embedding          BLOB,
            bbox_x1                 INTEGER,
            bbox_y1                 INTEGER,
            bbox_x2                 INTEGER,
            bbox_y2                 INTEGER,
            match_status            TEXT NOT NULL,
            match_confidence        REAL NOT NULL,
            match_method            TEXT NOT NULL,
            plate_similarity        REAL,
            reid_similarity         REAL,
            transit_speed_kmh       REAL,
            distance_km             REAL,
            match_reason            TEXT,
            created_at              TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (global_id) REFERENCES global_vehicles(global_id),
            FOREIGN KEY (camera_id) REFERENCES cameras(camera_id),
            UNIQUE (camera_id, local_track_id)
        )
    """)

    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_detections_plate ON detections(plate_text)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_detections_camera ON detections(camera_id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_detections_time ON detections(timestamp)"
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_ack ON alerts(acknowledged)")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_jobs_status ON processing_jobs(status)"
    )

    # Phase 6B Indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_global_vehicles_plate ON global_vehicles(canonical_plate)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_global_vehicles_last_seen ON global_vehicles(last_seen_ts)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_global_vehicles_status ON global_vehicles(status)")

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_obs_global_id ON vehicle_observations(global_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_obs_cam_ts ON vehicle_observations(camera_id, last_timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_obs_plate ON vehicle_observations(canonical_plate)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_obs_ts ON vehicle_observations(last_timestamp)")

    conn.commit()
    return conn


def upsert_camera(
    conn: sqlite3.Connection,
    camera_id: str,
    name: str,
    lat: float,
    lon: float,
    description: str = None,
):
    """Insert or update a camera."""
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO cameras (camera_id, name, latitude, longitude, description)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(camera_id) DO UPDATE SET
            name=excluded.name,
            latitude=excluded.latitude,
            longitude=excluded.longitude,
            description=excluded.description
    """,
        (camera_id, name, lat, lon, description),
    )
    conn.commit()


def insert_detection(
    conn: sqlite3.Connection,
    plate_text: str,
    camera_id: str,
    timestamp: str,
    detection_conf: float = None,
    ocr_conf: float = None,
    bbox: tuple = None,
    track_id: int = None,
    frame_number: int = None,
):
    """Insert a single detection."""
    cursor = conn.cursor()
    x1, y1, x2, y2 = bbox if bbox else (None, None, None, None)
    cursor.execute(
        """
        INSERT INTO detections (
            plate_text, camera_id, timestamp, detection_conf, ocr_conf,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2, track_id, frame_number
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            plate_text,
            camera_id,
            timestamp,
            detection_conf,
            ocr_conf,
            x1,
            y1,
            x2,
            y2,
            track_id,
            frame_number,
        ),
    )
    conn.commit()


def _sanitize_row(row_dict: dict) -> dict:
    """Sanitize SQLite row dictionary converting bytes to strings for JSON serialization."""
    sanitized = {}
    for k, v in row_dict.items():
        if isinstance(v, bytes):
            try:
                sanitized[k] = v.decode('utf-8', errors='ignore').strip('\x00')
            except Exception:
                sanitized[k] = str(v)
        else:
            sanitized[k] = v
    return sanitized


def query_plate_history(conn: sqlite3.Connection, plate_text: str) -> list[dict]:
    """Get all sightings for a specific plate."""
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT d.*, c.name as camera_name, c.latitude, c.longitude
        FROM detections d
        JOIN cameras c ON d.camera_id = c.camera_id
        WHERE d.plate_text = ?
        ORDER BY d.timestamp ASC
    """,
        (plate_text,),
    )
    return [_sanitize_row(dict(row)) for row in cursor.fetchall()]


def query_camera_activity(
    conn: sqlite3.Connection,
    camera_id: str,
    start_time: str = None,
    end_time: str = None,
) -> list[dict]:
    """Get detections at a camera within a time range."""
    cursor = conn.cursor()
    query = "SELECT * FROM detections WHERE camera_id = ?"
    params = [camera_id]
    if start_time:
        query += " AND timestamp >= ?"
        params.append(start_time)
    if end_time:
        query += " AND timestamp <= ?"
        params.append(end_time)
    query += " ORDER BY timestamp ASC"
    cursor.execute(query, params)
    return [dict(row) for row in cursor.fetchall()]


def get_all_plates(
    conn: sqlite3.Connection, start_time: str = None, end_time: str = None
) -> list[str]:
    """Get all unique plate texts."""
    cursor = conn.cursor()
    query = "SELECT DISTINCT plate_text FROM detections WHERE 1=1"
    params = []
    if start_time:
        query += " AND timestamp >= ?"
        params.append(start_time)
    if end_time:
        query += " AND timestamp <= ?"
        params.append(end_time)
    cursor.execute(query, params)
    return [row["plate_text"] for row in cursor.fetchall()]


def get_detection_stats(conn: sqlite3.Connection) -> dict:
    """Get overall database statistics."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) as total_detections, COUNT(DISTINCT plate_text) as unique_plates, COUNT(DISTINCT camera_id) as unique_cameras, MIN(timestamp) as min_time, MAX(timestamp) as max_time FROM detections"
    )
    row = cursor.fetchone()
    return dict(row) if row else {}


def get_recent_detections(conn: sqlite3.Connection, limit: int = 15) -> list[dict]:
    """Get most recent detections with camera info."""
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT d.id, d.plate_text, d.camera_id, d.timestamp, d.ocr_conf, d.detection_conf, c.name as camera_name
        FROM detections d
        LEFT JOIN cameras c ON d.camera_id = c.camera_id
        ORDER BY d.id DESC
        LIMIT ?
    """,
        (limit,),
    )
    return [_sanitize_row(dict(row)) for row in cursor.fetchall()]


def get_detections_over_time(
    conn: sqlite3.Connection, bucket_minutes: int = 5
) -> list[dict]:
    """Count detections per time bucket."""
    cursor = conn.cursor()
    # Bucket into specific minute intervals using unix epoch arithmetic (replacing ISO 'T' with space)
    query = f"""
        SELECT
            datetime((strftime('%s', replace(timestamp, 'T', ' ')) / ({bucket_minutes} * 60)) * ({bucket_minutes} * 60), 'unixepoch') as bucket_time,
            COUNT(*) as count
        FROM detections
        WHERE timestamp IS NOT NULL AND timestamp != ''
        GROUP BY bucket_time
        ORDER BY bucket_time ASC
    """
    cursor.execute(query)
    return [dict(row) for row in cursor.fetchall()]


def get_camera_heatmap_data(conn: sqlite3.Connection) -> list[dict]:
    """Get detections count per camera."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT c.camera_id, c.name, c.latitude, c.longitude, COUNT(d.id) as count
        FROM cameras c
        LEFT JOIN detections d ON c.camera_id = d.camera_id
        GROUP BY c.camera_id, c.name, c.latitude, c.longitude
    """)
    return [dict(row) for row in cursor.fetchall()]


def get_top_routes(conn: sqlite3.Connection, limit: int = 10) -> list[dict]:
    """Most common consecutive camera-pair transitions for the same plate."""
    cursor = conn.cursor()
    cursor.execute(
        """
        WITH PlatePaths AS (
            SELECT
                plate_text,
                camera_id AS from_camera,
                timestamp AS from_time,
                LEAD(camera_id) OVER (PARTITION BY plate_text ORDER BY timestamp ASC) AS to_camera,
                LEAD(timestamp) OVER (PARTITION BY plate_text ORDER BY timestamp ASC) AS to_time
            FROM detections
        )
        SELECT
            p.from_camera,
            p.to_camera,
            c1.name AS from_name,
            c2.name AS to_name,
            COUNT(*) AS count,
            AVG(CAST(strftime('%s', replace(p.to_time, 'T', ' ')) AS INTEGER) - CAST(strftime('%s', replace(p.from_time, 'T', ' ')) AS INTEGER)) AS avg_travel_seconds
        FROM PlatePaths p
        JOIN cameras c1 ON p.from_camera = c1.camera_id
        JOIN cameras c2 ON p.to_camera = c2.camera_id
        WHERE p.to_camera IS NOT NULL AND p.from_camera != p.to_camera
        GROUP BY p.from_camera, p.to_camera
        ORDER BY count DESC
        LIMIT ?
    """,
        (limit,),
    )
    return [dict(row) for row in cursor.fetchall()]


def check_blacklist(conn: sqlite3.Connection, plate_text: str) -> str | None:
    """Check if a plate is blacklisted. Returns the reason if it is."""
    cursor = conn.cursor()
    cursor.execute("SELECT reason FROM blacklist WHERE plate_text = ?", (plate_text,))
    row = cursor.fetchone()
    return row["reason"] if row else None


def add_to_blacklist(conn: sqlite3.Connection, plate_text: str, reason: str = None):
    """Add a plate to the blacklist."""
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO blacklist (plate_text, reason)
        VALUES (?, ?)
        ON CONFLICT(plate_text) DO UPDATE SET reason=excluded.reason
    """,
        (plate_text, reason),
    )
    conn.commit()


def remove_from_blacklist(conn: sqlite3.Connection, plate_text: str):
    """Remove a plate from the blacklist."""
    cursor = conn.cursor()
    cursor.execute("DELETE FROM blacklist WHERE plate_text = ?", (plate_text,))
    conn.commit()


def get_blacklist(conn: sqlite3.Connection) -> list[dict]:
    """Get all blacklisted plates."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM blacklist ORDER BY added_at DESC")
    return [dict(row) for row in cursor.fetchall()]


def insert_alert(
    conn: sqlite3.Connection,
    plate_text: str,
    camera_id: str,
    timestamp: str,
    reason: str,
):
    """Insert a new alert."""
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO alerts (plate_text, camera_id, timestamp, reason)
        VALUES (?, ?, ?, ?)
    """,
        (plate_text, camera_id, timestamp, reason),
    )
    conn.commit()


def get_alerts(
    conn: sqlite3.Connection, only_unacknowledged: bool = False
) -> list[dict]:
    """Get alerts."""
    cursor = conn.cursor()
    query = "SELECT * FROM alerts"
    if only_unacknowledged:
        query += " WHERE acknowledged = 0"
    query += " ORDER BY timestamp DESC"
    cursor.execute(query)
    return [dict(row) for row in cursor.fetchall()]


def acknowledge_alert(conn: sqlite3.Connection, alert_id: int):
    """Mark an alert as acknowledged."""
    cursor = conn.cursor()
    cursor.execute("UPDATE alerts SET acknowledged = 1 WHERE id = ?", (alert_id,))
    conn.commit()


def load_blacklist_from_file(conn: sqlite3.Connection, filepath: str | Path):
    """Read a text file and add entries to the blacklist."""
    filepath = Path(filepath)
    if not filepath.exists():
        return
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            add_to_blacklist(conn, line, reason="Loaded from file")


def load_cameras_from_json(conn: sqlite3.Connection, filepath: str | Path):
    """Read a JSON file and upsert cameras."""
    filepath = Path(filepath)
    if not filepath.exists():
        return
    with open(filepath, "r") as f:
        cameras = json.load(f)
        for cam in cameras:
            upsert_camera(
                conn,
                cam["camera_id"],
                cam["name"],
                cam["latitude"],
                cam["longitude"],
                cam.get("description"),
            )


# ============================================================================
# Job Management (Background Processing)
# ============================================================================


def create_job(
    conn: sqlite3.Connection,
    job_id: str,
    camera_id: str,
    video_path: str,
    total_frames: int = 0,
) -> dict:
    """Create a new processing job."""
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO processing_jobs (job_id, camera_id, video_path, status, total_frames)
        VALUES (?, ?, ?, 'pending', ?)
    """,
        (job_id, camera_id, video_path, total_frames),
    )
    conn.commit()
    return get_job(conn, job_id)


def get_job(conn: sqlite3.Connection, job_id: str) -> dict | None:
    """Fetch a job by ID."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM processing_jobs WHERE job_id = ?", (job_id,))
    row = cursor.fetchone()
    return dict(row) if row else None


def update_job_progress(
    conn: sqlite3.Connection, job_id: str, progress: int, detections: int = 0
):
    """Update job progress."""
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE processing_jobs
        SET progress = ?, detections_found = ?
        WHERE job_id = ?
    """,
        (progress, detections, job_id),
    )
    conn.commit()


def mark_job_started(conn: sqlite3.Connection, job_id: str):
    """Mark a job as started."""
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE processing_jobs
        SET status = 'processing', started_at = datetime('now')
        WHERE job_id = ?
    """,
        (job_id,),
    )
    conn.commit()


def mark_job_completed(
    conn: sqlite3.Connection,
    job_id: str,
    output_video: str = None,
    detection_count: int = 0,
):
    """Mark a job as completed."""
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE processing_jobs
        SET status = 'completed', completed_at = datetime('now'),
            output_video = ?, detections_found = ?
        WHERE job_id = ?
    """,
        (output_video, detection_count, job_id),
    )
    conn.commit()


def mark_job_failed(conn: sqlite3.Connection, job_id: str, error_message: str):
    """Mark a job as failed."""
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE processing_jobs
        SET status = 'failed', completed_at = datetime('now'), error_message = ?
        WHERE job_id = ?
    """,
        (error_message, job_id),
    )
    conn.commit()


def get_pending_jobs(conn: sqlite3.Connection) -> list:
    """Get all pending jobs."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM processing_jobs WHERE status = ? ORDER BY created_at ASC",
        ("pending",),
    )
    return [dict(row) for row in cursor.fetchall()]


def get_job_status(conn: sqlite3.Connection, job_id: str) -> dict | None:
    """Get job status summary for UI."""
    job = get_job(conn, job_id)
    if not job:
        return None
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "progress": job["progress"],
        "detections_found": job["detections_found"],
        "total_frames": job["total_frames"],
        "error_message": job["error_message"],
        "output_video": job["output_video"],
        "created_at": job["created_at"],
        "completed_at": job["completed_at"],
    }


# ── Phase 6B: Global Vehicle & Observation Persistence ──


def serialize_embedding(embedding: Optional[np.ndarray]) -> Optional[bytes]:
    """Serialize a numpy embedding vector to raw bytes for BLOB storage."""
    if embedding is None or len(embedding) == 0:
        return None
    return np.asarray(embedding, dtype=np.float32).ravel().tobytes()


def deserialize_embedding(blob: Optional[bytes]) -> Optional[np.ndarray]:
    """Deserialize raw bytes back to a float32 numpy embedding vector."""
    if blob is None or len(blob) == 0:
        return None
    return np.frombuffer(blob, dtype=np.float32)


def save_global_identity(conn: sqlite3.Connection, identity: Any) -> None:
    """
    Insert or update a GlobalVehicleIdentity in the database.
    """
    cursor = conn.cursor()
    emb_blob = serialize_embedding(identity.representative_embedding)
    cursor.execute(
        """
        INSERT INTO global_vehicles (
            global_id, canonical_plate, plate_confidence, vehicle_type,
            first_seen_ts, last_seen_ts, first_camera_id, last_camera_id,
            sighting_count, status, representative_embedding, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(global_id) DO UPDATE SET
            canonical_plate = excluded.canonical_plate,
            plate_confidence = excluded.plate_confidence,
            vehicle_type = excluded.vehicle_type,
            last_seen_ts = excluded.last_seen_ts,
            last_camera_id = excluded.last_camera_id,
            sighting_count = excluded.sighting_count,
            status = excluded.status,
            representative_embedding = COALESCE(excluded.representative_embedding, global_vehicles.representative_embedding),
            updated_at = datetime('now')
        """,
        (
            identity.global_id,
            identity.canonical_plate,
            float(identity.plate_confidence),
            identity.vehicle_type,
            float(identity.first_seen_ts),
            float(identity.last_seen_ts),
            identity.first_camera_id,
            identity.last_camera_id,
            int(identity.sighting_count),
            identity.status,
            emb_blob,
        ),
    )
    conn.commit()


def record_vehicle_observation(
    conn: sqlite3.Connection,
    obs: Any,
    match_result: Any,
    first_timestamp: Optional[float] = None,
) -> bool:
    """
    Insert or update a finalized vehicle observation (idempotent on camera_id, local_track_id).
    """
    cursor = conn.cursor()
    emb_blob = serialize_embedding(obs.best_reid_embedding)
    first_ts = float(first_timestamp) if first_timestamp is not None else float(obs.timestamp)
    last_ts = float(obs.timestamp)

    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = (None, None, None, None)
    if obs.bbox is not None and len(obs.bbox) == 4:
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = (int(v) for v in obs.bbox)

    cursor.execute(
        """
        INSERT INTO vehicle_observations (
            global_id, camera_id, local_track_id, first_timestamp, last_timestamp,
            vehicle_type, canonical_plate, plate_confidence, crop_quality,
            reid_embedding, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
            match_status, match_confidence, match_method, plate_similarity,
            reid_similarity, transit_speed_kmh, distance_km, match_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(camera_id, local_track_id) DO UPDATE SET
            global_id = excluded.global_id,
            last_timestamp = excluded.last_timestamp,
            canonical_plate = excluded.canonical_plate,
            plate_confidence = excluded.plate_confidence,
            crop_quality = excluded.crop_quality,
            reid_embedding = COALESCE(excluded.reid_embedding, vehicle_observations.reid_embedding),
            match_status = excluded.match_status,
            match_confidence = excluded.match_confidence,
            match_method = excluded.match_method,
            plate_similarity = excluded.plate_similarity,
            reid_similarity = excluded.reid_similarity,
            transit_speed_kmh = excluded.transit_speed_kmh,
            distance_km = excluded.distance_km,
            match_reason = excluded.match_reason
        """,
        (
            match_result.global_id,
            obs.camera_id,
            int(obs.track_id),
            first_ts,
            last_ts,
            obs.vehicle_type,
            obs.canonical_plate,
            float(obs.plate_confidence),
            float(obs.crop_quality),
            emb_blob,
            bbox_x1,
            bbox_y1,
            bbox_x2,
            bbox_y2,
            match_result.status,
            float(match_result.confidence),
            match_result.match_method,
            float(match_result.plate_similarity) if match_result.plate_similarity is not None else None,
            float(match_result.reid_similarity) if match_result.reid_similarity is not None else None,
            float(match_result.transit_speed_kmh) if match_result.transit_speed_kmh is not None else None,
            float(match_result.distance_km) if match_result.distance_km is not None else None,
            match_result.reason,
        ),
    )
    conn.commit()
    return True


def get_global_vehicle(conn: sqlite3.Connection, global_id: str) -> Optional[dict]:
    """Retrieve a global vehicle record by global_id."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM global_vehicles WHERE global_id = ?", (global_id,))
    row = cursor.fetchone()
    if not row:
        return None
    d = dict(row)
    if d.get("representative_embedding"):
        d["representative_embedding"] = deserialize_embedding(d["representative_embedding"])
    return d


def get_vehicle_trajectory(conn: sqlite3.Connection, global_id: str) -> List[dict]:
    """
    Retrieve all sightings/observations for a global vehicle, ordered chronologically.
    """
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT * FROM vehicle_observations
        WHERE global_id = ?
        ORDER BY first_timestamp ASC
        """,
        (global_id,),
    )
    rows = cursor.fetchall()
    results = []
    for r in rows:
        d = dict(r)
        if d.get("reid_embedding"):
            d["reid_embedding"] = deserialize_embedding(d["reid_embedding"])
        results.append(d)
    return results


def get_all_global_vehicles(
    conn: sqlite3.Connection,
    limit: int = 100,
    status: Optional[str] = None,
) -> List[dict]:
    """Retrieve list of global vehicle identities."""
    cursor = conn.cursor()
    if status:
        cursor.execute(
            "SELECT * FROM global_vehicles WHERE status = ? ORDER BY last_seen_ts DESC LIMIT ?",
            (status, limit),
        )
    else:
        cursor.execute(
            "SELECT * FROM global_vehicles ORDER BY last_seen_ts DESC LIMIT ?",
            (limit,),
        )
    rows = cursor.fetchall()
    results = []
    for r in rows:
        d = dict(r)
        if d.get("representative_embedding"):
            d["representative_embedding"] = deserialize_embedding(d["representative_embedding"])
        results.append(d)
    return results
