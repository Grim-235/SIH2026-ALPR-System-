import json
import sqlite3
from pathlib import Path


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
    return [dict(row) for row in cursor.fetchall()]


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
    return [dict(row) for row in cursor.fetchall()]


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
