"""
Phase 9A -- Acceptance Tests for End-to-End Multi-Camera Validation, Looping & Supervisor Resilience.

Verifies:
1. CameraSource Video Looping: Seamless EOF rewind, loop_count increment, and monotonic capture timestamps.
2. CameraWorker Loop Continuity: Sustained frame processing past EOF without premature thread termination.
3. PipelineOrchestrator Telemetry & Health: Accurate collection of CPU, RAM, DB metrics, and latency percentiles (P50/P95).
4. DB Concurrency Metrics: Thread-safe tracking of total transactions, retries, and lock errors.
5. Supervisor Fault Injection & Auto-Recovery: Automated detection and respawn of dropped worker thread.
6. Architectural Boundary Verification: Zero math formulas in Flask REST route handlers.
"""

import inspect
import json
import logging
import os
import sqlite3
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from alpr.camera import CameraSource
from alpr.database import (
    init_db,
    get_thread_connection,
    execute_with_retry,
    get_db_concurrency_metrics,
    reset_db_concurrency_metrics,
)
from workers.camera_worker import CameraWorker
from workers.orchestrator import PipelineOrchestrator, CameraTelemetry
from app import app


def test_camera_source_video_looping():
    """Verify CameraSource loops video seamlessly with advancing timestamps."""
    video_path = "inputs/cam01.mp4"
    assert os.path.exists(video_path), f"Test video not found: {video_path}"

    # 1. Non-looping should stop at EOF (61 frames)
    cam_no_loop = CameraSource(video_path, "TEST-NO-LOOP", loop=False)
    assert cam_no_loop.connect()
    frame_count = 0
    while True:
        ok, frame, ts = cam_no_loop.read_frame()
        if not ok:
            break
        frame_count += 1
    cam_no_loop.release()
    assert frame_count == 61
    assert cam_no_loop.loop_count == 0

    # 2. Looping should continue past 61 frames
    cam_loop = CameraSource(video_path, "TEST-LOOP", loop=True)
    assert cam_loop.connect()
    last_ts = 0.0
    for i in range(130):  # More than 2x the video length
        ok, frame, ts = cam_loop.read_frame()
        assert ok, f"Read failed at frame {i}"
        assert frame is not None
        assert ts >= last_ts, f"Non-monotonic timestamp at frame {i}: {ts} < {last_ts}"
        last_ts = ts

    assert cam_loop.frames_read == 130
    assert cam_loop.loop_count >= 2
    cam_loop.release()


def test_camera_worker_loop_continuity():
    """Verify CameraWorker continues running across video file loop boundaries."""
    video_path = "inputs/cam01.mp4"
    assert os.path.exists(video_path)

    worker = CameraWorker(
        camera_id="CAM-TEST-LOOP",
        source=video_path,
        loop=True,
    )

    thread = threading.Thread(target=worker.start, daemon=True)
    thread.start()

    # Wait until worker processes at least 70 frames (> 61 original frames)
    t0 = time.time()
    while worker.frames_processed < 70 and time.time() - t0 < 8.0:
        time.sleep(0.1)

    assert worker.frames_processed >= 70, f"Worker processed only {worker.frames_processed} frames"
    assert thread.is_alive(), "Worker thread terminated prematurely"

    worker.stop()
    thread.join(timeout=3.0)


def test_orchestrator_loop_propagation_and_telemetry():
    """Verify PipelineOrchestrator propagates loop_video and exposes complete telemetry."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name

    try:
        init_db(db_path)
        orchestrator = PipelineOrchestrator(
            config_path="configs/cameras.json",
            camera_graph_path="configs/camera_graph.json",
            db_path=db_path,
            use_stream=False,
            loop_video=True,
            mode="ingest",
        )

        worker = orchestrator._create_worker_for_camera("CAM-001")
        assert worker.loop is True, "Orchestrator failed to propagate loop_video to worker"

        # Check telemetry aggregation
        health = orchestrator.get_health()
        assert "cpu_percent" in health
        assert "memory_mb" in health
        assert "db_metrics" in health
        assert "p50_latency_ms" in health
        assert "p95_latency_ms" in health
        assert health["status"] == "offline"

        statuses = orchestrator.get_camera_statuses()
        assert len(statuses) == 4
        for s in statuses:
            assert "p50_latency_ms" in s
            assert "p95_latency_ms" in s
            assert "loop_count" in s
    finally:
        if os.path.exists(db_path):
            try:
                os.unlink(db_path)
            except Exception:
                pass


def test_db_concurrency_metrics_counters():
    """Verify thread-safe DB concurrency metrics recording and resetting."""
    reset_db_concurrency_metrics()
    metrics0 = get_db_concurrency_metrics()
    assert metrics0["total_transactions"] == 0
    assert metrics0["retries"] == 0
    assert metrics0["lock_errors"] == 0

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name

    try:
        conn = init_db(db_path)

        def sample_write():
            c = get_thread_connection(db_path)
            c.execute("INSERT OR REPLACE INTO blacklist (plate_text, reason) VALUES ('TEST01', 'testing')")
            c.commit()

        execute_with_retry(sample_write)
        metrics1 = get_db_concurrency_metrics()
        assert metrics1["total_transactions"] >= 1

        reset_db_concurrency_metrics()
        metrics2 = get_db_concurrency_metrics()
        assert metrics2["total_transactions"] == 0
        conn.close()
    finally:
        if os.path.exists(db_path):
            try:
                os.unlink(db_path)
            except Exception:
                pass


def test_supervisor_fault_recovery():
    """Verify supervisor detects terminated worker and triggers auto-recovery."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name

    try:
        init_db(db_path)
        orchestrator = PipelineOrchestrator(
            config_path="configs/cameras.json",
            camera_graph_path="configs/camera_graph.json",
            db_path=db_path,
            use_stream=False,
            loop_video=True,
            mode="ingest",
            heartbeat_interval=0.5,
            max_worker_restarts=3,
        )

        orchestrator.start()
        time.sleep(1.5)

        # Inject failure into CAM-002
        worker_cam02 = orchestrator.workers.get("CAM-002")
        old_thread = orchestrator.worker_threads.get("CAM-002")
        assert worker_cam02 is not None
        assert old_thread is not None and old_thread.is_alive()

        # Terminate worker artificially
        worker_cam02.stop()
        old_thread.join(timeout=2.0)
        assert not old_thread.is_alive(), "Target worker thread did not stop after worker.stop()"

        # Wait for supervisor auto-restart (heartbeat 0.5s)
        t0 = time.time()
        recovered = False
        while time.time() - t0 < 8.0:
            telem = orchestrator.telemetry.get("CAM-002")
            new_th = orchestrator.worker_threads.get("CAM-002")
            new_worker = orchestrator.workers.get("CAM-002")
            if (
                telem
                and telem.restarts > 0
                and new_th is not None
                and new_th != old_thread
                and new_th.is_alive()
                and new_worker
                and new_worker._running
            ):
                recovered = True
                break
            time.sleep(0.3)

        assert recovered, "Supervisor failed to detect and restart dropped worker CAM-002"

        orchestrator.stop()
    finally:
        if os.path.exists(db_path):
            try:
                os.unlink(db_path)
            except Exception:
                pass


def test_zero_math_in_flask_routes():
    """Verify Flask routes maintain zero-math invariant with new telemetry endpoints."""
    import app as app_module

    routes_to_inspect = [
        app_module.api_system_health,
        app_module.api_system_cameras,
        app_module.api_alerts_list,
        app_module.api_alerts_summary,
        app_module.api_analytics_summary,
    ]

    forbidden_patterns = ["np.percentile", "percentile", "psutil.", "math.", "speed =", "tti ="]
    for route_fn in routes_to_inspect:
        src = inspect.getsource(route_fn).lower()
        for pat in forbidden_patterns:
            assert pat not in src, f"Route {route_fn.__name__} violates architectural boundary: contains '{pat}'"
