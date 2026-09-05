"""
Core abstraction for video source handling in the ALPR system.

This module provides the CameraSource class, which unifies RTSP streams,
video files, and webcams under a single interface with built-in reconnection,
FPS throttling, and statistics tracking.
"""

import cv2
import numpy as np
import time
import threading
import logging
import os
from typing import Tuple, Union, Optional
from collections import deque

# Setup logger for the camera module
logger = logging.getLogger("alpr.camera")

# Ensure OpenCV uses TCP for RTSP streams to prevent UDP packet loss artifacts.
# This flag is set globally at the module level before any VideoCapture is opened.
if 'OPENCV_FFMPEG_CAPTURE_OPTIONS' not in os.environ:
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'


class CameraSource:
    """
    A unified interface for accessing video streams, files, and webcams.
    
    Provides thread-safe frame reading, automatic reconnection for streams,
    FPS throttling, and performance monitoring.
    """

    def __init__(self, source: Union[str, int], camera_id: str, fps_target: float = 0.0, 
                 reconnect_max_retries: int = 10, reconnect_base_delay: float = 1.0,
                 loop: bool = False):
        """
        Initialize the CameraSource.
        
        Args:
            source: RTSP URL string, file path string, webcam integer index, or HTTP URL.
            camera_id: Logical camera identifier like 'CAM-001'.
            fps_target: If > 0, throttle frame reads to this FPS.
            reconnect_max_retries: Max reconnection attempts before giving up.
            reconnect_base_delay: Initial delay for exponential backoff reconnection.
            loop: If True and source is a video file, rewind to beginning upon reaching EOF.
        """
        self.source = source
        self.camera_id = camera_id
        self.fps_target = fps_target
        self.reconnect_max_retries = reconnect_max_retries
        self.reconnect_base_delay = reconnect_base_delay
        self.loop = loop
        self.loop_count = 0

        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()

        # State tracking
        self._connected = False
        self._status = "unknown"
        self.frames_read = 0
        self.frames_dropped = 0
        self.last_frame_time = 0.0
        self._connect_time = 0.0

        # FPS calculation window (last 30 frames)
        self._frame_times: deque = deque(maxlen=30)
        self._min_frame_interval = 1.0 / self.fps_target if self.fps_target > 0 else 0.0

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    @property
    def is_stream(self) -> bool:
        """
        Returns True for RTSP/HTTP sources (as opposed to files/webcams).
        Useful because streams typically need reconnection logic.
        """
        if isinstance(self.source, str):
            lower_source = self.source.lower()
            return lower_source.startswith(('rtsp://', 'http://', 'https://'))
        return False

    @property
    def connected(self) -> bool:
        """Current connection state."""
        return self._connected

    @property
    def status(self) -> str:
        """Current dynamic status: 'unknown', 'connecting', 'online', 'reconnecting', 'offline'."""
        return self._status

    @property
    def connection_uptime(self) -> float:
        """Seconds since last successful connect."""
        if self._connected and self._connect_time > 0:
            return time.time() - self._connect_time
        return 0.0

    def connect(self) -> bool:
        """
        Open the VideoCapture. Returns True on success.
        
        Logs connection info including resolution and native FPS.
        
        Returns:
            bool: True if connection was successful, False otherwise.
        """
        with self._lock:
            if self._cap is not None:
                self._cap.release()
            
            if self._status != "reconnecting":
                self._status = "connecting"
            logger.info(f"[{self.camera_id}] Connecting to source: {self.source}")
            self._cap = cv2.VideoCapture(self.source)
            
            if not self._cap.isOpened():
                logger.error(f"[{self.camera_id}] Failed to open source.")
                self._connected = False
                self._status = "offline"
                return False
                
            self._connected = True
            self._status = "online"
            self._connect_time = time.time()
            self._frame_times.clear()
            
            # Log connection info
            res = self.get_resolution(acquire_lock=False)
            fps = self.get_native_fps(acquire_lock=False)
            logger.info(f"[{self.camera_id}] Connected. Resolution: {res[0]}x{res[1]}, Native FPS: {fps:.2f}")
            
            return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Read a frame from the source.
        
        If fps_target is set, applies throttling to match the target.
        
        Returns:
            Tuple[bool, Optional[np.ndarray], float]: (success, frame, capture_timestamp).
            capture_timestamp is time.time() at the moment the frame is grabbed.
            Returns (False, None, 0.0) on failure.
        """
        if not self._connected or self._cap is None:
            return False, None, 0.0
            
        current_time = time.time()
        # Apply FPS throttling
        if self._min_frame_interval > 0:
            time_since_last = current_time - self.last_frame_time
            if time_since_last < self._min_frame_interval:
                time.sleep(self._min_frame_interval - time_since_last)
                
        with self._lock:
            try:
                if self._cap is None or not self._cap.isOpened():
                    return False, None, 0.0
                ret, frame = self._cap.read()
                capture_timestamp = time.time()
                # If file reached EOF and looping is enabled, rewind to beginning
                if (not ret or frame is None) and not self.is_stream and self.loop:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self._cap.read()
                    if ret and frame is not None:
                        self.loop_count += 1
                        capture_timestamp = time.time()
            except Exception as e:
                logger.error(f"[{self.camera_id}] Exception reading frame: {e}")
                ret = False
                frame = None
                capture_timestamp = 0.0

        if ret and frame is not None:
            self.frames_read += 1
            self.last_frame_time = capture_timestamp
            self._frame_times.append(capture_timestamp)
            return True, frame, capture_timestamp
        else:
            self.frames_dropped += 1
            return False, None, 0.0

    def reconnect(self) -> bool:
        """
        Exponential backoff reconnection.
        Doubles delay each attempt up to 30 seconds max.
        
        Returns:
            bool: True when reconnected, False if max retries exceeded.
        """
        logger.warning(f"[{self.camera_id}] Attempting to reconnect...")
        self._connected = False
        self._status = "reconnecting"
        
        delay = self.reconnect_base_delay
        for attempt in range(1, self.reconnect_max_retries + 1):
            logger.info(f"[{self.camera_id}] Reconnection attempt {attempt}/{self.reconnect_max_retries} "
                        f"in {delay:.1f}s...")
            time.sleep(delay)
            
            if self.connect():
                logger.info(f"[{self.camera_id}] Successfully reconnected on attempt {attempt}.")
                return True
                
            # Exponential backoff capped at 30 seconds
            delay = min(delay * 2.0, 30.0)
            
        logger.error(f"[{self.camera_id}] Failed to reconnect after {self.reconnect_max_retries} attempts.")
        self._status = "offline"
        return False

    def is_alive(self) -> bool:
        """
        Check if the underlying VideoCapture is opened and responsive.
        
        Returns:
            bool: True if alive, False otherwise.
        """
        with self._lock:
            return self._cap is not None and self._cap.isOpened()

    def release(self) -> None:
        """Clean release of VideoCapture resources."""
        logger.info(f"[{self.camera_id}] Releasing camera resources.")
        self._connected = False
        self._status = "offline"
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None

    def get_fps(self) -> float:
        """
        Calculate measured FPS (frames actually read per second),
        updated as a rolling average of the last 30 frames.
        
        Returns:
            float: Measured FPS.
        """
        if len(self._frame_times) < 2:
            return 0.0
            
        times = list(self._frame_times)
        time_diff = times[-1] - times[0]
        
        if time_diff <= 0:
            return 0.0
            
        return (len(times) - 1) / time_diff

    def get_native_fps(self, acquire_lock: bool = True) -> float:
        """
        Returns the FPS reported by the source via CAP_PROP_FPS.
        
        Args:
            acquire_lock: If True, acquires the lock before querying. Internal use.
            
        Returns:
            float: Native FPS as reported by the source.
        """
        def _get_fps():
            if self._cap is not None and self._cap.isOpened():
                return self._cap.get(cv2.CAP_PROP_FPS)
            return 0.0

        if acquire_lock:
            with self._lock:
                return _get_fps()
        return _get_fps()

    def get_resolution(self, acquire_lock: bool = True) -> Tuple[int, int]:
        """
        Returns the resolution of the video source.
        
        Args:
            acquire_lock: If True, acquires the lock before querying. Internal use.
            
        Returns:
            Tuple[int, int]: (width, height)
        """
        def _get_res():
            if self._cap is not None and self._cap.isOpened():
                width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                return width, height
            return 0, 0

        if acquire_lock:
            with self._lock:
                return _get_res()
        return _get_res()
