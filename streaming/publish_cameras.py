"""
FFmpeg Stream Publisher

This script reads camera configurations and uses FFmpeg to publish MP4 files
as RTSP streams through MediaMTX.
"""

import argparse
import json
import logging
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('streaming.publish')

class StreamPublisher:
    """Manages FFmpeg processes for publishing streams."""
    
    def __init__(
        self,
        config_path: str,
        mediamtx_host: str,
        mediamtx_port: int,
        auto_restart: bool,
        ffmpeg_path: str
    ):
        self.config_path = Path(config_path)
        self.mediamtx_host = mediamtx_host
        self.mediamtx_port = mediamtx_port
        self.auto_restart = auto_restart
        self.ffmpeg_path = ffmpeg_path
        self.processes: Dict[str, subprocess.Popen] = {}
        self.cameras: List[Dict[str, Any]] = []
        self.running = True

    def load_config(self) -> None:
        """Loads camera configurations from JSON file."""
        if not self.config_path.exists():
            logger.error(f"Configuration file not found: {self.config_path}")
            sys.exit(1)
            
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                
                # Assuming the config has a 'cameras' list, fallback to using root list if it is one
                if isinstance(config_data, dict) and 'cameras' in config_data:
                    self.cameras = config_data['cameras']
                elif isinstance(config_data, list):
                    self.cameras = config_data
                else:
                    logger.error("Invalid configuration format. Expected list or dict with 'cameras' key.")
                    sys.exit(1)
                    
                if not self.cameras:
                    logger.warning("No cameras found in configuration.")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON configuration: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error reading configuration: {e}")
            sys.exit(1)

    def start_stream(self, camera: Dict[str, Any]) -> None:
        """Starts an FFmpeg process for a specific camera."""
        cam_id = camera.get('camera_id')
        video_path = camera.get('video')
        stream_url = camera.get('stream_url')

        if not all([cam_id, video_path, stream_url]):
            logger.warning(f"Skipping camera due to missing fields: {camera}")
            return

        logger.info(f"Starting stream for camera: {cam_id}")
        
        # Fallback command with transcoding
        cmd_transcode = [
            self.ffmpeg_path,
            '-re',
            '-stream_loop', '-1',
            '-i', video_path,
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-c:a', 'aac',
            '-f', 'rtsp',
            stream_url
        ]
        
        # Primary command with copy codec
        cmd_copy = [
            self.ffmpeg_path,
            '-re',
            '-stream_loop', '-1',
            '-i', video_path,
            '-c', 'copy',
            '-f', 'rtsp',
            stream_url
        ]

        try:
            # Start with copy
            process = subprocess.Popen(
                cmd_copy,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Wait a brief moment to see if it fails immediately due to codec issues
            time.sleep(1.0)
            if process.poll() is not None:
                logger.warning(f"Copy codec failed for {cam_id}, falling back to transcoding...")
                process = subprocess.Popen(
                    cmd_transcode,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                
            self.processes[cam_id] = process
            logger.info(f"Successfully started FFmpeg for {cam_id} (PID: {process.pid})")
            
        except FileNotFoundError:
            logger.error(f"FFmpeg executable not found at '{self.ffmpeg_path}'. Is it installed and in PATH?")
            self.running = False
        except Exception as e:
            logger.error(f"Failed to start FFmpeg for {cam_id}: {e}")

    def display_summary(self) -> None:
        """Prints a summary table of all camera streams."""
        print(f"\n{'='*80}")
        print(f"{'Camera ID':<15} | {'Video Path':<30} | {'Stream URL':<20} | {'Status'}")
        print(f"{'-'*80}")
        for cam in self.cameras:
            cam_id = cam.get('camera_id', 'Unknown')
            video = cam.get('video', 'N/A')
            url = cam.get('stream_url', 'N/A')
            
            # Truncate long paths
            if len(video) > 27:
                video = "..." + video[-24:]
                
            status = "Running" if cam_id in self.processes and self.processes[cam_id].poll() is None else "Failed/Stopped"
            if not cam.get('video') or not cam.get('stream_url'):
                status = "Skipped (Missing Info)"
                
            print(f"{cam_id:<15} | {video:<30} | {url:<20} | {status}")
        print(f"{'='*80}\n")

    def run(self) -> None:
        """Main loop to monitor and potentially restart streams."""
        self.load_config()
        
        for cam in self.cameras:
            if not self.running:
                break
            self.start_stream(cam)
            
        if not self.running:
            return
            
        self.display_summary()
        
        try:
            while self.running:
                for cam in self.cameras:
                    cam_id = cam.get('camera_id')
                    if not cam_id or cam_id not in self.processes:
                        continue
                        
                    process = self.processes[cam_id]
                    if process.poll() is not None:
                        logger.warning(f"FFmpeg process for {cam_id} terminated unexpectedly (Exit code: {process.returncode})")
                        
                        if self.auto_restart and self.running:
                            logger.info(f"Restarting stream for {cam_id}...")
                            self.start_stream(cam)
                        else:
                            del self.processes[cam_id]
                            
                time.sleep(5)
                
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt.")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Terminates all running FFmpeg processes."""
        logger.info("Shutting down stream publisher...")
        self.running = False
        
        for cam_id, process in list(self.processes.items()):
            if process.poll() is None:
                logger.info(f"Terminating FFmpeg for {cam_id} (PID: {process.pid})")
                try:
                    process.terminate()
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Process {cam_id} did not terminate gracefully, forcing kill...")
                    process.kill()
                except Exception as e:
                    logger.error(f"Error terminating process {cam_id}: {e}")
                    
        self.processes.clear()
        logger.info("Cleanup complete.")


def main():
    parser = argparse.ArgumentParser(description="Publish MP4 files as RTSP streams using FFmpeg.")
    parser.add_argument('--config', type=str, default='configs/cameras.json',
                        help='Path to cameras.json (default: configs/cameras.json)')
    parser.add_argument('--mediamtx-host', type=str, default='localhost',
                        help='MediaMTX hostname (default: localhost)')
    parser.add_argument('--mediamtx-port', type=int, default=8554,
                        help='MediaMTX RTSP port (default: 8554)')
    parser.add_argument('--auto-restart', action='store_true',
                        help='Auto-restart failed FFmpeg processes')
    parser.add_argument('--ffmpeg', type=str, default='ffmpeg',
                        help='Path to ffmpeg binary (default: ffmpeg)')
                        
    args = parser.parse_args()
    
    publisher = StreamPublisher(
        config_path=args.config,
        mediamtx_host=args.mediamtx_host,
        mediamtx_port=args.mediamtx_port,
        auto_restart=args.auto_restart,
        ffmpeg_path=args.ffmpeg
    )
    
    def signal_handler(sig, frame):
        logger.info("Signal received, initiating shutdown...")
        publisher.running = False
        
    # Handle SIGINT (Ctrl+C) gracefully
    signal.signal(signal.SIGINT, signal_handler)
    # Register SIGBREAK for Windows if available, otherwise SIGTERM
    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, signal_handler)
    else:
        try:
            signal.signal(signal.SIGTERM, signal_handler)
        except AttributeError:
            pass
        
    publisher.run()


if __name__ == '__main__':
    main()
