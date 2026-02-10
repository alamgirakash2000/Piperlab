#!/usr/bin/env python3
"""Multi-camera capture manager for synchronized recording."""

import time
import threading
import numpy as np
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass
from queue import Queue, Empty
import cv2
import os
import sys
from contextlib import contextmanager

# Suppress libjpeg warnings by setting OpenCV log level
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")


@dataclass
class CameraFrame:
    """Container for a captured camera frame."""
    timestamp: float
    frame: np.ndarray
    camera_name: str
    width: int
    height: int
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.frame.shape


class Camera:
    """Single camera capture handler."""
    
    def __init__(
        self,
        name: str,
        device_index: int,
        width: int = 1280,
        height: int = 720,
        fps: int = 30
    ):
        self.name = name
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps
        
        self._capture: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._last_frame: Optional[CameraFrame] = None
    
    def open(self) -> bool:
        """Open the camera device."""
        with self._lock:
            if self._capture is not None:
                return True
            
            try:
                # Try V4L2 backend on Linux for lower latency
                self._capture = cv2.VideoCapture(self.device_index, cv2.CAP_V4L2)
                
                if not self._capture.isOpened():
                    # Fallback to default backend
                    self._capture = cv2.VideoCapture(self.device_index)
                
                if not self._capture.isOpened():
                    print(f"Error: Could not open camera {self.name} (device {self.device_index})")
                    self._capture = None
                    return False
                
                # Configure camera - set format first for better compatibility
                # Use MJPEG for higher frame rates (and suppress warnings)
                self._capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                
                # Set resolution and FPS
                self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self._capture.set(cv2.CAP_PROP_FPS, self.fps)
                
                # Reduce buffer size to minimize latency (critical for wrist camera)
                self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Auto-exposure: 1 = manual, 3 = auto
                self._capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
                
                # Flush buffer to get fresh frames
                for _ in range(10):
                    self._capture.grab()
                
                # Read actual settings
                actual_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = self._capture.get(cv2.CAP_PROP_FPS)
                
                print(f"Camera {self.name}: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")
                
                # Update our width/height to actual values
                self.width = actual_width
                self.height = actual_height
                
                return True
                
            except Exception as e:
                print(f"Error opening camera {self.name}: {e}")
                self._capture = None
                return False
    
    def close(self):
        """Close the camera device."""
        with self._lock:
            if self._capture is not None:
                self._capture.release()
                self._capture = None
    
    @property
    def is_open(self) -> bool:
        """Check if camera is open."""
        with self._lock:
            return self._capture is not None and self._capture.isOpened()
    
    def read(self) -> Optional[CameraFrame]:
        """Read a frame from the camera."""
        with self._lock:
            if self._capture is None:
                return None
            
            timestamp = time.perf_counter()
            ret, frame = self._capture.read()
            
            if not ret or frame is None:
                return None
            
            camera_frame = CameraFrame(
                timestamp=timestamp,
                frame=frame,
                camera_name=self.name,
                width=frame.shape[1],
                height=frame.shape[0]
            )
            
            self._last_frame = camera_frame
            return camera_frame
    
    @property
    def last_frame(self) -> Optional[CameraFrame]:
        """Get the last captured frame."""
        return self._last_frame


class CameraManager:
    """Manager for multiple synchronized cameras."""
    
    def __init__(self):
        self._cameras: Dict[str, Camera] = {}
        self._capture_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.RLock()
        
        # Frame buffers for each camera
        self._frame_buffers: Dict[str, CameraFrame] = {}
        
        # Callbacks
        self._frame_callbacks: List[Callable[[Dict[str, CameraFrame]], None]] = []
    
    def add_camera(
        self,
        name: str,
        device_index: int,
        width: int = 1280,
        height: int = 720,
        fps: int = 30
    ) -> bool:
        """Add a camera to the manager."""
        with self._lock:
            if name in self._cameras:
                print(f"Warning: Camera {name} already exists")
                return False
            
            camera = Camera(name, device_index, width, height, fps)
            self._cameras[name] = camera
            return True
    
    def open_all(self) -> Dict[str, bool]:
        """Open all cameras. Returns dict of name -> success."""
        results = {}
        with self._lock:
            for name, camera in self._cameras.items():
                results[name] = camera.open()
        return results
    
    def close_all(self):
        """Close all cameras."""
        self.stop_capture()
        with self._lock:
            for camera in self._cameras.values():
                camera.close()
    
    def get_camera(self, name: str) -> Optional[Camera]:
        """Get a camera by name."""
        return self._cameras.get(name)
    
    @property
    def camera_names(self) -> List[str]:
        """Get list of camera names."""
        return list(self._cameras.keys())
    
    def read_all(self) -> Dict[str, Optional[CameraFrame]]:
        """Read frames from all cameras."""
        frames = {}
        with self._lock:
            for name, camera in self._cameras.items():
                frames[name] = camera.read()
                if frames[name] is not None:
                    self._frame_buffers[name] = frames[name]
        return frames
    
    def get_latest_frames(self) -> Dict[str, Optional[CameraFrame]]:
        """Get the latest frame from each camera without blocking."""
        with self._lock:
            return self._frame_buffers.copy()
    
    def register_frame_callback(self, callback: Callable[[Dict[str, CameraFrame]], None]):
        """Register a callback for new frames."""
        self._frame_callbacks.append(callback)
    
    def start_capture(self, target_fps: float = 30.0):
        """Start continuous capture in background thread."""
        if self._running:
            return
        
        self._running = True
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            args=(target_fps,),
            daemon=True
        )
        self._capture_thread.start()
    
    def stop_capture(self):
        """Stop continuous capture."""
        self._running = False
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None
    
    def _capture_loop(self, target_fps: float):
        """Background capture loop."""
        target_period = 1.0 / target_fps
        
        while self._running:
            loop_start = time.perf_counter()
            
            # Capture from all cameras
            frames = self.read_all()
            
            # Filter to only successful captures
            valid_frames = {k: v for k, v in frames.items() if v is not None}
            
            # Call callbacks
            if valid_frames:
                for callback in self._frame_callbacks:
                    try:
                        callback(valid_frames)
                    except Exception as e:
                        print(f"Frame callback error: {e}")
            
            # Maintain target FPS
            elapsed = time.perf_counter() - loop_start
            sleep_time = target_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    @property
    def is_running(self) -> bool:
        """Check if capture is running."""
        return self._running
    
    def get_camera_info(self) -> Dict[str, dict]:
        """Get info about all cameras."""
        info = {}
        with self._lock:
            for name, camera in self._cameras.items():
                info[name] = {
                    "device_index": camera.device_index,
                    "width": camera.width,
                    "height": camera.height,
                    "fps": camera.fps,
                    "is_open": camera.is_open
                }
        return info
