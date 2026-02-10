#!/usr/bin/env python3
"""Synchronized data recorder for VLA demonstration capture."""

import time
import threading
import numpy as np
from typing import Optional, Dict, List, Callable, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import cv2

from .robot_controller import RobotController, RobotState
from .camera_manager import CameraManager, CameraFrame
from ..utils.sync import SyncClock, RateController, SyncBuffer


@dataclass
class RecordedStep:
    """Single recorded step in a demonstration."""
    timestamp: float  # Seconds from episode start
    frame_index: int
    
    # Visual observations (stored as references, frames saved separately)
    images: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Robot state
    joint_positions: np.ndarray = field(default_factory=lambda: np.zeros(6))
    joint_velocities: np.ndarray = field(default_factory=lambda: np.zeros(6))
    gripper_position: float = 0.0
    
    # Computed actions (deltas from previous step)
    joint_position_delta: np.ndarray = field(default_factory=lambda: np.zeros(6))
    gripper_action: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary (without images)."""
        return {
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
            "observation.state.joint_positions": self.joint_positions.tolist(),
            "observation.state.joint_velocities": self.joint_velocities.tolist(),
            "observation.state.gripper_position": self.gripper_position,
            "action.joint_position_delta": self.joint_position_delta.tolist(),
            "action.gripper_action": self.gripper_action,
        }


@dataclass
class Episode:
    """Complete recorded episode (demonstration)."""
    episode_id: int
    task: str
    start_time: float
    steps: List[RecordedStep] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        """Get episode duration in seconds."""
        if not self.steps:
            return 0.0
        return self.steps[-1].timestamp - self.steps[0].timestamp
    
    @property
    def num_frames(self) -> int:
        """Get number of frames."""
        return len(self.steps)
    
    @property
    def fps(self) -> float:
        """Calculate actual FPS."""
        if self.duration <= 0 or self.num_frames <= 1:
            return 0.0
        return (self.num_frames - 1) / self.duration
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "episode_id": self.episode_id,
            "task": self.task,
            "start_time": self.start_time,
            "duration": self.duration,
            "num_frames": self.num_frames,
            "fps": self.fps,
            "metadata": self.metadata,
        }


class DataRecorder:
    """Synchronized recorder for robot state and camera frames."""
    
    def __init__(
        self,
        robot: RobotController,
        camera_manager: CameraManager,
        target_fps: float = 30.0
    ):
        self.robot = robot
        self.camera_manager = camera_manager
        self.target_fps = target_fps
        
        self._clock = SyncClock()
        self._rate_controller = RateController(target_fps)
        
        # Recording state
        self._is_recording = False
        self._current_episode: Optional[Episode] = None
        self._episode_counter = 0
        self._record_thread: Optional[threading.Thread] = None
        
        # Previous state for delta calculation
        self._prev_joint_positions: Optional[np.ndarray] = None
        self._prev_gripper: Optional[float] = None
        
        # Callbacks
        self._on_step_recorded: List[Callable[[RecordedStep], None]] = []
        self._on_recording_complete: List[Callable[[Episode], None]] = []
        
        self._lock = threading.RLock()
    
    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._is_recording
    
    @property
    def current_episode(self) -> Optional[Episode]:
        """Get the current episode being recorded."""
        return self._current_episode
    
    def register_step_callback(self, callback: Callable[[RecordedStep], None]):
        """Register callback for each recorded step."""
        self._on_step_recorded.append(callback)
    
    def register_complete_callback(self, callback: Callable[[Episode], None]):
        """Register callback for when recording completes."""
        self._on_recording_complete.append(callback)
    
    def start_recording(self, task: str) -> bool:
        """Start recording a new episode."""
        with self._lock:
            if self._is_recording:
                return False
            
            self._episode_counter += 1
            self._current_episode = Episode(
                episode_id=self._episode_counter,
                task=task,
                start_time=time.perf_counter(),
                metadata={
                    "camera_names": self.camera_manager.camera_names,
                    "target_fps": self.target_fps,
                    "robot_connected": self.robot.is_connected,
                }
            )
            
            self._prev_joint_positions = None
            self._prev_gripper = None
            self._is_recording = True
            self._clock.start()
            self._rate_controller.reset()
            
            return True
    
    def stop_recording(self) -> Optional[Episode]:
        """Stop recording and return the completed episode."""
        with self._lock:
            if not self._is_recording:
                return None
            
            self._is_recording = False
            self._clock.stop()
            
            episode = self._current_episode
            self._current_episode = None
            
            if episode:
                # Add final metadata
                episode.metadata["actual_fps"] = episode.fps
                episode.metadata["total_frames"] = episode.num_frames
                episode.metadata["duration_seconds"] = episode.duration
                episode.metadata["timing_overruns"] = self._rate_controller.overrun_count
                
                # Call completion callbacks
                for callback in self._on_recording_complete:
                    try:
                        callback(episode)
                    except Exception as e:
                        print(f"Recording complete callback error: {e}")
            
            return episode
    
    def record_step(self) -> Optional[RecordedStep]:
        """Record a single synchronized step."""
        if not self._is_recording or self._current_episode is None:
            return None
        
        # Get current time
        timestamp = self._clock.get_time()
        frame_index = len(self._current_episode.steps)
        
        # Capture robot state
        robot_state = self.robot.get_state()
        
        # Capture camera frames
        camera_frames = self.camera_manager.get_latest_frames()
        images = {}
        for name, frame in camera_frames.items():
            if frame is not None:
                images[name] = frame.frame.copy()
        
        # Calculate deltas
        if self._prev_joint_positions is not None:
            joint_delta = robot_state.joint_positions - self._prev_joint_positions
        else:
            joint_delta = np.zeros(6)
        
        if self._prev_gripper is not None:
            gripper_action = robot_state.gripper_position - self._prev_gripper
        else:
            gripper_action = 0.0
        
        # Normalize gripper action to [-1, 1]
        gripper_action_normalized = np.clip(gripper_action / 100.0, -1.0, 1.0)
        
        # Create step
        step = RecordedStep(
            timestamp=timestamp,
            frame_index=frame_index,
            images=images,
            joint_positions=robot_state.joint_positions.copy(),
            joint_velocities=robot_state.joint_velocities.copy(),
            gripper_position=robot_state.gripper_position,
            joint_position_delta=joint_delta,
            gripper_action=gripper_action_normalized,
        )
        
        # Update previous state
        self._prev_joint_positions = robot_state.joint_positions.copy()
        self._prev_gripper = robot_state.gripper_position
        
        # Add to episode
        with self._lock:
            if self._current_episode is not None:
                self._current_episode.steps.append(step)
        
        # Call callbacks
        for callback in self._on_step_recorded:
            try:
                callback(step)
            except Exception as e:
                print(f"Step callback error: {e}")
        
        return step
    
    def record_continuously(self, stop_condition: Optional[Callable[[], bool]] = None):
        """
        Record continuously until stop_condition returns True or stop_recording is called.
        This runs in the calling thread (blocking).
        """
        while self._is_recording:
            if stop_condition and stop_condition():
                break
            
            self.record_step()
            self._rate_controller.sleep()
    
    def start_continuous_recording(
        self,
        task: str,
        stop_condition: Optional[Callable[[], bool]] = None
    ) -> bool:
        """Start recording in a background thread."""
        if not self.start_recording(task):
            return False
        
        self._record_thread = threading.Thread(
            target=self.record_continuously,
            args=(stop_condition,),
            daemon=True
        )
        self._record_thread.start()
        return True
    
    def wait_for_recording(self, timeout: Optional[float] = None):
        """Wait for recording thread to complete."""
        if self._record_thread is not None:
            self._record_thread.join(timeout=timeout)
            self._record_thread = None
    
    def get_recording_stats(self) -> dict:
        """Get current recording statistics."""
        with self._lock:
            if self._current_episode is None:
                return {
                    "is_recording": False,
                    "frames": 0,
                    "duration": 0.0,
                    "fps": 0.0
                }
            
            return {
                "is_recording": True,
                "episode_id": self._current_episode.episode_id,
                "task": self._current_episode.task,
                "frames": self._current_episode.num_frames,
                "duration": self._clock.get_time(),
                "target_fps": self.target_fps,
                "timing_overruns": self._rate_controller.overrun_count
            }
