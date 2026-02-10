#!/usr/bin/env python3
"""Teach-and-play manager for demonstration collection workflow."""

import time
import threading
from typing import Optional, List, Callable, Any
from dataclasses import dataclass
from enum import Enum, auto

from .robot_controller import RobotController, TrajectoryPoint, RobotMode
from .camera_manager import CameraManager
from .data_recorder import DataRecorder, Episode


class WorkflowState(Enum):
    """States in the teach-play-record workflow."""
    IDLE = auto()
    TEACHING = auto()
    READY_TO_PLAY = auto()
    PLAYING = auto()
    RECORDING = auto()
    REVIEW = auto()


@dataclass
class WorkflowStatus:
    """Current status of the workflow."""
    state: WorkflowState
    trajectory_points: int = 0
    playback_progress: float = 0.0
    recording_frames: int = 0
    recording_duration: float = 0.0
    message: str = ""
    error: Optional[str] = None


class TeachPlayManager:
    """
    Manages the teach-play-record workflow for VLA demonstration collection.
    
    Workflow:
    1. IDLE -> Enter teach mode
    2. TEACHING -> User drags robot, points are recorded
    3. READY_TO_PLAY -> User presses Play
    4. PLAYING+RECORDING -> Robot replays motion while recording data
    5. REVIEW -> User reviews and saves/discards the demo
    """
    
    def __init__(
        self,
        robot: RobotController,
        camera_manager: CameraManager,
        recorder: DataRecorder,
        teach_record_hz: float = 30.0
    ):
        self.robot = robot
        self.camera_manager = camera_manager
        self.recorder = recorder
        self.teach_record_hz = teach_record_hz
        
        self._state = WorkflowState.IDLE
        self._trajectory: List[TrajectoryPoint] = []
        self._current_task: str = ""
        self._last_episode: Optional[Episode] = None
        
        # Thread control
        self._teach_thread: Optional[threading.Thread] = None
        self._play_thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        
        # Progress tracking
        self._playback_index = 0
        
        # Callbacks
        self._on_state_change: List[Callable[[WorkflowState], None]] = []
        self._on_playback_progress: List[Callable[[float, int], None]] = []
        self._on_recording_complete: List[Callable[[Episode], None]] = []
        
        self._lock = threading.RLock()
    
    @property
    def state(self) -> WorkflowState:
        """Get current workflow state."""
        return self._state
    
    @property
    def trajectory(self) -> List[TrajectoryPoint]:
        """Get the recorded trajectory."""
        return self._trajectory
    
    @property
    def last_episode(self) -> Optional[Episode]:
        """Get the last recorded episode."""
        return self._last_episode
    
    def _set_state(self, new_state: WorkflowState):
        """Set state and notify callbacks."""
        self._state = new_state
        for callback in self._on_state_change:
            try:
                callback(new_state)
            except Exception as e:
                print(f"State change callback error: {e}")
    
    def register_state_callback(self, callback: Callable[[WorkflowState], None]):
        """Register callback for state changes."""
        self._on_state_change.append(callback)
    
    def register_playback_callback(self, callback: Callable[[float, int], None]):
        """Register callback for playback progress (progress 0-1, point_index)."""
        self._on_playback_progress.append(callback)
    
    def register_recording_complete_callback(self, callback: Callable[[Episode], None]):
        """Register callback for when recording completes."""
        self._on_recording_complete.append(callback)
    
    def get_status(self) -> WorkflowStatus:
        """Get current workflow status."""
        with self._lock:
            recording_stats = self.recorder.get_recording_stats()
            
            return WorkflowStatus(
                state=self._state,
                trajectory_points=len(self._trajectory),
                playback_progress=self._playback_index / max(len(self._trajectory), 1),
                recording_frames=recording_stats.get("frames", 0),
                recording_duration=recording_stats.get("duration", 0.0),
                message=self._get_state_message()
            )
    
    def _get_state_message(self) -> str:
        """Get human-readable message for current state."""
        messages = {
            WorkflowState.IDLE: "Ready - Press 'Teach' to start",
            WorkflowState.TEACHING: f"Teaching... ({len(self._trajectory)} points)",
            WorkflowState.READY_TO_PLAY: f"Ready to play ({len(self._trajectory)} points)",
            WorkflowState.PLAYING: "Playing trajectory...",
            WorkflowState.RECORDING: "Recording demonstration...",
            WorkflowState.REVIEW: "Review the recorded demo",
        }
        return messages.get(self._state, "Unknown state")
    
    def start_teaching(self, task: str) -> bool:
        """Start the teaching phase."""
        with self._lock:
            if self._state not in [WorkflowState.IDLE, WorkflowState.READY_TO_PLAY, WorkflowState.REVIEW]:
                return False
            
            if not self.robot.is_connected:
                return False
            
            # Clear previous trajectory
            self._trajectory = []
            self._current_task = task
            self._stop_flag.clear()
            
            # Enable teach mode on robot
            if not self.robot.enable_teach_mode():
                return False
            
            self._set_state(WorkflowState.TEACHING)
            
            # Start recording trajectory in background
            self._teach_thread = threading.Thread(
                target=self._teach_loop,
                daemon=True
            )
            self._teach_thread.start()
            
            return True
    
    def _teach_loop(self):
        """Background loop to record trajectory during teaching."""
        period = 1.0 / self.teach_record_hz
        
        while not self._stop_flag.is_set() and self._state == WorkflowState.TEACHING:
            try:
                # Get current robot state
                state = self.robot.get_state()
                
                # Create trajectory point
                point = TrajectoryPoint(
                    timestamp=time.perf_counter(),
                    joint_positions=state.joint_positions.copy(),
                    gripper_position=state.gripper_position
                )
                
                with self._lock:
                    self._trajectory.append(point)
                
                time.sleep(period)
                
            except Exception as e:
                print(f"Teach loop error: {e}")
                break
    
    def stop_teaching(self) -> int:
        """Stop teaching and return number of recorded points."""
        with self._lock:
            if self._state != WorkflowState.TEACHING:
                return 0
            
            self._stop_flag.set()
        
        # Wait for teach thread
        if self._teach_thread is not None:
            self._teach_thread.join(timeout=2.0)
            self._teach_thread = None
        
        # Disable teach mode
        self.robot.disable_teach_mode()
        
        with self._lock:
            num_points = len(self._trajectory)
            
            if num_points > 0:
                self._set_state(WorkflowState.READY_TO_PLAY)
            else:
                self._set_state(WorkflowState.IDLE)
            
            return num_points
    
    def start_playback_with_recording(self, speed: int = 30) -> bool:
        """Start playing back the trajectory while recording data."""
        with self._lock:
            if self._state != WorkflowState.READY_TO_PLAY:
                return False
            
            if not self._trajectory:
                return False
            
            self._stop_flag.clear()
            self._playback_index = 0
            self._set_state(WorkflowState.PLAYING)
        
        # Start playback in background
        self._play_thread = threading.Thread(
            target=self._playback_loop,
            args=(speed,),
            daemon=True
        )
        self._play_thread.start()
        
        return True
    
    def _playback_loop(self, speed: int):
        """Background loop for playback with recording."""
        try:
            # Start recording
            if not self.recorder.start_recording(self._current_task):
                print("Failed to start recording")
                self._set_state(WorkflowState.READY_TO_PLAY)
                return
            
            self._set_state(WorkflowState.RECORDING)
            
            # Calculate base time for trajectory timing
            if not self._trajectory:
                return
            
            base_time = self._trajectory[0].timestamp
            playback_start = time.perf_counter()
            
            for i, point in enumerate(self._trajectory):
                if self._stop_flag.is_set():
                    break
                
                self._playback_index = i
                
                # Move robot to this point
                self.robot.set_joint_positions(point.joint_positions.tolist(), speed=speed)
                self.robot.set_gripper(point.gripper_position)
                
                # Record data step
                self.recorder.record_step()
                
                # Notify progress callbacks
                progress = (i + 1) / len(self._trajectory)
                for callback in self._on_playback_progress:
                    try:
                        callback(progress, i)
                    except Exception:
                        pass
                
                # Wait for next point timing
                if i < len(self._trajectory) - 1:
                    next_point = self._trajectory[i + 1]
                    target_time = (next_point.timestamp - base_time)
                    elapsed = time.perf_counter() - playback_start
                    wait_time = target_time - elapsed
                    
                    if wait_time > 0:
                        time.sleep(wait_time)
            
            # Wait a moment at the end for final position
            time.sleep(0.3)
            
            # Stop recording
            self._last_episode = self.recorder.stop_recording()
            
            if self._last_episode:
                for callback in self._on_recording_complete:
                    try:
                        callback(self._last_episode)
                    except Exception as e:
                        print(f"Recording complete callback error: {e}")
            
            self._set_state(WorkflowState.REVIEW)
            
        except Exception as e:
            print(f"Playback error: {e}")
            import traceback
            traceback.print_exc()
            self.recorder.stop_recording()
            self._set_state(WorkflowState.READY_TO_PLAY)
    
    def stop_playback(self):
        """Stop current playback."""
        self._stop_flag.set()
        
        if self._play_thread is not None:
            self._play_thread.join(timeout=2.0)
            self._play_thread = None
        
        self.recorder.stop_recording()
        
        with self._lock:
            if self._trajectory:
                self._set_state(WorkflowState.READY_TO_PLAY)
            else:
                self._set_state(WorkflowState.IDLE)
    
    def accept_recording(self) -> Optional[Episode]:
        """Accept the last recording for saving to dataset."""
        with self._lock:
            if self._state != WorkflowState.REVIEW:
                return None
            
            episode = self._last_episode
            self._last_episode = None
            self._trajectory = []
            self._set_state(WorkflowState.IDLE)
            
            return episode
    
    def discard_recording(self):
        """Discard the last recording."""
        with self._lock:
            self._last_episode = None
            self._set_state(WorkflowState.READY_TO_PLAY)
    
    def reset(self):
        """Reset the workflow to idle state."""
        self.stop_playback()
        self.stop_teaching()
        
        with self._lock:
            self._trajectory = []
            self._last_episode = None
            self._current_task = ""
            self._set_state(WorkflowState.IDLE)
