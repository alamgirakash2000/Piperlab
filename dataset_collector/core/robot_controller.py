#!/usr/bin/env python3
"""Robot controller wrapper for AgileX Piper robot."""

import time
import json
import subprocess
import threading
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto

try:
    from piper_sdk import C_PiperInterface_V2
except ImportError:
    C_PiperInterface_V2 = None
    print("Warning: piper_sdk not found. Robot functionality will be limited.")


class RobotMode(Enum):
    """Robot operating modes."""
    DISCONNECTED = auto()
    IDLE = auto()
    TEACH = auto()
    PLAYBACK = auto()
    MOVING = auto()


@dataclass
class RobotState:
    """Complete robot state at a single timestamp."""
    timestamp: float
    joint_positions: np.ndarray  # 6 joint angles in degrees
    joint_velocities: np.ndarray  # 6 joint velocities in deg/s
    gripper_position: float  # 0-100 (mm or percentage)
    mode: RobotMode = RobotMode.DISCONNECTED
    
    def to_radians(self) -> np.ndarray:
        """Get joint positions in radians."""
        return np.deg2rad(self.joint_positions)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "joint_positions": self.joint_positions.tolist(),
            "joint_velocities": self.joint_velocities.tolist(),
            "gripper_position": float(self.gripper_position),
            "mode": self.mode.name
        }


@dataclass
class TrajectoryPoint:
    """Single point in a trajectory."""
    timestamp: float
    joint_positions: np.ndarray
    gripper_position: float


class RobotController:
    """Controller for AgileX Piper 6-DOF robot arm with gripper."""
    
    # Conversion constants
    DEG_TO_UNITS = 1000  # Degrees to SDK units
    MM_TO_UNITS = 1000   # Millimeters to SDK units
    
    def __init__(self, can_interface: str = "can0"):
        self.can_interface = can_interface
        self._piper: Optional[C_PiperInterface_V2] = None
        self._mode = RobotMode.DISCONNECTED
        self._lock = threading.RLock()
        
        # State tracking
        self._last_state: Optional[RobotState] = None
        self._prev_positions: Optional[np.ndarray] = None
        self._prev_time: Optional[float] = None
        
        # Trajectory recording
        self._trajectory: List[TrajectoryPoint] = []
        self._is_recording_trajectory = False
        
        # Teach mode control
        self._teach_thread: Optional[threading.Thread] = None
        self._teach_stop_flag = threading.Event()
        
        # Offsets (from calibration)
        self._offsets = np.zeros(6)
        self._offsets_path = Path(__file__).parent.parent.parent / "zero_offsets.json"
        self._load_offsets()
        
        # Callbacks
        self._state_callbacks: List[Callable[[RobotState], None]] = []
    
    def _load_offsets(self):
        """Load joint zero offsets from file."""
        if self._offsets_path.exists():
            try:
                data = json.loads(self._offsets_path.read_text())
                if isinstance(data, list) and len(data) == 6:
                    self._offsets = np.array(data, dtype=np.float64)
            except Exception as e:
                print(f"Warning: Could not load offsets: {e}")
    
    def _setup_can(self) -> bool:
        """Setup CAN interface with sudo if needed."""
        try:
            result = subprocess.run(
                ["ip", "link", "show", self.can_interface],
                capture_output=True, text=True
            )
            if "UP" in result.stdout:
                return True
            
            # Try to bring up CAN interface
            for cmd in [
                ["sudo", "ip", "link", "set", self.can_interface, "down"],
                ["sudo", "ip", "link", "set", self.can_interface, "type", "can", "bitrate", "1000000"],
                ["sudo", "ip", "link", "set", self.can_interface, "up"],
            ]:
                subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
            
            result = subprocess.run(
                ["ip", "link", "show", self.can_interface],
                capture_output=True, text=True
            )
            return "UP" in result.stdout
        except Exception as e:
            print(f"CAN setup failed: {e}")
            return False
    
    def connect(self) -> bool:
        """Connect to the robot and enable control."""
        with self._lock:
            if self._piper is not None:
                return True
            
            if C_PiperInterface_V2 is None:
                print("Error: piper_sdk not available")
                return False
            
            if not self._setup_can():
                print("Error: Could not setup CAN interface")
                return False
            
            try:
                self._piper = C_PiperInterface_V2()
                self._piper.ConnectPort()
                
                # Wait for enable - use infinite loop like original robot_connection.py
                while not self._piper.EnablePiper():
                    time.sleep(0.01)
                
                time.sleep(0.5)
                self._mode = RobotMode.IDLE
                print("✓ Robot connected and enabled")
                return True
                
            except Exception as e:
                print(f"Error connecting to robot: {e}")
                self._piper = None
                return False
    
    def connect_passive(self) -> bool:
        """Connect to the robot in passive mode (read-only, no enable command).
        
        Use this when the physical button is being used for control.
        The app will only observe robot state, not send commands.
        """
        with self._lock:
            if self._piper is not None:
                return True
            
            if C_PiperInterface_V2 is None:
                print("Error: piper_sdk not available")
                return False
            
            if not self._setup_can():
                print("Error: Could not setup CAN interface")
                return False
            
            try:
                self._piper = C_PiperInterface_V2()
                self._piper.ConnectPort()
                
                # Do NOT enable - just connect to read state
                time.sleep(0.3)
                self._mode = RobotMode.IDLE
                print("✓ Robot connected in PASSIVE mode (read-only)")
                return True
                
            except Exception as e:
                print(f"Error connecting to robot: {e}")
                self._piper = None
                return False
    
    def re_enable(self) -> bool:
        """Re-enable the robot after physical button use or mode change."""
        with self._lock:
            if self._piper is None:
                return False
            
            try:
                # First exit any drag mode
                for _ in range(5):
                    self._piper.MotionCtrl_1(0x00, 0x00, 0x02)
                    time.sleep(0.05)
                
                # Reset to position control mode
                self._piper.MotionCtrl_2(0x01, 0x01, 30, 0x00)
                time.sleep(0.1)
                
                # Re-enable the robot (try multiple times)
                enabled = False
                for _ in range(100):
                    if self._piper.EnablePiper():
                        enabled = True
                        break
                    time.sleep(0.02)
                
                if not enabled:
                    print("Warning: EnablePiper did not return True, but continuing...")
                
                # Send another motion control command to ensure position mode
                self._piper.MotionCtrl_2(0x01, 0x01, 30, 0x00)
                time.sleep(0.2)
                
                self._mode = RobotMode.IDLE
                print("✓ Robot re-enabled")
                return True
            except Exception as e:
                print(f"Error re-enabling robot: {e}")
                return False
    
    def disconnect(self):
        """Disconnect from the robot."""
        self.disable_teach_mode()  # Stop teach thread if running
        with self._lock:
            self._piper = None
            self._mode = RobotMode.DISCONNECTED
    
    @property
    def is_connected(self) -> bool:
        """Check if robot is connected."""
        with self._lock:
            return self._piper is not None
    
    @property
    def mode(self) -> RobotMode:
        """Get current robot mode."""
        return self._mode
    
    def get_joint_positions_raw(self) -> np.ndarray:
        """Get raw joint positions in degrees (without offset correction)."""
        with self._lock:
            if self._piper is None:
                return np.zeros(6)
            
            try:
                msg = self._piper.GetArmJointMsgs()
                try:
                    joint_state = msg[2]
                except (TypeError, IndexError):
                    joint_state = getattr(msg, "joint_state", msg)
                
                positions = np.array([
                    getattr(joint_state, f"joint_{i}", 0) / self.DEG_TO_UNITS
                    for i in range(1, 7)
                ], dtype=np.float64)
                
                return positions
            except Exception as e:
                print(f"Error reading joints: {e}")
                return np.zeros(6)
    
    def get_joint_positions(self) -> np.ndarray:
        """Get joint positions in degrees (with offset correction)."""
        raw = self.get_joint_positions_raw()
        return raw - self._offsets
    
    def get_gripper_position(self) -> float:
        """Get gripper position (0-100)."""
        with self._lock:
            if self._piper is None:
                return 0.0
            
            try:
                msg = self._piper.GetArmGripperMsgs()
                try:
                    gripper_state = msg[2]
                except (TypeError, IndexError):
                    gripper_state = getattr(msg, "gripper_state", msg)
                
                # Get gripper position in mm and convert to percentage
                pos_mm = getattr(gripper_state, "grippers_angle", 0) / self.MM_TO_UNITS
                return float(np.clip(pos_mm, 0, 100))
            except Exception:
                return 0.0
    
    def get_state(self) -> RobotState:
        """Get complete robot state."""
        now = time.perf_counter()
        positions = self.get_joint_positions()
        gripper = self.get_gripper_position()
        
        # Calculate velocities from position change
        if self._prev_positions is not None and self._prev_time is not None:
            dt = now - self._prev_time
            if dt > 0.001:
                velocities = (positions - self._prev_positions) / dt
            else:
                velocities = np.zeros(6)
        else:
            velocities = np.zeros(6)
        
        self._prev_positions = positions.copy()
        self._prev_time = now
        
        state = RobotState(
            timestamp=now,
            joint_positions=positions,
            joint_velocities=velocities,
            gripper_position=gripper,
            mode=self._mode
        )
        
        self._last_state = state
        
        # Call registered callbacks
        for callback in self._state_callbacks:
            try:
                callback(state)
            except Exception:
                pass
        
        return state
    
    def register_state_callback(self, callback: Callable[[RobotState], None]):
        """Register a callback to be called when state is updated."""
        self._state_callbacks.append(callback)
    
    def set_joint_positions(self, positions: List[float], speed: int = 30):
        """Set joint positions in degrees."""
        with self._lock:
            if self._piper is None:
                return
            
            try:
                # Re-enable motion control each time to recover from physical button
                self._piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)
                time.sleep(0.02)
                
                # Apply offsets and convert to SDK units
                pos_with_offsets = [p + o for p, o in zip(positions, self._offsets)]
                units = [int(p * self.DEG_TO_UNITS) for p in pos_with_offsets]
                
                self._piper.JointCtrl(*units)
            except Exception as e:
                print(f"Joint control error (try re-enabling): {e}")
    
    def set_gripper(self, position: float, effort: int = 2000):
        """Set gripper position (0-100 mm)."""
        with self._lock:
            if self._piper is None:
                return
            
            pos_units = int(np.clip(position, 0, 100) * self.MM_TO_UNITS)
            self._piper.GripperCtrl(pos_units, effort, 1, 0)
    
    def go_home(self, speed: int = 20, timeout: float = 6.0) -> bool:
        """Move robot to home position."""
        print("→ Moving to home position...")
        
        with self._lock:
            if self._piper is None:
                print("Error: Robot not connected")
                return False
            
            try:
                # Home position (user-specified)
                home_pos = [0.04, 0.38, 0.11, 3.98, -18.01, 2.29]
                home_gripper = 50.0
                
                # Apply offsets and convert to SDK units
                pos_with_offsets = [p + o for p, o in zip(home_pos, self._offsets)]
                units = [int(p * self.DEG_TO_UNITS) for p in pos_with_offsets]
                
                # Set position control mode and send joint command
                self._piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)
                time.sleep(0.05)
                self._piper.JointCtrl(*units)
                
                # Set gripper
                gripper_units = int(home_gripper * self.MM_TO_UNITS)
                self._piper.GripperCtrl(gripper_units, 2000, 1, 0)
                
            except Exception as e:
                print(f"Error sending home command: {e}")
                return False
        
        # Wait for movement
        time.sleep(timeout)
        print("✓ Home position reached")
        return True
    
    def enable_teach_mode(self) -> bool:
        """Enable teach/drag mode (allowing manual positioning).
        
        Uses a background thread to continuously send drag-teach commands,
        mimicking holding the physical teach button.
        """
        with self._lock:
            if self._piper is None:
                return False
            
            # Stop any existing teach thread
            if self._teach_thread is not None:
                self._teach_stop_flag.set()
                self._teach_thread.join(timeout=1.0)
            
            self._teach_stop_flag.clear()
            self._mode = RobotMode.TEACH
        
        # Start continuous teach mode thread
        self._teach_thread = threading.Thread(
            target=self._teach_mode_loop,
            daemon=True
        )
        self._teach_thread.start()
        
        print("✓ Drag-teach mode enabled - you can now move the robot by hand")
        return True
    
    def _teach_mode_loop(self):
        """Continuously send teach mode commands (like holding physical button)."""
        while not self._teach_stop_flag.is_set():
            try:
                with self._lock:
                    if self._piper is not None:
                        # Continuously send drag-teach enable command
                        self._piper.MotionCtrl_1(0x00, 0x00, 0x01)
            except Exception:
                pass
            # Send command at ~20 Hz
            time.sleep(0.05)
    
    def disable_teach_mode(self) -> bool:
        """Disable teach mode (return to position control)."""
        # Stop the teach thread first
        self._teach_stop_flag.set()
        if self._teach_thread is not None:
            self._teach_thread.join(timeout=1.0)
            self._teach_thread = None
        
        with self._lock:
            if self._piper is None:
                return False
            
            try:
                # Exit drag-teach mode: grag_teach = 0x02
                # Send multiple times to ensure it takes effect
                for i in range(20):
                    self._piper.MotionCtrl_1(0x00, 0x00, 0x02)
                    time.sleep(0.03)
                
                # Re-enable the robot for position control
                for _ in range(50):
                    self._piper.EnablePiper()
                    time.sleep(0.02)
                
                # Set position control mode
                self._piper.MotionCtrl_2(0x01, 0x01, 30, 0x00)
                time.sleep(0.1)
                
                print("✓ Drag-teach mode disabled - robot ready for commands")
                self._mode = RobotMode.IDLE
                return True
            except Exception as e:
                print(f"Error disabling teach mode: {e}")
                return False
    
    def start_trajectory_recording(self):
        """Start recording a trajectory during teach mode."""
        self._trajectory = []
        self._is_recording_trajectory = True
    
    def stop_trajectory_recording(self) -> List[TrajectoryPoint]:
        """Stop recording and return the recorded trajectory."""
        self._is_recording_trajectory = False
        return self._trajectory.copy()
    
    def record_trajectory_point(self):
        """Record the current position as a trajectory point."""
        if not self._is_recording_trajectory:
            return
        
        state = self.get_state()
        point = TrajectoryPoint(
            timestamp=state.timestamp,
            joint_positions=state.joint_positions.copy(),
            gripper_position=state.gripper_position
        )
        self._trajectory.append(point)
    
    def play_trajectory(
        self,
        trajectory: List[TrajectoryPoint],
        speed: int = 30,
        on_point: Optional[Callable[[int, TrajectoryPoint], None]] = None
    ) -> bool:
        """
        Play back a recorded trajectory.
        
        Args:
            trajectory: List of trajectory points to play
            speed: Motion speed (1-100)
            on_point: Callback called at each point (index, point)
        
        Returns:
            True if playback completed successfully
        """
        if not trajectory:
            return False
        
        with self._lock:
            if self._piper is None:
                return False
            self._mode = RobotMode.PLAYBACK
        
        try:
            # Calculate relative timestamps
            base_time = trajectory[0].timestamp
            
            for i, point in enumerate(trajectory):
                if self._mode != RobotMode.PLAYBACK:
                    break  # Playback was interrupted
                
                # Move to this point
                self.set_joint_positions(point.joint_positions.tolist(), speed=speed)
                self.set_gripper(point.gripper_position)
                
                # Call progress callback
                if on_point:
                    on_point(i, point)
                
                # Wait for timing if not the last point
                if i < len(trajectory) - 1:
                    next_point = trajectory[i + 1]
                    dt = next_point.timestamp - point.timestamp
                    if dt > 0:
                        time.sleep(dt)
                else:
                    # Wait a bit for the last point to complete
                    time.sleep(0.5)
            
            return True
            
        except Exception as e:
            print(f"Error during playback: {e}")
            return False
        finally:
            self._mode = RobotMode.IDLE
    
    def stop_playback(self):
        """Stop trajectory playback."""
        self._mode = RobotMode.IDLE
    
    @property
    def trajectory(self) -> List[TrajectoryPoint]:
        """Get the currently recorded trajectory."""
        return self._trajectory
