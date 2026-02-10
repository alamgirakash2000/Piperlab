#!/usr/bin/env python3
"""Main window for VLA Dataset Collector - Simplified for physical button workflow.

This version is designed to work with the AligeX Piper's physical teach button:
1. User teaches robot using physical button (single press to drag)
2. User double-presses physical button to replay
3. User clicks RECORD in app during replay to capture data
4. User clicks STOP when done, then saves or discards
"""

import sys
import time
import threading
from pathlib import Path
from typing import Optional, Dict

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFrame, QComboBox, QListWidget, QListWidgetItem,
    QProgressBar, QGroupBox, QSizePolicy,
    QMessageBox, QApplication
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QPixmap, QImage, QFont

import numpy as np
import cv2

from .styles import STYLESHEET, COLORS
from ..core.robot_controller import RobotController, RobotState
from ..core.camera_manager import CameraManager, CameraFrame
from ..core.data_recorder import DataRecorder, Episode
from ..dataset.lerobot_writer import LeRobotWriter
from ..utils.config import load_config


class SignalBridge(QObject):
    """Bridge for thread-safe signal emission."""
    update_robot_state = pyqtSignal(object)
    update_camera_frame = pyqtSignal(str, object)
    recording_complete = pyqtSignal(object)


class AnimatedLabel(QLabel):
    """Label with pulsing animation for recording indicator."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._opacity = 1.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._pulse)
        self._increasing = False
    
    def start_pulse(self):
        self._timer.start(50)
    
    def stop_pulse(self):
        self._timer.stop()
        self._opacity = 1.0
        self.setStyleSheet("")
    
    def _pulse(self):
        if self._increasing:
            self._opacity += 0.05
            if self._opacity >= 1.0:
                self._opacity = 1.0
                self._increasing = False
        else:
            self._opacity -= 0.05
            if self._opacity <= 0.3:
                self._opacity = 0.3
                self._increasing = True
        
        self.setStyleSheet(f"color: rgba(255, 51, 68, {self._opacity});")


class CameraWidget(QLabel):
    """Widget to display camera feed with overlay."""
    
    def __init__(self, camera_name: str, parent=None):
        super().__init__(parent)
        self.camera_name = camera_name
        self.setMinimumSize(320, 240)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(f"""
            background-color: {COLORS["bg_card"]};
            border-radius: 12px;
            border: 2px solid rgba(255, 255, 255, 0.1);
        """)
        self.setText(f"📷 {camera_name.upper()}\nWaiting for feed...")
        
        self._recording = False
    
    def update_frame(self, frame: np.ndarray):
        """Update displayed frame."""
        if frame is None:
            return
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        h, w = rgb.shape[:2]
        widget_w = self.width() - 4
        widget_h = self.height() - 4
        
        scale = min(widget_w / w, widget_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        if new_w > 0 and new_h > 0:
            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            if self._recording:
                cv2.circle(rgb, (20, 20), 10, (255, 0, 0), -1)
                cv2.putText(rgb, "REC", (35, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
            h, w = rgb.shape[:2]
            bytes_per_line = 3 * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.setPixmap(QPixmap.fromImage(qimg))
    
    def set_recording(self, recording: bool):
        """Set recording state for overlay."""
        self._recording = recording


class RobotStatusWidget(QFrame):
    """Widget showing robot connection status and joint positions."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # Header
        header_layout = QHBoxLayout()
        title = QLabel("🤖 ROBOT STATUS")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {COLORS['cyan']};")
        header_layout.addWidget(title)
        
        self.status_label = QLabel("Disconnected")
        self.status_label.setObjectName("status_disconnected")
        header_layout.addStretch()
        header_layout.addWidget(self.status_label)
        layout.addLayout(header_layout)
        
        # Joint positions
        self.joint_labels = []
        joints_group = QGroupBox("Joint Positions (degrees)")
        joints_layout = QGridLayout(joints_group)
        
        for i in range(6):
            name_label = QLabel(f"J{i+1}:")
            name_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
            value_label = QLabel("---.--")
            value_label.setFont(QFont("Consolas", 12))
            value_label.setStyleSheet(f"color: {COLORS['lime']};")
            
            joints_layout.addWidget(name_label, i // 2, (i % 2) * 2)
            joints_layout.addWidget(value_label, i // 2, (i % 2) * 2 + 1)
            self.joint_labels.append(value_label)
        
        layout.addWidget(joints_group)
        
        # Gripper
        gripper_layout = QHBoxLayout()
        gripper_label = QLabel("Gripper:")
        gripper_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        self.gripper_value = QLabel("---.--")
        self.gripper_value.setFont(QFont("Consolas", 12))
        self.gripper_value.setStyleSheet(f"color: {COLORS['orange']};")
        gripper_layout.addWidget(gripper_label)
        gripper_layout.addWidget(self.gripper_value)
        gripper_layout.addStretch()
        layout.addLayout(gripper_layout)
        
        layout.addStretch()
    
    def update_state(self, state: RobotState):
        """Update displayed robot state."""
        if state is None:
            return
        
        for i, label in enumerate(self.joint_labels):
            label.setText(f"{state.joint_positions[i]:7.2f}°")
        
        self.gripper_value.setText(f"{state.gripper_position:.1f}")
    
    def set_connected(self, connected: bool):
        """Set connection status."""
        if connected:
            self.status_label.setText("✓ Connected")
            self.status_label.setObjectName("status_connected")
        else:
            self.status_label.setText("✗ Disconnected")
            self.status_label.setObjectName("status_disconnected")
        
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)


class ControlPanel(QFrame):
    """Main control panel with simplified Record/Stop buttons."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("panel")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)
        
        # Header
        header = QLabel("⚡ VLA DEMO COLLECTOR ⚡")
        header.setObjectName("header")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        layout.addWidget(header)
        
        # Instructions
        instructions = QLabel(
            "📋 Workflow:\n"
            "1. Use physical button to teach robot movements\n"
            "2. Double-press physical button to replay\n"
            "3. Click START RECORDING during replay\n"
            "4. Click STOP when done, then Save or Discard"
        )
        instructions.setStyleSheet(f"""
            color: {COLORS['text_secondary']};
            padding: 16px;
            background: {COLORS['bg_elevated']};
            border-radius: 8px;
            font-size: 14px;
        """)
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Task selector
        task_layout = QVBoxLayout()
        task_label = QLabel("📝 Task Description:")
        task_label.setObjectName("subheader")
        task_layout.addWidget(task_label)
        
        self.task_combo = QComboBox()
        self.task_combo.addItem("pick the white cup and place it on the red cup")
        self.task_combo.setEditable(True)
        self.task_combo.setMinimumWidth(400)
        task_layout.addWidget(self.task_combo)
        layout.addLayout(task_layout)
        
        # Main action buttons
        buttons_layout = QVBoxLayout()
        buttons_layout.setSpacing(16)
        
        # Home button
        self.home_btn = QPushButton("🏠 HOME POSITION")
        self.home_btn.setObjectName("primary")
        self.home_btn.setToolTip("Move robot to home position")
        self.home_btn.setMinimumHeight(50)
        buttons_layout.addWidget(self.home_btn)
        
        # Record button (large and prominent)
        self.record_btn = QPushButton("🔴 START RECORDING")
        self.record_btn.setObjectName("danger")
        self.record_btn.setToolTip("Start recording camera and robot data")
        self.record_btn.setMinimumHeight(80)
        self.record_btn.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        buttons_layout.addWidget(self.record_btn)
        
        # Stop button
        self.stop_btn = QPushButton("⏹ STOP RECORDING")
        self.stop_btn.setObjectName("teach")
        self.stop_btn.setToolTip("Stop recording")
        self.stop_btn.setMinimumHeight(60)
        self.stop_btn.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self.stop_btn.setEnabled(False)
        buttons_layout.addWidget(self.stop_btn)
        
        layout.addLayout(buttons_layout)
        
        # Status section
        status_group = QGroupBox("📊 Recording Status")
        status_layout = QVBoxLayout(status_group)
        
        self.status_label = QLabel("Ready - Start replay then click Record")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet(f"""
            font-size: 16px;
            padding: 12px;
            background: {COLORS["bg_elevated"]};
            border-radius: 8px;
        """)
        status_layout.addWidget(self.status_label)
        
        # Recording indicator
        rec_layout = QHBoxLayout()
        self.record_indicator = AnimatedLabel("● REC")
        self.record_indicator.setFont(QFont("Consolas", 16, QFont.Weight.Bold))
        self.record_indicator.setStyleSheet(f"color: {COLORS['record_red']};")
        self.record_indicator.setVisible(False)
        rec_layout.addWidget(self.record_indicator)
        
        self.frames_label = QLabel("Frames: 0")
        self.frames_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        rec_layout.addStretch()
        rec_layout.addWidget(self.frames_label)
        status_layout.addLayout(rec_layout)
        
        layout.addWidget(status_group)
        
        # Duration timer
        time_layout = QHBoxLayout()
        time_label = QLabel("⏱️ Duration:")
        self.duration_label = QLabel("00:00.0")
        self.duration_label.setFont(QFont("Consolas", 18))
        self.duration_label.setStyleSheet(f"color: {COLORS['cyan']};")
        time_layout.addWidget(time_label)
        time_layout.addWidget(self.duration_label)
        time_layout.addStretch()
        layout.addLayout(time_layout)
        
        layout.addStretch()
    
    def set_recording(self, is_recording: bool):
        """Update UI for recording state."""
        self.record_btn.setEnabled(not is_recording)
        self.stop_btn.setEnabled(is_recording)
        self.home_btn.setEnabled(not is_recording)
        
        if is_recording:
            self.status_label.setText("● Recording in progress...")
            self.record_indicator.setVisible(True)
            self.record_indicator.start_pulse()
        else:
            self.status_label.setText("Ready - Start replay then click Record")
            self.record_indicator.stop_pulse()
            self.record_indicator.setVisible(False)
    
    def set_reviewing(self, is_reviewing: bool):
        """Update UI for review state."""
        if is_reviewing:
            self.status_label.setText("📋 Review recording - Save or Discard")
            self.record_btn.setEnabled(False)
        else:
            self.status_label.setText("Ready - Start replay then click Record")
            self.record_btn.setEnabled(True)
    
    def update_duration(self, seconds: float):
        """Update duration display."""
        mins = int(seconds) // 60
        secs = seconds % 60
        self.duration_label.setText(f"{mins:02d}:{secs:05.2f}")
    
    def update_frames(self, count: int):
        """Update frame counter."""
        self.frames_label.setText(f"Frames: {count}")


class RecordingsPanel(QFrame):
    """Panel showing recorded demos with Save/Discard actions."""
    
    save_requested = pyqtSignal()
    delete_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("panel")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        
        # Header
        header = QLabel("📹 RECORDINGS")
        header.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        header.setStyleSheet(f"color: {COLORS['magenta']};")
        layout.addWidget(header)
        
        # Demo list
        self.demo_list = QListWidget()
        self.demo_list.setMaximumHeight(200)
        layout.addWidget(self.demo_list)
        
        # Review section (initially hidden)
        self.review_group = QGroupBox("📋 Review Last Recording")
        review_layout = QVBoxLayout(self.review_group)
        
        self.review_info = QLabel("No recording to review")
        self.review_info.setWordWrap(True)
        review_layout.addWidget(self.review_info)
        
        review_buttons = QHBoxLayout()
        self.save_btn = QPushButton("✓ SAVE")
        self.save_btn.setObjectName("play")
        self.save_btn.clicked.connect(self.save_requested.emit)
        review_buttons.addWidget(self.save_btn)
        
        self.discard_btn = QPushButton("✗ DISCARD")
        self.discard_btn.setObjectName("danger")
        self.discard_btn.clicked.connect(self.delete_requested.emit)
        review_buttons.addWidget(self.discard_btn)
        
        review_layout.addLayout(review_buttons)
        layout.addWidget(self.review_group)
        self.review_group.setVisible(False)
        
        # Progress counter
        counter_layout = QHBoxLayout()
        counter_label = QLabel("Progress:")
        counter_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        self.counter_value = QLabel("0 / 20 demos")
        self.counter_value.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self.counter_value.setStyleSheet(f"color: {COLORS['lime']};")
        counter_layout.addWidget(counter_label)
        counter_layout.addWidget(self.counter_value)
        counter_layout.addStretch()
        layout.addLayout(counter_layout)
        
        # Dataset progress bar
        self.dataset_progress = QProgressBar()
        self.dataset_progress.setRange(0, 20)
        self.dataset_progress.setValue(0)
        self.dataset_progress.setFormat("%v / %m demos")
        layout.addWidget(self.dataset_progress)
        
        layout.addStretch()
    
    def show_review(self, episode: Episode):
        """Show review section for episode."""
        self.review_group.setVisible(True)
        self.review_info.setText(
            f"Episode #{episode.episode_id}\n"
            f"Duration: {episode.duration:.2f}s\n"
            f"Frames: {episode.num_frames}\n"
            f"FPS: {episode.fps:.1f}"
        )
    
    def hide_review(self):
        """Hide review section."""
        self.review_group.setVisible(False)
    
    def add_demo(self, episode_id: int, task: str, duration: float):
        """Add demo to list."""
        item = QListWidgetItem(f"Demo #{episode_id}: {duration:.2f}s")
        self.demo_list.addItem(item)
    
    def update_counter(self, count: int, target: int = 20):
        """Update progress counter."""
        self.counter_value.setText(f"{count} / {target} demos")
        self.dataset_progress.setValue(count)
        self.dataset_progress.setMaximum(target)


class MainWindow(QMainWindow):
    """Main application window - simplified for physical button workflow."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("⚡ VLA Demo Collector - Physical Button Mode")
        self.setMinimumSize(1400, 900)
        
        self.setStyleSheet(STYLESHEET)
        
        self.config = load_config()
        
        # Initialize components
        self.robot: Optional[RobotController] = None
        self.camera_manager: Optional[CameraManager] = None
        self.recorder: Optional[DataRecorder] = None
        self.dataset_writer: Optional[LeRobotWriter] = None
        
        # State
        self._is_recording = False
        self._last_episode: Optional[Episode] = None
        self._recording_start_time = 0.0
        
        # Signal bridge for thread safety
        self.signals = SignalBridge()
        self.signals.update_robot_state.connect(self._on_robot_state)
        self.signals.update_camera_frame.connect(self._on_camera_frame)
        self.signals.recording_complete.connect(self._on_recording_complete)
        
        # Timers
        self.state_timer = QTimer(self)
        self.state_timer.timeout.connect(self._poll_robot_state)
        
        self.duration_timer = QTimer(self)
        self.duration_timer.timeout.connect(self._update_duration)
        
        # Build UI
        self._setup_ui()
        
        # Initialize components after a short delay
        QTimer.singleShot(500, self._initialize_components)
    
    def _setup_ui(self):
        """Build the main UI layout."""
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)
        
        # Left panel - Robot status & cameras
        left_panel = QVBoxLayout()
        left_panel.setSpacing(16)
        
        self.robot_widget = RobotStatusWidget()
        left_panel.addWidget(self.robot_widget)
        
        # Camera feeds
        cameras_widget = QFrame()
        cameras_widget.setObjectName("card")
        cameras_layout = QVBoxLayout(cameras_widget)
        cameras_layout.setContentsMargins(12, 12, 12, 12)
        
        cam_label = QLabel("📷 CAMERA FEEDS")
        cam_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        cam_label.setStyleSheet(f"color: {COLORS['orange']};")
        cameras_layout.addWidget(cam_label)
        
        self.camera_widgets: Dict[str, CameraWidget] = {}
        for cam_config in self.config.cameras:
            cam_widget = CameraWidget(cam_config.name)
            cam_widget.setMinimumHeight(180)
            cameras_layout.addWidget(cam_widget)
            self.camera_widgets[cam_config.name] = cam_widget
        
        left_panel.addWidget(cameras_widget)
        left_panel.setStretch(0, 0)
        left_panel.setStretch(1, 1)
        
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setMaximumWidth(400)
        main_layout.addWidget(left_widget)
        
        # Center - Main controls
        self.control_panel = ControlPanel()
        main_layout.addWidget(self.control_panel, 1)
        
        # Right - Recordings
        self.recordings_panel = RecordingsPanel()
        self.recordings_panel.setMaximumWidth(350)
        self.recordings_panel.save_requested.connect(self._on_save_demo)
        self.recordings_panel.delete_requested.connect(self._on_discard_demo)
        main_layout.addWidget(self.recordings_panel)
        
        # Connect button signals
        self.control_panel.home_btn.clicked.connect(self._on_home_clicked)
        self.control_panel.record_btn.clicked.connect(self._on_record_clicked)
        self.control_panel.stop_btn.clicked.connect(self._on_stop_clicked)
        
        # Status bar
        self.statusBar().showMessage("Initializing...")
    
    def _initialize_components(self):
        """Initialize robot, cameras, and other components."""
        self.statusBar().showMessage("Connecting to robot...")
        
        # Initialize robot controller in PASSIVE mode (read-only)
        # Physical button handles teach/playback, app just observes
        self.robot = RobotController()
        if self.robot.connect_passive():
            self.robot_widget.set_connected(True)
            self.statusBar().showMessage("Robot connected!")
            
            # Start polling robot state
            self.state_timer.start(33)  # ~30 Hz
        else:
            self.robot_widget.set_connected(False)
            self.statusBar().showMessage("Robot connection failed - check CAN interface")
        
        # Initialize cameras
        self.camera_manager = CameraManager()
        for cam_config in self.config.cameras:
            self.camera_manager.add_camera(
                name=cam_config.name,
                device_index=cam_config.device_index,
                width=cam_config.width,
                height=cam_config.height,
                fps=cam_config.fps
            )
        
        cam_results = self.camera_manager.open_all()
        for name, success in cam_results.items():
            if success:
                self.statusBar().showMessage(f"Camera {name} connected")
            else:
                self.statusBar().showMessage(f"Camera {name} failed to connect")
        
        # Start camera capture
        self.camera_manager.register_frame_callback(self._on_new_frames)
        self.camera_manager.start_capture(target_fps=30.0)
        
        # Initialize recorder
        self.recorder = DataRecorder(
            robot=self.robot,
            camera_manager=self.camera_manager,
            target_fps=self.config.recording.fps
        )
        
        # Initialize dataset writer
        task = self.control_panel.task_combo.currentText()
        dataset_name = task.replace(" ", "_").lower()[:50]
        self.dataset_writer = LeRobotWriter(
            base_path=Path("datasets"),
            dataset_name=dataset_name,
            task=task,
            fps=self.config.recording.fps
        )
        
        # Update counters
        self.recordings_panel.update_counter(self.dataset_writer.num_episodes)
        
        self.statusBar().showMessage("Ready! Use physical button to teach, then click Record during replay.")
    
    def _poll_robot_state(self):
        """Poll robot state at regular intervals."""
        if self.robot and self.robot.is_connected:
            state = self.robot.get_state()
            self.signals.update_robot_state.emit(state)
    
    def _on_new_frames(self, frames: Dict[str, CameraFrame]):
        """Callback for new camera frames (from camera thread)."""
        for name, frame in frames.items():
            self.signals.update_camera_frame.emit(name, frame)
    
    def _on_robot_state(self, state: RobotState):
        """Handle robot state update (UI thread)."""
        self.robot_widget.update_state(state)
    
    def _on_camera_frame(self, name: str, frame: CameraFrame):
        """Handle camera frame update (UI thread)."""
        if name in self.camera_widgets:
            self.camera_widgets[name].update_frame(frame.frame)
    
    def _on_recording_complete(self, episode: Episode):
        """Handle recording completion."""
        self.statusBar().showMessage(f"Recording complete: {episode.num_frames} frames, {episode.duration:.2f}s")
    
    def _update_duration(self):
        """Update duration display during recording."""
        elapsed = time.time() - self._recording_start_time
        self.control_panel.update_duration(elapsed)
        
        # Update frame count
        if self.recorder:
            stats = self.recorder.get_recording_stats()
            self.control_panel.update_frames(stats.get("frames", 0))
    
    def _on_home_clicked(self):
        """Handle Home button click."""
        if not self.robot or not self.robot.is_connected:
            return
        
        self.statusBar().showMessage("Moving to home position...")
        self.control_panel.home_btn.setEnabled(False)
        
        def go_home():
            success = self.robot.go_home()
            self.statusBar().showMessage(
                "Home position reached" if success else "Failed to reach home"
            )
            self.control_panel.home_btn.setEnabled(True)
        
        threading.Thread(target=go_home, daemon=True).start()
    
    def _on_record_clicked(self):
        """Handle Record button click - start passive recording."""
        if self._is_recording:
            return
        
        task = self.control_panel.task_combo.currentText()
        
        # Start recording in background
        if not self.recorder.start_recording(task):
            self.statusBar().showMessage("Failed to start recording")
            return
        
        self._is_recording = True
        self._recording_start_time = time.time()
        
        # Update UI
        self.control_panel.set_recording(True)
        for widget in self.camera_widgets.values():
            widget.set_recording(True)
        
        # Start timers
        self.duration_timer.start(100)
        
        # Start recording loop in background
        def record_loop():
            while self._is_recording:
                self.recorder.record_step()
                time.sleep(1.0 / self.config.recording.fps)
        
        self._record_thread = threading.Thread(target=record_loop, daemon=True)
        self._record_thread.start()
        
        self.statusBar().showMessage("Recording... Click STOP when done.")
    
    def _on_stop_clicked(self):
        """Handle Stop button click - stop recording and show review."""
        if not self._is_recording:
            return
        
        self._is_recording = False
        self.duration_timer.stop()
        
        # Wait for record thread
        if hasattr(self, '_record_thread') and self._record_thread.is_alive():
            self._record_thread.join(timeout=1.0)
        
        # Stop recording and get episode
        self._last_episode = self.recorder.stop_recording()
        
        # Update UI
        self.control_panel.set_recording(False)
        for widget in self.camera_widgets.values():
            widget.set_recording(False)
        
        if self._last_episode:
            self.recordings_panel.show_review(self._last_episode)
            self.control_panel.set_reviewing(True)
            self.statusBar().showMessage(
                f"Recording stopped: {self._last_episode.num_frames} frames, "
                f"{self._last_episode.duration:.2f}s - Save or Discard?"
            )
        else:
            self.statusBar().showMessage("Recording stopped - no data captured")
    
    def _on_save_demo(self):
        """Handle Save demo button click."""
        if not self._last_episode or not self.dataset_writer:
            return
        
        if self.dataset_writer.save_episode(self._last_episode):
            self.statusBar().showMessage(f"Demo saved! Total: {self.dataset_writer.num_episodes}")
            self.recordings_panel.add_demo(
                episode_id=self._last_episode.episode_id,
                task=self._last_episode.task,
                duration=self._last_episode.duration
            )
            self.recordings_panel.update_counter(self.dataset_writer.num_episodes)
            self.recordings_panel.hide_review()
            self.control_panel.set_reviewing(False)
            self._last_episode = None
        else:
            self.statusBar().showMessage("Failed to save demo")
            QMessageBox.warning(self, "Save Failed", "Could not save the demo to dataset.")
    
    def _on_discard_demo(self):
        """Handle Discard demo button click."""
        self._last_episode = None
        self.recordings_panel.hide_review()
        self.control_panel.set_reviewing(False)
        self.statusBar().showMessage("Demo discarded - ready to record again")
    
    def closeEvent(self, event):
        """Handle window close."""
        self._is_recording = False
        
        self.state_timer.stop()
        self.duration_timer.stop()
        
        if self.camera_manager:
            self.camera_manager.close_all()
        
        if self.robot:
            self.robot.disconnect()
        
        event.accept()


def run_app():
    """Run the application."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())
