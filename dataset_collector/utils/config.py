#!/usr/bin/env python3
"""Configuration management for VLA Dataset Collector."""

import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class CameraConfig:
    """Camera configuration."""
    name: str
    device_index: int
    width: int = 1280
    height: int = 720
    fps: int = 30


@dataclass
class RobotConfig:
    """Robot configuration."""
    can_interface: str = "can0"
    num_joints: int = 6
    gripper_min: float = 0.0
    gripper_max: float = 100.0
    home_position: List[float] = field(default_factory=lambda: [0.04, 0.38, 0.11, 3.98, -18.01, 2.29])
    home_gripper: float = 50.0
    motion_speed: int = 30


@dataclass
class RecordingConfig:
    """Recording configuration."""
    fps: int = 30
    video_codec: str = "mp4v"
    video_quality: int = 23  # CRF for x264 (lower = better quality)


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    base_path: Path = field(default_factory=lambda: Path("datasets"))
    default_task: str = "pick the white cup and place it on the red cup"
    max_demos: int = 20
    lerobot_version: str = "2.1"


@dataclass
class AppConfig:
    """Main application configuration."""
    cameras: List[CameraConfig] = field(default_factory=lambda: [
        CameraConfig(name="global", device_index=0, width=1280, height=720),
        CameraConfig(name="wrist", device_index=2, width=1280, height=720),
    ])
    robot: RobotConfig = field(default_factory=RobotConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    
    def save(self, path: Path):
        """Save configuration to JSON file."""
        data = {
            "cameras": [asdict(c) for c in self.cameras],
            "robot": asdict(self.robot),
            "recording": asdict(self.recording),
            "dataset": {
                **asdict(self.dataset),
                "base_path": str(self.dataset.base_path)
            }
        }
        path.write_text(json.dumps(data, indent=2))
    
    @classmethod
    def load(cls, path: Path) -> "AppConfig":
        """Load configuration from JSON file."""
        if not path.exists():
            return cls()
        
        data = json.loads(path.read_text())
        config = cls()
        
        if "cameras" in data:
            config.cameras = [CameraConfig(**c) for c in data["cameras"]]
        if "robot" in data:
            config.robot = RobotConfig(**data["robot"])
        if "recording" in data:
            config.recording = RecordingConfig(**data["recording"])
        if "dataset" in data:
            d = data["dataset"]
            d["base_path"] = Path(d["base_path"])
            config.dataset = DatasetConfig(**d)
        
        return config


# Default configuration instance
DEFAULT_CONFIG = AppConfig()


def get_config_path() -> Path:
    """Get the configuration file path."""
    return Path(__file__).parent.parent.parent / "config.json"


def load_config() -> AppConfig:
    """Load or create default configuration."""
    config_path = get_config_path()
    if config_path.exists():
        return AppConfig.load(config_path)
    return DEFAULT_CONFIG
