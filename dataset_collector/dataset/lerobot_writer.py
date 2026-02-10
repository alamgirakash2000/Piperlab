#!/usr/bin/env python3
"""LeRobot V2.1 format dataset writer."""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not available. Install with: pip install pandas pyarrow")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass 
class DatasetInfo:
    """Dataset metadata."""
    name: str
    version: str = "2.1"
    task: str = ""
    robot_type: str = "agilex_piper"
    fps: int = 30
    num_episodes: int = 0
    total_frames: int = 0
    camera_names: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "task": self.task,
            "robot_type": self.robot_type,
            "fps": self.fps,
            "num_episodes": self.num_episodes,
            "total_frames": self.total_frames,
            "camera_names": self.camera_names,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }


class LeRobotWriter:
    """
    Writes episodes to LeRobot V2.1 format.
    
    Structure:
    dataset_name/
    ├── data/
    │   └── chunk-000/
    │       ├── episode_000000.parquet
    │       └── ...
    ├── videos/
    │   └── chunk-000/
    │       ├── observation.images.global/
    │       │   └── episode_000000.mp4
    │       └── observation.images.wrist/
    │           └── episode_000000.mp4
    └── meta/
        ├── info.json
        ├── episodes.jsonl
        ├── tasks.jsonl
        └── stats.json
    """
    
    def __init__(
        self,
        base_path: Path,
        dataset_name: str,
        task: str,
        fps: int = 30,
        video_codec: str = "mp4v",
        video_quality: int = 23
    ):
        self.base_path = Path(base_path)
        self.dataset_name = dataset_name
        self.task = task
        self.fps = fps
        self.video_codec = video_codec
        self.video_quality = video_quality
        
        # Create directory structure
        self.dataset_path = self.base_path / self._sanitize_name(dataset_name)
        self.data_path = self.dataset_path / "data" / "chunk-000"
        self.videos_path = self.dataset_path / "videos" / "chunk-000"
        self.meta_path = self.dataset_path / "meta"
        
        self._ensure_directories()
        
        # Track episodes
        self._episodes_info: List[dict] = []
        self._tasks: Dict[int, str] = {0: task}  # Task ID -> description
        self._camera_names: List[str] = []
        self._stats: Dict[str, dict] = {}
        
        # Load existing data if present
        self._load_existing()
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize dataset name for filesystem."""
        return name.lower().replace(" ", "_").replace("/", "_")
    
    def _ensure_directories(self):
        """Create all necessary directories."""
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.videos_path.mkdir(parents=True, exist_ok=True)
        self.meta_path.mkdir(parents=True, exist_ok=True)
    
    def _load_existing(self):
        """Load existing episode info if present."""
        episodes_file = self.meta_path / "episodes.jsonl"
        if episodes_file.exists():
            with open(episodes_file, 'r') as f:
                for line in f:
                    if line.strip():
                        self._episodes_info.append(json.loads(line))
        
        tasks_file = self.meta_path / "tasks.jsonl"
        if tasks_file.exists():
            with open(tasks_file, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        self._tasks[data["task_index"]] = data["task"]
    
    @property
    def num_episodes(self) -> int:
        """Get number of episodes in dataset."""
        return len(self._episodes_info)
    
    def save_episode(self, episode) -> bool:
        """
        Save an episode to the dataset.
        
        Args:
            episode: Episode object from data_recorder
        
        Returns:
            True if saved successfully
        """
        if not PANDAS_AVAILABLE:
            print("Error: pandas is required for saving episodes")
            return False
        
        if not episode.steps:
            print("Error: Episode has no steps")
            return False
        
        episode_index = self.num_episodes
        episode_id = f"episode_{episode_index:06d}"
        
        try:
            # 1. Save tabular data to Parquet
            self._save_parquet(episode, episode_index)
            
            # 2. Save videos for each camera
            self._save_videos(episode, episode_index)
            
            # 3. Update episode info
            episode_info = {
                "episode_index": episode_index,
                "tasks": [self.task],
                "length": episode.num_frames,
            }
            self._episodes_info.append(episode_info)
            
            # 4. Update camera names
            if episode.steps and episode.steps[0].images:
                for cam_name in episode.steps[0].images.keys():
                    if cam_name not in self._camera_names:
                        self._camera_names.append(cam_name)
            
            # 5. Save metadata files
            self._save_metadata()
            
            print(f"Saved episode {episode_index}: {episode.num_frames} frames, {episode.duration:.2f}s")
            return True
            
        except Exception as e:
            print(f"Error saving episode: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _save_parquet(self, episode, episode_index: int):
        """Save episode tabular data to Parquet file."""
        rows = []
        
        for step in episode.steps:
            row = {
                "episode_index": episode_index,
                "frame_index": step.frame_index,
                "index": episode_index * 10000 + step.frame_index,  # Global unique ID
                "timestamp": step.timestamp,
                
                # State observations
                "observation.state": np.concatenate([
                    step.joint_positions,
                    [step.gripper_position]
                ]).tolist(),
                
                # Actions
                "action": np.concatenate([
                    step.joint_position_delta,
                    [step.gripper_action]
                ]).tolist(),
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        parquet_path = self.data_path / f"episode_{episode_index:06d}.parquet"
        df.to_parquet(parquet_path, index=False)
        
        # Update statistics
        self._update_stats(df)
    
    def _save_videos(self, episode, episode_index: int):
        """Save camera frames as MP4 videos."""
        if not CV2_AVAILABLE:
            print("Warning: OpenCV not available, skipping video saving")
            return
        
        # Calculate ACTUAL FPS from episode timestamps (not target FPS)
        actual_fps = episode.fps if episode.fps > 0 else self.fps
        print(f"  Saving videos at {actual_fps:.1f} FPS (actual recorded rate)")
        
        # Group frames by camera
        camera_frames: Dict[str, List[np.ndarray]] = {}
        
        for step in episode.steps:
            for cam_name, frame in step.images.items():
                if cam_name not in camera_frames:
                    camera_frames[cam_name] = []
                camera_frames[cam_name].append(frame)
        
        # Save each camera's frames as video
        for cam_name, frames in camera_frames.items():
            if not frames:
                continue
            
            # Create camera directory
            cam_dir = self.videos_path / f"observation.images.{cam_name}"
            cam_dir.mkdir(parents=True, exist_ok=True)
            
            video_path = cam_dir / f"episode_{episode_index:06d}.mp4"
            
            # Get frame dimensions
            height, width = frames[0].shape[:2]
            
            # Create video writer with ACTUAL FPS
            fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
            writer = cv2.VideoWriter(
                str(video_path),
                fourcc,
                actual_fps,  # Use actual FPS, not target
                (width, height)
            )
            
            for frame in frames:
                # Ensure BGR format for OpenCV
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                writer.write(frame)
            
            writer.release()
    
    def _update_stats(self, df: 'pd.DataFrame'):
        """Update running statistics from episode data."""
        for col in ["observation.state", "action"]:
            if col not in df.columns:
                continue
            
            # Convert list column to array for stats
            try:
                values = np.array(df[col].tolist())
                
                if col not in self._stats:
                    self._stats[col] = {
                        "min": values.min(axis=0).tolist(),
                        "max": values.max(axis=0).tolist(),
                        "mean": values.mean(axis=0).tolist(),
                        "std": values.std(axis=0).tolist(),
                    }
                else:
                    # Update running stats (simplified - just track min/max)
                    current = self._stats[col]
                    current["min"] = np.minimum(current["min"], values.min(axis=0)).tolist()
                    current["max"] = np.maximum(current["max"], values.max(axis=0)).tolist()
            except Exception:
                pass
    
    def _save_metadata(self):
        """Save all metadata files."""
        # info.json
        info = DatasetInfo(
            name=self.dataset_name,
            task=self.task,
            fps=self.fps,
            num_episodes=self.num_episodes,
            total_frames=sum(ep["length"] for ep in self._episodes_info),
            camera_names=self._camera_names,
        )
        
        info_path = self.meta_path / "info.json"
        with open(info_path, 'w') as f:
            json.dump(info.to_dict(), f, indent=2)
        
        # episodes.jsonl
        episodes_path = self.meta_path / "episodes.jsonl"
        with open(episodes_path, 'w') as f:
            for ep_info in self._episodes_info:
                f.write(json.dumps(ep_info) + "\n")
        
        # tasks.jsonl
        tasks_path = self.meta_path / "tasks.jsonl"
        with open(tasks_path, 'w') as f:
            for task_id, task_desc in self._tasks.items():
                f.write(json.dumps({"task_index": task_id, "task": task_desc}) + "\n")
        
        # stats.json
        if self._stats:
            stats_path = self.meta_path / "stats.json"
            with open(stats_path, 'w') as f:
                json.dump(self._stats, f, indent=2)
    
    def delete_episode(self, episode_index: int) -> bool:
        """Delete an episode from the dataset."""
        try:
            # Delete parquet file
            parquet_path = self.data_path / f"episode_{episode_index:06d}.parquet"
            if parquet_path.exists():
                parquet_path.unlink()
            
            # Delete video files
            for cam_name in self._camera_names:
                video_path = self.videos_path / f"observation.images.{cam_name}" / f"episode_{episode_index:06d}.mp4"
                if video_path.exists():
                    video_path.unlink()
            
            # Remove from episodes info (but don't renumber)
            self._episodes_info = [
                ep for ep in self._episodes_info 
                if ep["episode_index"] != episode_index
            ]
            
            # Update metadata
            self._save_metadata()
            
            return True
            
        except Exception as e:
            print(f"Error deleting episode: {e}")
            return False
    
    def get_episode_info(self) -> List[dict]:
        """Get info about all episodes."""
        return self._episodes_info.copy()
    
    def get_dataset_summary(self) -> dict:
        """Get summary of the dataset."""
        return {
            "name": self.dataset_name,
            "task": self.task,
            "path": str(self.dataset_path),
            "num_episodes": self.num_episodes,
            "total_frames": sum(ep["length"] for ep in self._episodes_info),
            "camera_names": self._camera_names,
            "fps": self.fps,
        }
