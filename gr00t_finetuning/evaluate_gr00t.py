#!/usr/bin/env python3
"""
GR00T N1.6 Evaluation Script for AgileX Piper Robot
====================================================

Runs the fine-tuned GR00T model to control the physical Piper robot.
Similar to replay_demo.py but uses the VLA model for action generation.

Usage:
    python evaluate_gr00t.py --checkpoint checkpoints/piper_finetune
    python evaluate_gr00t.py --checkpoint checkpoints/piper_finetune --task "pick the white cup"
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np

try:
    import cv2
except ImportError:
    print("Error: OpenCV required. Install with: pip install opencv-python")
    sys.exit(1)


class GR00TPolicy:
    """Wrapper for GR00T model inference."""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = None
        self.processor = None
        
    def load(self):
        """Load the fine-tuned GR00T model."""
        print(f"Loading GR00T model from: {self.checkpoint_path}")
        
        # Add Isaac-GR00T to path
        groot_path = Path(__file__).parent / "Isaac-GR00T"
        sys.path.insert(0, str(groot_path))
        
        try:
            from gr00t.model.gr00t import GR00TModel
            from gr00t.data.processor import GR00TProcessor
            
            self.model = GR00TModel.from_pretrained(self.checkpoint_path)
            self.model.to(self.device)
            self.model.eval()
            
            self.processor = GR00TProcessor()
            print("✓ Model loaded successfully")
            
        except ImportError as e:
            print(f"Error: Could not import GR00T modules: {e}")
            print("Make sure to run setup first: cd Isaac-GR00T && uv sync --python 3.10")
            return False
            
        return True
    
    def get_action(
        self,
        images: Dict[str, np.ndarray],
        state: np.ndarray,
        task: str
    ) -> np.ndarray:
        """
        Get action from the model given observations.
        
        Args:
            images: Dict mapping camera names to RGB images
            state: Current robot state (7D: joints + gripper)
            task: Task instruction string
            
        Returns:
            Action array (7D: joint deltas + gripper action)
        """
        import torch
        
        # Prepare inputs
        inputs = self.processor(
            images=images,
            state=state,
            task=task,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate action
        with torch.no_grad():
            outputs = self.model.generate(**inputs, action_horizon=8)
            action = outputs["action"][0, 0].cpu().numpy()  # First action
        
        return action


def run_evaluation(
    robot,
    camera_manager,
    policy: GR00TPolicy,
    task: str,
    max_steps: int = 300,
    hz: float = 10.0
):
    """
    Run closed-loop evaluation with the GR00T policy.
    
    Args:
        robot: RobotController instance
        camera_manager: CameraManager instance  
        policy: Loaded GR00T policy
        task: Task instruction string
        max_steps: Maximum steps before timeout
        hz: Control frequency
    """
    print(f"\n{'='*50}")
    print(f"Running evaluation")
    print(f"Task: {task}")
    print(f"Max steps: {max_steps}")
    print(f"Control Hz: {hz}")
    print(f"{'='*50}\n")
    
    step_duration = 1.0 / hz
    
    for step in range(max_steps):
        step_start = time.perf_counter()
        
        # Get camera frames
        frames = camera_manager.get_latest_frames()
        images = {}
        for cam_name, frame in frames.items():
            if frame is not None:
                images[f"observation.images.{cam_name}"] = frame.frame
        
        if len(images) < 2:
            print(f"Warning: Missing camera frames at step {step}")
            time.sleep(step_duration)
            continue
        
        # Get robot state
        robot_state = robot.get_state()
        state = np.array(robot_state.joint_positions + [robot_state.gripper_position])
        
        # Get action from policy
        action = policy.get_action(images, state, task)
        
        # Parse action (6 joint deltas + 1 gripper action)
        joint_deltas = action[:6]
        gripper_action = action[6]
        
        # Apply action
        new_joint_positions = [
            state[i] + joint_deltas[i] for i in range(6)
        ]
        new_gripper = state[6] + gripper_action
        new_gripper = max(0, min(100, new_gripper))  # Clamp to valid range
        
        robot.set_joint_positions(new_joint_positions, speed=50)
        robot.set_gripper(new_gripper)
        
        # Progress
        elapsed = time.perf_counter() - step_start
        print(f"\rStep {step+1}/{max_steps} | Action norm: {np.linalg.norm(action):.3f}", end="", flush=True)
        
        # Maintain control frequency
        sleep_time = step_duration - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        
        # Check for keyboard interrupt
        # (In real deployment, add success detection here)
    
    print(f"\n\n{'='*50}")
    print("Evaluation complete!")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description="Run GR00T policy on physical Piper robot",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default="checkpoints/piper_finetune",
        help="Path to fine-tuned checkpoint"
    )
    parser.add_argument(
        "--task", "-t",
        type=str,
        default="pick the white cup and place it on the red cup",
        help="Task instruction"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=300,
        help="Maximum steps before timeout"
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=10.0,
        help="Control frequency"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load model but don't run on robot"
    )
    
    args = parser.parse_args()
    
    # Check checkpoint exists
    checkpoint_path = Path(__file__).parent / args.checkpoint
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Run training first: ./train_gr00t.sh")
        return 1
    
    # Load policy
    policy = GR00TPolicy(str(checkpoint_path))
    if not policy.load():
        return 1
    
    if args.dry_run:
        print("\n[DRY RUN] Model loaded successfully, not running on robot")
        return 0
    
    # Connect to robot and cameras
    print("\nConnecting to robot and cameras...")
    
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    try:
        from dataset_collector.core.robot_controller import RobotController
        from dataset_collector.core.camera_manager import CameraManager
        from dataset_collector.utils.config import load_config
    except ImportError as e:
        print(f"Error importing robot modules: {e}")
        return 1
    
    # Load config
    config = load_config(project_root / "config.json")
    
    # Initialize robot
    robot = RobotController()
    if not robot.connect():
        print("Error: Could not connect to robot")
        return 1
    print("✓ Robot connected")
    
    # Initialize cameras
    camera_manager = CameraManager()
    for cam_config in config.cameras:
        camera_manager.add_camera(
            cam_config.name,
            cam_config.device_index,
            cam_config.width,
            cam_config.height,
            cam_config.fps
        )
    camera_manager.open_all()
    camera_manager.start_capture(30.0)
    print("✓ Cameras connected")
    
    # Confirm before running
    print(f"\n⚠️  Ready to run GR00T policy on physical robot")
    print(f"Task: {args.task}")
    response = input("Press ENTER to start, or 'q' to quit: ")
    if response.lower() == 'q':
        print("Cancelled.")
        robot.disconnect()
        camera_manager.close_all()
        return 0
    
    try:
        run_evaluation(
            robot=robot,
            camera_manager=camera_manager,
            policy=policy,
            task=args.task,
            max_steps=args.max_steps,
            hz=args.hz
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        camera_manager.close_all()
        robot.disconnect()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
