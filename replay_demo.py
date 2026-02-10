#!/usr/bin/env python3
"""
Demo Replay Script for AgileX Piper Robot
==========================================

Replays a saved demonstration from the LeRobot dataset format to the physical robot.
Uses trajectory interpolation for smooth motion.

Usage:
    python replay_demo.py --demo 0          # Replay demo #0
    python replay_demo.py --demo 3 --speed 0.5   # Replay at half speed
    python replay_demo.py --demo 1 --speed 2.0   # Replay at double speed
    python replay_demo.py --list            # List all available demos

Requirements:
    - conda activate aeropiper
    - Robot connected via CAN
"""

import argparse
import sys
import time
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
except ImportError:
    print("Error: pandas is required. Install with: pip install pandas pyarrow")
    sys.exit(1)


def find_dataset_path() -> Optional[Path]:
    """Find the most recent dataset directory."""
    base_path = Path("datasets")
    if not base_path.exists():
        return None
    
    for dataset_dir in sorted(base_path.iterdir(), reverse=True):
        if dataset_dir.is_dir():
            data_path = dataset_dir / "data" / "chunk-000"
            if data_path.exists():
                return dataset_dir
    
    return None


def list_demos(dataset_path: Path) -> List[dict]:
    """List all available demos in the dataset."""
    demos = []
    data_path = dataset_path / "data" / "chunk-000"
    
    for parquet_file in sorted(data_path.glob("episode_*.parquet")):
        try:
            episode_idx = int(parquet_file.stem.split("_")[1])
            df = pd.read_parquet(parquet_file)
            num_frames = len(df)
            
            if 'timestamp' in df.columns:
                duration = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
            else:
                duration = num_frames / 30.0
            
            demos.append({
                "index": episode_idx,
                "frames": num_frames,
                "duration": duration,
                "path": parquet_file
            })
        except Exception as e:
            print(f"Warning: Could not read {parquet_file}: {e}")
    
    return demos


def load_demo(dataset_path: Path, demo_index: int) -> Optional[pd.DataFrame]:
    """Load a specific demo from the dataset."""
    parquet_path = dataset_path / "data" / "chunk-000" / f"episode_{demo_index:06d}.parquet"
    
    if not parquet_path.exists():
        print(f"Error: Demo #{demo_index} not found at {parquet_path}")
        return None
    
    return pd.read_parquet(parquet_path)


def interpolate_trajectory(
    states: np.ndarray,
    timestamps: np.ndarray,
    command_hz: float = 100.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate trajectory to higher frequency for smooth motion.
    
    Args:
        states: Array of robot states (N x state_dim)
        timestamps: Array of timestamps (N,)
        command_hz: Target command frequency
    
    Returns:
        Tuple of (interpolated_states, interpolated_timestamps)
    """
    if len(timestamps) < 2:
        return states, timestamps
    
    # Create high-frequency timeline
    duration = timestamps[-1] - timestamps[0]
    num_points = int(duration * command_hz) + 1
    new_timestamps = np.linspace(timestamps[0], timestamps[-1], num_points)
    
    # Interpolate each state dimension
    state_dim = states.shape[1]
    new_states = np.zeros((num_points, state_dim))
    
    for dim in range(state_dim):
        new_states[:, dim] = np.interp(new_timestamps, timestamps, states[:, dim])
    
    return new_states, new_timestamps


def replay_demo(df: pd.DataFrame, robot, speed: float = 1.0, smooth: bool = True):
    """Replay a demo to the robot with smooth interpolated motion."""
    
    # Extract state observations (joint positions + gripper)
    states = np.array(df['observation.state'].tolist())
    
    # Extract timestamps
    if 'timestamp' in df.columns:
        timestamps = df['timestamp'].values
    else:
        timestamps = np.arange(len(df)) / 30.0
    
    # Normalize timestamps to start from 0
    timestamps = timestamps - timestamps[0]
    
    # Interpolate for smooth motion
    if smooth:
        command_hz = 100.0  # Send commands at 100 Hz for smooth motion
        states, timestamps = interpolate_trajectory(states, timestamps, command_hz)
        print(f"Interpolated to {len(states)} points at {command_hz:.0f} Hz")
    
    print(f"\n{'='*50}")
    print(f"Starting replay with speed multiplier: {speed}x")
    print(f"Total points: {len(states)}")
    print(f"Duration: {timestamps[-1] / speed:.2f}s")
    print(f"{'='*50}\n")
    
    # Replay loop
    start_time = time.perf_counter()
    last_print_time = start_time
    
    for i, (state, ts) in enumerate(zip(states, timestamps)):
        # Extract joint positions (first 6) and gripper (last 1)
        joint_positions = state[:6].tolist()
        gripper_position = state[6] if len(state) > 6 else 50.0
        
        # Send to robot (high speed value for responsiveness)
        robot.set_joint_positions(joint_positions, speed=100)
        robot.set_gripper(gripper_position)
        
        # Progress display (update every 0.5s to avoid console spam)
        now = time.perf_counter()
        if now - last_print_time > 0.5:
            progress = (i + 1) / len(states) * 100
            elapsed = now - start_time
            print(f"\rProgress: {progress:5.1f}% | Time: {elapsed:.1f}s / {timestamps[-1]/speed:.1f}s", end="", flush=True)
            last_print_time = now
        
        # Wait for next command timing (adjusted by speed)
        if i < len(timestamps) - 1:
            target_time = timestamps[i + 1] / speed
            elapsed = time.perf_counter() - start_time
            wait_time = target_time - elapsed
            if wait_time > 0:
                time.sleep(wait_time)
    
    print(f"\n\n{'='*50}")
    print(f"Replay complete!")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description="Replay saved demos to the AgileX Piper robot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python replay_demo.py --list              # List all demos
    python replay_demo.py --demo 0            # Replay demo #0
    python replay_demo.py --demo 3 --speed 0.5  # Replay at half speed
    python replay_demo.py --demo 0 --no-smooth  # Replay without interpolation
        """
    )
    parser.add_argument("--demo", "-d", type=int, help="Demo index to replay")
    parser.add_argument("--speed", "-s", type=float, default=1.0, help="Speed multiplier (default: 1.0)")
    parser.add_argument("--list", "-l", action="store_true", help="List all available demos")
    parser.add_argument("--dataset", type=str, help="Path to dataset directory (optional)")
    parser.add_argument("--dry-run", action="store_true", help="Don't send commands to robot, just simulate")
    parser.add_argument("--no-smooth", action="store_true", help="Disable trajectory interpolation")
    
    args = parser.parse_args()
    
    # Find dataset
    if args.dataset:
        dataset_path = Path(args.dataset)
    else:
        dataset_path = find_dataset_path()
    
    if not dataset_path or not dataset_path.exists():
        print("Error: No dataset found. Run the collector first to create demos.")
        print("Expected path: datasets/<task_name>/")
        return 1
    
    print(f"Using dataset: {dataset_path}")
    
    # List demos
    if args.list:
        demos = list_demos(dataset_path)
        if not demos:
            print("No demos found in dataset.")
            return 1
        
        print(f"\n{'='*50}")
        print(f"Available Demos ({len(demos)} total)")
        print(f"{'='*50}")
        
        for demo in demos:
            print(f"  Demo #{demo['index']:3d}: {demo['frames']:4d} frames, {demo['duration']:.2f}s")
        
        print()
        return 0
    
    # Replay specific demo
    if args.demo is None:
        parser.print_help()
        return 1
    
    # Load demo data
    df = load_demo(dataset_path, args.demo)
    if df is None:
        return 1
    
    print(f"\nLoaded demo #{args.demo}:")
    print(f"  Frames: {len(df)}")
    if 'timestamp' in df.columns:
        duration = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
        print(f"  Duration: {duration:.2f}s")
    
    if args.dry_run:
        print("\n[DRY RUN] Would replay demo without sending to robot")
        return 0
    
    # Connect to robot
    print("\nConnecting to robot...")
    print("  (Make sure the collector app is closed and robot is not in teach mode)")
    
    sys.path.insert(0, str(Path(__file__).parent))
    
    try:
        from dataset_collector.core.robot_controller import RobotController
    except ImportError:
        print("Error: Could not import RobotController. Make sure you're in the project directory.")
        return 1
    
    robot = RobotController()
    
    # Try to connect with timeout
    print("  Enabling robot (this may take a moment)...")
    connected = robot.connect()
    
    if not connected:
        print("\n" + "="*60)
        print("ERROR: Could not connect to robot!")
        print("="*60)
        print("\nTroubleshooting steps:")
        print("  1. Close the VLA Demo Collector app if it's running")
        print("  2. If the physical button light is on, press it to exit teach mode")
        print("  3. Power cycle the robot if needed")
        print("  4. Check CAN interface: ip link show can0")
        print("="*60)
        return 1
    
    print("✓ Robot connected and enabled")
    
    # Confirm before replay
    smooth_str = "with interpolation" if not args.no_smooth else "without interpolation"
    print(f"\n⚠️  Ready to replay demo #{args.demo} at {args.speed}x speed ({smooth_str}).")
    response = input("Press ENTER to start, or 'q' to quit: ")
    if response.lower() == 'q':
        print("Cancelled.")
        robot.disconnect()
        return 0
    
    try:
        replay_demo(df, robot, speed=args.speed, smooth=not args.no_smooth)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        robot.disconnect()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
