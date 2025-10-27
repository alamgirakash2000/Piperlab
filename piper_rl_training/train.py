#!/usr/bin/env python
"""Training script for Piper robot reaching task using RSL-RL PPO.

This script trains a PPO agent to control the Piper robot to reach
random target positions in 3D space.

Usage:
    ./isaaclab.sh -p piper_rl_training/train.py --num_envs 4096 --headless
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Train Piper robot with RSL-RL PPO")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Reach-Piper-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="Maximum iterations for training.")

# Append AppLauncher arguments
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch the simulator
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Import after launching app
import sys
from pathlib import Path

# Add piper_rl_training to path
piper_rl_path = Path(__file__).parent.resolve()
if str(piper_rl_path) not in sys.path:
    sys.path.insert(0, str(piper_rl_path))

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401

# Import environment configs
from piper_reach_env_cfg import PiperReachEnvCfg, PiperReachEnvCfg_PLAY  # noqa: F401

# Register gym environments
gym.register(
    id="Isaac-Reach-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": PiperReachEnvCfg,
        "rsl_rl_cfg_entry_point": "agents.rsl_rl_ppo_cfg:PiperReachPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Reach-Piper-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": PiperReachEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": "agents.rsl_rl_ppo_cfg:PiperReachPPORunnerCfg",
    },
    disable_env_checker=True,
)

from isaaclab_rl.rsl_rl.wrappers import RslRlVecEnvWrapper, RslRlOnPolicyRunner


def main():
    """Train with RSL-RL PPO agent."""
    
    # Parse configuration
    env_cfg = gym.make(args.task).unwrapped.cfg
    agent_cfg = gym.spec(args.task).kwargs["rsl_rl_cfg_entry_point"]
    
    # Override configuration parameters if provided
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    if args.seed is not None:
        agent_cfg.seed = args.seed
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    
    # Set device (from AppLauncher)
    agent_cfg.device = args.device if hasattr(args, 'device') and args.device else "cuda:0"
    
    # Create environment
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    
    # Wrap environment for RSL-RL
    env = RslRlVecEnvWrapper(env)
    
    # Print training info
    print("\n" + "="*80)
    print("PIPER ROBOT - PPO TRAINING")
    print("="*80)
    print(f"Task: {args.task}")
    print(f"Number of environments: {env_cfg.scene.num_envs}")
    print(f"Environment spacing: {env_cfg.scene.env_spacing}")
    print(f"Episode length: {env_cfg.episode_length_s}s")
    print(f"Decimation: {env_cfg.decimation}")
    print(f"Max iterations: {agent_cfg.max_iterations}")
    print(f"Device: {agent_cfg.device}")
    print(f"Seed: {agent_cfg.seed}")
    print("="*80 + "\n")
    
    # Create runner from rsl-rl
    runner = RslRlOnPolicyRunner(env, agent_cfg, log_dir=None, device=agent_cfg.device)
    
    # Set up video recording if requested
    if args.video:
        print(f"Video recording enabled: length={args.video_length}, interval={args.video_interval}")
        runner.learn(
            num_learning_iterations=agent_cfg.max_iterations,
            init_at_random_ep_len=True,
            log_interval=agent_cfg.log_interval if hasattr(agent_cfg, 'log_interval') else 1,
        )
    else:
        # Standard training
        runner.learn(
            num_learning_iterations=agent_cfg.max_iterations,
            init_at_random_ep_len=True,
            log_interval=agent_cfg.log_interval if hasattr(agent_cfg, 'log_interval') else 1,
        )
    
    # Close the environment
    env.close()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED!")
    print(f"Logs saved to: {runner.log_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR: {e}")
        print(f"{'='*80}\n")
        raise
    finally:
        simulation_app.close()

