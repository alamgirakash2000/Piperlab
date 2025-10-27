#!/usr/bin/env python
"""Training script for Piper robot cube stacking task using RSL-RL PPO.

This script trains a PPO agent to control the Piper robot to stack
three colored cubes on top of each other.

Usage:
    ./isaaclab.sh -p piper_rl_training/stack_task/train.py --num_envs 1024 --headless
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Train Piper robot cube stacking with RSL-RL PPO")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Stack-Piper-v0", help="Name of the task.")
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

# Add stack_task to path
stack_task_path = Path(__file__).parent.resolve()
if str(stack_task_path) not in sys.path:
    sys.path.insert(0, str(stack_task_path))

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401

# Import environment configs
from piper_stack_env_cfg import PiperCubeStackEnvCfg, PiperCubeStackEnvCfg_PLAY  # noqa: F401

# Register gym environments
gym.register(
    id="Isaac-Stack-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": PiperCubeStackEnvCfg,
        "rsl_rl_cfg_entry_point": "agents.rsl_rl_ppo_cfg:PiperStackPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Piper-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": PiperCubeStackEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": "agents.rsl_rl_ppo_cfg:PiperStackPPORunnerCfg",
    },
    disable_env_checker=True,
)

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner


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
    print("PIPER ROBOT - CUBE STACKING PPO TRAINING")
    print("="*80)
    print(f"Task: {args.task}")
    print(f"Number of environments: {env_cfg.scene.num_envs}")
    print(f"Environment spacing: {env_cfg.scene.env_spacing}")
    print(f"Episode length: {env_cfg.episode_length_s}s")
    print(f"Decimation: {env_cfg.decimation}")
    print(f"Max iterations: {agent_cfg.max_iterations}")
    print(f"Device: {agent_cfg.device}")
    print(f"Seed: {agent_cfg.seed}")
    print("\nObjective: Stack red cube on blue cube, then green cube on red cube")
    print("="*80 + "\n")
    
    # Create runner from rsl-rl
    # Build explicit runner config for rsl_rl
    runner_cfg_dict = {
        "seed": agent_cfg.seed,
        "device": agent_cfg.device,
        "num_steps_per_env": agent_cfg.num_steps_per_env,
        "max_iterations": agent_cfg.max_iterations,
        "clip_actions": getattr(agent_cfg, "clip_actions", None),
        "save_interval": getattr(agent_cfg, "save_interval", 100),
        "experiment_name": agent_cfg.experiment_name,
        "run_name": agent_cfg.run_name,
        "logger": agent_cfg.logger,
        "resume": getattr(agent_cfg, "resume", False),
        "load_run": getattr(agent_cfg, "load_run", ".*"),
        "load_checkpoint": getattr(agent_cfg, "load_checkpoint", "model_.*.pt"),
        "obs_groups": {"policy": ["policy"], "critic": ["policy"]},
        "policy": {
            "class_name": agent_cfg.policy.class_name,
            "init_noise_std": agent_cfg.policy.init_noise_std,
            "noise_std_type": agent_cfg.policy.noise_std_type,
            "actor_obs_normalization": agent_cfg.policy.actor_obs_normalization,
            "critic_obs_normalization": agent_cfg.policy.critic_obs_normalization,
            "actor_hidden_dims": agent_cfg.policy.actor_hidden_dims,
            "critic_hidden_dims": agent_cfg.policy.critic_hidden_dims,
            "activation": agent_cfg.policy.activation,
        },
        "algorithm": {
            "class_name": agent_cfg.algorithm.class_name,
            "num_learning_epochs": agent_cfg.algorithm.num_learning_epochs,
            "num_mini_batches": agent_cfg.algorithm.num_mini_batches,
            "learning_rate": agent_cfg.algorithm.learning_rate,
            "schedule": agent_cfg.algorithm.schedule,
            "gamma": agent_cfg.algorithm.gamma,
            "lam": agent_cfg.algorithm.lam,
            "entropy_coef": agent_cfg.algorithm.entropy_coef,
            "desired_kl": agent_cfg.algorithm.desired_kl,
            "max_grad_norm": agent_cfg.algorithm.max_grad_norm,
            "value_loss_coef": agent_cfg.algorithm.value_loss_coef,
            "use_clipped_value_loss": agent_cfg.algorithm.use_clipped_value_loss,
            "clip_param": agent_cfg.algorithm.clip_param,
        },
    }
    import os as _os
    log_dir = _os.path.join(_os.getcwd(), "logs", "rsl_rl", agent_cfg.experiment_name)
    _os.makedirs(log_dir, exist_ok=True)
    runner = OnPolicyRunner(env, runner_cfg_dict, log_dir=log_dir, device=agent_cfg.device)
    
    # Training
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

