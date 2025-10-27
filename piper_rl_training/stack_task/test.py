#!/usr/bin/env python
"""Testing/evaluation script for trained Piper cube stacking policy.

This script loads a trained checkpoint and evaluates the policy
by visualizing the robot's performance on the stacking task.

Usage:
    ./isaaclab.sh -p piper_rl_training/stack_task/test.py --checkpoint /path/to/model.pt
"""

import argparse
import os
import torch

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Test trained Piper cube stacking policy")
parser.add_argument(
    "--checkpoint", 
    type=str, 
    required=True, 
    help="Path to model checkpoint (e.g., logs/rsl_rl/piper_stack/model_5000.pt)"
)
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Stack-Piper-Play-v0", help="Name of the task.")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to run.")

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

from isaaclab_rl.rsl_rl.wrappers import RslRlVecEnvWrapper


def main():
    """Evaluate trained policy."""
    
    # Parse configuration
    env_cfg = gym.make(args.task).unwrapped.cfg
    
    # Override configuration
    env_cfg.scene.num_envs = args.num_envs
    
    # Create environment
    env = gym.make(args.task, cfg=env_cfg)
    
    # Wrap environment
    env = RslRlVecEnvWrapper(env)
    
    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    print("\n" + "="*80)
    print("PIPER ROBOT - CUBE STACKING POLICY EVALUATION")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Task: {args.task}")
    print(f"Number of environments: {args.num_envs}")
    print(f"Episodes: {args.num_episodes}")
    device = args.device if hasattr(args, 'device') and args.device else "cuda:0"
    print(f"Device: {device}")
    print("="*80 + "\n")
    
    # Load the policy
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Extract policy from checkpoint
    from rsl_rl.modules import ActorCritic
    
    # Get observation and action dimensions
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Create policy network
    policy = ActorCritic(
        num_obs=obs_dim,
        num_critic_obs=obs_dim,
        num_actions=action_dim,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    ).to(device)
    
    # Load policy weights
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    
    print("Policy loaded successfully!\n")
    print("Running evaluation...\n")
    
    # Run evaluation
    obs, _ = env.reset()
    episode_count = 0
    episode_rewards = []
    successes = []
    current_episode_reward = torch.zeros(args.num_envs, device=args.device)
    
    while episode_count < args.num_episodes:
        with torch.no_grad():
            actions = policy.act_inference(obs)
        
        obs, rewards, dones, infos = env.step(actions)
        current_episode_reward += rewards
        
        # Check for episode completion
        if dones.any():
            for idx in torch.where(dones)[0]:
                episode_rewards.append(current_episode_reward[idx].item())
                
                # Check if it was a success termination
                if "log" in infos and "terminations" in infos["log"]:
                    success = infos["log"]["terminations"]["success"][idx].item()
                    successes.append(success)
                
                current_episode_reward[idx] = 0.0
                episode_count += 1
                
                if episode_count <= args.num_episodes:
                    status = "SUCCESS" if (successes and successes[-1]) else "INCOMPLETE"
                    print(f"Episode {episode_count}/{args.num_episodes} - Reward: {episode_rewards[-1]:.2f} - {status}")
                
                if episode_count >= args.num_episodes:
                    break
    
    # Print statistics
    episode_rewards = torch.tensor(episode_rewards[:args.num_episodes])
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Episodes: {len(episode_rewards)}")
    print(f"Mean reward: {episode_rewards.mean():.2f}")
    print(f"Std reward: {episode_rewards.std():.2f}")
    print(f"Min reward: {episode_rewards.min():.2f}")
    print(f"Max reward: {episode_rewards.max():.2f}")
    if successes:
        success_rate = sum(successes[:args.num_episodes]) / len(successes[:args.num_episodes]) * 100
        print(f"Success rate: {success_rate:.1f}%")
    print("="*80 + "\n")
    
    # Close environment
    env.close()


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

