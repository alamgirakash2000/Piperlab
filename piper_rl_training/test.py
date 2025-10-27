#!/usr/bin/env python
"""Testing/evaluation script for trained Piper robot policy.

This script loads a trained checkpoint and evaluates the policy
by visualizing the robot's performance on the reaching task.

Usage:
    ./isaaclab.sh -p piper_rl_training/test.py --checkpoint /path/to/model.pt
"""

import argparse
import os
import torch

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Test trained Piper robot policy")
parser.add_argument("--checkpoint", type=str, default="", help="Explicit path to a .pt checkpoint")
parser.add_argument("--experiment_name", type=str, default="piper_reach", help="Experiment folder under logs/rsl_rl")
parser.add_argument("--run_name", type=str, default="", help="Run folder name (if empty, pick latest)")
parser.add_argument("--num_envs", type=int, default=50, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Reach-Piper-Play-v0", help="Name of the task.")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to run.")
parser.add_argument("--episode_seconds", type=float, default=None, help="Override episode length in seconds.")
parser.add_argument("--max_steps", type=int, default=0, help="Run for a fixed number of steps (continuous mode).")

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

import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.envs import ManagerBasedRLEnv
from piper_reach_env_cfg import PiperReachEnvCfg_PLAY


def main():
    """Evaluate trained policy."""
    
    # Build env directly (GUI-capable)
    env_cfg = PiperReachEnvCfg_PLAY()
    env_cfg.scene.num_envs = args.num_envs
    if args.episode_seconds is not None:
        env_cfg.episode_length_s = float(args.episode_seconds)
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    
    # Resolve checkpoint: explicit path > logs/rsl_rl/{experiment}/{run}/model.pt (latest run if run_name empty)
    ckpt_path = args.checkpoint
    if not ckpt_path:
        base = os.path.join(os.getcwd(), "logs", "rsl_rl", args.experiment_name)
        if not os.path.isdir(base):
            raise FileNotFoundError(f"Experiment folder not found: {base}")
        run_dir = args.run_name
        if not run_dir:
            runs = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
            if not runs:
                raise FileNotFoundError(f"No runs found under: {base}")
            run_dir = max(runs, key=lambda d: os.path.getmtime(os.path.join(base, d)))
        ckpt_path = os.path.join(base, run_dir, "model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    print("\n" + "="*80)
    print("PIPER ROBOT - POLICY EVALUATION")
    print("="*80)
    print(f"Checkpoint: {ckpt_path}")
    print(f"Task: {args.task}")
    print(f"Number of environments: {args.num_envs}")
    print(f"Episodes: {args.num_episodes}")
    device = args.device if hasattr(args, 'device') and args.device else "cuda:0"
    print(f"Device: {device}")
    print("="*80 + "\n")
    
    # Load the policy
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Build a minimal RSL-RL runner to construct a compatible policy, then load weights
    from rsl_rl.runners import OnPolicyRunner
    runner_cfg_dict = {
        "seed": 42,
        "device": device,
        "num_steps_per_env": 1,
        "max_iterations": 1,
        "clip_actions": None,
        "save_interval": 1,
        "experiment_name": "eval",
        "run_name": "",
        "logger": "tensorboard",
        "resume": False,
        "load_run": ".*",
        "load_checkpoint": "model_.*.pt",
        "obs_groups": {"policy": ["policy"], "critic": ["policy"]},
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 1.0,
            "noise_std_type": "scalar",
            "actor_obs_normalization": True,
            "critic_obs_normalization": True,
            "actor_hidden_dims": [256, 128, 64],
            "critic_hidden_dims": [256, 128, 64],
            "activation": "elu",
        },
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
            "learning_rate": 3.0e-4,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "entropy_coef": 0.0,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
        },
    }
    eval_log_dir = os.path.join(os.getcwd(), "logs", "eval")
    os.makedirs(eval_log_dir, exist_ok=True)
    runner = OnPolicyRunner(env, runner_cfg_dict, log_dir=eval_log_dir, device=device)
    # Retrieve policy from runner (version-agnostic)
    policy = None
    for attr in ("actor_critic", "ac", "policy"):
        if hasattr(runner, attr):
            policy = getattr(runner, attr)
            break
    if policy is None:
        for container in ("alg", "algo", "algorithm"):
            obj = getattr(runner, container, None)
            if obj is not None:
                for attr in ("actor_critic", "ac", "policy"):
                    if hasattr(obj, attr):
                        policy = getattr(obj, attr)
                        break
            if policy is not None:
                break
    if policy is None:
        raise RuntimeError("Could not access ActorCritic from OnPolicyRunner (checked runner and alg containers).")
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    
    print("Policy loaded successfully!\n")
    print("Running evaluation...\n")
    
    # Run evaluation
    obs_td, _ = env.reset()
    obs = obs_td["policy"].to(device)
    episode_count = 0
    episode_rewards = []
    current_episode_reward = torch.zeros(args.num_envs, device=device)

    if args.max_steps and args.max_steps > 0:
        # Continuous mode: ignore episodes, run fixed number of steps
        steps_left = int(args.max_steps)
        while steps_left > 0:
            with torch.no_grad():
                try:
                    actions = policy.act_inference({"policy": obs})
                except Exception:
                    actions = policy.act_inference(obs)
            obs_td, rewards, dones, infos = env.step(actions)
            obs = obs_td["policy"].to(device)
            steps_left -= 1
        # No stats in continuous mode
        env.close()
        return
    else:
        # Episode mode
        while episode_count < args.num_episodes:
            with torch.no_grad():
                try:
                    actions = policy.act_inference({"policy": obs})
                except Exception:
                    actions = policy.act_inference(obs)
            
            obs_td, rewards, dones, infos = env.step(actions)
            obs = obs_td["policy"].to(device)
            rewards = rewards.view(-1)
            dones = dones.view(-1).bool()
            current_episode_reward += rewards
            
            # Check for episode completion
            if dones.any():
                done_indices = torch.nonzero(dones, as_tuple=False).squeeze(-1)
                for i in done_indices.tolist():
                    episode_rewards.append(current_episode_reward[i].item())
                    current_episode_reward[i] = 0.0
                    episode_count += 1
                    if episode_count <= args.num_episodes:
                        print(f"Episode {episode_count}/{args.num_episodes} - Reward: {episode_rewards[-1]:.2f}")
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

