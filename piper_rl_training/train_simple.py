#!/usr/bin/env python
"""Simple standalone training script for Piper robot reaching task.

Usage:
    ./isaaclab.sh -p piper_rl_training/train_simple.py --num_envs 256 --headless
"""

import argparse
import os
import glob
from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Train Piper robot reaching with RSL-RL PPO")
parser.add_argument("--num_envs", type=int, default=256, help="Number of environments")
parser.add_argument("--max_iterations", type=int, default=100, help="Max training iterations")
parser.add_argument("--run_name", type=str, default="", help="Run folder name suffix (version tag)")
parser.add_argument("--experiment_name", type=str, default="piper_reach", help="Top-level experiment folder")

# Append AppLauncher arguments  
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch simulator
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Import after app launch
import torch
from piper_reach_env_cfg import PiperReachEnvCfg
from agents.rsl_rl_ppo_cfg import PiperReachPPORunnerCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def main():
    """Train with RSL-RL PPO."""
    
    try:
        # Create environment config
        env_cfg = PiperReachEnvCfg()
        env_cfg.scene.num_envs = args.num_envs
        
        print("\n" + "="*80)
        print("PIPER ROBOT - PPO REACHING TRAINING")
        print("="*80)
        print(f"Creating environment with {args.num_envs} parallel environments...")
        
        # Create environment
        env = ManagerBasedRLEnv(cfg=env_cfg)
        print(f"✓ Environment created successfully!")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        
        # Wrap for RSL-RL
        env = RslRlVecEnvWrapper(env)
        print(f"✓ Environment wrapped for RSL-RL")
        
        # Get agent config
        agent_cfg = PiperReachPPORunnerCfg()
        agent_cfg.max_iterations = args.max_iterations
        # allow overriding experiment/run names from CLI
        if args.experiment_name:
            agent_cfg.experiment_name = args.experiment_name
        if args.run_name:
            agent_cfg.run_name = args.run_name
        
        print(f"\nTraining configuration:")
        print(f"  Max iterations: {args.max_iterations}")
        print(f"  Episode length: {env_cfg.episode_length_s}s")
        print(f"  Policy network: {agent_cfg.policy.actor_hidden_dims}")
        print("="*80 + "\n")
        
        # Ensure device selection
        agent_cfg.device = args.device if hasattr(args, "device") and args.device else "cuda:0"

        # Build RSL-RL runner config dict explicitly (avoids MISSING issues)
        from rsl_rl.runners import OnPolicyRunner

        # Auto-resume from last checkpoint (if present)
        exp_dir = os.path.join(os.getcwd(), "logs", "rsl_rl", agent_cfg.experiment_name)
        run_dir_name = agent_cfg.run_name if agent_cfg.run_name else None
        ckpt_file_name = None
        if agent_cfg.run_name:
            # Look inside specific run folder
            run_path = os.path.join(exp_dir, agent_cfg.run_name)
            if os.path.isdir(run_path):
                ckpts = glob.glob(os.path.join(run_path, "model_*.pt"))
                if ckpts:
                    ckpt_file_name = os.path.basename(max(ckpts, key=os.path.getmtime))
        else:
            # Pick latest run subfolder under experiment
            if os.path.isdir(exp_dir):
                try:
                    run_dirs = [d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))]
                    if run_dirs:
                        run_dir_name = max(run_dirs, key=lambda d: os.path.getmtime(os.path.join(exp_dir, d)))
                        ckpts = glob.glob(os.path.join(exp_dir, run_dir_name, "model_*.pt"))
                        if ckpts:
                            ckpt_file_name = os.path.basename(max(ckpts, key=os.path.getmtime))
                except Exception:
                    run_dir_name = None
                    ckpt_file_name = None

        runner_cfg_dict = {
            "seed": agent_cfg.seed,
            "device": agent_cfg.device,
            "num_steps_per_env": agent_cfg.num_steps_per_env,
            "max_iterations": agent_cfg.max_iterations,
            "clip_actions": getattr(agent_cfg, "clip_actions", None),
            # ensure checkpoints every 100 iterations (or at least once at the end)
            "save_interval": min(100, agent_cfg.max_iterations),
            "experiment_name": agent_cfg.experiment_name,
            "run_name": agent_cfg.run_name,
            "logger": agent_cfg.logger,
            # auto-resume fields
            "resume": bool(ckpt_file_name),
            "load_run": run_dir_name if ckpt_file_name else agent_cfg.load_run,
            "load_checkpoint": ckpt_file_name if ckpt_file_name else agent_cfg.load_checkpoint,
            # Our env exposes a single observation group named 'policy'
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

        # Resolve a concrete log directory (per run folder when run_name provided)
        log_dir = os.path.join(exp_dir, agent_cfg.run_name) if agent_cfg.run_name else exp_dir
        os.makedirs(log_dir, exist_ok=True)
        runner = OnPolicyRunner(env, runner_cfg_dict, log_dir=log_dir, device=agent_cfg.device)
        print("✓ PPO runner created successfully!")
        print(f"\nStarting training for {agent_cfg.max_iterations} iterations...\n")

        # Train in chunks to enforce clean 100,200,300... checkpoint names and update model.pt
        import shutil
        total = agent_cfg.max_iterations
        chunk = 100
        completed = 0
        run_dir = runner.log_dir
        while completed < total:
            step = min(chunk, total - completed)
            runner.learn(num_learning_iterations=step, init_at_random_ep_len=True)
            completed += step
            # after each window, copy latest checkpoint to both model.pt and model_{completed}.pt
            try:
                ckpts = glob.glob(os.path.join(run_dir, "model_*.pt"))
                if ckpts:
                    latest = max(ckpts, key=os.path.getmtime)
                    # best/latest pointer
                    shutil.copy2(latest, os.path.join(run_dir, "model.pt"))
                    # enforce clean numbering
                    numbered = os.path.join(run_dir, f"model_{completed}.pt")
                    if not os.path.exists(numbered):
                        shutil.copy2(latest, numbered)
                    print(f"[INFO] Saved checkpoints: model.pt and model_{completed}.pt")
                    # remove off-cycle checkpoints (keep only multiples of 100)
                    for f in ckpts:
                        try:
                            n = int(os.path.splitext(os.path.basename(f))[0].split("_")[-1])
                            if n % 100 != 0:
                                os.remove(f)
                        except Exception:
                            pass
            except Exception:
                pass
        
        print("\n" + "="*80)
        print("✓ TRAINING COMPLETED SUCCESSFULLY!")
        print(f"  Logs saved to: {runner.log_dir}")
        print("="*80 + "\n")
        
        env.close()
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"✗ ERROR OCCURRED:")
        print(f"{'='*80}")
        print(f"{e}")
        print("="*80)
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
