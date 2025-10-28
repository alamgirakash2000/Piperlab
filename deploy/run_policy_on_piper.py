#!/usr/bin/env python3
"""
Run trained PPO policy on the physical Piper robot (no export required).

This script:
- Loads an RSL-RL PPO checkpoint (logs/rsl_rl/.../model.pt)
- Builds the policy headlessly using Isaac Lab's OnPolicyRunner (to match training shapes/normalization)
- Closes the simulator and enters a hardware control loop using piper_sdk

Usage example:

  # 1) Bring up CAN and test robot
  sudo bash piper_sdk/scripts/1_setup_can.sh
  python3 piper_sdk/scripts/test_robot.py

  # 2) Run policy on hardware (slow and safe!)
  python3 deploy/run_policy_on_piper.py \
    --experiment_name piper_reach --run_name v1 \
    --rate_hz 50 --speed_percent 20 --duration_s 60 \
    --target "0.35 0.00 0.20 0.00" --device cpu --headless

Notes:
- Target (x y z yaw) is in robot base frame. Pitch is fixed down (pi), roll=0.
- Start with low speed (10-30%) and keep clear workspace.
"""

import argparse
import math
import os
import signal
import sys
import time
from pathlib import Path

import torch


# --------------------------------------------------------------------------------------
# Helper: robust import paths (repo-root so we can import piper_rl_training, isaaclab_rl)
# --------------------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# --------------------------------------------------------------------------------------
# Isaac Lab imports (after adding CLI flags)
# --------------------------------------------------------------------------------------
from isaaclab.app import AppLauncher


def add_args():
    parser = argparse.ArgumentParser("Deploy PPO policy to Piper hardware (joint position control)")

    # Checkpoint resolution (either explicit path, or experiment/run)
    parser.add_argument("--checkpoint", type=str, default="", help="Path to model.pt (optional)")
    parser.add_argument("--experiment_name", type=str, default="piper_reach", help="Experiment under logs/rsl_rl/")
    parser.add_argument("--run_name", type=str, default="", help="Run folder name (if empty, pick latest)")

    # Control loop
    parser.add_argument("--rate_hz", type=float, default=50.0, help="Control frequency")
    parser.add_argument("--speed_percent", type=int, default=20, help="Robot speed percent (1-100)")
    parser.add_argument("--duration_s", type=float, default=60.0, help="Run time limit (seconds)")

    # Target selection (base frame): x y z yaw; roll=0, pitch=pi (point down)
    parser.add_argument(
        "--target",
        type=str,
        default="0.35 0.00 0.20 0.00",
        help="Fixed target: 'x y z yaw' in meters/radians (base frame)",
    )
    # Optional random resampling like in sim
    parser.add_argument(
        "--resample_sec",
        type=float,
        default=0.0,
        help="If >0, resample target every N seconds inside --box (sim-like)",
    )
    parser.add_argument(
        "--box",
        type=str,
        default="0.2 0.45 -0.25 0.25 0.1 0.35",
        help="Sampling box: 'x_min x_max y_min y_max z_min z_max' (m) in base frame",
    )

    # Add AppLauncher args; we force headless by default
    AppLauncher.add_app_launcher_args(parser)
    # Avoid potential Kit close hang by keeping it open (optional)
    parser.add_argument("--keep_kit_open", action="store_true", help="Keep Kit running after policy build")
    # Built-in multi-target sequence
    parser.add_argument("--run_sequence", action="store_true", help="Run built-in 5 targets then resting pose")
    parser.add_argument("--target_hold_s", type=float, default=6.0, help="Seconds to hold each sequence target")
    parser.add_argument("--reach_tol_m", type=float, default=0.02, help="Position tolerance (meters) to count as reached")
    parser.add_argument("--target_timeout_s", type=float, default=12.0, help="Max seconds per target before skipping")
    # Motion smoothing / rate limiting
    parser.add_argument("--smooth_alpha", type=float, default=0.2, help="EMA smoothing factor for joint targets (0..1)")
    parser.add_argument("--max_step_rad", type=float, default=0.02, help="Max joint change per step (rad)")
    parser.add_argument("--policy_hz", type=float, default=10.0, help="How often to update the policy (Hz)")
    parser.add_argument("--command_deadband_rad", type=float, default=0.002, help="Min joint change to actually send (rad)")
    return parser


# -----------------------------------------------------
# PPO policy bootstrap (headless env just to build net)
# -----------------------------------------------------
def build_policy_and_load(ckpt_path: str, device: str, launcher_args):
    # Launch a minimal app to construct the policy correctly using provided args
    app = AppLauncher(launcher_args).app
    try:
        import isaaclab_tasks  # noqa: F401
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from piper_rl_training.piper_reach_env_cfg import PiperReachEnvCfg_PLAY
        from rsl_rl.runners import OnPolicyRunner

        # Tiny env
        env_cfg = PiperReachEnvCfg_PLAY()
        env_cfg.scene.num_envs = 8
        env = ManagerBasedRLEnv(cfg=env_cfg)
        env = RslRlVecEnvWrapper(env)

        # Runner config mirrors training architecture
        runner_cfg = {
            "seed": 42,
            "device": device,
            "num_steps_per_env": 1,
            "max_iterations": 1,
            "clip_actions": None,
            "save_interval": 1,
            "experiment_name": "deploy",
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

        runner = OnPolicyRunner(env, runner_cfg, log_dir=str(REPO_ROOT / "logs" / "deploy"), device=device)

        # Access policy
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
            raise RuntimeError("Could not access ActorCritic from OnPolicyRunner.")

        checkpoint = torch.load(ckpt_path, map_location=device)
        policy.load_state_dict(checkpoint["model_state_dict"])  # includes normalizers
        policy.eval()

        # Detach from any env/runner specifics; we only need the torch module
        return policy
    finally:
        # Optionally keep Kit running to avoid close hang on some systems
        if not getattr(launcher_args, "keep_kit_open", False):
            app.close()


# --------------------------------------
# Utility: target parsing and conversions
# --------------------------------------
def parse_target(s: str):
    parts = [float(p) for p in (s.split(",") if "," in s else s.split())]
    if len(parts) != 4:
        raise ValueError("--target must be 4 numbers: x y z yaw")
    return parts


def euler_rpy_to_quat_wxyz(roll, pitch, yaw):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    if w < 0:
        w, x, y, z = -w, -x, -y, -z
    return (w, x, y, z)


# ----------------------------------------
# Hardware interface via piper_sdk helpers
# ----------------------------------------
RAD_TO_UNITS = 57295.7795  # rad -> 0.001 deg

JOINT_LIMITS = [
    (-2.6179, 2.6179),   # J1
    (0.0, 3.14),         # J2
    (-2.9670, 0.0),      # J3
    (-1.7450, 1.7450),   # J4
    (-1.2200, 1.2200),   # J5
    (-2.09439, 2.09439), # J6
]


def read_joints_rad(piper):
    msg = piper.GetArmJointMsgs()
    js = msg.joint_state
    vals = [js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6]
    return [v / RAD_TO_UNITS for v in vals]


def clamp_joint_limits(q):
    out = []
    for i, qi in enumerate(q):
        lo, hi = JOINT_LIMITS[i]
        out.append(min(max(qi, lo), hi))
    return out


def resolve_checkpoint(exp_name: str, run_name: str) -> str:
    base = REPO_ROOT / "logs" / "rsl_rl" / exp_name
    if not base.is_dir():
        raise FileNotFoundError(f"Experiment folder not found: {base}")
    run = run_name
    if not run:
        runs = [d for d in base.iterdir() if d.is_dir()]
        if not runs:
            raise FileNotFoundError(f"No runs found under: {base}")
        run = max(runs, key=lambda d: d.stat().st_mtime).name
    ckpt = base / run / "model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return str(ckpt)


def maybe_sample_target(args):
    # Sample within box if resampling enabled
    if args.resample_sec and args.resample_sec > 0.0:
        import random
        parts = [float(p) for p in (args.box.split(",") if "," in args.box else args.box.split())]
        if len(parts) != 6:
            raise ValueError("--box must be 6 numbers: x_min x_max y_min y_max z_min z_max")
        x_min, x_max, y_min, y_max, z_min, z_max = parts
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        z = random.uniform(z_min, z_max)
        yaw = random.uniform(-math.pi, math.pi)
        return x, y, z, yaw
    return parse_target(args.target)


def main():
    parser = add_args()
    args = parser.parse_args()

    # Resolve checkpoint path
    ckpt_path = args.checkpoint or resolve_checkpoint(args.experiment_name, args.run_name)

    # Build policy (includes normalizers); device can be cpu for hardware
    device = args.device if hasattr(args, "device") else "cpu"
    print(f"Loading policy from: {ckpt_path} (device={device})")
    policy = build_policy_and_load(ckpt_path, device=device, launcher_args=args)

    # Import here to avoid requiring piper_sdk for non-hardware steps
    try:
        from piper_sdk import C_PiperInterface_V2, C_PiperForwardKinematics
    except Exception:
        try:
            from piper_sdk.interface.piper_interface_v2 import C_PiperInterface_V2
            from piper_sdk.kinematics.piper_fk import C_PiperForwardKinematics
        except Exception:
            # As a last resort, add local SDK path to sys.path
            sdk_path = REPO_ROOT / "piper_sdk"
            if str(sdk_path) not in sys.path:
                sys.path.insert(0, str(sdk_path))
            from piper_sdk.interface.piper_interface_v2 import C_PiperInterface_V2
            from piper_sdk.kinematics.piper_fk import C_PiperForwardKinematics

    # Connect to robot
    print("Connecting to Piper (can0)...")
    piper = C_PiperInterface_V2()
    piper.ConnectPort()
    while not piper.EnablePiper():
        time.sleep(0.01)
    time.sleep(0.3)
    spd = max(1, min(int(args.speed_percent), 100))
    piper.MotionCtrl_2(0x01, 0x01, spd, 0x00)
    print("✓ Connected and enabled. Starting control loop.")

    # Infer action and observation dims from the actor network
    actor_mod = getattr(policy, "actor", None) or getattr(policy, "student", None)
    act_dim = None
    obs_dim = None
    if actor_mod is not None:
        linear_layers = [m for m in actor_mod.modules() if isinstance(m, torch.nn.Linear)]
        if linear_layers:
            obs_dim = getattr(linear_layers[0], "in_features", None)
            act_dim = getattr(linear_layers[-1], "out_features", None)
    if act_dim is None or obs_dim is None:
        raise RuntimeError("Could not infer action/observation dims from policy actor.")

    # Validate expected structure: obs = [q(aj), dq(aj), target(7), last_action(aj)] -> 3*aj+7
    if obs_dim != (3 * act_dim + 7):
        print(f"[WARN] Unexpected obs_dim ({obs_dim}) vs 3*act_dim+7 ({3*act_dim+7}). Proceeding anyway.")

    # Target setup (base frame). Pitch=pi, roll=0 as in training (point down)
    tx, ty, tz, yaw = maybe_sample_target(args)
    qw, qx, qy, qz = euler_rpy_to_quat_wxyz(0.0, math.pi, yaw)
    torch_device = torch.device(args.device if hasattr(args, "device") else "cpu")
    target_vec = torch.tensor([tx, ty, tz, qw, qx, qy, qz], dtype=torch.float32, device=torch_device)

    # Control loop parameters
    rate_hz = max(5.0, float(args.rate_hz))
    dt = 1.0 / rate_hz
    policy_hz = max(1.0, float(args.policy_hz))
    policy_update_interval = max(1, int(round(rate_hz / policy_hz)))
    command_deadband = max(0.0, float(args.command_deadband_rad))
    a_prev = torch.zeros(act_dim, dtype=torch.float32, device=torch_device)
    q_prev = read_joints_rad(piper)
    q_initial = list(q_prev)
    dq_prev = [0.0] * 6
    alpha = 0.2  # velocity low-pass
    # Commanded joint target (smoothed); start from current pose
    q_cmd_prev = list(q_prev)
    last_sent_cmd = list(q_prev)

    running = True
    def on_sigint(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, on_sigint)

    t_end = time.monotonic() + float(args.duration_s)
    t_next_resample = time.monotonic() + float(args.resample_sec) if args.resample_sec and args.resample_sec > 0 else float("inf")

    # Built-in target sequence (5 close targets then a resting pose)
    sequence_targets = [
        (0.34, -0.03, 0.20,  0.00),
        (0.24,  0.2, 0.20,  0.10),
        (0.35,  0.03, 0.32, -0.10),
        (0.33, -0.12, 0.18,  0.00),
        (0.37,  0.02, 0.11,  0.00),
    ]
    # Resting pose will be the pose at script start (q_initial)

    fk = C_PiperForwardKinematics()

    def ee_pose_from_joints(q_rad):
        # Returns (x,y,z [m], roll,pitch,yaw [rad]) using SDK FK
        ee = fk.CalFK(q_rad)[-1]
        x_m, y_m, z_m = ee[0] / 1000.0, ee[1] / 1000.0, ee[2] / 1000.0
        r_rad, p_rad, y_rad = math.radians(ee[3]), math.radians(ee[4]), math.radians(ee[5])
        return x_m, y_m, z_m, r_rad, p_rad, y_rad

    def run_to_target(target_xyz_yaw):
        nonlocal a_prev, q_prev, dq_prev, target_vec, q_cmd_prev, last_sent_cmd
        tx, ty, tz, yaw = target_xyz_yaw
        qw, qx, qy, qz = euler_rpy_to_quat_wxyz(0.0, math.pi, yaw)
        target_vec = torch.tensor([tx, ty, tz, qw, qx, qy, qz], dtype=torch.float32, device=torch_device)

        reached = False
        hold_until = None
        t_deadline = time.monotonic() + float(args.target_timeout_s)
        policy_counter = 0
        q_des_prev = list(q_cmd_prev)
        while running:
            t0 = time.monotonic()

            # Read joints and estimate velocities
            q = read_joints_rad(piper)
            dq_meas = [(qi - qi_prev) / dt for qi, qi_prev in zip(q, q_prev)]
            dq = [alpha * dqm + (1.0 - alpha) * dqp for dqm, dqp in zip(dq_meas, dq_prev)]

            # Build observation
            q_pad = q + [0.0] * max(0, act_dim - len(q))
            dq_pad = dq + [0.0] * max(0, act_dim - len(dq))
            obs = torch.tensor(q_pad + dq_pad, dtype=torch.float32, device=torch_device)
            obs = torch.cat([obs, target_vec, a_prev], dim=0).unsqueeze(0)

            # Update policy at a reduced rate to avoid high-frequency jitter
            if policy_counter % policy_update_interval == 0:
                with torch.no_grad():
                    try:
                        actions = policy.act_inference({"policy": obs})
                    except Exception:
                        actions = policy.act_inference(obs)
                    act = actions.squeeze(0).float()
                act_list = act.tolist()
                q_des_prev = [0.5 * float(ai) for ai in act_list[:6]]
            q_des = q_des_prev
            policy_counter += 1
            q_des = clamp_joint_limits(q_des)

            # EMA smoothing then rate limit
            sa = max(0.0, min(1.0, float(args.smooth_alpha)))
            q_ema = [(1.0 - sa) * q_cmd_prev[i] + sa * q_des[i] for i in range(6)]
            max_step = max(0.001, float(args.max_step_rad))
            q_cmd = []
            for i in range(6):
                delta = q_ema[i] - q_cmd_prev[i]
                if delta > max_step:
                    delta = max_step
                elif delta < -max_step:
                    delta = -max_step
                q_cmd.append(q_cmd_prev[i] + delta)
            q_cmd_prev = q_cmd

            # Deadband: only send when change exceeds threshold
            if max(abs(q_cmd[i] - last_sent_cmd[i]) for i in range(6)) >= command_deadband:
                cmd = [int(round(v * RAD_TO_UNITS)) for v in q_cmd]
                # Do not spam MotionCtrl_2 each cycle; it was set once above
                piper.JointCtrl(cmd[0], cmd[1], cmd[2], cmd[3], cmd[4], cmd[5])
                last_sent_cmd = q_cmd

            a_prev = act.detach()
            q_prev = q
            dq_prev = dq

            # Verify reach using FK
            x_m, y_m, z_m, _, _, yaw_m = ee_pose_from_joints(q)
            pos_err = math.sqrt((x_m - tx) ** 2 + (y_m - ty) ** 2 + (z_m - tz) ** 2)
            # Print brief status
            print(f"target=({tx:.3f},{ty:.3f},{tz:.3f},{yaw:.2f}) curr=({x_m:.3f},{y_m:.3f},{z_m:.3f},{yaw_m:.2f}) err={(pos_err*100):.1f}cm",
                  end="\r", flush=True)

            # Reached criteria
            if pos_err <= float(args.reach_tol_m):
                if hold_until is None:
                    hold_until = time.monotonic() + float(args.target_hold_s)
                elif time.monotonic() >= hold_until:
                    print()  # newline after carriage
                    return True

            # Timeout protection
            if time.monotonic() >= t_deadline:
                print("\n[WARN] Target timeout reached, moving to next.")
                return False

            # Rate keeping
            elapsed = time.monotonic() - t0
            to_sleep = dt - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)

    def go_to_joint_positions(q_target_rad, tol_rad=0.02, timeout_s=12.0):
        # Command target joints once, then wait until within tolerance or timeout
        cmd = [int(round(v * RAD_TO_UNITS)) for v in q_target_rad]
        piper.MotionCtrl_2(0x01, 0x01, spd, 0x00)
        piper.JointCtrl(cmd[0], cmd[1], cmd[2], cmd[3], cmd[4], cmd[5])
        t_deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < t_deadline:
            q_now = read_joints_rad(piper)
            max_err = max(abs(a - b) for a, b in zip(q_now, q_target_rad))
            if max_err <= tol_rad:
                return True
            time.sleep(0.1)
        return False

    if args.run_sequence:
        for idx, tgt in enumerate(sequence_targets, 1):
            print(f"\n[SEQ] Target {idx}/5: {tgt}")
            run_to_target(tgt)
        print("\n[SEQ] Returning to initial joint pose (rest)")
        ok = go_to_joint_positions(q_initial, tol_rad=0.02, timeout_s=12.0)
        if ok:
            print("Returned to initial pose.")
        else:
            print("[WARN] Return-to-initial timed out; robot left at last pose.")
        print("Sequence complete.")
        return

    # Default: free-run for duration with (optional) resampling
    policy_counter = 0
    q_des_prev = list(q_cmd_prev)
    while running and time.monotonic() < t_end:
        t0 = time.monotonic()

        # Optional target resampling
        if time.monotonic() >= t_next_resample:
            tx, ty, tz, yaw = maybe_sample_target(args)
            qw, qx, qy, qz = euler_rpy_to_quat_wxyz(0.0, math.pi, yaw)
            target_vec = torch.tensor([tx, ty, tz, qw, qx, qy, qz], dtype=torch.float32, device=torch_device)
            t_next_resample = time.monotonic() + float(args.resample_sec)

        # Read joints and estimate velocities
        q = read_joints_rad(piper)
        dq_meas = [(qi - qi_prev) / dt for qi, qi_prev in zip(q, q_prev)]
        dq = [alpha * dqm + (1.0 - alpha) * dqp for dqm, dqp in zip(dq_meas, dq_prev)]

        # Build observation [q(act_dim), dq(act_dim), target(7), last_action(act_dim)]
        q_pad = q + [0.0] * max(0, act_dim - len(q))
        dq_pad = dq + [0.0] * max(0, act_dim - len(dq))
        obs = torch.tensor(q_pad + dq_pad, dtype=torch.float32, device=torch_device)
        obs = torch.cat([obs, target_vec, a_prev], dim=0).unsqueeze(0)  # shape (1, obs_dim)

        # Update policy at reduced rate
        if policy_counter % policy_update_interval == 0:
            with torch.no_grad():
                try:
                    actions = policy.act_inference({"policy": obs})
                except Exception:
                    actions = policy.act_inference(obs)
                act = actions.squeeze(0).float()  # (act_dim,)
            act_list = act.tolist()
            q_des_prev = [0.5 * float(ai) for ai in act_list[:6]]
        q_des = q_des_prev
        policy_counter += 1
        q_des = clamp_joint_limits(q_des)

        # Smoothing + rate limit
        sa = max(0.0, min(1.0, float(args.smooth_alpha)))
        q_ema = [(1.0 - sa) * q_cmd_prev[i] + sa * q_des[i] for i in range(6)]
        max_step = max(0.001, float(args.max_step_rad))
        q_cmd = []
        for i in range(6):
            delta = q_ema[i] - q_cmd_prev[i]
            if delta > max_step:
                delta = max_step
            elif delta < -max_step:
                delta = -max_step
            q_cmd.append(q_cmd_prev[i] + delta)
        q_cmd_prev = q_cmd

        # Deadband to avoid chattering
        if max(abs(q_cmd[i] - last_sent_cmd[i]) for i in range(6)) >= command_deadband:
            cmd = [int(round(v * RAD_TO_UNITS)) for v in q_cmd]
            piper.JointCtrl(cmd[0], cmd[1], cmd[2], cmd[3], cmd[4], cmd[5])
            last_sent_cmd = q_cmd

        # Update loop state
        a_prev = act.detach()
        q_prev = q
        dq_prev = dq

        # Rate keeping
        elapsed = time.monotonic() - t0
        to_sleep = dt - elapsed
        if to_sleep > 0:
            time.sleep(to_sleep)

    print("\nStopping. Leaving robot at last commanded pose.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


