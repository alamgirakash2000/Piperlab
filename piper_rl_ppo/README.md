# Piper PPO Training (IsaacLab)

This doc gives you only what you asked for: commands to train/test PPO policies for Piper in IsaacLab (conda env: `isaacenv`) plus a brief note on policy and rewards. It reuses the already-working Piper reaching and cube-stacking tasks in this repo.

## Prerequisites

```bash
conda activate isaacenv
pip install -e ./source/isaaclab
pip install -e ./source/isaaclab_assets
pip install -e ./source/isaaclab_tasks
pip install -e ./source/isaaclab_rl
```

## Quick sanity check (random demo)

```bash
./isaaclab.sh -p piper_rl_project/demo_moves.py
```

---

## Task A: Reaching (recommended first)

Train PPO to reach random 3D poses with Piper.

### Train

```bash
# short test
./isaaclab.sh -p piper_rl_training/train_simple.py \
  --num_envs 64 --max_iterations 10

# headless medium run (~minutes)
./isaaclab.sh -p piper_rl_training/train_simple.py \
  --num_envs 256 --max_iterations 200 --headless

# full run (~1h depending on GPU)
./isaaclab.sh -p piper_rl_training/train_simple.py \
  --num_envs 512 --max_iterations 2000 --headless
```

### Test

```bash
# replace with your checkpoint path
./isaaclab.sh -p piper_rl_training/test.py \
  --checkpoint logs/rsl_rl/piper_reach/YYYY-MM-DD_HH-MM-SS/model_2000.pt
```

### Policy/Rewards (brief)

- Policy: 3-layer MLP [256, 128, 64], ELU, normalized obs, PPO (RSL-RL).
- Obs: joint pos/vel, target pose, previous action.
- Reward: shaped distance-to-target (+tanh proximity), small penalties on action rate and joint velocity.

---

## Task B: Cube Stacking (Franka-style adapted to Piper)

PPO to stack three cubes (blue base, red middle, green top). Includes gripper open/close actions.

### Train

```bash
# quick check
./isaaclab.sh -p piper_rl_training/stack_task/train.py \
  --num_envs 64 --max_iterations 10

# headless training (longer task; expect hours)
./isaaclab.sh -p piper_rl_training/stack_task/train.py \
  --num_envs 512 --max_iterations 5000 --headless
```

### Test

```bash
./isaaclab.sh -p piper_rl_training/stack_task/test.py \
  --checkpoint logs/rsl_rl/piper_stack/YYYY-MM-DD_HH-MM-SS/model_5000.pt
```

### Policy/Rewards (brief)

- Policy: larger MLP [512, 256, 128], ELU, PPO (RSL-RL).
- Obs: joint pos/vel, gripper, end-effector pose, each cube pose, relative distances; subtask flags for grasp/stack.
- Rewards/Success: dense shaping via object and EE features; episode success when cubes are aligned and height thresholds met (termination condition logs success).

---

## Logging and Monitoring

```bash
tensorboard --logdir logs/rsl_rl
# open http://localhost:6006
```

## Troubleshooting

- OOM: reduce `--num_envs` and keep `--headless`.
- Imports: ensure the editable installs above were run inside `isaacenv`.

---

## Notes

- All commands assume you run them from the repo root with `conda activate isaacenv`.
- Training artifacts go under `logs/rsl_rl/{piper_reach|piper_stack}/`.


