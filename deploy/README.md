Piper hardware policy deployment
================================

Quick start
-----------

1) Prepare the robot (once per boot):

```bash
sudo bash piper_sdk/scripts/setup_can.sh
python3 piper_sdk/scripts/test_robot.py
python3 piper_sdk/scripts/move_robot.py
```

2) Run the trained policy on the robot:

```bash
# Uses the latest run under logs/rsl_rl/piper_reach by default
python3 deploy/run_policy_on_piper.py   --experiment_name piper_reach --run_name v1   --rate_hz 50 --policy_hz 15   --smooth_alpha 0.1 --max_step_rad 0.005   --command_deadband_rad 0.001   --speed_percent 20 --device cpu --headless --run_sequence --keep_kit_open
```

What it does
------------

- Loads `logs/rsl_rl/<experiment>/<run>/model.pt` directly (no export needed).
- Instantiates the PPO policy headlessly to match training normalization and network shape.
- Closes the simulator and streams joint position targets to the Piper via `piper_sdk` at the requested rate.
- Observation on hardware mirrors training: `[q, dq, target(7), last_action]`.

Arguments
---------

- `--checkpoint`: explicit path to `model.pt` (otherwise resolved via experiment/run).
- `--experiment_name`, `--run_name`: where to find your checkpoint under `logs/rsl_rl`.
- `--target "x y z yaw"`: fixed target in robot base frame (meters, radians). Pitch is fixed to π (tool down), roll=0.
- `--rate_hz`: control loop frequency (default 50 Hz).
- `--speed_percent`: Piper speed percent (default 20; start low!).
- `--duration_s`: run time limit.
- `--resample_sec`: if >0, resample target every N seconds inside `--box` bounds (sim-like behavior).
- `--box "x_min x_max y_min y_max z_min z_max"`: sampling bounds in base frame (meters).

Safety notes
------------

- Start at low speed (10–30%), keep the workspace clear, remain ready to Ctrl+C.
- Targets are in the robot base frame; ensure they are reachable and collision-free.
- Real hardware differs from sim; test nearby targets first, increase speed gradually.

Troubleshooting
---------------

- If you see CAN errors, re-run: `sudo bash piper_sdk/scripts/1_setup_can.sh`.
- If no checkpoint is found, verify `logs/rsl_rl/<experiment>/<run>/model.pt` exists.
- For slower machines, use `--device cpu` and `--rate_hz 20` initially.


