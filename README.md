# Piper Robot Control in Isaac Lab

Simulation and control framework for Piper robot using NVIDIA Isaac Lab and Isaac Sim.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Physical Robot Control](#physical-robot-control)
- [Simulation Controls](#simulation-controls)

---

## Quick Start

### Simulation Commands

```bash
# Activate conda environment (always required)
conda activate isaacenv

# 1. Static robot (no movement)
./isaaclab.sh -p piper_rl_project/load_piper.py

# 2. Real-time control (type commands during runtime)
./isaaclab.sh -p piper_rl_project/realtime_control.py

# 3. Demo random movements
./isaaclab.sh -p piper_rl_project/demo_moves.py

# 4. Interactive view (free camera control)
./isaaclab.sh -p piper_rl_project/interactive_view.py
```

---

## Physical Robot Control

**Three commands to get started with the physical robot:**

```bash
# 1. Activate CAN interface (run once after boot/USB reconnect)
sudo bash piper_sdk/scripts/1_setup_can.sh

# 2. Test connection - safe, reads sensor data only
python3 piper_sdk/scripts/test_robot.py

# 3. Execute demo movements - robot will physically move ⚠️
python3 piper_sdk/scripts/3_move_robot.py

# Alternative: Use GUI for interactive control
bash piper_sdk/gui/launch_gui.sh
```

---


## Prerequisites

Before setting up this project, ensure you have:

1. **Operating System**: Ubuntu 20.04 or 22.04 (Linux x86_64)
2. **NVIDIA GPU**: RTX 2000 series or newer with updated drivers
3. **CUDA**: Version 12.x or compatible with your Isaac Sim version
4. **Conda/Miniconda**: For environment management
5. **Isaac Sim**: Version 4.5.0 or 5.0.0 installed

### Isaac Sim Installation

Download and install Isaac Sim from NVIDIA:
- [Isaac Sim Download Page](https://developer.nvidia.com/isaac-sim)
- Follow the official installation guide for your system

---

## Installation

### Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd IsaacLab
```

### Step 2: Create Conda Environment

Create the conda environment using the provided `environment.yml` file:

```bash
# Create environment from yml file
conda env create -f environment.yml

# Activate the environment
conda activate isaacenv
```


### Step 3: Install Isaac Lab Packages

After activating the conda environment, install the Isaac Lab source packages in editable mode:

```bash
# Make sure you're in the IsaacLab root directory
cd /path/to/IsaacLab

# Install Isaac Lab core packages
pip install -e ./source/isaaclab
pip install -e ./source/isaaclab_assets
pip install -e ./source/isaaclab_tasks
pip install -e ./source/isaaclab_rl
pip install -e ./source/isaaclab_mimic

# Install Piper SDK
pip install -e ./piper_sdk
```

### Step 4: Configure Isaac Sim Path

Ensure Isaac Sim is properly linked or set the path:

```bash
# Option 1: If Isaac Sim is installed via pip
# (already configured if installed correctly)

# Option 2: If Isaac Sim is installed as standalone
export ISAACSIM_PATH="/path/to/isaac-sim"
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
```

### Step 5: Verify Installation

Test the installation by running the demo:

```bash
./isaaclab.sh -p piper_rl_project/demo_moves.py
```

## Simulation Controls

### Mouse Controls

- **Left Click + Drag** = Rotate camera
- **Middle Click + Drag** = Pan camera
- **Right Click + Drag** = Zoom
- **Double-click robot** = Focus camera

---

## Troubleshooting

### Common Issues

1. **Import Error: No module named 'isaaclab'**
   ```bash
   # Reinstall Isaac Lab packages
   pip install -e ./source/isaaclab
   ```

2. **CUDA/GPU Issues**
   ```bash
   # Check CUDA availability
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **Isaac Sim Not Found**
   ```bash
   # Verify Isaac Sim installation
   ./isaaclab.sh --help
   ```

4. **CAN Interface Issues (Physical Robot)**
   ```bash
   # Check CAN devices
   ip link show
   
   # Reconfigure CAN interface
   sudo bash piper_sdk/scripts/1_setup_can.sh
   ```

---

## Environment Management

### Exporting Your Current Environment

If you've installed additional packages:

```bash
# Export exact environment with all packages
conda env export > environment_full.yml

# Export only manually installed packages
conda env export --from-history > environment_minimal.yml
```

### Updating Environment

```bash
# Update environment from yml file
conda env update -f environment.yml --prune
```

### Removing Environment

```bash
# Deactivate if active
conda deactivate

# Remove environment
conda env remove -n isaacenv
```

---

## Project Structure

```
IsaacLab/
├── environment.yml           # Conda environment specification
├── .gitignore               # Git ignore rules
├── isaaclab.sh              # Main launcher script
├── piper_rl_project/        # Your simulation scripts
│   ├── demo_moves.py        # Random movement demo
│   ├── load_piper.py        # Static robot loader
│   ├── realtime_control.py  # Interactive control
│   └── interactive_view.py  # Free camera view
├── piper_sdk/               # Piper robot SDK
│   ├── scripts/             # Robot control scripts
│   └── gui/                 # GUI interface
├── piper_isaac_sim/         # Robot models and assets
│   └── usd/                 # USD files for simulation
└── source/                  # Isaac Lab source packages
    ├── isaaclab/
    ├── isaaclab_assets/
    ├── isaaclab_tasks/
    ├── isaaclab_rl/
    └── isaaclab_mimic/
```

---

## License

See `LICENSE` file for details.

---

## Contributing

This project uses Isaac Lab framework. Please refer to the [Isaac Lab documentation](https://isaac-sim.github.io/IsaacLab/) for more information.
