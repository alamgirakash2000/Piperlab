# Piper Robot Control - Setup & Usage Guide

## Quick Start

**Three commands to get started:**

```bash
# 1. Activate CAN interface (run once after boot/USB reconnect)
sudo bash scripts/1_setup_can.sh

# 2. Test connection - safe, reads sensor data only
python3 scripts/test_robot.py

# 3. Execute demo movements - robot will physically move ⚠️
python3 scripts/3_move_robot.py

# Alternative: Use GUI for interactive control
bash gui/launch_gui.sh
```

---

## Overview

This repository contains the Piper SDK for controlling the Agile X Piper robotic arm via CAN interface. This guide covers setup, configuration, and control of the physical Piper robot on Ubuntu 24.04.

---

## File Structure

```
piper_sdk/
│
├── scripts/                          CONTROL SCRIPTS
│   ├── 1_setup_can.sh                # CAN interface activation (sudo required)
│   ├── test_robot.py                 # Connection test (reads data only)
│   └── 3_move_robot.py               # Movement demonstrations
│
├── gui/                              GRAPHICAL INTERFACE
│   ├── robot_control_gui.py          # GUI application
│   ├── launch_gui.sh                 # GUI launcher
│   └── README.md                     # GUI documentation
│
├── piper_sdk/                        SDK PACKAGE
│   ├── interface/                    # Robot interface classes
│   ├── hardware_port/                # CAN communication layer
│   ├── demo/V2/                      # Official demo scripts
│   ├── kinematics/                   # Forward/inverse kinematics
│   └── piper_msgs/                   # Message definitions
│
├── asserts/                          HARDWARE DOCUMENTATION
│   ├── can_config.MD                 # CAN configuration guide
│   ├── Q&A.MD                        # Common questions
│   └── wire_connection.PNG           # Wiring diagram
│
├── docs/                             ORIGINAL SDK DOCS
│   ├── README.MD                     # Original SDK readme
│   └── CHANGELOG.MD                  # Version history
│
├── README.md                         THIS FILE
├── setup.py                          # SDK installation script
└── LICENSE                           # License file
```

---

## Prerequisites & Installation

### Hardware Setup

**Required Components:**
- Piper robotic arm
- CAN-to-USB adapter
- USB cable
- Power supply (max 26V, minimum 10A)

**Connection Steps:**
1. Wire CAN_H and CAN_L from robot to CAN-to-USB adapter
2. Connect CAN-to-USB adapter to computer via USB
3. Power on robot (ensure voltage ≤26V, current ≥10A)

### Software Dependencies

**Package Installation:**

```bash
# Install CAN utilities
sudo apt-get update
sudo apt-get install can-utils iproute2

# Install Python CAN library
pip3 install python-can
```

**Verify Installation:**
```bash
python3 -c "import can; print('python-can version:', can.__version__)"
```

---

## Control Methods

### Option A: Graphical Interface (Recommended for Beginners)

Launch the GUI for interactive control:

```bash
bash gui/launch_gui.sh
```

**GUI Features:**
- Visual sliders for all 6 joints
- Gripper control (open/close buttons and position slider)
- Preset positions (Reset, Stand, Rest, Ready)
- Real-time position monitoring (joints + gripper)
- Speed control
- Custom preset save/load

See `gui/README.md` for complete GUI documentation.

### Option B: Command Line Scripts

## Usage Instructions

### Step 1: Activate CAN Interface

**Command (requires sudo):**

```bash
sudo bash scripts/1_setup_can.sh
```

**When to run:**
- After system boot
- After USB device reconnection
- When error "CAN port can0 is not UP" appears

### Step 2: Test Robot Connection

```bash
python3 scripts/test_robot.py
```

**Functionality:**
- Establishes CAN connection to robot
- Enables all joints
- Displays real-time joint positions and gripper status (200 Hz)
- Press Ctrl+C to terminate

Successful connection shows updating joint values.

### Step 3: Execute Movement Demonstrations

⚠️ **WARNING:** Robot will physically move. Clear workspace before execution.

```bash
python3 scripts/3_move_robot.py
```

**Movement Sequence:**
- Joint 1: +5° → return to zero
- Joint 2: +5° → return to zero
- Joint 5: +10° → return to zero
- Gripper: close → open

**Safety Features:**
- Reduced speed (30% of maximum)
- Small movement increments
- Emergency stop via Ctrl+C

---

## Technical Reference

### CAN Interface

**Communication Parameters:**
- Protocol: CAN 2.0
- Baud Rate: 1000000 (fixed)
- Interface: can0

**Status Check:**
```bash
ip link show can0
```

States:
- `UP` = Active and operational
- `DOWN` = Inactive, requires activation

### Joint Specifications

| Joint | Range (radians) | Range (degrees) |
|-------|-----------------|-----------------|
| Joint 1 | [-2.6179, 2.6179] | [-150°, 150°] |
| Joint 2 | [0, 3.14] | [0°, 180°] |
| Joint 3 | [-2.967, 0] | [-170°, 0°] |
| Joint 4 | [-1.745, 1.745] | [-100°, 100°] |
| Joint 5 | [-1.22, 1.22] | [-70°, 70°] |
| Joint 6 | [-2.09439, 2.09439] | [-120°, 120°] |

**Angle Units:**
- SDK uses 0.001 degree units
- Conversion: `angle_units = radians × 57295.7795`
- Example: 0.1 rad = 5.73° = 5729 units

---

## Programming Guide

### Basic Control Template

```python
#!/usr/bin/env python3
from piper_sdk import *
import time

# Initialize connection
piper = C_PiperInterface_V2()
piper.ConnectPort()

# Enable motors
piper.EnableArm(7)  # 6 joints + gripper
time.sleep(1)

# Configure control mode (CAN control, Joint mode, 50% speed)
piper.MotionCtrl_2(0x01, 0x01, 50, 0x00)
time.sleep(0.1)

# Move joint (angles in 0.001 degree units)
j1 = round(0.087 * 57295.7795)  # 5 degrees
piper.JointCtrl(j1, 0, 0, 0, 0, 0)
time.sleep(2)

# Return to zero position
piper.JointCtrl(0, 0, 0, 0, 0, 0)
time.sleep(2)

# Gripper control (position, speed, force_flag, force)
piper.GripperCtrl(800000, 1000, 0x01, 0)  # Close
time.sleep(2)
piper.GripperCtrl(0, 1000, 0x01, 0)       # Open
```

### Core API Functions

**Connection Management:**
```python
piper = C_PiperInterface_V2()    # Create interface instance
piper.ConnectPort()               # Connect to can0
```

**Motion Control:**
```python
piper.EnableArm(7)                              # Enable all motors
piper.MotionCtrl_2(ctrl, move, speed, mit)      # Set control mode
piper.JointCtrl(j1, j2, j3, j4, j5, j6)         # Command joint positions
piper.GripperCtrl(pos, spd, flag, force)        # Control gripper
```

**State Reading:**
```python
joints = piper.GetArmJointMsgs()     # Read joint positions
gripper = piper.GetArmGripperMsgs()  # Read gripper status
pose = piper.GetArmEndPose()         # Read end-effector pose
status = piper.GetArmStatus()        # Read robot status
```

---

## Official Demo Scripts

Advanced examples located in `piper_sdk/demo/V2/`:

**Notable Examples:**
- `piper_read_joint_state.py` - Continuous joint state monitoring
- `piper_ctrl_joint.py` - Joint control demonstration
- `piper_ctrl_end_pose.py` - End-effector control
- `piper_ctrl_gripper.py` - Gripper operation
- `piper_read_firmware.py` - Firmware version query

**Execution:**
```bash
cd piper_sdk/demo/V2
python3 piper_read_joint_state.py
```

Complete list available in `piper_sdk/demo/V2/README.MD`

---

## Troubleshooting

### Error: "CAN port can0 is not UP"

**Solution:**
```bash
sudo bash scripts/1_setup_can.sh
```

### Error: "CAN socket can0 does not exist"

**Common Causes:**
- USB adapter disconnected
- Device not recognized by system

**Resolution:**
1. Verify USB physical connection
2. Reconnect USB adapter
3. Execute: `sudo bash scripts/1_setup_can.sh`

### Error: "No module named 'can'"

**Solution:**
```bash
pip3 install python-can
```

### Robot Not Responding

**Diagnostic Checklist:**
1. Robot power status (verify LED indicators)
2. CAN interface status: `ip link show can0`
3. USB cable connection
4. CAN wiring (verify CAN_H, CAN_L connections)
5. CAN interface activation: `sudo bash scripts/1_setup_can.sh`

### Robot Movement Failure

**Common Issues:**
- Motors not enabled: execute `piper.EnableArm(7)` before motion commands
- Control mode not configured: call `piper.MotionCtrl_2()` before `JointCtrl()`
- Emergency stop activated: check physical buttons
- Joint limits violated: verify target angles within specifications

---

## Safety Guidelines

### Workspace Safety
- Clear workspace before robot operation
- Maintain safe distance during movement
- Emergency stop: Ctrl+C in terminal

### Movement Protocol
- Begin with minimal movements (5-10 degrees)
- Use reduced speeds (20-50%) during testing
- Test joint mode before end-effector mode

### Power Specifications
- Maximum voltage: 26V
- Minimum current: 10A
- Maximum payload: 1.5kg

### Emergency Procedures
- Software stop: Ctrl+C in terminal
- No automatic collision detection available
- Continuous supervision required during operation

---

## Additional Resources

- **Hardware Documentation:** `asserts/` directory
- **CAN Configuration:** `asserts/can_config.MD`
- **FAQ:** `asserts/Q&A.MD`
- **Original SDK Documentation:** `docs/` directory
- **Version History:** `docs/CHANGELOG.MD`
- **Official Repository:** https://github.com/agilexrobotics/piper_sdk

---

## Standard Workflow

```bash
# 1. Activate CAN interface (once per boot/USB reconnect)
sudo bash scripts/1_setup_can.sh

# 2. Test connection (safe, no movement)
python3 scripts/test_robot.py

# 3. Execute control script
python3 scripts/3_move_robot.py
```

---

## Quick Reference Table

| Command | Function | Requires Sudo |
|---------|----------|---------------|
| `sudo bash scripts/1_setup_can.sh` | Activate CAN interface | Yes |
| `python3 scripts/test_robot.py` | Test connection | No |
| `python3 scripts/3_move_robot.py` | Execute demo movements | No |
| `ip link show can0` | Check CAN status | No |

**System Specifications:**
- Communication Rate: 200 Hz (5ms cycle)
- Baud Rate: 1000000 (fixed)
- Protocol: CAN 2.0
- SDK Version: See `piper_sdk/version.py`

---

*Documentation for Agile X Piper Robot*  
*Platform: Ubuntu 24.04*  
*Last Updated: 2025-10-20*
