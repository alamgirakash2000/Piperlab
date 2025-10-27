# Piper Robot Control GUI

A graphical user interface for controlling the Piper robotic arm with real-time joint control and preset positions.

## Features

### Joint Control
- **6 Independent Sliders**: Control each joint individually with visual feedback
- **Angle Limits**: Sliders automatically constrained to safe joint limits
- **Real-time Values**: See target angles before moving
- **Speed Control**: Adjust movement speed from 5% to 50%
- **⚡ Live Control Mode**: Robot moves in real-time as you adjust sliders (NEW!)

### Preset Positions
- **Reset**: All joints to 0° (home position)
- **Stand**: Standing posture [0°, 45°, -45°, 0°, 0°, 0°]
- **Rest**: Resting position [0°, 90°, -90°, 0°, 0°, 0°]
- **Ready**: Ready for operation [0°, 30°, -30°, 0°, 20°, 0°]
- **Custom Preset**: Save and load custom positions

### Gripper Control
- **🔓 Open Gripper**: Instantly open the gripper to maximum position (800000)
- **🔒 Close Gripper**: Close the gripper to fully closed position (0)
- **Position Slider**: Fine control of gripper position (0 = closed, 800000 = open)
- **Status Display**: Real-time gripper position and state feedback

### Live Monitoring
- **Current Position Display**: Real-time robot joint positions
- **Auto-refresh**: Optional 1Hz position monitoring
- **Status Log**: Timestamped activity log

### Safety Features
- **Connection Status**: Visual indicator (red/green)
- **Emergency Stop**: Immediate disconnect
- **Movement Confirmation**: Clear feedback for all actions

## Installation

No additional installation required! The GUI uses tkinter which comes with Python.

## Usage

### Quick Start

```bash
# 1. Activate CAN interface (required first time)
sudo bash scripts/1_setup_can.sh

# 2. Launch GUI
bash gui/launch_gui.sh
```

Or run directly:
```bash
python3 gui/robot_control_gui.py
```

### Step-by-Step

1. **Connect to Robot**
   - Click "Connect" button
   - Wait for "Connected" status (green)
   - Motors will automatically enable

2. **Choose Control Mode**
   
   **Option A: Manual Mode (Default)**
   - Move sliders to desired angles
   - Adjust speed if needed
   - Click "▶ Move to Position" to execute
   
   **Option B: Live Control Mode ⚡ (Real-time)**
   - Enable "⚡ Live Control Mode" checkbox
   - Robot moves immediately as you adjust sliders
   - Great for fine-tuning positions
   - Disable when done

3. **Use Presets**
   - Click any preset button (Reset, Stand, Rest, Ready)
   - Sliders will update to preset values
   - Click "▶ Move to Position" to execute

4. **Monitor Position**
   - Click "↻ Refresh" to read current position
   - Or enable "Auto-refresh" for continuous monitoring

5. **Save Custom Preset**
   - Set sliders to desired position
   - Click "Save Current" in Custom Preset section
   - Click "Load Custom" to restore later

6. **Control Gripper**
   - Click "🔓 Open Gripper" to fully open
   - Click "🔒 Close Gripper" to fully close
   - Or use the position slider for fine control
   - Click "Move to Position" to move gripper to slider value
   - Gripper status updates automatically with position refresh

## Joint Limits

| Joint | Minimum | Maximum | Description |
|-------|---------|---------|-------------|
| Joint 1 | -150° | +150° | Base rotation |
| Joint 2 | 0° | +180° | Shoulder |
| Joint 3 | -170° | 0° | Elbow |
| Joint 4 | -100° | +100° | Wrist rotation |
| Joint 5 | -70° | +70° | Wrist bend |
| Joint 6 | -120° | +120° | End effector rotation |

## Preset Positions Explained

### Reset (Home)
All joints at 0° - standard reference position for calibration and startup.

### Stand
Vertical standing posture with slight forward reach. Useful for inspection and as a safe intermediate position.

### Rest
Compact resting position with joints folded. Minimal workspace footprint, good for storage or standby.

### Ready
Positioned for typical pick-and-place operations. Balanced posture ready for forward movements.

## Troubleshooting

### "Failed to connect to robot"

**Check:**
1. CAN interface is UP: `ip link show can0`
2. Run: `sudo bash scripts/1_setup_can.sh`
3. Robot is powered on
4. USB cable is connected

### Robot doesn't move

**Check:**
1. Connection status shows green "Connected"
2. Joint angles are within limits
3. Speed is not set too low (< 5%)
4. Robot power supply is adequate

### Position display shows raw values

The current position display shows encoder values from the robot. These are internal motor positions and may differ from degree values.

### GUI won't launch

**Check:**
1. Python 3 is installed: `python3 --version`
2. tkinter is available: `python3 -c "import tkinter"`
3. Running from correct directory

## Safety Guidelines

- **Start Slow**: Use low speeds (10-20%) when testing new positions
- **Clear Workspace**: Ensure robot path is clear before moving
- **Emergency Stop**: Use emergency stop button if unexpected behavior occurs
- **Gradual Changes**: Make incremental position changes, not large jumps
- **Monitor Movement**: Watch robot during movement execution

## Control Modes Explained

### Manual Mode (Default)
- Set all joint angles with sliders
- Click "Move to Position" to execute
- Robot moves smoothly to target position
- **Best for**: Precise positioning, complex movements

### Live Control Mode ⚡
- Enable checkbox: "⚡ Live Control Mode"
- Robot responds immediately to slider changes
- Continuous real-time control at 10 Hz update rate
- **Best for**: Fine-tuning, teaching positions, dynamic control
- **Warning**: Start with low speed (10-15%) in live mode

## Tips

1. **Smooth Movements**: Use speeds between 15-25% for smooth operation
2. **Test First**: Try preset positions before custom movements
3. **Live Mode**: Start with low speed and small movements when using live control
4. **Auto-refresh**: Enable when tuning positions to see real-time feedback
5. **Intermediate Positions**: Move to preset positions before large movements
6. **Teaching Mode**: Use live control with auto-refresh for position teaching
7. **Gripper Control**: Test gripper open/close without objects first
8. **Gripper Position**: Use slider for partial opening when gripping delicate objects

## Keyboard Shortcuts

- `Ctrl+C` in terminal: Emergency exit
- Window close: Disconnects and exits safely

## Files

```
gui/
├── robot_control_gui.py    # Main GUI application
├── launch_gui.sh           # Launcher script
└── README.md               # This file
```

## Technical Details

- **Framework**: tkinter (Python standard library)
- **Communication**: CAN bus via piper_sdk
- **Update Rate**: 1 Hz position monitoring (when enabled)
- **Thread Safe**: Background monitoring in separate thread
- **Unit Conversion**: Automatic degrees ↔ SDK units (0.001°)

## Advanced Usage

### Running in Background

```bash
nohup python3 gui/robot_control_gui.py &
```

### Debug Mode

```bash
python3 -u gui/robot_control_gui.py 2>&1 | tee gui_log.txt
```

## Support

For issues or questions:
- Check main README.md in project root
- Review SDK documentation in docs/
- See hardware manual in asserts/

---

*GUI for Piper Robot Control System*  
*Compatible with Ubuntu 24.04*

