#!/usr/bin/env python3
"""
Piper Robot Movement Demonstration
Prerequisite: CAN interface must be active (run: sudo bash scripts/1_setup_can.sh)

This script demonstrates:
1. Reading current joint positions
2. Moving to a predefined home position
3. Executing coordinated movements
4. Gripper control
"""

import time
from piper_sdk import *

# Conversion factor: radians to 0.001 degrees
# 1 radian = 180/π degrees = 57.2958 degrees = 57295.7795 units (0.001 degree)
RAD_TO_UNITS = 57295.7795

def deg_to_units(degrees):
    """Convert degrees to SDK units (0.001 degrees)"""
    return round(degrees * 1000)

def print_joint_positions(piper, label="Current"):
    """Read and display current joint positions"""
    joints = piper.GetArmJointMsgs()
    print(f"\n{label} Joint Positions:")
    print(f"  Joint 1: {joints.joint_state.joint_1}")
    print(f"  Joint 2: {joints.joint_state.joint_2}")
    print(f"  Joint 3: {joints.joint_state.joint_3}")
    print(f"  Joint 4: {joints.joint_state.joint_4}")
    print(f"  Joint 5: {joints.joint_state.joint_5}")
    print(f"  Joint 6: {joints.joint_state.joint_6}")
    return joints

def move_to_position(piper, j1, j2, j3, j4, j5, j6, speed=15, wait_time=0.5, description=""):
    """Execute joint movement with optional description"""
    if description:
        print(f"\n→ {description}")
    piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)
    time.sleep(0.05)
    piper.JointCtrl(j1, j2, j3, j4, j5, j6)
    
    # Continuous monitoring while moving
    start_time = time.time()
    while time.time() - start_time < wait_time:
        time.sleep(0.1)

# ============================================================================
# MAIN PROGRAM
# ============================================================================

print("=" * 70)
print("PIPER ROBOT - MOVEMENT DEMONSTRATION")
print("=" * 70)
print("\n⚠️  WARNING: Robot will physically move. Clear workspace before proceeding.")
print("Press Ctrl+C at any time to emergency stop.\n")

# Countdown
for i in range(3, 0, -1):
    print(f"Starting in {i}...")
    time.sleep(1)

try:
    # ------------------------------------------------------------------------
    # PHASE 1: INITIALIZATION
    # ------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 1: INITIALIZATION")
    print("=" * 70)
    
    print("\nConnecting to robot...")
    piper = C_PiperInterface_V2()
    piper.ConnectPort()
    print("✓ Connection established")
    
    print("\nEnabling motors...")
    # Use EnablePiper() for V2 interface - it returns True when successful
    while not piper.EnablePiper():
        time.sleep(0.01)
    time.sleep(0.5)
    print("✓ Motors enabled")
    
    # ------------------------------------------------------------------------
    # PHASE 2: READ CURRENT POSITION
    # ------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 2: READING CURRENT POSITION")
    print("=" * 70)
    
    initial_joints = print_joint_positions(piper, "Initial")
    time.sleep(1)
    
    # ------------------------------------------------------------------------
    # PHASE 3: MOVE TO HOME POSITION
    # ------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 3: MOVING TO HOME POSITION")
    print("=" * 70)
    
    print("\nHome position: All joints at 0 degrees")
    move_to_position(
        piper, 0, 0, 0, 0, 0, 0,
        speed=10,
        wait_time=4,
        description="Moving to home position (all joints to 0°)"
    )
    print_joint_positions(piper, "Home Position")
    time.sleep(0.5)  # Brief pause at home
    
    # ------------------------------------------------------------------------
    # PHASE 4: EXECUTE MOVEMENT SEQUENCE
    # ------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 4: EXECUTING MOVEMENT SEQUENCE")
    print("=" * 70)
    
    print("\nSequence: Continuous pick-and-place movements (slow and smooth)")
    print("Speed: 15% of maximum for safety and visibility\n")
    
    # Movement 1: Ready position
    move_to_position(
        piper,
        deg_to_units(10),   # J1: 10°
        deg_to_units(20),   # J2: 20°
        deg_to_units(-15),  # J3: -15°
        deg_to_units(0),    # J4: 0°
        deg_to_units(15),   # J5: 15°
        deg_to_units(0),    # J6: 0°
        speed=15,
        wait_time=3.5,
        description="Movement 1: Ready position"
    )
    
    # Movement 2: Reach forward (continuous from previous)
    move_to_position(
        piper,
        deg_to_units(10),   # J1: 10°
        deg_to_units(40),   # J2: 40°
        deg_to_units(-30),  # J3: -30°
        deg_to_units(0),    # J4: 0°
        deg_to_units(20),   # J5: 20°
        deg_to_units(0),    # J6: 0°
        speed=15,
        wait_time=3,
        description="Movement 2: Reach forward"
    )
    
    # Movement 3: Lower down (continuous)
    move_to_position(
        piper,
        deg_to_units(10),   # J1: 10°
        deg_to_units(50),   # J2: 50°
        deg_to_units(-40),  # J3: -40°
        deg_to_units(0),    # J4: 0°
        deg_to_units(25),   # J5: 25°
        deg_to_units(0),    # J6: 0°
        speed=12,
        wait_time=3.5,
        description="Movement 3: Lower down (simulate pick)"
    )
    
    # Brief pause to simulate picking
    print("   Simulating object pick...")
    time.sleep(1)
    
    # Movement 4: Lift up (continuous)
    move_to_position(
        piper,
        deg_to_units(10),   # J1: 10°
        deg_to_units(20),   # J2: 20°
        deg_to_units(-15),  # J3: -15°
        deg_to_units(0),    # J4: 0°
        deg_to_units(15),   # J5: 15°
        deg_to_units(0),    # J6: 0°
        speed=15,
        wait_time=3.5,
        description="Movement 4: Lift up with object"
    )
    
    # Movement 5: Rotate to place position (continuous)
    move_to_position(
        piper,
        deg_to_units(-15),  # J1: -15° (rotate base)
        deg_to_units(30),   # J2: 30°
        deg_to_units(-25),  # J3: -25°
        deg_to_units(0),    # J4: 0°
        deg_to_units(20),   # J5: 20°
        deg_to_units(0),    # J6: 0°
        speed=15,
        wait_time=3.5,
        description="Movement 5: Rotate to place position"
    )
    
    # Movement 6: Lower to place (continuous)
    move_to_position(
        piper,
        deg_to_units(-15),  # J1: -15°
        deg_to_units(45),   # J2: 45°
        deg_to_units(-35),  # J3: -35°
        deg_to_units(0),    # J4: 0°
        deg_to_units(25),   # J5: 25°
        deg_to_units(0),    # J6: 0°
        speed=12,
        wait_time=3.5,
        description="Movement 6: Lower to place position"
    )
    
    # Brief pause to simulate placing
    print("   Simulating object release...")
    time.sleep(1)
    
    # Movement 7: Retract slightly before returning
    move_to_position(
        piper,
        deg_to_units(-15),  # J1: -15°
        deg_to_units(25),   # J2: 25°
        deg_to_units(-20),  # J3: -20°
        deg_to_units(0),    # J4: 0°
        deg_to_units(15),   # J5: 15°
        deg_to_units(0),    # J6: 0°
        speed=15,
        wait_time=3,
        description="Movement 7: Retract from place position"
    )
    
    # Movement 8: Return to home (continuous)
    move_to_position(
        piper, 0, 0, 0, 0, 0, 0,
        speed=12,
        wait_time=4,
        description="Movement 8: Return to home position"
    )
    print_joint_positions(piper, "Final Position")
    
    print("\n✓ Movement sequence completed successfully")
    
    # ------------------------------------------------------------------------
    # PHASE 5: GRIPPER CONTROL
    # ------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 5: GRIPPER CONTROL DEMONSTRATION")
    print("=" * 70)
    
    try:
        print("\n→ Closing gripper...")
        piper.GripperCtrl(800000, 1000, 0x01, 0)
        time.sleep(2)
        
        gripper_status = piper.GetArmGripperMsgs()
        print(f"Gripper status: {gripper_status}")
        
        print("\n→ Opening gripper...")
        piper.GripperCtrl(0, 1000, 0x01, 0)
        time.sleep(2)
        
        gripper_status = piper.GetArmGripperMsgs()
        print(f"Gripper status: {gripper_status}")
        
        print("✓ Gripper control successful")
        
    except Exception as e:
        print(f"Gripper control error: {e}")
    
    # ------------------------------------------------------------------------
    # COMPLETION
    # ------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nRobot returned to home position and ready for next operation.")

except KeyboardInterrupt:
    print("\n\n⚠️  Emergency stop activated by user")
    print("Robot stopped at current position")
    
except Exception as e:
    print(f"\n\n✗ Error occurred during operation: {e}")
    import traceback
    traceback.print_exc()
    print("\nRecommendations:")
    print("  1. Check CAN interface: ip link show can0")
    print("  2. Verify robot power and connections")
    print("  3. Review joint limits in documentation")

finally:
    print("\n" + "=" * 70)
    print("Additional Resources:")
    print("  - Official demos: piper_sdk/demo/V2/")
    print("  - Documentation: piper_sdk/demo/V2/README.MD")
    print("  - Hardware manual: asserts/")
    print("=" * 70)
