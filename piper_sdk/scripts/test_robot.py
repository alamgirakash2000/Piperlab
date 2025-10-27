#!/usr/bin/env python3
# Simple script to test your Piper robot

import time
from piper_sdk import *

print("=" * 50)
print("PIPER ROBOT TEST SCRIPT")
print("=" * 50)

# Connect to the robot
print("\n1. Connecting to robot...")
try:
    piper = C_PiperInterface_V2()
    piper.ConnectPort()
    print("✓ Connected successfully!")
except Exception as e:
    print(f"✗ Connection failed: {e}")
    print("\nTroubleshooting:")
    print("- Make sure CAN interface is up: sudo ip link set can0 up type can bitrate 1000000")
    print("- Check robot is powered on")
    print("- Check USB cable is connected")
    exit(1)

# Enable the robot
print("\n2. Enabling robot...")
try:
    # Use EnablePiper() for V2 interface
    while not piper.EnablePiper():
        time.sleep(0.01)
    time.sleep(0.5)
    print("✓ Robot enabled!")
except Exception as e:
    print(f"✗ Enable failed: {e}")
    exit(1)

# Read current status
print("\n3. Reading robot status...")
print("Reading joint positions for 5 seconds...")
print("(You should see joint angles updating)\n")

try:
    for i in range(50):  # Read for 5 seconds
        joint_msg = piper.GetArmJointMsgs()
        gripper_msg = piper.GetArmGripperMsgs()
        
        print(f"[{i+1}/50] Joints: {joint_msg}")
        print(f"        Gripper: {gripper_msg}\n")
        
        time.sleep(0.1)
        
except KeyboardInterrupt:
    print("\n\nStopped by user")
except Exception as e:
    print(f"\n✗ Error reading status: {e}")

print("\n" + "=" * 50)
print("TEST COMPLETE!")
print("=" * 50)
print("\nIf you saw joint angles updating above, your robot is working!")
print("\nNext steps:")
print("- Try the demo scripts in: piper_sdk/demo/V2/")
print("- Start with: piper_ctrl_joint.py to move the robot")
print("- Read README: piper_sdk/demo/V2/README.MD")

