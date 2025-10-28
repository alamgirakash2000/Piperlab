#!/bin/bash
# Run this WITHOUT sudo - it will use your conda environment

cd /home/alamgir/Downloads/piper_sdk
echo "Starting robot test..."
echo "Make sure you ran: sudo bash 1_setup_can.sh first!"
echo ""
python3 test_robot.py

