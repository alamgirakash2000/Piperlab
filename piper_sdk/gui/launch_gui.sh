#!/bin/bash
# Launcher script for Piper Robot Control GUI

echo "======================================"
echo "Piper Robot Control GUI"
echo "======================================"
echo ""

# Check if CAN is up
if ip link show can0 2>/dev/null | grep -q "UP"; then
    echo "✓ CAN interface is UP"
else
    echo "⚠ CAN interface is not UP!"
    echo ""
    echo "Please run first:"
    echo "  sudo bash scripts/1_setup_can.sh"
    echo ""
    read -p "Continue anyway? (y/n): " continue
    if [[ ! $continue =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "Launching GUI..."
echo ""

cd "$(dirname "$0")"
python3 robot_control_gui.py

