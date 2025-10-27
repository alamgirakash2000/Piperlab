#!/bin/bash
# Simple script to activate CAN interface - run this ONCE with sudo

echo "Setting up CAN interface..."
ip link set can0 down 2>/dev/null
ip link set can0 type can bitrate 1000000
ip link set can0 up

if ip link show can0 | grep -q "UP"; then
    echo "✓ CAN interface is UP and ready!"
    ip link show can0
else
    echo "✗ Failed to activate CAN"
    exit 1
fi

