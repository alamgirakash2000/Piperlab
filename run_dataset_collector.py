#!/usr/bin/env python3
"""
VLA Dataset Collector Entry Point
==================================
A colorful, animated application for collecting robot manipulation
demonstrations to fine-tune VLA models (gr00t N1.6, OpenVLA, InternVLA).

Usage:
    conda activate aeropiper
    python run_dataset_collector.py

Requirements:
    - PyQt6
    - OpenCV (cv2)
    - pandas
    - pyarrow
    - numpy
    - piper_sdk

Author: Generated for AgileX Piper Robot
"""

import sys
import os

# Suppress OpenCV JPEG warnings (must be set before cv2 is imported anywhere)
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

# Ensure we can find our package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_dependencies():
    """Check that all required dependencies are installed."""
    missing = []
    
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        missing.append("PyQt6 (pip install PyQt6)")
    
    try:
        import cv2
    except ImportError:
        missing.append("opencv-python (pip install opencv-python)")
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy (pip install numpy)")
    
    try:
        import pandas
    except ImportError:
        missing.append("pandas (pip install pandas)")
    
    try:
        import pyarrow
    except ImportError:
        missing.append("pyarrow (pip install pyarrow)")
    
    if missing:
        print("❌ Missing required dependencies:")
        for dep in missing:
            print(f"   • {dep}")
        print("\nInstall with: pip install PyQt6 opencv-python numpy pandas pyarrow")
        return False
    
    return True


def print_banner():
    """Print a colorful ASCII banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║   ⚡ VLA DEMO COLLECTOR ⚡                                       ║
    ║                                                                  ║
    ║   🤖 Robot:   AgileX Piper 6-DOF                                 ║
    ║   📷 Cameras: Global + Wrist (Logitech C920)                     ║
    ║   📊 Output:  LeRobot V2.1 format                                ║
    ║                                                                  ║
    ║   Collect demonstrations for:                                    ║
    ║   • NVIDIA gr00t N1.6                                            ║
    ║   • OpenVLA                                                      ║
    ║   • InternVLA                                                    ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """Main entry point."""
    print_banner()
    
    print("Checking dependencies...")
    if not check_dependencies():
        return 1
    
    print("✓ All dependencies found\n")
    print("Starting VLA Demo Collector...")
    print("─" * 50)
    
    # Suppress OpenCV JPEG warnings (common with some USB cameras)
    import os
    os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
    
    try:
        from dataset_collector.ui.main_window import run_app
        run_app()
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
