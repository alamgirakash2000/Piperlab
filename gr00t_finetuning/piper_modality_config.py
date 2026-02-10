# AgileX Piper Robot Modality Configuration for GR00T N1.6
# This file defines the state and action spaces for the Piper 6-DOF arm + gripper

from dataclasses import dataclass
from gr00t.data.modality import ModalityConfig


def get_piper_modality_config():
    """
    Create modality config for AgileX Piper robot.
    
    State space: 7D (6 joint angles + 1 gripper position)
    Action space: 7D (6 joint deltas + 1 gripper action)
    Images: global (overhead) + wrist cameras
    """
    
    config = ModalityConfig(
        # State observation configuration
        state_config={
            "state_keys": ["observation.state"],
            "state_dims": {
                "joint_positions": 6,  # 6 DOF arm
                "gripper_position": 1,  # Gripper position (0-100%)
            },
            "state_normalization": {
                "joint_positions": {"min": -180.0, "max": 180.0},  # degrees
                "gripper_position": {"min": 0.0, "max": 100.0},   # percentage
            }
        },
        
        # Action configuration
        action_config={
            "action_keys": ["action"],
            "action_dims": {
                "joint_position_delta": 6,  # Delta joint positions
                "gripper_action": 1,         # Gripper action
            },
            "action_normalization": {
                "joint_position_delta": {"min": -10.0, "max": 10.0},  # degrees
                "gripper_action": {"min": -50.0, "max": 50.0},       # percentage
            }
        },
        
        # Image observation configuration  
        image_config={
            "image_keys": [
                "observation.images.global",   # Overhead camera view
                "observation.images.wrist",    # Wrist-mounted camera
            ],
            "image_height": 360,
            "image_width": 640,
        },
        
        # Total dimensions
        state_dim=7,   # 6 joints + 1 gripper
        action_dim=7,  # 6 joint deltas + 1 gripper action
    )
    
    return config


# For direct import
PIPER_MODALITY_CONFIG = get_piper_modality_config()
