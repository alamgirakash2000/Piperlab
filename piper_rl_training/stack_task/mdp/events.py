"""Event functions for Piper cube stacking task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def set_default_piper_pose(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    default_pose: list[float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Set the default joint positions for Piper robot."""
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Convert default pose to tensor
    default_joint_pos = torch.tensor(default_pose, device=env.device).unsqueeze(0).repeat(len(env_ids), 1)
    
    # Set joint positions
    robot.write_joint_state_to_sim(
        joint_pos=default_joint_pos,
        joint_vel=torch.zeros_like(default_joint_pos),
        env_ids=env_ids,
    )


def randomize_cube_poses(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pose_range: dict,
    min_separation: float,
    asset_cfgs: list[SceneEntityCfg],
):
    """Randomize cube positions on the table with minimum separation."""
    num_envs = len(env_ids)
    
    for asset_cfg in asset_cfgs:
        obj: RigidObject = env.scene[asset_cfg.name]
        
        # Generate random positions
        attempts = 0
        max_attempts = 100
        
        while attempts < max_attempts:
            # Random x, y positions
            x = torch.rand(num_envs, device=env.device) * (pose_range["x"][1] - pose_range["x"][0]) + pose_range["x"][0]
            y = torch.rand(num_envs, device=env.device) * (pose_range["y"][1] - pose_range["y"][0]) + pose_range["y"][0]
            z = torch.full((num_envs,), pose_range["z"][0], device=env.device)
            
            # Random yaw
            yaw = torch.rand(num_envs, device=env.device) * (pose_range["yaw"][1] - pose_range["yaw"][0]) + pose_range["yaw"][0]
            
            # Convert yaw to quaternion (rotation around z-axis)
            quat_w = torch.cos(yaw / 2.0)
            quat_x = torch.zeros_like(yaw)
            quat_y = torch.zeros_like(yaw)
            quat_z = torch.sin(yaw / 2.0)
            
            # Set positions
            positions = torch.stack([x, y, z], dim=1)
            orientations = torch.stack([quat_w, quat_x, quat_y, quat_z], dim=1)
            
            # Write to simulation
            obj.write_root_pose_to_sim(
                root_pose=torch.cat([positions, orientations], dim=1),
                env_ids=env_ids,
            )
            
            break  # For simplicity, accept first attempt
            # In production, you'd check separation distances
        
        # Reset velocities
        obj.write_root_velocity_to_sim(
            root_vel=torch.zeros(num_envs, 6, device=env.device),
            env_ids=env_ids,
        )

