"""Termination functions for Piper cube stacking task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def cubes_stacked(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
) -> torch.Tensor:
    """Check if all three cubes are successfully stacked.
    
    Success criteria:
    - Cube 2 is stacked on Cube 1
    - Cube 3 is stacked on Cube 2
    """
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]
    cube_3: RigidObject = env.scene[cube_3_cfg.name]

    # Get positions
    cube_1_pos = cube_1.data.root_pos_w
    cube_2_pos = cube_2.data.root_pos_w
    cube_3_pos = cube_3.data.root_pos_w

    # Check cube 2 on cube 1
    height_diff_21 = cube_2_pos[:, 2] - cube_1_pos[:, 2]
    xy_distance_21 = torch.norm(cube_2_pos[:, :2] - cube_1_pos[:, :2], dim=1)
    stack_21 = (height_diff_21 > 0.035) & (height_diff_21 < 0.05) & (xy_distance_21 < 0.03)

    # Check cube 3 on cube 2
    height_diff_32 = cube_3_pos[:, 2] - cube_2_pos[:, 2]
    xy_distance_32 = torch.norm(cube_3_pos[:, :2] - cube_2_pos[:, :2], dim=1)
    stack_32 = (height_diff_32 > 0.035) & (height_diff_32 < 0.05) & (xy_distance_32 < 0.03)

    # Success if both are stacked
    success = stack_21 & stack_32

    return success

