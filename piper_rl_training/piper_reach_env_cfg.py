"""Environment configuration for Piper robot reaching task.

This configuration sets up a reaching task where the Piper robot learns to move
its end-effector to random target positions in 3D space.
"""

import math
import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.assets import Articulation
from isaaclab.utils.math import combine_frame_transforms

import isaaclab.envs.mdp as mdp
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.managers.manager_base import ManagerTermBase


##
# Scene definition
##


@configclass
class PiperSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with Piper robot and target marker."""

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # Lights
    dome_light = AssetBaseCfg(
        prim_path="/World/dome_light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    distant_light = AssetBaseCfg(
        prim_path="/World/distant_light",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0, angle=0.5),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.738, 0.477, 0.477, 0.0)),
    )

    # Piper robot
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=UsdFileCfg(
            usd_path=os.path.join(
                os.getenv("ISAACLAB_PATH", os.path.dirname(os.path.dirname(__file__))),
                "piper_isaac_sim/usd/piper_description.usd",
            ),
            activate_contact_sensors=False,
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "piper": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                velocity_limit=100.0,
                effort_limit=87.0,
                stiffness=800.0,
                damping=40.0,
            ),
        },
    )

    # Target marker (visual indicator of target position) - non-physics asset
    target = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.05, 0.05, 0.05),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.3, 0.0, 0.2)),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="gripper_base",  # End-effector link on Piper USD
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.2, 0.45),
            pos_y=(-0.25, 0.25),
            pos_z=(0.1, 0.35),
            roll=(0.0, 0.0),
            pitch=(math.pi, math.pi),  # Point down
            yaw=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        target_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Task rewards: Reaching the target
    reaching_object = RewTerm(
        func=lambda env, std, command_name, asset_cfg: _position_command_error_tanh(
            env, std=std, command_name=command_name, asset_cfg=asset_cfg
        ),
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["gripper_base"]),
            "std": 0.1,
            "command_name": "ee_pose",
        },
    )

    position_tracking = RewTerm(
        func=lambda env, command_name, asset_cfg: _position_command_error(env, command_name=command_name, asset_cfg=asset_cfg),
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["gripper_base"]),
            "command_name": "ee_pose",
        },
    )

    # Regularization rewards: Penalize large actions and velocities
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    joint_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate_l2", "weight": -0.05, "num_steps": 10000},
    )


##
# Environment configuration
##


@configclass
class PiperReachEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Piper reaching environment."""

    # Scene settings
    scene: PiperSceneCfg = PiperSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 12.0
        # simulation settings
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        # viewer settings
        self.viewer.eye = (2.5, 2.5, 2.5)
        self.viewer.lookat = (0.0, 0.0, 0.0)


@configclass
class PiperReachEnvCfg_PLAY(PiperReachEnvCfg):
    """Configuration for playing/testing the Piper reaching environment."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot further away for better visualization
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)
        # disable randomization for play
        self.observations.policy.enable_corruption = False


# ---------------------------------------------------------------------------------
# Local helper reward functions (compatible with current IsaacLab API in this repo)
# ---------------------------------------------------------------------------------

def _position_command_error(env, command_name: str, asset_cfg: SceneEntityCfg):
    """L2 distance between EE and commanded target position (world frame)."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    return (curr_pos_w - des_pos_w).norm(dim=1)


def _position_command_error_tanh(env, std: float, command_name: str, asset_cfg: SceneEntityCfg):
    """Tanh-shaped reward based on EE distance to commanded target position (world frame)."""
    dist = _position_command_error(env, command_name, asset_cfg)
    return 1 - (dist / std).tanh()

