"""Environment configuration for Piper robot cube stacking task.

This configuration sets up a cube stacking task where the Piper robot learns to
stack three colored cubes (blue, red, green) on top of each other.

Task: Stack cube_2 on cube_1, then stack cube_3 on cube_2.
"""

import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.markers.config import FRAME_MARKER_CFG

import isaaclab.envs.mdp as mdp
try:
    from . import mdp as stack_mdp
except Exception:
    # Fallback when file is executed outside package context
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(__file__))
    import mdp as stack_mdp  # type: ignore


##
# Scene definition
##


@configclass
class PiperStackSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with Piper robot, table, and three cubes."""

    # Piper robot
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=UsdFileCfg(
            usd_path=os.path.join(
                os.getenv("ISAACLAB_PATH", os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                "piper_isaac_sim/usd/piper_description.usd",
            ),
            activate_contact_sensors=False,
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                "joint1": 0.0,
                "joint2": -0.3,
                "joint3": -0.2,
                "joint4": 0.0,
                "joint5": -0.5,
                "joint6": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "piper": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-6]"],
                velocity_limit=100.0,
                effort_limit=87.0,
                stiffness=800.0,
                damping=40.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["gripper_.*"],
                velocity_limit=100.0,
                effort_limit=50.0,
                stiffness=1000.0,
                damping=50.0,
            ),
        },
    )

    # End-effector frame transformer
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_CFG.replace(prim_path="/Visuals/FrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/link_eef",
                name="end_effector",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.06],
                ),
            ),
        ],
    )

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # Ground plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # Cube properties
    cube_properties = RigidBodyPropertiesCfg(
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=1,
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=5.0,
        disable_gravity=False,
    )

    # Cube 1 (Blue - base cube)
    cube_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube_1",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0.0, 0.0203], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=cube_properties,
            semantic_tags=[("class", "cube_1")],
        ),
    )

    # Cube 2 (Red - middle cube)
    cube_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube_2",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.55, 0.05, 0.0203], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=cube_properties,
            semantic_tags=[("class", "cube_2")],
        ),
    )

    # Cube 3 (Green - top cube)
    cube_3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube_3",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.60, -0.1, 0.0203], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/green_block.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=cube_properties,
            semantic_tags=[("class", "cube_3")],
        ),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint[1-6]"],
        scale=0.5,
        use_default_offset=True,
    )

    gripper_action: mdp.BinaryJointPositionActionCfg = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper_.*"],
        open_command_expr={"gripper_.*": 0.04},
        close_command_expr={"gripper_.*": 0.0},
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Proprioceptive observations
        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        
        # Object observations
        object = ObsTerm(func=stack_mdp.object_obs)
        cube_positions = ObsTerm(func=stack_mdp.cube_positions_in_world_frame)
        cube_orientations = ObsTerm(func=stack_mdp.cube_orientations_in_world_frame)
        
        # End-effector observations
        eef_pos = ObsTerm(func=stack_mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=stack_mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=stack_mdp.gripper_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask tracking."""

        # Track if cube_2 is grasped
        grasp_1 = ObsTerm(
            func=stack_mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_2"),
            },
        )
        
        # Track if cube_2 is stacked on cube_1
        stack_1 = ObsTerm(
            func=stack_mdp.object_stacked,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "upper_object_cfg": SceneEntityCfg("cube_2"),
                "lower_object_cfg": SceneEntityCfg("cube_1"),
            },
        )
        
        # Track if cube_3 is grasped
        grasp_2 = ObsTerm(
            func=stack_mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_3"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # Reset robot to default pose
    init_piper_arm_pose = EventTerm(
        func=stack_mdp.set_default_piper_pose,
        mode="reset",
        params={
            "default_pose": [0.0, -0.3, -0.2, 0.0, -0.5, 0.0, 0.04, 0.04],  # 6 joints + 2 gripper
        },
    )

    # Randomize joint state slightly
    randomize_piper_joint_state = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint[1-6]"]),
        },
    )

    # Randomize cube positions on table
    randomize_cube_positions = EventTerm(
        func=stack_mdp.randomize_cube_poses,
        mode="reset",
        params={
            "pose_range": {"x": (0.35, 0.65), "y": (-0.15, 0.15), "z": (0.0203, 0.0203), "yaw": (-1.0, 1.0)},
            "min_separation": 0.10,
            "asset_cfgs": [SceneEntityCfg("cube_1"), SceneEntityCfg("cube_2"), SceneEntityCfg("cube_3")],
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Terminate if cubes fall off the table
    cube_1_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_1")},
    )

    cube_2_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_2")},
    )

    cube_3_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_3")},
    )

    # Success termination
    success = DoneTerm(func=stack_mdp.cubes_stacked)


##
# Environment configuration
##


@configclass
class PiperCubeStackEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Piper cube stacking environment."""

    # Scene settings
    scene: PiperStackSceneCfg = PiperStackSceneCfg(num_envs=1024, env_spacing=2.5, replicate_physics=False)
    
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()

    # Unused managers for demonstration (no explicit rewards for this task)
    commands = None
    rewards = None
    curriculum = None

    # Gripper utility parameters
    gripper_joint_names = ["gripper_.*"]
    gripper_open_val = 0.04
    gripper_threshold = 0.005

    def __post_init__(self):
        """Post initialization."""
        # General settings
        self.decimation = 5
        self.episode_length_s = 30.0
        
        # Simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = 2
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        
        # Viewer settings
        self.viewer.eye = (2.5, 2.5, 2.0)
        self.viewer.lookat = (0.5, 0.0, 0.0)


@configclass
class PiperCubeStackEnvCfg_PLAY(PiperCubeStackEnvCfg):
    """Configuration for playing/testing the Piper stacking environment."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # Make a smaller scene for play
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.0
        
        # Spawn the robot further away for better visualization
        self.viewer.eye = (3.0, 3.0, 2.5)
        self.viewer.lookat = (0.5, 0.0, 0.2)
        
        # Disable randomization for play
        self.observations.policy.enable_corruption = False
        
        # Longer episodes for demonstration
        self.episode_length_s = 60.0

