"""Control Piper Robot Joints - SIMPLE VERSION

Just change the numbers in JOINT_VALUES below and run the script.
The robot will move to those positions.

Usage:
    ./isaaclab.sh -p piper_rl_project/control_joints.py
"""

import argparse
import os
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.sim import SimulationContext
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass


# ==========================================
# CHANGE THESE VALUES TO MOVE THE ROBOT
# ==========================================
# Set the position for each joint (in radians)
# Positive = one direction, Negative = other direction
# Example: 0.5 = about 28 degrees

JOINT_VALUES = [
    0.0,   # Joint 0
    0.0,   # Joint 1
    0.0,   # Joint 2
    0.0,   # Joint 3
    0.0,   # Joint 4
    0.0,   # Joint 5
]

# Try changing one at a time, like:
# JOINT_VALUES = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]  # Move joint 0
# JOINT_VALUES = [0.0, 0.8, 0.0, 0.0, 0.0, 0.0]  # Move joint 1
# JOINT_VALUES = [0.5, 0.3, 0.2, 0.0, 0.0, 0.0]  # Move multiple joints
# ==========================================


@configclass
class PiperSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/Piper",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(os.getenv("ISAACLAB_PATH", os.path.dirname(os.path.dirname(__file__))), 
                                  "piper_isaac_sim/usd/piper_description.usd"),
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
        ),
        actuators={
            "piper_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=800.0,
                damping=40.0,
            ),
        },
    )
    
    dome_light = AssetBaseCfg(
        prim_path="/World/dome_light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )


def main():
    # Setup simulation
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.5, 1.5, 1.0], target=[0.0, 0.0, 0.3])
    
    scene_cfg = PiperSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    
    robot = scene["robot"]
    
    print("=" * 60)
    print("PIPER ROBOT - JOINT CONTROL")
    print("=" * 60)
    print(f"Number of joints: {robot.num_joints}")
    print("\nJoint names:")
    for i, name in enumerate(robot.data.joint_names):
        print(f"  [{i}] {name}")
    
    print("\nTarget joint positions:")
    for i, val in enumerate(JOINT_VALUES[:robot.num_joints]):
        print(f"  Joint {i}: {val:.2f} rad ({val*57.3:.1f}°)")
    
    print("\nMoving to target positions...")
    print("Press Ctrl+C to stop\n")
    
    # Convert to tensor
    joint_targets = torch.tensor([JOINT_VALUES[:robot.num_joints]], device=sim_cfg.device)
    
    count = 0
    try:
        while simulation_app.is_running():
            # Apply the joint values
            robot.set_joint_position_target(joint_targets)
            robot.write_data_to_sim()
            
            # Step simulation
            scene.update(dt=sim_cfg.dt)
            sim.step()
            
            # Print current position every 2 seconds
            if count % 200 == 0:
                current = robot.data.joint_pos[0].cpu().numpy()
                print(f"Current: {current}")
            
            count += 1
            
    except KeyboardInterrupt:
        print("\nStopped")
    
    simulation_app.close()


if __name__ == "__main__":
    main()


