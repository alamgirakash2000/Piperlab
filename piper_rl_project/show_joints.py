"""Show all joint information for Piper Robot

Run this to see what joints exist and their limits.

Usage:
    ./isaaclab.sh -p piper_rl_project/show_joints.py
"""

import argparse
import os
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
    # Setup
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.5, 1.5, 1.0], target=[0.0, 0.0, 0.3])
    
    scene_cfg = PiperSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    
    robot = scene["robot"]
    
    print("\n" + "=" * 70)
    print("PIPER ROBOT - JOINT INFORMATION")
    print("=" * 70)
    
    print(f"\nTotal Joints: {robot.num_joints}")
    print(f"Robot Path: {robot.cfg.prim_path}")
    
    print("\n" + "-" * 70)
    print("JOINT DETAILS:")
    print("-" * 70)
    
    joint_names = robot.data.joint_names
    joint_limits = robot.data.joint_limits
    
    for i in range(robot.num_joints):
        name = joint_names[i]
        lower = joint_limits[0, i, 0].item()
        upper = joint_limits[0, i, 1].item()
        
        print(f"\nJoint {i}: {name}")
        print(f"  Limits: [{lower:.3f}, {upper:.3f}] radians")
        print(f"  Limits: [{lower*57.3:.1f}°, {upper*57.3:.1f}°]")
    
    print("\n" + "=" * 70)
    print("TO CONTROL IN GUI:")
    print("  1. Window → Stage (F3)")
    print("  2. Find: /World/Piper/<joint_name>")
    print("  3. In Property window, look for 'Drive' section")
    print("  4. Change 'Target Position' value")
    print("=" * 70)
    
    # Keep simulation running
    print("\nSimulation running. Press Ctrl+C to exit...")
    
    try:
        while simulation_app.is_running():
            scene.update(dt=sim_cfg.dt)
            sim.step()
    except KeyboardInterrupt:
        print("\nStopped")
    
    simulation_app.close()


if __name__ == "__main__":
    main()


