"""Piper Robot - Interactive Viewing

This script just loads the robot so you can freely navigate with mouse.
No automatic movements - pure manual camera control.

Mouse Controls in Isaac Sim:
  - Left Click + Drag     = Rotate camera
  - Middle Click + Drag   = Pan camera  
  - Right Click + Drag    = Zoom
  - Scroll Wheel          = Zoom
  - Double-click object   = Focus on it
  - F key                 = Frame selected object

Usage:
    ./isaaclab.sh -p piper_rl_project/interactive_view.py
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
            pos=(0.0, 0.0, 0.5),  # Raise robot up a bit
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
    
    # Set initial camera view
    sim.set_camera_view(eye=[1.5, 1.5, 1.0], target=[0.0, 0.0, 0.5])
    
    scene_cfg = PiperSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    
    robot = scene["robot"]
    
    print("\n" + "=" * 70)
    print("PIPER ROBOT - INTERACTIVE VIEW MODE")
    print("=" * 70)
    print("\nMOUSE CONTROLS (in viewport window):")
    print("  Left Click + Drag       → Rotate camera around robot")
    print("  Middle Click + Drag     → Pan camera left/right/up/down")
    print("  Right Click + Drag      → Zoom in/out")
    print("  Scroll Wheel            → Zoom in/out")
    print("  Double-click robot      → Focus camera on robot")
    print("  F key                   → Frame selected object")
    print("\nALTERNATIVE (if no middle mouse):")
    print("  Alt + Left Click + Drag → Pan camera")
    print("  Alt + Right Click       → Zoom")
    print("=" * 70)
    
    print("\nTIP: Double-click on the robot to focus the camera!")
    print("Robot is at position: /World/Piper")
    print("\nSimulation running... Press Ctrl+C to exit\n")
    
    # Hold robot in a visible pose
    joint_targets = torch.tensor([[0.3, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0]], device=sim_cfg.device)
    
    try:
        while simulation_app.is_running():
            # Keep robot in pose
            robot.set_joint_position_target(joint_targets)
            robot.write_data_to_sim()
            
            # Step simulation
            scene.update(dt=sim_cfg.dt)
            sim.step()
            
    except KeyboardInterrupt:
        print("\nStopped")
    
    simulation_app.close()


if __name__ == "__main__":
    main()

