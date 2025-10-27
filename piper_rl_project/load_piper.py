"""Load Piper Robotic Arm in Isaac Lab"""

import argparse
import os
from isaaclab.app import AppLauncher

# Create argument parser
parser = argparse.ArgumentParser(description="Load Piper Arm")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Import after app is created
from isaaclab.sim import SimulationContext
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

@configclass
class PiperSceneCfg(InteractiveSceneCfg):
    """Configuration for Piper scene"""
    
    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    
    # Piper robot
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/Piper",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(os.getenv("ISAACLAB_PATH", os.path.dirname(os.path.dirname(__file__))), 
                                  "piper_isaac_sim/usd/piper_description.usd"),
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                ".*": 0.0,  # All joints start at 0
            },
        ),
        actuators={
            "piper_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],  # All joints
                stiffness=800.0,
                damping=40.0,
            ),
        },
    )
    
    # Dome light for better visualization
    dome_light = AssetBaseCfg(
        prim_path="/World/dome_light",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0,
            color=(0.75, 0.75, 0.75),
        ),
    )


def main():
    """Main function"""
    
    # Create simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    sim = SimulationContext(sim_cfg)
    
    # Set camera view
    sim.set_camera_view(eye=[1.5, 1.5, 1.0], target=[0.0, 0.0, 0.3])
    
    # Create scene with Piper
    scene_cfg = PiperSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # Play simulation
    sim.reset()
    print("=" * 50)
    print("✓ Piper arm loaded successfully!")
    print("✓ Robot has", scene["robot"].num_joints, "joints")
    print("=" * 50)
    
    # Simulation loop
    sim_time = 0.0
    count = 0
    
    print("Running simulation... (Press Ctrl+C to exit)")
    
    try:
        while simulation_app.is_running():
            # Step simulation
            scene.update(dt=sim_cfg.dt)
            sim.step()
            
            # Print joint positions every 2 seconds
            if count % 200 == 0:
                joint_pos = scene["robot"].data.joint_pos
                print(f"[{sim_time:.1f}s] Joint positions: {joint_pos[0].cpu().numpy()}")
            
            sim_time += sim_cfg.dt
            count += 1
            
    except KeyboardInterrupt:
        print("\nStopping simulation...")
    
    # Close
    simulation_app.close()
    print("✓ Done!")


if __name__ == "__main__":
    main()