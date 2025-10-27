"""Real-time Control for Piper Robot

Type joint values in the terminal while simulation is running.

Usage:
    ./isaaclab.sh -p piper_rl_project/realtime_control.py --headless

Commands while running:
    - Enter 6 numbers: e.g., "0.5 0.3 0.2 0.0 0.0 0.0"
    - 'h' or 'home': Go to home position (all zeros)
    - 'q' or 'quit': Exit
    - Just press Enter: Keep current position
"""

import argparse
import os
import torch
import sys
import select
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


def get_input_non_blocking():
    """Check if there's input available without blocking"""
    if select.select([sys.stdin], [], [], 0.0)[0]:
        return sys.stdin.readline().strip()
    return None


def main():
    # Setup
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.5, 1.5, 1.0], target=[0.0, 0.0, 0.3])
    
    scene_cfg = PiperSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    
    robot = scene["robot"]
    num_joints = robot.num_joints
    
    print("\n" + "=" * 70)
    print("PIPER ROBOT - REAL-TIME CONTROL")
    print("=" * 70)
    print(f"Number of joints: {num_joints}")
    print("\nJoint names:")
    for i, name in enumerate(robot.data.joint_names):
        print(f"  [{i}] {name}")
    
    print("\n" + "=" * 70)
    print("HOW TO CONTROL:")
    print("  - Type joint values: 0.5 0.3 0.2 0.0 0.0 0.0 0.0 0.0")
    print("  - Or use test shortcuts:")
    print("      't1' = Move first joint")
    print("      't2' = Move second joint")
    print("      't3' = Move multiple joints")
    print("      't4' = Mixed positive/negative")
    print("      't5' = Move end joints")
    print("      'h'  = Home position")
    print("      'q'  = Quit")
    print("=" * 70)
    print("\nSimulation running. Type command and press Enter:\n")
    
    # Define test positions
    test_positions = {
        't1': [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # First joint
        't2': [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Second joint
        't3': [0.3, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0],  # Multiple joints
        't4': [0.5, -0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0], # Mixed directions
        't5': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0],  # End joints
    }
    
    # Start at home
    joint_targets = torch.zeros((1, num_joints), device=sim_cfg.device)
    
    count = 0
    prompt_shown = False
    
    try:
        while simulation_app.is_running():
            # Show prompt every 100 steps (~1 second)
            if count % 100 == 0 and not prompt_shown:
                current = robot.data.joint_pos[0].cpu().numpy()
                print(f"\nCurrent: {' '.join([f'{x:.2f}' for x in current])}")
                print(">>> ", end='', flush=True)
                prompt_shown = True
            
            # Check for user input
            user_input = get_input_non_blocking()
            if user_input is not None:
                prompt_shown = False
                
                if user_input.lower() in ['q', 'quit', 'exit']:
                    print("Exiting...")
                    break
                
                elif user_input.lower() in ['h', 'home']:
                    joint_targets = torch.zeros((1, num_joints), device=sim_cfg.device)
                    print("→ Moving to HOME")
                
                elif user_input.lower() in test_positions:
                    values = test_positions[user_input.lower()][:num_joints]
                    joint_targets = torch.tensor([values], device=sim_cfg.device)
                    print(f"→ Test position: {' '.join([f'{x:.2f}' for x in values])}")
                
                elif user_input.strip():  # Any other non-empty input
                    try:
                        # Remove brackets and commas to handle both formats
                        cleaned = user_input.replace('[', '').replace(']', '').replace(',', ' ')
                        values = [float(x) for x in cleaned.split()]
                        
                        if len(values) == num_joints:
                            joint_targets = torch.tensor([values], device=sim_cfg.device)
                            print(f"→ Moving to: {' '.join([f'{x:.2f}' for x in values])}")
                        else:
                            print(f"ERROR: Need {num_joints} values, got {len(values)}")
                    except ValueError:
                        print("ERROR: Invalid input. Examples:")
                        print("  - Space separated: 0.5 0.3 0.2 0.0 0.0 0.0 0.0 0.0")
                        print("  - Or try: t1, t2, t3, t4, t5")
            
            # Apply targets
            robot.set_joint_position_target(joint_targets)
            robot.write_data_to_sim()
            
            # Step simulation
            scene.update(dt=sim_cfg.dt)
            sim.step()
            
            count += 1
            
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    
    simulation_app.close()


if __name__ == "__main__":
    main()


