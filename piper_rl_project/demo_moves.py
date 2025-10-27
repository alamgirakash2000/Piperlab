"""Demo Random Moves for Piper Robot

Generates 8-10 random joint positions and executes them one by one.
Final move returns to home position.

Usage:
    ./isaaclab.sh -p piper_rl_project/demo_moves.py
"""

import argparse
import os
import torch
import random
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


def generate_random_positions(num_joints, num_moves=10):
    """Generate random joint positions within safe limits"""
    positions = []
    
    for i in range(num_moves):
        # Generate random values between -0.8 and 0.8 radians
        pos = [random.uniform(-0.8, 0.8) for _ in range(num_joints)]
        positions.append(pos)
    
    # Final position is home (all zeros)
    positions.append([0.0] * num_joints)
    
    return positions


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
    
    # Generate random moves
    num_moves = random.randint(8, 10)
    positions = generate_random_positions(num_joints, num_moves)
    
    print("\n" + "=" * 70)
    print("PIPER ROBOT - DEMO MOVES")
    print("=" * 70)
    print(f"Joints: {num_joints}")
    print(f"Random moves: {num_moves} + 1 home position")
    print("=" * 70)
    
    print("\nGenerated positions:")
    for i, pos in enumerate(positions[:-1]):
        print(f"  Move {i+1}: [{', '.join([f'{x:6.2f}' for x in pos])}]")
    print(f"  Move {len(positions)} (Final): [Home Position]")
    
    print("\n" + "=" * 70)
    print("Starting demo... Press Ctrl+C to stop")
    print("=" * 70 + "\n")
    
    # Execute moves
    current_move = 0
    hold_time = 3.0  # Hold each position for 3 seconds
    sim_time = 0.0
    move_start_time = 0.0
    
    try:
        while simulation_app.is_running() and current_move < len(positions):
            # Check if it's time to move to next position
            if sim_time - move_start_time >= hold_time:
                current_move += 1
                move_start_time = sim_time
                
                if current_move < len(positions):
                    if current_move == len(positions) - 1:
                        print(f"\n[Move {current_move + 1}] FINAL - Returning to HOME position")
                    else:
                        pos = positions[current_move]
                        print(f"\n[Move {current_move + 1}] Target: [{', '.join([f'{x:6.2f}' for x in pos])}]")
            
            # Set current target
            if current_move < len(positions):
                joint_targets = torch.tensor([positions[current_move]], device=sim_cfg.device)
                robot.set_joint_position_target(joint_targets)
                robot.write_data_to_sim()
            
            # Step simulation
            scene.update(dt=sim_cfg.dt)
            sim.step()
            
            sim_time += sim_cfg.dt
        
        print("\n" + "=" * 70)
        print("Demo completed!")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\nDemo stopped by user")
    
    simulation_app.close()


if __name__ == "__main__":
    main()

