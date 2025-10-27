"""Piper Robot Cube Stacking Task

This package contains the cube stacking task for Piper robot where the robot
learns to stack three colored cubes on top of each other.
"""

import gymnasium as gym

# Import without relative imports for standalone script execution
try:
    from .piper_stack_env_cfg import PiperCubeStackEnvCfg, PiperCubeStackEnvCfg_PLAY
except ImportError:
    from piper_stack_env_cfg import PiperCubeStackEnvCfg, PiperCubeStackEnvCfg_PLAY

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Stack-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": PiperCubeStackEnvCfg,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:rsl_rl_ppo_cfg.PiperStackPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Piper-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": PiperCubeStackEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:rsl_rl_ppo_cfg.PiperStackPPORunnerCfg",
    },
    disable_env_checker=True,
)

