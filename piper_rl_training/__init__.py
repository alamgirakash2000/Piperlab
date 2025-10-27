"""Piper Robot Reinforcement Learning Training Package

This package contains PPO training pipeline for Piper robot reaching task.
"""

import gymnasium as gym

# Import without relative imports for standalone script execution
try:
    from .piper_reach_env_cfg import PiperReachEnvCfg, PiperReachEnvCfg_PLAY
except ImportError:
    from piper_reach_env_cfg import PiperReachEnvCfg, PiperReachEnvCfg_PLAY

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Reach-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": PiperReachEnvCfg,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:rsl_rl_ppo_cfg.PiperReachPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Reach-Piper-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": PiperReachEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:rsl_rl_ppo_cfg.PiperReachPPORunnerCfg",
    },
    disable_env_checker=True,
)

