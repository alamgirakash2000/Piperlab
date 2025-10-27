"""PPO agent configuration using RSL-RL for Piper robot reaching task.

This configuration defines the PPO hyperparameters, network architecture,
and training settings for the Piper robot reaching task.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PiperReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Runner configuration for PPO with Piper robot."""

    # Training parameters
    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 100
    experiment_name = "piper_reach"
    run_name = ""
    
    # Logging
    logger = "tensorboard"
    log_interval = 10
    
    # Checkpointing
    resume = False
    load_run = None
    load_checkpoint = None
    
    # Policy network configuration
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    
    # PPO algorithm configuration
    algorithm = RslRlPpoAlgorithmCfg(
        # Loss coefficients
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        
        # Training parameters
        num_learning_epochs=8,
        num_mini_batches=8,
        learning_rate=3.0e-4,
        schedule="adaptive",
        
        # Advantage estimation
        gamma=0.99,
        lam=0.95,
        
        # Policy updates
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

