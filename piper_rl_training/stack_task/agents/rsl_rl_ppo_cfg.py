"""PPO agent configuration using RSL-RL for Piper cube stacking task.

This configuration defines the PPO hyperparameters for learning to stack cubes.
The task is more complex than reaching, so we use a larger network and more training.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PiperStackPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Runner configuration for PPO with Piper cube stacking."""

    # Training parameters
    num_steps_per_env = 32
    max_iterations = 5000  # Longer training for complex task
    save_interval = 200
    experiment_name = "piper_stack"
    run_name = ""
    
    # Logging
    logger = "tensorboard"
    log_interval = 10
    
    # Checkpointing
    resume = False
    load_run = None
    load_checkpoint = None
    
    # Policy network configuration (larger for complex task)
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],  # Larger network
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    
    # PPO algorithm configuration
    algorithm = RslRlPpoAlgorithmCfg(
        # Loss coefficients
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,  # Higher entropy for exploration
        
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

