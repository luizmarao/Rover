import os

import numpy as np

from stable_baselines3.ppo import PPO
from Rover.utils.env_register import register_rover_environments
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from Rover.utils.networks import RoverMetaLearningNet, RoverMetaLearningActorCriticNet

register_rover_environments()

if __name__ == "__main__":
    os.environ["MUJOCO_GL"] = 'egl'  # Set mujoco rendering to dedicated GPU
    EXPERIMENT_NAME = 'TestMetaLearning'
    os.environ["SB3_LOGDIR"] = os.path.join(os.path.dirname(__file__), 'Experiments')
    assert isinstance(os.environ["SB3_LOGDIR"], str)
    os.makedirs(os.environ["SB3_LOGDIR"], exist_ok=True)

    # set up logger
    logger = configure(folder=None, format_strings=["stdout", "csv", "tensorboard"])

    ## ENVIRONMENT PARAMETERS ##
    rover_env = "Rover4WMetaLearning-v0"
    num_environments = 32

    ## NETWORK STRUCTURE PARAMETERS ##
    Networks_Architecture = RoverMetaLearningNet
    net_arch = dict(pi=[64, 64], vf=[64, 64])  # mlp extractor after conv layers (feature extractor)

    ## PPO PARAMETERS ##
    total_learning_timesteps = int(5e6)
    n_steps = 2048  # for each env per update
    seed = None
    learning_rate = 0.001
    gamma = 0.99
    n_epochs = 10  # networks training epochs
    gae_lambda = 0.95
    batch_size = 32  # was 64 in default algo
    clip_range = 0.3  # was 0.3 in default algo
    normalize_advantage = True
    ent_coef = 0.0
    max_grad_norm = 0.5  # was 0.5 in default algo
    use_sde = False
    target_kl = None  # was None in default algo
    stats_window_size = 100
    clear_ep_info_buffer_every_iteration = True

    policy_kwargs = dict(
        net_arch=net_arch,
    )

    envs = make_vec_env(rover_env, n_envs=num_environments)
    model = PPO(RoverMetaLearningActorCriticNet, envs, policy_kwargs=policy_kwargs, verbose=1,
                      n_steps=n_steps,
                      learning_rate=learning_rate,
                      gamma=gamma,
                      n_epochs=n_epochs,
                      gae_lambda=gae_lambda,
                      batch_size=batch_size,
                      clip_range=clip_range,
                      normalize_advantage=normalize_advantage,
                      ent_coef=ent_coef,
                      max_grad_norm=max_grad_norm,
                      # use_sde=use_sde,
                      target_kl=target_kl,
                      seed=seed,
                      # stats_window_size=stats_window_size
                      )
    model.set_logger(logger)
    model.learn(total_timesteps=total_learning_timesteps, progress_bar=True)

    model.save(path=os.path.join(logger.get_dir(), "saved_model"), include=None, exclude=None)