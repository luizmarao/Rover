import torch as th
import torch.nn as nn
from gymnasium import spaces
import numpy as np
import os

from Rover.utils.env_util import make_vec_env
from Rover.utils.networks import RovernetClassic
from Rover.utils.logger import configure
from Rover.utils.env_register import register_rover_environments
from Rover.algos.ppo.ppo import PPO_Rover
from Rover.utils.networks_ranking import RoverRankingSystem

register_rover_environments()

if __name__ == "__main__":
    os.environ["MUJOCO_GL"] = 'egl'  # Set mujoco rendering to dedicated GPU
    EXPERIMENT_NAME = 'TestRoverEncodedSB3'
    os.environ["SB3_LOGDIR"] = os.path.join(os.path.dirname(__file__), 'Experiments')
    assert isinstance(os.environ["SB3_LOGDIR"], str)
    os.makedirs(os.environ["SB3_LOGDIR"], exist_ok=True)

    # set up logger
    logger = configure(folder=None, format_strings=["stdout", "csv", "tensorboard"], exp_name=EXPERIMENT_NAME)

    ## ENVIRONMENT PARAMETERS ##
    rover_env = "Rover4WeEncodedVision-v0"
    img_red_size = (32, 32)
    rover_2cam_and_packed_images = False
    dynamic_obs_size = 14
    num_environments = 4
    env_kwargs = {"render_mode": "rgb_array", "width": 440, "height": 270, "encoder_name": 'EC g32_1c_4b_128f_128a'}
    start_at_initpos = False
    end_after_current_goal = True
    random_current_goal = True
    monitor_kwargs = dict(info_keywords=['death', 'goal_reached_flag', 'timeout'], reset_keywords=['current_goal'])

    ## NETWORK STRUCTURE PARAMETERS ##
    net_arch = dict(pi=[64, 64], vf=[64, 64])  # mlp extractor


    ## PPO PARAMETERS ##
    total_learning_timesteps = int(5e5)
    n_steps = 2048  # for each env per update
    seed = None
    learning_rate = 0.0001
    gamma = 0.99
    n_epochs = 10  # networks training epochs
    gae_lambda = 0.95
    batch_size = 256  # was 64in default algo
    clip_range = 0.1  # was 0.3 in default algo
    normalize_advantage = True
    ent_coef = 0.0
    max_grad_norm = 1.0  # was 0.5 in default algo
    use_sde = False
    target_kl = 0.1  # was None in default algo
    stats_window_size = 100
    clear_ep_info_buffer_every_iteration = True

    policy_kwargs = dict(
        # use_sde=use_sde,
        net_arch=net_arch,
    )

    envs = make_vec_env(rover_env, env_kwargs=env_kwargs, n_envs=num_environments, monitor_kwargs=monitor_kwargs)
    model = PPO_Rover("MlpPolicy", envs, policy_kwargs=policy_kwargs, verbose=1,
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
                      clear_ep_info_buffer_every_iteration=clear_ep_info_buffer_every_iteration
                      )
    rover_rankings = RoverRankingSystem(networks_limit_per_ranking=5, num_rankings=3,
                                       save_path=logger.get_dir(), networks_subfolder='saved_networks', verbose=1)
    model.set_ranking_system(rover_rankings)
    model.set_logger(logger)
    model.learn(total_timesteps=total_learning_timesteps, progress_bar=True)

    model.save(path=os.path.join(logger.get_dir(), "saved_model"), include=None, exclude=None)

    # loaded_model = PPO_Rover.load(path=logger.get_dir() + "saved_model")
