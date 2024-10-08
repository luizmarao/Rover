import os

import numpy as np

from Rover.algos.ppo.ppo import PPO_Rover
from Rover.utils.env_register import register_rover_environments
from Rover.utils.env_util import make_vec_env
from Rover.utils.logger import configure
from Rover.utils.lr_schedulers import linear_schedule
from Rover.utils.networks import RovernetClassic
from Rover.utils.networks_ranking import RoverRankingSystem

register_rover_environments()

if __name__ == "__main__":
    os.environ["MUJOCO_GL"] = 'egl'  # Set mujoco rendering to dedicated GPU
    EXPERIMENT_NAME = 'TestRoverPackedSB3'
    os.environ["SB3_LOGDIR"] = os.path.join(os.path.dirname(__file__), 'Experiments')
    assert isinstance(os.environ["SB3_LOGDIR"], str)
    os.makedirs(os.environ["SB3_LOGDIR"], exist_ok=True)

    # set up logger
    logger = configure(folder=None, format_strings=["stdout", "csv", "tensorboard"], exp_name=EXPERIMENT_NAME)

    ## ENVIRONMENT PARAMETERS ##
    rover_env = "Rover4We-v2"
    img_red_size = (64, 64)
    rover_2cam_and_packed_images = False
    dynamic_obs_size = 14
    num_environments = 8
    save_images = False
    start_at_initpos = False
    end_after_current_goal = True
    random_current_goal = True
    monitor_kwargs = dict(info_keywords=['death', 'goal_reached_flag', 'timeout'], reset_keywords=['current_goal'])
    env_verbose = 0

    ## NETWORK STRUCTURE PARAMETERS ##
    Networks_Architecture = RovernetClassic
    conv_layers = [  # (filters, size, stride)
        (16, 8, 2),
        (32, 4, 2),
        (64, 2, 1)
    ]
    features_extractor_lin_layers = None  # linear layers after conv layers and before mlp layers [size] ex:[64, 64]
    share_features_extractor = False
    net_arch = dict(pi=[64, 64], vf=[64, 64])  # mlp extractor after conv layers (feature extractor)
    normalize_images = False  # Already done in env

    ## PPO PARAMETERS ##
    total_learning_timesteps = int(5e6)
    n_steps = 2048  # for each env per update
    seed = None
    learning_rate = linear_schedule(0.00005)
    gamma = 0.99
    n_epochs = 20  # networks training epochs
    gae_lambda = 0.95
    batch_size = 256  # was 64 in default algo
    clip_range = 0.1  # was 0.3 in default algo
    normalize_advantage = True
    ent_coef = 0.0
    max_grad_norm = 0.5  # was 0.5 in default algo
    use_sde = False
    target_kl = 0.1  # was None in default algo
    stats_window_size = 100
    clear_ep_info_buffer_every_iteration = True

    policy_kwargs = dict(
        features_extractor_class=Networks_Architecture,
        # use_sde=use_sde,
        share_features_extractor=share_features_extractor,
        normalize_images=normalize_images,
        net_arch=net_arch,
        features_extractor_kwargs=dict(
            img_red_size=img_red_size,
            rover_2cam_and_packed_images=rover_2cam_and_packed_images,
            dynamic_obs_size=dynamic_obs_size,
            conv_layers=conv_layers,
            lin_layers=features_extractor_lin_layers,
        ),
    )

    env_kwargs = {"render_mode": "rgb_array", "width": 440, "height": 270, "save_images": save_images, "verbose": env_verbose,
                  "img_reduced_size": img_red_size, "gamma": gamma}
    envs = make_vec_env(rover_env, env_kwargs=env_kwargs, n_envs=num_environments, monitor_kwargs=monitor_kwargs)
    model = PPO_Rover("CnnPolicy", envs, policy_kwargs=policy_kwargs, verbose=1,
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
                                       save_path=logger.get_dir(), networks_subfolder='saved_networks', verbose=0)
    model.set_ranking_system(rover_rankings)
    model.set_logger(logger)
    model.learn(total_timesteps=total_learning_timesteps, progress_bar=True)

    model.save(path=os.path.join(logger.get_dir(), "saved_model"), include=None, exclude=None)

    # loaded_model = PPO_Rover.load(path=logger.get_dir() + "saved_model")
