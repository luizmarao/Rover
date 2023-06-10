import os

import numpy as np

from Rover.algos.ppo.ppo import PPO_Rover
from Rover.utils.env_register import register_rover_environments
from Rover.utils.env_util import make_vec_env
from Rover.utils.logger import configure
from Rover.utils.lr_schedulers import linear_schedule
from Rover.utils.networks_ranking import RoverRankingSystem

register_rover_environments()

if __name__ == "__main__":
    os.environ["MUJOCO_GL"] = 'egl'  # Set mujoco rendering to dedicated GPU
    EXPERIMENT_NAME = 'TestRover4Wv1_4env_4096ns_bs8192_60e_newrewsys_seed10'  # ns=num steps, bs=batch size, e=epochs, gr=goal_rwd
    os.environ["SB3_LOGDIR"] = os.path.join(os.path.dirname(__file__), 'Experiments')
    assert isinstance(os.environ["SB3_LOGDIR"], str)
    os.makedirs(os.environ["SB3_LOGDIR"], exist_ok=True)

    # set up logger
    logger = configure(folder=None, format_strings=["stdout", "csv", "tensorboard"], exp_name=EXPERIMENT_NAME)

    ## ENVIRONMENT PARAMETERS ##
    rover_env = "Rover4W-v1"
    num_environments = 4
    env_kwargs = {"render_mode": "rgb_array", 'goal_rwd': 5, 'control_cost': 0.0007, 'time_pnlt': 0.00012,
                  'svv_rew':  0.0}
    start_at_initpos = False
    end_after_current_goal = True
    random_current_goal = True
    monitor_kwargs = dict(info_keywords=['death', 'goal_reached_flag', 'timeout'], reset_keywords=['current_goal'])

    ## NETWORK STRUCTURE PARAMETERS ##
    net_arch = dict(pi=[64, 64], vf=[64, 64])

    ## PPO PARAMETERS ##
    total_learning_timesteps = int(1e7)
    n_steps = 4096  # for each env per update
    seed = 20
    learning_rate = linear_schedule(0.00005)
    gamma = 0.99
    n_epochs = 60  # networks training epochs
    gae_lambda = 0.95
    batch_size = 4*2048  # was 64 in default algo
    clip_range = 0.08  # was 0.3 in default algo
    normalize_advantage = True
    ent_coef = 0.1
    max_grad_norm = 0.5  # was 0.5 in default algo
    use_sde = False
    target_kl = 0.08  # was None in default algo
    stats_window_size = 100
    clear_ep_info_buffer_every_iteration = True

    policy_kwargs = dict(
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
                                       save_path=logger.get_dir(), networks_subfolder='saved_networks', verbose=0)
    model.set_ranking_system(rover_rankings)
    model.set_logger(logger)
    model.learn(total_timesteps=total_learning_timesteps, progress_bar=True)

    model.save(path=os.path.join(logger.get_dir(), "saved_model"), include=None, exclude=None)

    # loaded_model = PPO_Rover.load(path=logger.get_dir() + "saved_model")
