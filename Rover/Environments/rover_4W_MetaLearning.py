from os import path
from typing import Optional, Union

import numpy as np

import gymnasium as gym
from gymnasium import error, logger, spaces
from gymnasium.spaces import Space

from Rover.Environments.rover_4W import Rover4Wv1Env
import os

import numpy as np

from Rover.algos.ppo.ppo import PPO_Rover
from Rover.utils.env_register import register_rover_environments
from Rover.utils.env_util import make_vec_env
from Rover.utils.logger import configure
from Rover.utils.lr_schedulers import linear_schedule
from Rover.utils.networks_ranking import RoverRankingSystem


class RoverMetaLearningEnv(gym.Env):

    def __init__(self):
        """
        This environment's actions are the parameters sent to create new instances of Rover4W envs and PPO_Rover.
        The action space must be as following:
        [0] PPO_Rover   learning_rate       [0.0 - 1.0]*1.0e-4
        [1] PPO_Rover   batch_size          2**round(10*[0.0 - 1.0])
        [2] PPO_Rover   gamma               [0.0 - 1.0]
        [3] PPO_Rover   gae_lambda          [0.0 - 1.0]
        [4] PPO_Rover   clip_range          [0.0 - 1.0]
        [5] PPO_Rover   ent_coef            [0.0 - 1.0]
        [6] PPO_Rover   target_kl           [0.0 - 1.0]
        [7] Rover4W     death_circle_dist   [0.0 - 1.0]
        [8] Rover4W     death_circle_time   [0.0 - 1.0]*20
        [9] Rover4W     fwd_rew            [0.0 - 1.0]*1.0e-1
        [10] Rover4W     control_cost       [0.0 - 1.0]*1.0e-2
        [11] Rover4W     time_pnlt          [0.0 - 1.0]*1.0e-2

        The environment runs a few Rover environments, with parameters given by 'action', for a number of steps,
        and then evaluate the learning process and calculates a reward. Thus, every episode ends with a single step, and
        begins exactly the same (except for networks weigths), with no reason for distinct observations.
        """

        register_rover_environments()
        os.environ["MUJOCO_GL"] = 'egl'  # Set mujoco rendering to dedicated GPU
        EXPERIMENT_NAME = 'TestRover4Wv1MetaLearning'
        # os.environ["SB3_LOGDIR"] = None  # os.path.join(os.path.dirname(__file__), 'Experiments')

        self.action_space = spaces.Box(low=0, high=1024, shape=(12,), dtype=np.float64)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)

        # set up logger
        self.logger = configure(folder=None, format_strings=["stdout"], exp_name=None)

        ## ROVER ENVIRONMENTS PARAMETERS ##
        self.rover_env = "Rover4W-v1"
        self.num_environments = 4
        self.env_kwargs = {"render_mode": "rgb_array"}
        self.start_at_initpos = False
        self.end_after_current_goal = True
        self.random_current_goal = True
        self.monitor_kwargs = dict(info_keywords=['death', 'goal_reached_flag', 'timeout'],
                                   reset_keywords=['current_goal'])

        ## NETWORK STRUCTURE PARAMETERS ##
        self.net_arch = dict(pi=[64, 64], vf=[64, 64])  # mlp extractor after conv layers (feature extractor)

        ## ROVER ENVIRONMENTS PPO PARAMETERS ##
        self.total_learning_timesteps = int(5e6)
        self.n_steps = 2048  # for each env per update
        self.seed = None
        self.policy_kwargs = dict(
            net_arch=self.net_arch,
        )

    def step(self, action: np.ndarray):
        learning_rate = linear_schedule(action[0])
        gamma = action[2]
        n_epochs = 15  # networks training epochs
        gae_lambda = action[3]
        batch_size = action[1] if action[1] >= 32 else 32  # was 64 in default algo
        clip_range = action[4]  # was 0.3 in default algo
        normalize_advantage = True
        ent_coef = action[5]
        max_grad_norm = 0.5  # was 0.5 in default algo
        use_sde = False
        target_kl = action[6]  # was None in default algo
        stats_window_size = 100
        clear_ep_info_buffer_every_iteration = True

        env_kwargs = self.env_kwargs.copy()
        env_kwargs['death_circ_dist'] = action[7]
        env_kwargs['death_circ_time'] = action[8]
        env_kwargs['fwd_rew'] = action[9]
        env_kwargs['control_cost'] = action[10]
        env_kwargs['time_pnlt'] = action[11]


        envs = make_vec_env(self.rover_env, env_kwargs=env_kwargs, n_envs=self.num_environments,
                            monitor_kwargs=self.monitor_kwargs)
        model = PPO_Rover("MlpPolicy", envs, policy_kwargs=self.policy_kwargs, verbose=0,
                          n_steps=self.n_steps,
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
                          seed=self.seed,
                          # stats_window_size=stats_window_size
                          clear_ep_info_buffer_every_iteration=clear_ep_info_buffer_every_iteration
                          )
        rover_rankings = RoverRankingSystem(networks_limit_per_ranking=1000, num_rankings=3, verbose=1)
        model.set_ranking_system(rover_rankings)
        model.set_logger(logger)
        model.learn(total_timesteps=self.total_learning_timesteps, progress_bar=False)

        success_rates = [rank for rank in rover_rankings[2]]
        global_success_mean = np.mean(success_rates)

        reward = global_success_mean
        terminated = True
        truncated = False
        obs = (1.0,)

        return obs, reward, terminated, truncated, {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options=None,
    ):
        return (1.0,), {}
