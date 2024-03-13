from typing import Dict, List, Tuple, Union

import gymnasium as gym
import numpy as np
from tqdm import tqdm
from Rover.utils.logger import safe_print

from stable_baselines3.common import type_aliases
from stable_baselines3.common.vec_env import VecEnv


def evaluate_policy(
        model: "type_aliases.PolicyPredictor",
        env: Union[gym.Env, VecEnv],
        net_num: str = None,
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
        goals: List = [-1, 1, 2],
        progress_bar_full = None
) -> Dict:
    """
    Runs Rover's policy for ``n_eval_episodes`` episodes and returns tracked data.

    :param net_num: Agent's network number, to be in the returned dict
    :param progress_bar_full: A progress bar to be updated if there is a loop calling this function
    :param goals: the listo of goals in which the agent will be evaluated. -1 means complete challenge track
    :param model: The Rover agent you want to evaluate. This can be any object
        that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The Rover ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :return: dict with several evaluated information.
    """

    if net_num is None:
        print("This function needs the network name to run properly")
    n_envs = env.num_envs
    eval_results = {}

    pbar_net = tqdm(desc='Current Net Episodes', total=n_eval_episodes * len(goals), position=1, unit='episodes')
    pbar_goal = tqdm(desc='Current Net and Goal', total=n_eval_episodes, position=0, unit='episodes')
    pbars_list = [pbar_net, pbar_goal]
    if progress_bar_full is not None:
        pbars_list.append(progress_bar_full)

    def pbars_update(n):
        [bar.update(n) for bar in pbars_list]

    for goal in goals:
        if goal == -1:
            env.env_method('set_init_configs_full')
            safe_print("Evaluating Network {} in complete mode".format(net_num))
        else:
            safe_print("Evaluating Network {} in goal_{} mode".format(net_num, goal))
        if goal == 0:
            env.env_method('set_init_configs_g0')
        if goal == 1:
            env.env_method('set_init_configs_g1')
        if goal == 2:
            env.env_method('set_init_configs_g2')

        episode_rewards = []
        episode_lengths = []
        current_rewards = np.zeros(n_envs)
        current_lengths = np.zeros(n_envs, dtype="int")
        observations = env.reset()
        episode_counts = np.zeros(n_envs, dtype="int")
        timeouts = 0
        deaths = 0
        goal_reached_episodes = [0, 0, 0]

        # Divides episodes among different sub environments in the vector as evenly as possible
        episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")
        while (episode_counts < episode_count_targets).any():
            actions, _ = model.predict(observation=observations, deterministic=deterministic)
            observations, rewards, dones, infos = env.step(actions)
            current_rewards += rewards
            current_lengths += 1
            if render:
                env.render()
            for i in range(n_envs):
                if episode_counts[i] < episode_count_targets[i]:
                    info = infos[i]
                    if dones[i]:
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                            pbars_update(1)
                        else:
                            episode_rewards.append(current_rewards[i])
                            episode_lengths.append(current_lengths[i])
                            episode_counts[i] += 1
                            pbars_update(1)
                        current_rewards[i] = 0
                        current_lengths[i] = 0

                    if "timeout" in info.keys():
                        timeouts += 1
                    if "goal_reached_flag" in info.keys():
                        goal_reached = info['goal_reached_flag']
                        goal_reached_episodes[int(goal_reached)] += 1
                    if "death" in info.keys():
                        deaths += 1

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        min_reward = np.min(episode_rewards)
        max_reward = np.max(episode_rewards)
        total_eps = len(episode_rewards)
        mean_length = np.mean(episode_lengths)
        std_length = np.std(episode_lengths)
        min_length = np.min(episode_lengths)
        max_length = np.max(episode_lengths)
        eval_results[goal] = {'Network': net_num,
                              'ep_rew_mean': mean_reward,
                              'ep_rew_std': std_reward,
                              'ep_rew_min': min_reward,
                              'ep_rew_max': max_reward,
                              'total_eps': total_eps,
                              'timeout_eps': timeouts,
                              'deaths': deaths,
                              'deaths (%)': 100 * deaths / total_eps,
                              'goal_reached_episodes': goal_reached_episodes,
                              'ep_len_mean': mean_length,
                              'ep_len_std': std_length,
                              'ep_len_min': min_length,
                              'ep_len_max': max_length,
                              }

        pbar_goal.reset()

    return eval_results
