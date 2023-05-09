import io
import os
import pathlib
import sys
import time
from typing import Any, Dict, Optional, Type, TypeVar, Union
from typing import Iterable

import numpy as np
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.save_util import recursive_getattr, save_to_zip_file
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, safe_mean
from stable_baselines3.ppo.ppo import PPO
from torch.nn import functional as F

from Rover.utils.lr_schedulers import linear_schedule
from Rover.utils.networks_ranking import RoverRankingSystem

SelfPPO = TypeVar("SelfPPO", bound="PPO")


class PPO_Rover(PPO):
    """
        Proximal Policy Optimization algorithm (PPO) (clip version)

        Paper: https://arxiv.org/abs/1707.06347
        Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
        https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
        Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

        Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

        :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
        :param env: The environment to learn from (if registered in Gym, can be str)
        :param learning_rate: The learning rate, it can be a function
            of the current progress remaining (from 1 to 0)
        :param n_steps: The number of steps to run for each environment per update
            (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
            NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
            See https://github.com/pytorch/pytorch/issues/29372
        :param batch_size: Minibatch size
        :param n_epochs: Number of epoch when optimizing the surrogate loss
        :param gamma: Discount factor
        :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        :param clip_range: Clipping parameter, it can be a function of the current progress
            remaining (from 1 to 0).
        :param clip_range_vf: Clipping parameter for the value function,
            it can be a function of the current progress remaining (from 1 to 0).
            This is a parameter specific to the OpenAI implementation. If None is passed (default),
            no clipping will be done on the value function.
            IMPORTANT: this clipping depends on the reward scaling.
        :param normalize_advantage: Whether to normalize or not the advantage
        :param ent_coef: Entropy coefficient for the loss calculation
        :param vf_coef: Value function coefficient for the loss calculation
        :param max_grad_norm: The maximum value for the gradient clipping
        :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
            instead of action noise exploration (default: False)
        :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
            Default: -1 (only sample at the beginning of the rollout)
        :param target_kl: Limit the KL divergence between updates,
            because the clipping is not enough to prevent large update
            see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
            By default, there is no limit on the kl div.
        :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
            the reported success rate, mean episode length, and mean reward over
        :param tensorboard_log: the log location for tensorboard (if None, no logging)
        :param policy_kwargs: additional arguments to be passed to the policy on creation
        :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
            debug messages
        :param seed: Seed for the pseudo random generators
        :param device: Device (cpu, cuda, ...) on which the code should be run.
            Setting it to auto, the code will be run on the GPU if possible.
        :param _init_setup_model: Whether or not to build the network at the creation of the instance
        :param clear_ep_info_buffer_every_iteration: the ep_info_buffer will be cleared after every update
        """

    def __init__(
            self,
            policy: Union[str, Type[ActorCriticPolicy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = linear_schedule(0.00005),
            n_steps: int = 2048,
            batch_size: int = 64,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: Union[float, Schedule] = 0.2,
            clip_range_vf: Union[None, float, Schedule] = None,
            normalize_advantage: bool = True,
            ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            target_kl: Optional[float] = None,
            stats_window_size: int = 100,
            clear_ep_info_buffer_every_iteration=True,
            tensorboard_log: Optional[str] = None,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )
        self.clear_ep_info_buffer_every_iteration = clear_ep_info_buffer_every_iteration
        self.rover_rankings = None
    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                # if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                # actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
            self: SelfPPO,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            tb_log_name: str = "OnPolicyAlgorithm",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ) -> SelfPPO:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            time_before_stepping = time.time_ns()
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer,
                                                      n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Personalized Rover info processing
            minEpRet, maxEpRet, r, death_rate, avg_per_MEA, success_rate, goal_episodes, goal_reached_episodes = self.eval_rover_performance()

            time_after_stepping = time.time_ns()
            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                if self.rover_rankings is not None:  # default ranking system [non-dying (%), Avg÷MEA (%), success (%)]
                    new_network = self.rover_rankings.Network(iteration,
                                                              [100.0 - death_rate, avg_per_MEA, success_rate])
                    should_save = self.rover_rankings.rank_new_network(new_network)
                    self.rover_rankings.print_rankings()
                    if should_save:
                        self.save(os.path.join(self.rover_rankings.save_path, self.rover_rankings.networks_subfolder,
                                               '%.5i'%iteration), exclude=None)
                        #self.save(os.path.join('saved_networks',  '%.5i'%iteration), exclude=None)
                        self.rover_rankings.write_ranking_files()

                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", r)
                    self.logger.record("rollout/ep_rew_min", minEpRet)
                    self.logger.record("rollout/ep_rew_max", maxEpRet)
                    self.logger.record("rollout/ep_len_mean",
                                       safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/death_rate (%)", death_rate)
                    self.logger.record("rollout/Avg÷MEA (%)", avg_per_MEA)
                    self.logger.record("rollout/success_rate (%)", success_rate)
                    for i in range(3):
                        self.logger.record("rollout/goal" + str(i) + '_episodes', goal_episodes[i])
                        self.logger.record("rollout/goal" + str(i) + '_reached', goal_reached_episodes[i])

            self.train()

            time_elapsed = max((time_after_stepping - self.start_time) / 1e9, sys.float_info.epsilon)
            time_stepping = max((time_after_stepping - time_before_stepping) / 1e9, sys.float_info.epsilon)
            fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
            time_after_training = time.time_ns()
            time_training = max((time_after_training - time_after_stepping) / 1e9, sys.float_info.epsilon)
            if log_interval is not None and iteration % log_interval == 0:
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_stepping", time_stepping)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.record("time/time_training", time_training)
                self.logger.dump(step=self.num_timesteps)

            if self.clear_ep_info_buffer_every_iteration:
                self.ep_info_buffer.clear()
        callback.on_training_end()

        return self

    def eval_rover_performance(self):
        epinfobuf = self.ep_info_buffer
        r = [epinfo['r'] for epinfo in epinfobuf]
        avgEpRet = safe_mean(r)
        minEpRet = np.nan if len(r) == 0 else np.min(r)
        maxEpRet = np.nan if len(r) == 0 else np.max(r)
        deaths = 0
        for epinfo in epinfobuf:
            if epinfo.get("death") is not None:
                deaths += 1
        goal_episodes = [0, 0, 0]
        goal_reached_episodes = [0, 0, 0]
        for epinfo in epinfobuf:
            goal_episode = epinfo.get('current_goal')
            goal_reached = epinfo.get('goal_reached_flag')
            if goal_episode is not None:
                goal_episodes[int(goal_episode)] += 1
            if goal_reached is not None:
                goal_reached_episodes[int(goal_reached)] += 1

        num_episodes = 1.0 * np.sum(goal_episodes)
        # print("g eh:",g)
        # guarantee that we will not have a NaN if the rover does not complete any episodes (being by death or success)
        # BUT may get an (improbable) error if it happens on first episode (will get undefined variables)
        if not num_episodes == 0:
            # MEA = np.dot([100.0, 50.0, 70.0], goal_episodes) / num_episodes  # TODO: use eng goal rew to compensate effects
            MEA = np.dot([110.0, 60.0, 80.0], goal_episodes) / num_episodes # Test with goal_rwd = 10.0
            death_rate = 100.0 * deaths / num_episodes
            avg_per_MEA = 100.0 * avgEpRet / MEA
            success_rate = 100.0 * np.sum(goal_reached_episodes) / num_episodes
        else:
            death_rate = np.nan
            avg_per_MEA = np.nan
            success_rate = np.nan
        return minEpRet, maxEpRet, avgEpRet, death_rate, avg_per_MEA, success_rate, goal_episodes, goal_reached_episodes

    def save(
            self,
            path: Union[str, pathlib.Path, io.BufferedIOBase],
            exclude: Optional[Iterable[str]] = None,
            include: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Save all the attributes of the object and the model parameters in a zip-file.

        :param path: path to the file where the rl agent should be saved
        :param exclude: name of parameters that should be excluded in addition to the default ones
        :param include: name of parameters that might be excluded but should be included anyway
        """
        # Copy parameter list so we don't mutate the original dict
        data = self.__dict__.copy()

        # Exclude is union of specified parameters (if any) and standard exclusions
        if exclude is None:
            exclude = []
        exclude = set(exclude).union(self._excluded_save_params())

        # Do not exclude params if they are specifically included
        if include is not None:
            exclude = exclude.difference(include)

        state_dicts_names, torch_variable_names = self._get_torch_save_params()
        all_pytorch_variables = state_dicts_names + torch_variable_names
        for torch_var in all_pytorch_variables:
            # We need to get only the name of the top most module as we'll remove that
            var_name = torch_var.split(".")[0]
            # Any params that are in the save vars must not be saved by data
            exclude.add(var_name)

        # Remove parameter entries of parameters which are to be excluded
        for param_name in exclude:
            data.pop(param_name, None)

        # Build dict of torch variables
        pytorch_variables = None
        if torch_variable_names is not None:
            pytorch_variables = {}
            for name in torch_variable_names:
                attr = recursive_getattr(self, name)
                pytorch_variables[name] = attr

        # Build dict of state_dicts
        params_to_save = self.get_parameters()

        save_to_zip_file(path, data=data, params=params_to_save, pytorch_variables=pytorch_variables)

    def set_ranking_system(self, rover_rankings: RoverRankingSystem):
        self.rover_rankings = rover_rankings