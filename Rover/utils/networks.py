from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, MlpExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import torch as th
import torch.nn as nn
from gymnasium import spaces
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union


class RovernetClassic(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256, img_red_size=(32,32),
                 rover_2cam_and_packed_images=False, dynamic_obs_size=14, conv_layers=[(16, 8, 2), (32, 4, 2)],
                 lin_layers=None):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        self.dynamic_obs_size = dynamic_obs_size
        self.img_red_size = img_red_size
        self.n_input_channels = 2 if rover_2cam_and_packed_images else 1
        self.cnn = nn.Sequential()
        channels = self.n_input_channels
        for (filters, size, stride) in conv_layers:
            self.cnn.append(nn.Conv2d(channels, filters, kernel_size=size, stride=stride, padding='valid'))
            self.cnn.append(nn.ReLU())
            channels = filters

        self.cnn.append(nn.Flatten())

        # Compute shape by doing one forward pass
        with th.no_grad():
            img = observation_space.sample()[dynamic_obs_size:].reshape(
                (self.n_input_channels, img_red_size[0], img_red_size[1]))[None]
            n_flatten = self.cnn(
                th.as_tensor(img).float()
            ).shape[1]

        self.linear = nn.Sequential()
        if lin_layers is not None:
            input_ = n_flatten + dynamic_obs_size
            for size in lin_layers:
                self.linear.append(nn.Linear(input_, size))
                self.linear.append(nn.Tanh())
                input_ = size
            self._features_dim = input_
        else:
            self._features_dim = n_flatten + dynamic_obs_size

    def forward(self, observations: th.Tensor) -> th.Tensor:
        images = observations[:, self.dynamic_obs_size:].reshape(
            (-1, self.n_input_channels, self.img_red_size[0], self.img_red_size[1]))
        dynamics = observations[:, :self.dynamic_obs_size]
        return self.linear(th.concat((self.cnn(images), dynamics), axis=1))


class RoverMetaLearningNet(MlpExtractor):
    '''
    This network outputs parameters for Rover4W env and PPO_Rover in the Rover Meta Learning env.
    Due to the need of squeezing some parameters, some different activations might be used for the neurons in the output
    layer.
    The sequence of the output parameters is as follows:
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
    '''

    def __init__(
            self,
            feature_dim: int,
            net_arch: Union[List[int], Dict[str, List[int]]],
            activation_fn: nn.Sigmoid,
            device: Union[th.device, str] = "auto",
    ):
        super().__init__(
            feature_dim=feature_dim,
            net_arch=net_arch,
            activation_fn=activation_fn,
            device=device
    )

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        latent_policy = super().forward_actor(features)

          # concatenate over feature dimension
        return out

class RoverMetaLearningActorCriticNet(ActorCriticPolicy):
    from stable_baselines3.common.distributions import DiagGaussianDistribution
    class DiagGaussianDistribution(DiagGaussianDistribution):
        def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0) -> Tuple[nn.Module, nn.Parameter]:
            """
            Create the layers and parameter that represent the distribution:
            one output will be the mean of the Gaussian, the other parameter will be the
            standard deviation (log std in fact to allow negative values)

            :param latent_dim: Dimension of the last layer of the policy (before the action layer)
            :param log_std_init: Initial value for the log standard deviation
            :return:
            """
            first_slice = latent_policy[:, 0]
            second_slice = latent_policy[:, 1]
            third_slice = latent_policy[:, 2:8]
            fourth_slice = latent_policy[:, 8]
            fifth_slice = latent_policy[:, 9]
            sixty_slice = latent_policy[:, 10:]
            tuple_of_adjusted_parts = (
                1e-4 * (first_slice),
                th.pow(2, th.round(second_slice)),
                (third_slice),
                20 * (fourth_slice),
                1e-1 * (fifth_slice),
                1e-2 * (sixty_slice)
            )
            out = th.cat(tuple_of_adjusted_parts, dim=1)

            mean_actions = nn.Linear(latent_dim, self.action_dim)
            log_std = nn.Parameter(th.ones(self.action_dim) * log_std_init, requires_grad=True)
            return mean_actions, log_std
    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = RoverMetaLearningNet(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=th.nn.Sigmoid,
            device=self.device,
        )