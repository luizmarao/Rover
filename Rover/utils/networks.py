from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
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

