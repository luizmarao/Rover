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
                 rover_2cam_and_combined_image=False, dynamic_obs_size=14, conv_layers=[(16, 8, 2), (32, 4, 2)],
                 lin_layers=None):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        self.dynamic_obs_size = dynamic_obs_size
        self.img_red_size = img_red_size
        self.n_input_channels = 2 if rover_2cam_and_combined_image else 1
        self.cnn = nn.Sequential()
        channels = self.n_input_channels
        for (filters, size, stride) in conv_layers:
            self.cnn.append(nn.Conv2d(channels, filters, kernel_size=size, stride=stride, padding=0))
            self.cnn.append(nn.ReLU())
            channels = filters

        self.cnn.append(nn.Flatten())

        # Compute shape by doing one forward pass
        with th.no_grad():
            img = observation_space.sample()[dynamic_obs_size:].reshape((1, img_red_size[0], img_red_size[1]))[None]
            n_flatten = self.cnn(
                th.as_tensor(img).float()
            ).shape[1]

        self.linear = nn.Sequential()
        if lin_layers is not None:
            input_ = n_flatten+dynamic_obs_size
            for size in lin_layers:
                self.linear.append(nn.Linear(input_, size))
                self.linear.append(nn.Tanh())
                input_ = size

    def forward(self, observations: th.Tensor) -> th.Tensor:
        images = observations[:, self.dynamic_obs_size:].reshape((-1, self.n_input_channels, self.img_red_size[0], self.img_red_size[1]))
        dynamics = observations[:, :self.dynamic_obs_size]
        return self.linear(th.concat((self.cnn(images), dynamics), axis=1))


def save_current_network(
    self,
    path: Union[str],
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
