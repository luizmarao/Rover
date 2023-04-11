# import gymnasium as gym
# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env
#
# env_k = {"render_mode":"rgb_array", "width":440, "height":270}
# envs = make_vec_env("Rover4We-v1", env_kwargs = env_k, n_envs=16)
# envs.reset()
# #env = gym.make("Rover4We-v1", render_mode="rgb_array", width=440, height=270)
# #venv = gym.vector.make("Rover4We-v1", render_mode="rgb_array", width=440, height=270, num_envs=3)
# #observation, info = env.reset(seed=42, options={})
# model = PPO("MlpPolicy", envs, verbose=1)
# model.learn(total_timesteps=20_000, progress_bar=True)

import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        # n_input_channels = observation_space.shape[0]
        n_input_channels = 1
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            img = observation_space.sample()[14:].reshape((1,32,32))[None]
            n_flatten = self.cnn(
                th.as_tensor(img).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten+14, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        images = observations[:,14:].reshape((-1,1,32,32))
        dynamics = observations[:,:14]
        return self.linear(th.concat((self.cnn(images),dynamics),axis=1))

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
)



env_k = {"render_mode":"rgb_array", "width":440, "height":270}
envs = make_vec_env("Rover4We-v1", env_kwargs = env_k, n_envs=3)
model = PPO("CnnPolicy", envs, policy_kwargs=policy_kwargs, verbose=1)
model.learn(total_timesteps=20_000, progress_bar=True)




#
# vec_env = model.get_env()
# obs = vec_env.reset()

# for i in range(1000):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = vec_env.step(action)
#     vec_env.render()

# obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

