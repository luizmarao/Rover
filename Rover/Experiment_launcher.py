import torch as th
import torch.nn as nn
from gymnasium import spaces
import numpy as np
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from Rover_customized_files.networks import RovernetClassic, save_current_network
from Rover_customized_files.logger import configure


os.environ["MUJOCO_GL"] = 'egl'  # Set mujoco rendering to dedicated GPU
EXPERIMENT_NAME = 'TestRoverSB3'
os.environ["SB3_LOGDIR"] = os.path.expanduser('~/Python_Packages/rover_sb3/Experiments')


# set up logger
logger = configure(folder=None, format_strings=["stdout", "csv", "tensorboard"], exp_name=EXPERIMENT_NAME)


## ENVIRONMENT PARAMETERS ##
rover_env = "Rover4We-v1"
img_red_size = (32, 32)
rover_2cam_and_combined_image = False
reduced_obs = False  # TODO: fix size bellow
dynamic_obs_size = 14 if not reduced_obs else 0
num_environments = 4
env_rendering_kwargs = {"render_mode": "rgb_array", "width": 440, "height": 270}
start_at_initpos = False
end_after_current_goal = True
random_current_goal = True

## NETWORK STRUCTURE PARAMETERS ##
Networks_Architecture = RovernetClassic
conv_layers = [  # (filters, size, stride)
    (16, 8, 2),
    (32, 4, 2),
    (64, 2, 1)
]
features_extractor_lin_layers = None  # linear layers after conv layers and before mlp layers [size] ex:[64, 64]
share_features_extractor = True
net_arch = dict(pi=[64, 64], vf=[64, 64])  # mlp extractor after conv layers (feature extractor)
if features_extractor_lin_layers is not None:
    features_dim = features_extractor_lin_layers[-1]
else:
    features_dim = np.product(img_red_size) + dynamic_obs_size
normalize_images = False  # Already done in env

## PPO PARAMETERS ##
total_learning_timesteps = 2e6
n_steps = 2048  # for each env per update
seed = None
learning_rate = 0.0001
gamma = 0.99
n_epochs = 40  # networks training epochs
gae_lambda = 0.95
batch_size = 256  # was 64in default algo
clip_range = 0.1  # was 0.3 in default algo
normalize_advantage = True
ent_coef = 0.0
max_grad_norm = 1.0  # was 0.5 in default algo
use_sde = False
target_kl = 0.1  # was None in default algo
stats_window_size = 100  # TODO: fix to use entire update data

policy_kwargs = dict(
    features_extractor_class=Networks_Architecture,
    #use_sde=use_sde,
    share_features_extractor=share_features_extractor,
    normalize_images=normalize_images,
    net_arch=net_arch,
    features_extractor_kwargs=dict(
        features_dim=features_dim,
        img_red_size=img_red_size,
        rover_2cam_and_combined_image=rover_2cam_and_combined_image,
        dynamic_obs_size=dynamic_obs_size,
        conv_layers=conv_layers,
        lin_layers=features_extractor_lin_layers,
    ),
)

envs = make_vec_env(rover_env, env_kwargs=env_rendering_kwargs, n_envs=num_environments)
model = PPO("CnnPolicy", envs, policy_kwargs=policy_kwargs, verbose=1,
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
            #use_sde=use_sde,
            target_kl=target_kl,
            seed=seed,
            #stats_window_size=stats_window_size
            )
model.set_logger(logger)
# model.learn(total_timesteps=total_learning_timesteps, progress_bar=True)

model.save_current_network = save_current_network
model.save(path=logger.get_dir()+"saved_model", include=None, exclude=None)

loaded_model = PPO.load(path=logger.get_dir()+"saved_model")
#
# vec_env = model.get_env()
# obs = vec_env.reset()

# for i in range(1000):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = vec_env.step(action)
#     vec_env.render()

# obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
