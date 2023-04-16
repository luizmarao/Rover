import torch as th
import torch.nn as nn
from gymnasium import spaces
import numpy as np
import os

from Rover.utils.env_util import make_vec_env
from Rover.utils.networks import RovernetClassic, save_current_network
from Rover.utils.logger import configure
from Rover.utils.env_register import register_rover_environments
from Rover.algos.ppo.ppo import PPO_Rover

os.environ["MUJOCO_GL"] = 'egl'  # Set mujoco rendering to dedicated GPU
EXPERIMENT_NAME = 'TestRoverSB3'
os.environ["SB3_LOGDIR"] = os.path.join(os.path.dirname(__file__), 'Experiments')
assert isinstance(os.environ["SB3_LOGDIR"], str)
os.makedirs(os.environ["SB3_LOGDIR"], exist_ok=True)

# set up logger
logger = configure(folder=None, format_strings=["stdout", "csv", "tensorboard"], exp_name=EXPERIMENT_NAME)

register_rover_environments()
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
monitor_kwargs = dict(info_keywords=['death', 'goal_reached_flag', 'timeout'], reset_keywords=['current_goal'])

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
n_steps = 512  # for each env per update
seed = None
learning_rate = 0.0001
gamma = 0.99
n_epochs = 10  # networks training epochs
gae_lambda = 0.95
batch_size = 256  # was 64in default algo
clip_range = 0.1  # was 0.3 in default algo
normalize_advantage = True
ent_coef = 0.0
max_grad_norm = 1.0  # was 0.5 in default algo
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
        features_dim=features_dim,
        img_red_size=img_red_size,
        rover_2cam_and_combined_image=rover_2cam_and_combined_image,
        dynamic_obs_size=dynamic_obs_size,
        conv_layers=conv_layers,
        lin_layers=features_extractor_lin_layers,
    ),
)

envs = make_vec_env(rover_env, env_kwargs=env_rendering_kwargs, n_envs=num_environments, monitor_kwargs=monitor_kwargs)
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
model.set_logger(logger)
model.learn(total_timesteps=total_learning_timesteps, progress_bar=True)

model.save_current_network = save_current_network
model.save(path=logger.get_dir() + "saved_model", include=None, exclude=None)

loaded_model = PPO_Rover.load(path=logger.get_dir() + "saved_model")
