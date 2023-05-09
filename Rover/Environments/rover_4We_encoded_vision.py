import os
from typing import Tuple

import keras
import numpy as np
from gymnasium import spaces

from Rover.Environments.rover_4We_v2 import RoverRobotrek4Wev2Env


class Rover4WeEncodedVisionv0Env(RoverRobotrek4Wev2Env):
    def __init__(
            self,
            use_ramps: bool = True,
            use_posts: bool = True,
            vectorized_obs: bool = True,
            flip_penalising: bool = True,
            flipped_time: float = 2.0,
            death_circle_penalising: bool = True,
            death_circ_dist: float = 0.8,
            death_circ_time: float = 8,
            fwd_rew: float = 0.1,
            control_cost: float = 0.0007,
            svv_rew: float = 0.0001,
            time_pnlt: float = 0.00012,
            leave_penalty: float = 10,
            circle_pnlt: float = 10,
            flip_pnlt: float = 10,
            goal_rwd: float = 1,
            sensors_error: float = 0.00,
            im_size: Tuple[int] = (440, 270),
            img_reduced_size: Tuple[int] = (32, 32),
            start_at_initpos: bool = False,
            force_goal: int = -1,
            random_start: bool = False,
            random_current_goal: bool = True,
            avoid_radius: float = 0.5,
            end_after_current_goal: bool = True,
            save_images: bool = False,
            verbose: int = 0,
            **kwargs
    ):
        self.encoder_name = kwargs.pop('encoder_name')
        encoder_input_size = self.encoder_name.split('g')[1].split('_', maxsplit=1)[0]
        assert img_reduced_size[0] == int(encoder_input_size), "Expected an encoder with name in the same format as " \
                                                               "'EC g32_1c_4b_128f_128a' and an squared image reduced size, " \
                                                               "like (32, 32)\nReduced image size{} incompatible with encoder " \
                                                               "input size{}".format(img_reduced_size,
                                                                                     encoder_input_size)
        assert self.encoder_name is not None, 'You have not chose any encoder. If you do not want to chose one, you ' \
                                              'should run Rover4We latest version instead of Rover4WeEncodedVision'
        encoder_path = os.path.join(os.path.dirname(__file__), 'assets', 'VisionEncoders', self.encoder_name)
        self.encoder = keras.models.load_model(encoder_path)
        self.encoder.compile()
        super().__init__(
            use_ramps=use_ramps,
            use_posts=use_posts,
            vectorized_obs=vectorized_obs,
            flip_penalising=flip_penalising,
            flipped_time=flipped_time,
            death_circle_penalising=death_circle_penalising,
            death_circ_dist=death_circ_dist,
            death_circ_time=death_circ_time,
            fwd_rew=fwd_rew,
            control_cost=control_cost,
            svv_rew=svv_rew,
            time_pnlt=time_pnlt,
            leave_penalty=leave_penalty,
            circle_pnlt=circle_pnlt,
            flip_pnlt=flip_pnlt,
            goal_rwd=goal_rwd,
            sensors_error=sensors_error,
            im_size=im_size,
            img_reduced_size=img_reduced_size,
            start_at_initpos=start_at_initpos,
            force_goal=force_goal,
            random_start=random_start,
            random_current_goal=random_current_goal,
            avoid_radius=avoid_radius,
            end_after_current_goal=end_after_current_goal,
            save_images=save_images,
            verbose=verbose,
            **kwargs
        )

    def make_env_observation_space(self):
        self.observation_space_size = np.product(self.encoder.output_shape[1:]) + 14
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_space_size,), dtype=np.float64)

    def camera_rendering(self):
        gray_normalized = super().camera_rendering()
        expanded_dim_gray_normalized = np.expand_dims(gray_normalized, (0, -1))
        encoded = np.asarray(self.encoder(expanded_dim_gray_normalized))
        flatten = np.ndarray.flatten(encoded)
        return flatten

    def format_obs(self, lin_obs, img_obs):
        return np.concatenate([lin_obs, img_obs])
