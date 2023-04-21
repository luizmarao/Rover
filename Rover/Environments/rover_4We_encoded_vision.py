from Rover.Environments.rover_4We_v2 import RoverRobotrek4Wev2Env
import keras
import os
import numpy as np

class Rover4WeEncodedVisionv0Env(RoverRobotrek4Wev2Env):
    def __init__(self, **kwargs):
        self.encoder_name = kwargs.pop('encoder_name')
        assert self.encoder_name is not None, 'You have not chose any encoder. If you do not want to chose one, you should run Rover4We latest version instead of Rover4WeEncodedVision'
        encoder_path = os.path.join(os.path.dirname(__file__), 'assets', 'VisionEncoders', self.encoder_name)  # TODO: check img_red_size compatible with encoder input size
        self.encoder = keras.models.load_model(encoder_path)
        self.encoder.compile()
        self.observation_space_size = np.product(self.encoder.output_shape[1:]) + 14
        super().__init__(**kwargs)


    def camera_rendering(self):
        gray_normalized = super().camera_rendering()
        expanded_dim_gray_normalized = np.expand_dims(gray_normalized, (0, -1))
        encoded = np.asarray(self.encoder(expanded_dim_gray_normalized))
        flatten = np.ndarray.flatten(encoded)
        return flatten

    def format_obs(self, lin_obs, img_obs):
        return np.asarray([lin_obs, img_obs])