'''
Arquivo de configuração do ambient rover4W do mujoco

Baseado em: https://github.com/openai/gym/tree/master/gym/envs/mujoco

qpos and qvel

qpos[0] = rover's x position
qpos[1] = rover's y position
qpos[2] = rover's z position
qpos[3] = rover's w quaternion vector
qpos[4] = rover's a quaternion vector
qpos[5] = rover's b quaternion vector
qpos[6] = rover's c quaternion vector
qpos[7] = rear left wheel rotation angle
qpos[8] = rear right wheel rotation angle
qpos[9] = steer-bar rotation angle
qpos[10]= front left wheel rotation angle
qpos[11]= front right wheel rotation angle
qpos[12]= drive motor rotation angle (the prism between rear wheels)

qvel[0] = rover's x velocity
qvel[1] = rover's y velocity
qvel[2] = rover's z velocity
qvel[3] = rover's x angular velocity
qvel[4] = rover's y angular velocity
qvel[5] = rover's z angular velocity
qvel[6] = rear left wheel angular velocity
qvel[7] = rear right wheel angular velocity
qvel[8] = steer-bar angular velocity
qvel[9]= front left wheel angular velocity
qvel[10]= front right wheel angular velocity
qvel[11]= drive motor angular velocity (the prism between rear wheels)

'''

import numpy as np
from gymnasium import spaces
from gymnasium import utils
from Rover.Environments.rover_4We_v2 import RoverRobotrek4Wev2Env
import mujoco
from Rover.utils.env_util import RoverMujocoEnv
import os


class Rover4Wv1Env(RoverRobotrek4Wev2Env):
    model_file_name = 'main-trekking-challenge-4wheels_diff-acker-double-front-wheel-no-cam.xml'

    def __init__(
            self,
            vectorized_obs: bool = True,
            flip_penalising: bool = True,
            flipped_time: float = 2.0,
            death_circle_penalising: bool = True,
            death_circ_dist: float = 0.8,
            death_circ_time: float = 8,
            fwd_rew: float = 0.1,
            control_cost: float = 0.001,
            svv_rew: float = 0.0001,
            time_pnlt: float = 0.0,
            leave_penalty: float = 10,
            circle_pnlt: float = 10,
            flip_pnlt: float = 10,
            goal_rwd: float = 0.0,
            sensors_error: float = 0.00,
            start_at_initpos: bool = False,
            force_goal: int = -1,
            random_start: bool = False,
            random_current_goal: bool = True,
            avoid_radius: float = 0.5,
            end_after_current_goal: bool = True,
            verbose: int = 0,
            gamma: float = 0.99,
            ** kwargs
    ):
        self.use_ramps = False
        self.use_posts = False
        self.vectorized_obs = vectorized_obs
        self.flip_penalising = flip_penalising
        self.flipped_time = flipped_time
        self.death_circle_penalising = death_circle_penalising
        self.death_circ_dist = death_circ_dist
        self.death_circ_time = death_circ_time
        self.fwd_rew = fwd_rew
        self.control_cost = control_cost
        self.svv_rew = svv_rew
        self.time_pnlt = time_pnlt
        self.leave_penalty = leave_penalty
        self.circle_pnlt = circle_pnlt
        self.flip_pnlt = flip_pnlt
        self.goal_rwd = goal_rwd
        self.sensors_error = sensors_error
        self.im_size = None
        self.img_reduced_size = None
        self.start_at_initpos = start_at_initpos
        self.force_goal = force_goal
        self.random_start = random_start
        self.random_current_goal = random_current_goal
        self.avoid_radius = avoid_radius
        self.gamma = gamma
        self.end_after_current_goal = end_after_current_goal
        self.save_images = None
        self.verbose = verbose

        observation_space = self.make_env_observation_space()

        model_path = os.path.join(os.path.dirname(__file__), 'assets', 'RoverOriginal')
        RoverMujocoEnv.__init__(
            self,
            model_path=os.path.join(model_path, self.model_file_name),
            frame_skip=4,
            observation_space=observation_space,
            **kwargs,
        )
        utils.EzPickle.__init__(self)
        self.env_config()

    def make_env_observation_space(self):
        self.observation_space_size = 14
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_space_size,), dtype=np.float64)

    def camera_rendering(self):
        pass

    def env_config(self):
        if self.random_current_goal:
            self.randomize_current_goal()
        self.body_names = [self.model.body(i).name for i in range(self.model.nbody)]
        self.body_name2id = lambda body_name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)

    def format_obs(self, lin_obs, img_obs):
        return np.asarray(lin_obs)

    def map_reset(self):
        pass

    def map_generator(self):
        pass

    def reset_model(self):
        self.step_counter = 0
        '''
        Reinicializa a simulação
        '''
        self.reseted = True
        self.map_reset()

        self.map_generator()

        if self.force_goal == -1:
            if self.random_current_goal:
                self.randomize_current_goal()
            else:
                self.current_goal = 0
        else:
            self.current_goal = self.force_goal

        if self.start_at_initpos:
            self.set_state(self.init_qpos, self.init_qvel)
        else:
            if self.random_start:
                rover_random_xy_pos = np.asarray([np.random.rand() * 44, np.random.rand() * 25])
                if self.verbose >= 1:
                    print(colorize("starting at random position {}".format(rover_random_xy_pos), 'blue', bold=False))
                quat = create_quat(np.random.rand() * 2 * np.pi, 0, 0, 1, is_radian=True)
                self.set_state(np.array([*rover_random_xy_pos, 0.2, *quat, *self.init_qpos[7:]]), self.init_qvel)
            else:
                if self.current_goal < 1:
                    self.set_state(self.init_qpos, self.init_qvel)
                else:
                    quat = create_quat(np.random.rand() * 2 * np.pi, 0, 0, 1, is_radian=True)
                    init_xy_rover = self.get_goal_xy_position(self.current_goal - 1)
                    self.set_state(np.asarray([init_xy_rover[0], init_xy_rover[1], 0.2, *quat, *self.init_qpos[7:]]),
                                   self.init_qvel)

                if self.verbose >= 1:
                    print(colorize("starting at {}".format(
                        'init_pos' if self.current_goal == 0 else 'goal {}'.format(self.current_goal - 1)), 'cyan',
                        bold=False))

        if self.verbose >= 1:
            print(colorize('current_goal: {}'.format(self.current_goal), 'yellow', bold=False))

        gps_exact, ob, img = self._get_obs()
        self.x_before = self.data.qpos[0:2].copy() + self.sensors_error * np.random.rand(2)

        self.last_15_pos = np.array([3, 3])
        self.last_15_time = 0

        obs = self.format_obs(ob, img)
        return obs, dict(current_goal=self.current_goal)


def create_quat(angle, x, y, z, is_radian=True):
    dir = np.array([x, y, z])
    dir = dir / np.linalg.norm(dir)
    if is_radian:
        return np.array([np.cos(angle / 2), *(dir * np.sin(angle / 2))])
    else:
        angle = angle * np.pi / 180
        return np.array([np.cos(angle / 2), *(dir * np.sin(angle / 2))])


def axisangle_from_quat(quat):
    angle_rad = 2 * np.arccos(quat[0])
    dir = quat[1:] / (np.arc(2 * angle_rad))
    return (angle_rad, dir)


# THIS COLORIZE FUNCTION WAS COPIED FROM SPINUP.UTILS.LOGX FILE # https://github.com/openai/spinningup/blob/master/spinup/utils/logx.py
color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)


def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.
    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)
