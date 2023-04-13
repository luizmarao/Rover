''' TODO: Check ids for rover 2 cam
Arquivo de configuração do ambient rover do mujoco

Endereço do arquivo : gym/gym/envs/mujoco

Baseado em: https://github.com/openai/gym/tree/master/gym/envs/mujoco

rover-4-wheels-diff-ackerman-double-front-wheel.xml qpos and qvel

body names:     ['rover', 'r-l-wheel', 'r-r-wheel', 'ghost-steer-wheel', 'f-l-wheel', 'f-l-axis', 'f-l-l-wheel', 'f-l-r-wheel', 'f-r-wheel', 'f-r-axis', 'f-r-l-wheel', 'f-r-r-wheel']
body id:        [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12]
body_geomadr:   [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

qpos[0] = rover's x position
qpos[1] = rover's y position
qpos[2] = rover's z position
qpos[3] = rover's w quaternion vector
qpos[4] = rover's a quaternion vector
qpos[5] = rover's b quaternion vector
qpos[6] = rover's c quaternion vector
qpos[7] = rear left wheel rotation angle
qpos[8] = rear right wheel rotation angle
qpos[9] = ghost steer hinge
qpos[10]= front left wheel steer angle
qpos[11]= front left wheel rotation 1 angle
qpos[12]= front left wheel rotation 2 angle
qpos[13]= front right wheel steer angle
qpos[14]= front right wheel rotation 3 angle
qpos[15]= front right wheel rotation 4 angle

qvel[0] = rover's x velocity
qvel[1] = rover's y velocity
qvel[2] = rover's z velocity
qvel[3] = rover's x angular velocity
qvel[4] = rover's y angular velocity
qvel[5] = rover's z angular velocity
qvel[6] = rear left wheel angular velocity
qvel[7] = rear right wheel angular velocity
qvel[8] = ghost steer hinge velocity
qvel[9]= front left wheel steer velocity
qvel[10]= front left wheel rotation 1 velocity
qvel[11]= front left wheel rotation 2 velocity
qvel[12]= front right wheel steer velocity
qvel[13]= front right wheel rotation 3 velocity
qvel[14]= front right wheel rotation 4 velocity



rover-4-wheels.xml qpos and qvel

qpos[0] = rover's x position
qpos[1] = rover's y position
qpos[2] = rover's z position
qpos[3] = rover's w quaternion vector
qpos[4] = rover's a quaternion vector
qpos[5] = rover's b quaternion vector
qpos[6] = rover's c quaternion vector
qpos[7] = rear left wheel rotation angle
qpos[8] = rear right wheel rotation angle
qpos[9] = steerbar rotation angle
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
qvel[8] = steerbar angular velocity
qvel[9]= front left wheel angular velocity
qvel[10]= front right wheel angular velocity
qvel[11]= drive motor angular velocity (the prism between rear wheels)


rover-4-wheels_loose.xml qpos and qvel

qpos[0] = rover's x position
qpos[1] = rover's y position
qpos[2] = rover's z position
qpos[3] = rover's w quaternion vector
qpos[4] = rover's a quaternion vector
qpos[5] = rover's b quaternion vector
qpos[6] = rover's c quaternion vector
qpos[7] = rear left wheel steer rotation angle		(very limited)
qpos[8] = rear left wheel rolling rotation angle
qpos[9] = rear right wheel steer rotation angle		(very limited)
qpos[10]= rear right wheel rolling rotation angle
qpos[11]= front left wheel steer rotation angle
qpos[12]= front left wheel rolling rotation angle
qpos[13]= front right wheel steer rotation angle
qpos[14]= front right wheel rolling rotation angle

qvel[0] = rover's x velocity
qvel[1] = rover's y velocity
qvel[2] = rover's z velocity
qvel[3] = rover's x angular velocity
qvel[4] = rover's y angular velocity
qvel[5] = rover's z angular velocity
qvel[6] = rear left wheel steer angular velocity
qvel[7] = rear left wheel rolling angular velocity
qvel[8] = rear right wheel steer angular velocity
qvel[9]= rear right wheel rolling angular velocity
qvel[10]= front left wheel steer angular velocity
qvel[11]= front left wheel rolling angular velocity
qvel[12]= front right wheel steer angular velocity
qvel[13]= front right wheel rolling angular velocity

'''

import numpy as np
import cv2
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces
#import matplotlib.pyplot as plt
import mujoco
import os
from Rover.Environments.rover_4We_v1 import RoverRobotrek4Wev1Env

# O campo tem dimensoes (x,y)=(44,25) [metros]
# É uma matriz (x,y) de (0,0) até (3, 4) elementos, em que cada elemento tem dimensão (x,y) de (44/4, 25/5) = (11.0 , 5.0) [metros]
# Na configuração A, os elementos (2, 1) e (2, 3) tem obstáculos
# Na configuração B, os elementos (1, 2) e (3, 2) tem obstáculos


class RoverRobotrek4Wev2Env(RoverRobotrek4Wev1Env):
    def __init__(self):
        if self.random_current_goal:
            self.randomize_current_goal()

        MujocoEnv.__init__(self,
                           'Rover4We-v1/main-trekking-challenge-4wheels_diff-acker-double-front-wheel-2cam.xml',
                           4)
        utils.EzPickle.__init__(self)

        for body in self.sim.model.body_names:
            if body[0:8] == 'long-bum':
                self.long_bump_storage.append(self.sim.model.body_name2id(body))

            if body[0:8] == 'circ-bum':
                self.circular_bump_storage.append(self.sim.model.body_name2id(body))

            if body[0:8] == 'square-l':
                self.square_long_hole_storage.append(self.sim.model.body_name2id(body))

            if body[0:8] == 'square-h':
                self.square_hole_storage.append(self.sim.model.body_name2id(body))

            if body[0:8] == 'box-tile':
                self.box_tile_storage.append(self.sim.model.body_name2id(body))

            if body[0:4] == 'ramp':
                self.ramp_storage.append(self.sim.model.body_name2id(body))

            if body[0:4] == 'post':
                self.post_storage.append(self.sim.model.body_name2id(body))

        self.holes = np.zeros((2, 11, 5))
        self.map_generator()
        # self.camera_renderer = mujoco_py.MjRenderContextOffscreen(self.sim)
        self.camera_r_id = self.sim.model.camera_name2id('first-person-r')
        self.camera_l_id = self.sim.model.camera_name2id('first-person-l')

    def _get_obs(self):
        '''
        Função que retorna os dados do observation_state
        '''
        # gps exato
        gps_exact = self.sim.data.qpos[0:2].copy()
        # vel exata
        speed_exact = [self.sim.data.qvel[0:2]].copy()

        # gps sensor com erro
        gps_sensor = gps_exact + self.gps_error * np.random.rand(2)
        # vel sensor com erro
        speed_sensor = np.asarray(speed_exact)  # + self.gps_error * np.random.rand(2)

        # orientation_rover
        orientation_rover = self.sim.data.body_xmat[1][0:2].copy()
        rover_ang_speed = self.sim.data.qvel[5].copy()
        ghost_steer_angle = self.sim.data.qpos[9]
        ghost_steer_angspeed = self.sim.data.qvel[8]

        # camera
        #### MUJOCO NOVO #####
        # self.mujoco_renderer.render("rgb_array", camera_id, camera_name)

        if os.getenv('MUJOCO_PY_FORCE_CPU') is not None:
            if not self.is_cam:
                self.sim.render(
                    mode='window')  # , camera_name='first-person', width=im_size[0], height=im_size[1], depth=False)
                self.is_cam = True
        if self.camera_renderer is None:
            gray = np.zeros(shape=self.img_reduced_size)
            self.camera_renderer_initializer()
        else:
            img_r = self.camera_rendering(width=self.im_size[0], height=self.im_size[1], cam_id=self.camera_r_id)
            img_l = self.camera_rendering(width=self.im_size[0], height=self.im_size[1], cam_id=self.camera_l_id)
            if self.save_images:
                cv2.imwrite("./running_images/full_size_r_{:0>7.3f}.png".format(self.sim.data.time),
                            cv2.cvtColor(img_r, cv2.COLOR_RGB2BGR))
                cv2.imwrite("./running_images/full_size_l_{:0>7.3f}.png".format(self.sim.data.time),
                            cv2.cvtColor(img_l, cv2.COLOR_RGB2BGR))
            img_r = cv2.resize(img_r, self.img_reduced_size)
            img_l = cv2.resize(img_l, self.img_reduced_size)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_RGB2GRAY)
            gray_l = cv2.cvtColor(img_l, cv2.COLOR_RGB2GRAY)
            if self.save_images:
                cv2.imwrite("./running_images/reduced_size_r_{:0>7.3f}.png".format(self.sim.data.time),
                            cv2.cvtColor(img_r, cv2.COLOR_RGB2BGR))
                cv2.imwrite("./running_images/reduced_size_l_{:0>7.3f}.png".format(self.sim.data.time),
                            cv2.cvtColor(img_l, cv2.COLOR_RGB2BGR))
                cv2.imwrite("./running_images/reduced_gray_r_{:0>7.3f}.png".format(self.sim.data.time), gray_r)
                cv2.imwrite("./running_images/reduced_gray_l_{:0>7.3f}.png".format(self.sim.data.time), gray_l)
            gray = np.asarray([gray_l, gray_r])

        if self.current_goal == 0:
            coordinates_goal = np.asarray([40, 20])
            goal = np.asarray([0, 0, 1])
        elif self.current_goal == 1:
            coordinates_goal = np.asarray([30, 2])
            goal = np.asarray([0, 1, 0])
        elif self.current_goal == 2:
            coordinates_goal = np.asarray([6, 18])
            goal = np.asarray([1, 0, 0])
        elif self.current_goal == -1:
            coordinates_goal = np.asarray([0, 0])
            goal = np.asarray([0, 0, 0])

        return gps_exact, np.concatenate(
            [gps_sensor.flat, orientation_rover.flat, [ghost_steer_angle], speed_sensor.flat, [rover_ang_speed],
             [ghost_steer_angspeed], coordinates_goal.flat, goal.flat]), gray / 255.0

    def format_obs(self, lin_obs, img_obs):
        img = np.reshape(img_obs, (-1))
        lin = lin_obs
        return np.concatenate([lin, img])


class RoverRobotrek4Wev1Env(MujocoEnv, utils.EzPickle):
    step_counter = 0
    # EXPERIMENTS LOG PARAMETERS #
    loose_version = False
    ackerman_version = True
    original_field = False
    use_ramps = True
    use_posts = True
    reduced_obs = False
    vectorized_obs = True
    flip_penalising = True
    flipped_time = 2.0
    death_circle_penalising = True
    death_circ_dist = 0.8
    death_circ_time = 8
    fwd_rew = 0.1
    control_cost = 0.0007
    svv_rew = 0.0001
    time_pnlt = 0.00012
    leave_penalty = 10

    # TODO: COLISION DETECTION NOT WORKING PROPERLY. SET HIGH VALUES TO AVOID
    # max_accxy = 100.0
    # max_accz = 50.0
    max_accxy = 500.0
    max_accz = 500.0
    colision_penalty = 10
    bumpiness_penalty = 1

    circle_pnlt = 10
    flip_pnlt = 10
    goal_rwd = 1
    gps_error = 0.00  # Erro gaussiano do GPS
    im_size = (440, 270)  # tamanho da imagem adquirida pelo mujoco
    img_reduced_size = (32, 32)  # tamanho da imagem reduzida

    # if start_at_initpos is True, random_start is not used.
    # if start_at_initpos is False, random_start is used.
    start_at_initpos = False

    # about random_start(rs) and random_current_goal(rcg)
    # rs True  and rcg True : rover starts (almost) anywhere and has a random goal (0, 1 or 2)
    # rs True  and rcg False: rover starts (almost) anywhere and has as current_goal the first goal (0)
    # rs False and rcg True : rover starts at the goal before its current_goal (and this one is random: 0, 1 or 2)
    # rs False and rcg False: rover starts at the goal before its current_goal (and this one is the first goal: 0)
    # force_goal -1 will not make anything, 0/1/2 will force goal to be 0/1/2
    force_goal = -1
    random_start = False
    random_current_goal = True
    avoid_radius = 0.5 # radius for obstacle avoidance in rover's position randomizer

    # defines if the episode will end after current_goal is reached
    end_after_current_goal = True

    current_obstacle_setup = None
    current_goal = 0  # inicializa a variável do objetivo atual
    x_before = [0, 0]  # Inicializa o vetor estado do carrinho [posição, orientação]
    is_cam = False
    last_15_pos = np.array([3, 3])
    last_15_time = 0
    last_straight_time = 0
    '''if reduced_obs:
        obs_size = (5,)
    elif loose_version:
        obs_size = (16,)
    else:
        obs_size = (14,)'''

    initial_pos_A = ((22.5, 15.5, 0), (22.5, 5.5, 0))
    initial_pos_B = ((11.5, 10.5, 0), (33.5, 10.5, 0))

    # block_step = 1

    long_bump_storage = []
    long_bump_used = []

    circular_bump_storage = []
    circular_bump_used = []

    square_long_hole_storage = []
    square_long_hole_used = []

    square_hole_storage = []
    square_hole_used = []

    box_tile_storage = []
    box_tile_used = []


    ramp_storage = []
    ramp_used = []


    post_storage = []
    post_used = []

    useds_lists_list = [long_bump_used, circular_bump_used, square_long_hole_used, square_hole_used, box_tile_used,
                        ramp_used, post_used]
    storage_lists_list = [long_bump_storage, circular_bump_storage, square_long_hole_storage, square_hole_storage, box_tile_storage,
                          ramp_storage, post_storage]

    camera_renderer = None
    camera_id = None
    flag_render = False
    save_images = False

    reseted = False # a flag for the first step after a reset, to send new goal flag in info dict

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 25,
    }

    def __init__(self, **kwargs): #TODO corrigir shape do obs_space
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(np.product(self.img_reduced_size)+14,), dtype=np.float64)
        if self.random_current_goal:
            self.randomize_current_goal()
        model_path = os.path.join(os.path.dirname(__file__), 'assets', 'Rover4We-v1')
        MujocoEnv.__init__(
            self,
            os.path.join(model_path, 'main-trekking-challenge-4wheels_diff-acker-double-front-wheel.xml'),
            4,
            observation_space=observation_space,
            **kwargs,
        )
        utils.EzPickle.__init__(self)

        self.body_names = [self.model.body(i).name for i in range(self.model.nbody)]

        self.body_name2id = lambda body_name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        self.camera_name2id = lambda camera_name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)

        for body in self.body_names:
            if body[0:8] == 'long-bum':
                self.long_bump_storage.append(self.body_name2id(body))

            elif body[0:8] == 'circ-bum':
                self.circular_bump_storage.append(self.body_name2id(body))

            elif body[0:8] == 'square-l':
                self.square_long_hole_storage.append(self.body_name2id(body))

            elif body[0:8] == 'square-h':
                self.square_hole_storage.append(self.body_name2id(body))

            elif body[0:8] == 'box-tile':
                self.box_tile_storage.append(self.body_name2id(body))

            elif body[0:4] == 'ramp':
                self.ramp_storage.append(self.body_name2id(body))

            elif body[0:4] == 'post':
                self.post_storage.append(self.body_name2id(body))

        self.holes = np.zeros((2, 11, 5))
        self.map_generator()
        # self.camera_renderer = mujoco_py.MjRenderContextOffscreen(self.sim)
        self.camera_id = self.camera_name2id('first-person')

    #def camera_renderer_initializer(self):
        #self.camera_renderer = mujoco_py.MjRenderContextOffscreen(self.sim)

def create_quat(angle, x, y, z, is_radian=True):
    dir = np.array([x, y, z])
    dir = dir/np.linalg.norm(dir)
    if is_radian:
        return np.array([np.cos(angle / 2), *(dir * np.sin(angle / 2))])
    else:
        angle = angle*np.pi/180
        return np.array([np.cos(angle / 2), *(dir * np.sin(angle / 2))])


def axisangle_from_quat(quat):
    angle_rad = 2*np.arccos(quat[0])
    dir = quat[1:]/(np.arc(2*angle_rad))
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
########################################################################################################################################

