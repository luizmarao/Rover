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
from Rover.Environments.rover_4We_v2 import RoverRobotrek4Wev2Env

# O campo tem dimensoes (x,y)=(44,25) [metros]
# É uma matriz (x,y) de (0,0) até (3, 4) elementos, em que cada elemento tem dimensão (x,y) de (44/4, 25/5) = (11.0 , 5.0) [metros]
# Na configuração A, os elementos (2, 1) e (2, 3) tem obstáculos
# Na configuração B, os elementos (1, 2) e (3, 2) tem obstáculos


class RoverRobotrek4Wev2Env(RoverRobotrek4Wev2Env):
    def __init__(self):
        if self.random_current_goal:
            self.randomize_current_goal()

        MujocoEnv.__init__(self,
                           'Rover4We/main-trekking-challenge-4wheels_diff-acker-double-front-wheel-2cam.xml',
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
