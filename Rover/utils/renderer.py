import time
from typing import Optional

import glfw
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer, WindowViewer, OffScreenViewer
import mujoco


class RoverOffScreenViewer(OffScreenViewer):
    fig = mujoco.MjvFigure()

    def render(
            self,
            render_mode: str,
            camera_id: Optional[int] = None,
            segmentation: bool = False,
    ):
        if camera_id is not None:
            if camera_id == -1:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.cam.fixedcamid = camera_id

        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.vopt,
            self.pert,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scn,
        )

        if segmentation:
            self.scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 1
            self.scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 1

        for marker_params in self._markers:
            self._add_marker_to_scene(marker_params)

        mujoco.mjr_render(self.viewport, self.scn, self.con)

        for gridpos, (text1, text2) in self._overlays.items():
            mujoco.mjr_overlay(
                mujoco.mjtFontScale.mjFONTSCALE_150,
                gridpos,
                self.viewport,
                text1.encode(),
                text2.encode(),
                self.con,
            )

        if segmentation:
            self.scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 0
            self.scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 0

        rgb_arr = np.zeros(
            3 * self.viewport.width * self.viewport.height, dtype=np.uint8
        )
        depth_arr = np.zeros(
            self.viewport.width * self.viewport.height, dtype=np.float32
        )

        mujoco.mjr_readPixels(rgb_arr, depth_arr, self.viewport, self.con)

        if render_mode == "depth_array":
            depth_img = depth_arr.reshape(self.viewport.height, self.viewport.width)
            # original image is upside-down, so flip it
            return depth_img[::-1, :]
        else:
            rgb_img = rgb_arr.reshape(self.viewport.height, self.viewport.width, 3)

            if segmentation:
                seg_img = (
                        rgb_img[:, :, 0]
                        + rgb_img[:, :, 1] * (2 ** 8)
                        + rgb_img[:, :, 2] * (2 ** 16)
                )
                seg_img[seg_img >= (self.scn.ngeom + 1)] = 0
                seg_ids = np.full(
                    (self.scn.ngeom + 1, 2), fill_value=-1, dtype=np.int32
                )

                for i in range(self.scn.ngeom):
                    geom = self.scn.geoms[i]
                    if geom.segid != -1:
                        seg_ids[geom.segid + 1, 0] = geom.objtype
                        seg_ids[geom.segid + 1, 1] = geom.objid
                rgb_img = seg_ids[seg_img]

            # original image is upside-down, so flip i
            return rgb_img[::-1, :, :]
class RoverWindowViewer(WindowViewer):

    fig = mujoco.MjvFigure()
    fig_viewport = mujoco.MjrRect(0, 0, 15, 15)
    def render(self):
        """
        Renders the environment geometries in the OpenGL glfw window:
            1. Create the overlay for the left side panel menu.
            2. Update the geometries used for rendering based on the current state of the model - `mujoco.mjv_updateScene()`.
            3. Add markers to scene, these are additional geometries to include in the model, i.e arrows, https://mujoco.readthedocs.io/en/latest/APIreference.html?highlight=arrow#mjtgeom.
                These markers are added with the `add_marker()` method before rendering.
            4. Render the 3D scene to the window context - `mujoco.mjr_render()`.
            5. Render overlays in the window context - `mujoco.mjr_overlay()`.
            6. Swap front and back buffer, https://www.glfw.org/docs/3.3/quick.html.
            7. Poll events like mouse clicks or keyboard input.
        """

        # mjv_updateScene, mjr_render, mjr_overlay
        def update():
            # fill overlay items
            self._create_overlay()

            render_start = time.time()
            if self.window is None:
                return
            elif glfw.window_should_close(self.window):
                glfw.destroy_window(self.window)
                glfw.terminate()
            self.viewport.width, self.viewport.height = glfw.get_framebuffer_size(
                self.window
            )
            # update scene
            mujoco.mjv_updateScene(
                self.model,
                self.data,
                self.vopt,
                mujoco.MjvPerturb(),
                self.cam,
                mujoco.mjtCatBit.mjCAT_ALL.value,
                self.scn,
            )

            # marker items
            for marker in self._markers:
                self._add_marker_to_scene(marker)

            self.fig.linedata = np.random.random((100, 2000))
            mujoco.mjr_figure(self.fig_viewport, self.fig, self.con)
            # render
            mujoco.mjr_render(self.viewport, self.scn, self.con)

            # overlay items
            if not self._hide_menu:
                for gridpos, [t1, t2] in self._overlays.items():
                    mujoco.mjr_overlay(
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        gridpos,
                        self.viewport,
                        t1,
                        t2,
                        self.con,
                    )

            glfw.swap_buffers(self.window)
            glfw.poll_events()
            self._time_per_render = 0.9 * self._time_per_render + 0.1 * (
                time.time() - render_start
            )

        if self._paused:
            while self._paused:
                update()
                if self._advance_by_one_step:
                    self._advance_by_one_step = False
                    break
        else:
            self._loop_count += self.model.opt.timestep / (
                self._time_per_render * self._run_speed
            )
            if self._render_every_frame:
                self._loop_count = 1
            while self._loop_count > 0:
                update()
                self._loop_count -= 1

        # clear overlay
        self._overlays.clear()
        # clear markers
        self._markers.clear()
class RoverRenderer(MujocoRenderer):
    def __init__(self,
        model: "mujoco.MjModel",
        data: "mujoco.MjData",
        default_cam_config: Optional[dict] = None,
    ):
        super().__init__(
            model,
            data,
            default_cam_config,
        )

    def _get_viewer(self, render_mode: str):
        """Initializes and returns a viewer class depending on the render_mode
        - `WindowViewer` class for "human" render mode
        - `OffScreenViewer` class for "rgb_array" or "depth_array" render mode
        """
        self.viewer = self._viewers.get(render_mode)
        if self.viewer is None:
            if render_mode == "human":
                self.viewer = RoverWindowViewer(self.model, self.data)

            elif render_mode in {"rgb_array", "depth_array"}:
                self.viewer = OffScreenViewer(self.model, self.data)
            else:
                raise AttributeError(
                    f"Unexpected mode: {render_mode}, expected modes: human, rgb_array, or depth_array"
                )
            # Add default camera parameters
            self._set_cam_config()
            self._viewers[render_mode] = self.viewer

        if len(self._viewers.keys()) > 1:
            # Only one context can be current at a time
            self.viewer.make_context_current()

        return self.viewer