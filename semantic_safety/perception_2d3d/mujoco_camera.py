import numpy as np
import mujoco


class MujocoCamera:
    def __init__(self, model, data, cam_name="camera", width=640, height=480):
        self.model = model
        self.data = data
        self.cam_name = cam_name
        self.width = width
        self.height = height

        self.renderer = mujoco.Renderer(model, height, width)

        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if self.cam_id == -1:
            raise ValueError(f"Camera '{cam_name}' not found in the MuJoCo XML.")

        fovy = model.cam_fovy[self.cam_id]
        fy = 0.5 * height / np.tan(np.deg2rad(fovy) / 2.0)
        fx = fy  # square pixels assumption

        self.intrinsics = {
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(width / 2.0),
            "cy": float(height / 2.0),
        }

    def get_frames(self):
        self.renderer.update_scene(self.data, camera=self.cam_name)

        # RGB
        self.renderer.disable_depth_rendering()
        color_image = self.renderer.render()

        # Depth
        self.renderer.enable_depth_rendering()
        depth_metric = self.renderer.render().copy()

        # Treat very far / invalid as background
        depth_metric[~np.isfinite(depth_metric)] = 0.0
        depth_metric[depth_metric <= 0.0] = 0.0

        return color_image, depth_metric, self.intrinsics