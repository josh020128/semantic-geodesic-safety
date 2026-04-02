import numpy as np
import mujoco

class MujocoCamera:
    def __init__(self, model, data, cam_name="camera", width=640, height=480):
        """
        Initializes the headless MuJoCo camera.
        It acts exactly like the RealSense wrapper so the downstream math 
        engine never knows it's in a simulation.
        """
        self.model = model
        self.data = data
        self.cam_name = cam_name
        self.width = width
        self.height = height

        # Setup the headless renderer (Requires MUJOCO_GL=egl in your environment)
        self.renderer = mujoco.Renderer(model, height, width)

        # Retrieve the internal camera ID
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if self.cam_id == -1:
            raise ValueError(f"Camera '{cam_name}' not found in the MuJoCo XML.")

        # ---------------------------------------------------------
        # MATHEMATICS: Calculate Camera Intrinsics
        # ---------------------------------------------------------
        # MuJoCo defines the camera via a vertical Field of View (fovy) in degrees.
        # We must convert this into the focal length (f) in pixels.
        fovy = model.cam_fovy[self.cam_id]
        f = 0.5 * height / np.tan(np.deg2rad(fovy) / 2.0)
        
        self.intrinsics = {
            "fx": f,
            "fy": f,
            "cx": width / 2.0,
            "cy": height / 2.0
        }

    def get_frames(self):
        """
        Renders the RGB and Depth matrices.
        Returns:
            color_image (np.ndarray): The 2D RGB matrix for Lang-SAM (H, W, 3).
            depth_image (np.ndarray): The 2D Depth matrix in true metric meters (H, W).
            intrinsics (dict): The focal length dictionary for 3D deprojection.
        """
        # Sync the renderer with the current physics state
        self.renderer.update_scene(self.data, camera=self.cam_name)

        # 1. Render RGB
        self.renderer.disable_depth_rendering()
        color_image = self.renderer.render()

        # 2. Render Raw Depth
        self.renderer.enable_depth_rendering()
        depth_raw = self.renderer.render()

        # 3. Convert non-linear [0, 1] depth buffer to true metric distance
        extent = self.model.stat.extent
        near = self.model.vis.map.znear * extent
        far = self.model.vis.map.zfar * extent

        # The MuJoCo depth linearization formula: 
        # d_metric = (near * far) / (far - d_raw * (far - near))
        depth_metric = near / (1.0 - depth_raw * (1.0 - near / far))

        # 4. Mask out the skybox (where depth_raw approaches 1.0)
        # We set the background distance to 0.0 to prevent infinite projections
        depth_metric[depth_raw >= 0.999] = 0.0

        return color_image, depth_metric, self.intrinsics