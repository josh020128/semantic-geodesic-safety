import numpy as np


class WorldTransform:
    def __init__(self, cam_pos: np.ndarray, cam_mat: np.ndarray):
        self.cam_pos = np.asarray(cam_pos, dtype=np.float64)
        self.cam_mat = np.asarray(cam_mat, dtype=np.float64).reshape(3, 3)

        # OpenCV camera: +X right, +Y down, +Z forward
        # MuJoCo camera: +X right, +Y up, -Z forward
        self.cv2_to_mj_camera = np.array([
            [1.0,  0.0,  0.0],
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0],
        ], dtype=np.float64)

    def to_world(self, camera_point: np.ndarray) -> np.ndarray:
        if camera_point is None:
            return None

        camera_point = np.asarray(camera_point, dtype=np.float64)

        point_mj_local = self.cv2_to_mj_camera @ camera_point
        world_point = self.cam_mat @ point_mj_local + self.cam_pos
        return world_point