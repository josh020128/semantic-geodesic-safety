import numpy as np

class WorldTransform:
    def __init__(self, cam_pos: np.ndarray, cam_mat: np.ndarray):
        """
        Initializes the transformer with the camera's global position and rotation.
        """
        self.cam_pos = np.array(cam_pos)
        
        # MuJoCo stores the 3x3 rotation matrix as a flat 9-element array, so we reshape it
        self.cam_mat = np.array(cam_mat).reshape(3, 3)
        
        # OpenCV's camera frame (+Z forward, +Y down) is different from 
        # MuJoCo's local camera frame (-Z forward, +Y up). We need an alignment matrix.
        self.cv2mj_rot = np.array([
            [1.0,  0.0,  0.0],
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0]
        ])

    def to_world(self, camera_point: np.ndarray) -> np.ndarray:
        """
        Transforms a 3D point from the OpenCV camera frame to the global MuJoCo World frame.
        """
        # 1. Align the OpenCV coordinate axes with MuJoCo's local camera axes
        point_mj_local = self.cv2mj_rot @ camera_point
        
        # 2. Rotate to match the World's orientation, then Translate to the global position
        world_point = (self.cam_mat @ point_mj_local) + self.cam_pos
        
        return world_point