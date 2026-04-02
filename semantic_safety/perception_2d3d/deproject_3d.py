import numpy as np

class CameraGeometry:
    def __init__(self, intrinsics: dict):
        """
        Initializes the math engine using the camera's internal focal lengths.
        """
        self.fx = intrinsics['fx']
        self.fy = intrinsics['fy']

    def get_3d_coordinate(self, u: int, v: int, depth_map: np.ndarray) -> np.ndarray:
        """
        Takes a 2D pixel (u, v) and the raw depth map, and returns the 3D (X, Y, Z) 
        coordinate relative to the camera frame.
        """
        # Read the exact dimensions dynamically from the depth map
        height, width = depth_map.shape
        cx = width / 2.0
        cy = height / 2.0
        
        # Ensure the pixel coordinates are within the image bounds
        u = np.clip(int(u), 0, width - 1)
        v = np.clip(int(v), 0, height - 1)
        
        # 1. Look up the depth (Z) at this exact pixel
        z = depth_map[v, u]
        
        # If the depth is 0.0, it hit the skybox mask!
        if z <= 0.0:
            print(f"Warning: Pixel ({u}, {v}) has no depth (Skybox or Edge Artifact).")
            return None

        # 2. Apply the Pinhole Camera Equation
        x = (u - cx) * z / self.fx
        y = (v - cy) * z / self.fy
        
        return np.array([x, y, z])