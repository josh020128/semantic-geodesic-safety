import numpy as np


class CameraGeometry:
    def __init__(self, intrinsics: dict):
        self.fx = float(intrinsics["fx"])
        self.fy = float(intrinsics["fy"])
        self.cx = float(intrinsics["cx"])
        self.cy = float(intrinsics["cy"])

    def get_3d_coordinate(self, u: int, v: int, depth_map: np.ndarray) -> np.ndarray | None:
        """
        Convert pixel (u, v) + metric depth into a 3D point in OpenCV-style camera frame:
          +X right, +Y down, +Z forward
        """
        height, width = depth_map.shape

        u = int(np.clip(u, 0, width - 1))
        v = int(np.clip(v, 0, height - 1))

        z = float(depth_map[v, u])

        if not np.isfinite(z) or z <= 0.0:
            print(f"Warning: Pixel ({u}, {v}) has invalid depth.")
            return None

        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy

        return np.array([x, y, z], dtype=np.float64)