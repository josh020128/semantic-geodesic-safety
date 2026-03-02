"""
Occupancy grid O(x): free space True (speed 1), obstacles False (speed 0).
Boundary seeding: 1-voxel thick surface S in free space adjacent to hazard (morphological dilation).
"""

from typing import Tuple

import numpy as np


def build_occupancy_grid(
    coords: np.ndarray,
    segment: np.ndarray,
    hazard_label: int,
    resolution: float = 0.02,
    padding: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build 3D occupancy grid from point cloud.
    - coords: (N, 3) xyz
    - segment: (N,) semantic labels
    - hazard_label: label id for the hazard object (e.g. laptop)
    Returns:
      - grid: 3D bool, True = traversable (free), False = obstacle
      - origin: (3,) world coords of grid[0,0,0]
      - shape: (nx, ny, nz) grid shape
      - hazard_voxels: (M, 3) int array of (i,j,k) voxel indices that are hazard
    """
    mn = coords.min(axis=0) - padding
    mx = coords.max(axis=0) + padding
    origin = mn
    shape = np.ceil((mx - mn) / resolution).astype(int)
    shape = np.maximum(shape, 1)

    vox_ijk = np.floor((coords - origin) / resolution).astype(int)
    vox_ijk = np.clip(vox_ijk, 0, np.array(shape) - 1)

    obstacle_voxels = set()
    hazard_voxel_set = set()
    for i in range(len(coords)):
        ix, iy, iz = vox_ijk[i]
        obstacle_voxels.add((ix, iy, iz))
        if segment[i] == hazard_label:
            hazard_voxel_set.add((ix, iy, iz))

    grid = np.ones(shape, dtype=bool)
    for (ix, iy, iz) in obstacle_voxels:
        if 0 <= ix < shape[0] and 0 <= iy < shape[1] and 0 <= iz < shape[2]:
            grid[ix, iy, iz] = False

    hazard_voxels = np.array(list(hazard_voxel_set), dtype=np.int32) if hazard_voxel_set else np.zeros((0, 3), dtype=np.int32)
    return grid, origin, np.array(shape), hazard_voxels


def extract_boundary_seeds(
    grid: np.ndarray,
    hazard_voxels: np.ndarray,
) -> np.ndarray:
    """
    Extract 1-voxel thick surface S: free-space voxels immediately adjacent to hazard.
    hazard_voxels: (M, 3) int array of (i,j,k) voxel indices that are hazard.
    Returns (K, 3) int array of seed voxel indices in free space adjacent to hazard.
    """
    from scipy import ndimage

    # Build binary hazard grid same shape as grid
    hazard_grid = np.zeros_like(grid, dtype=bool)
    for (i, j, k) in hazard_voxels:
        if 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1] and 0 <= k < grid.shape[2]:
            hazard_grid[i, j, k] = True

    # Dilate hazard by 1 voxel; seeds = dilated - hazard, restricted to traversable
    struct = ndimage.generate_binary_structure(3, 1)
    dilated = ndimage.binary_dilation(hazard_grid, structure=struct)
    boundary = dilated & ~hazard_grid & grid
    seeds = np.argwhere(boundary)
    return seeds
