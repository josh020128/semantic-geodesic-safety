"""
Fast Marching Method for geodesic distance from seed surface S.
Euclidean distance field from hazard centroid for shielding ratio A(x).
"""

from typing import Tuple

import numpy as np


def _neighbors_6(shape: Tuple[int, int, int], i: int, j: int, k: int):
    """6-neighborhood in 3D."""
    out = []
    if i > 0:
        out.append((i - 1, j, k))
    if i < shape[0] - 1:
        out.append((i + 1, j, k))
    if j > 0:
        out.append((i, j - 1, k))
    if j < shape[1] - 1:
        out.append((i, j + 1, k))
    if k > 0:
        out.append((i, j, k - 1))
    if k < shape[2] - 1:
        out.append((i, j, k + 1))
    return out


def geodesic_distance_field(
    traversable: np.ndarray,
    seed_indices: np.ndarray,
    resolution: float = 1.0,
) -> np.ndarray:
    """
    Isotropic FMM on 3D grid. traversable: bool grid, True = free (speed 1).
    seed_indices: (K, 3) int. Returns distance grid (same shape), np.inf where not reached.
    """
    from heapq import heappush, heappop

    shape = traversable.shape
    dist = np.full(shape, np.inf, dtype=np.float64)
    for idx in seed_indices:
        i, j, k = int(idx[0]), int(idx[1]), int(idx[2])
        if 0 <= i < shape[0] and 0 <= j < shape[1] and 0 <= k < shape[2] and traversable[i, j, k]:
            dist[i, j, k] = 0.0

    # Min-heap (d, i, j, k)
    heap = []
    for idx in seed_indices:
        i, j, k = int(idx[0]), int(idx[1]), int(idx[2])
        if 0 <= i < shape[0] and 0 <= j < shape[1] and 0 <= k < shape[2] and traversable[i, j, k]:
            heappush(heap, (0.0, i, j, k))

    while heap:
        d, i, j, k = heappop(heap)
        if d > dist[i, j, k]:
            continue
        for ni, nj, nk in _neighbors_6(shape, i, j, k):
            if not traversable[ni, nj, nk]:
                continue
            step = resolution  # isotropic
            nd = d + step
            if nd < dist[ni, nj, nk]:
                dist[ni, nj, nk] = nd
                heappush(heap, (nd, ni, nj, nk))

    return dist


def euclidean_distance_field(
    grid_shape: Tuple[int, int, int],
    origin: np.ndarray,
    centroid: np.ndarray,
    resolution: float,
) -> np.ndarray:
    """
    Euclidean distance from centroid to each voxel center.
    origin: (3,) world coords of grid[0,0,0]. centroid: (3,) world. resolution: voxel size.
    """
    nx, ny, nz = grid_shape
    ix = (np.arange(nx, dtype=np.float64) + 0.5) * resolution + origin[0]
    iy = (np.arange(ny, dtype=np.float64) + 0.5) * resolution + origin[1]
    iz = (np.arange(nz, dtype=np.float64) + 0.5) * resolution + origin[2]
    xx, yy, zz = np.meshgrid(ix, iy, iz, indexing="ij")
    cx, cy, cz = centroid[0], centroid[1], centroid[2]
    return np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2)
