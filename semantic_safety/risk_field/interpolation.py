"""
Continuous directional interpolation: W_hazard(x) from 6 LLM weights.
Soft sign gating (sigmoid) + generalized blending (exponent p).
"""

from typing import Tuple

import numpy as np


def sigmoid(v: np.ndarray, k: float = 10.0) -> np.ndarray:
    """σ(v) = 1 / (1 + exp(-k*v))."""
    return 1.0 / (1.0 + np.exp(-k * v))


def directional_weight_grid(
    grid_shape: Tuple[int, int, int],
    origin: np.ndarray,
    centroid: np.ndarray,
    resolution: float,
    weights_6: Tuple[float, float, float, float, float, float],
    sigmoid_steepness: float = 10.0,
    blend_exponent: float = 2.0,
) -> np.ndarray:
    """
    Compute W_hazard(x) for every voxel: directional prior from hazard centroid.
    weights_6 = (w_+x, w_-x, w_+y, w_-y, w_+z, w_-z).
    """
    w_px, w_mx, w_py, w_my, w_pz, w_mz = weights_6
    nx, ny, nz = grid_shape
    ix = (np.arange(nx, dtype=np.float64) + 0.5) * resolution + origin[0]
    iy = (np.arange(ny, dtype=np.float64) + 0.5) * resolution + origin[1]
    iz = (np.arange(nz, dtype=np.float64) + 0.5) * resolution + origin[2]
    xx, yy, zz = np.meshgrid(ix, iy, iz, indexing="ij")
    cx, cy, cz = centroid[0], centroid[1], centroid[2]
    dx = xx - cx
    dy = yy - cy
    dz = zz - cz
    norm = np.sqrt(dx * dx + dy * dy + dz * dz)
    eps = 1e-12
    ux = np.where(norm > eps, dx / (norm + eps), 0.0)
    uy = np.where(norm > eps, dy / (norm + eps), 0.0)
    uz = np.where(norm > eps, dz / (norm + eps), 0.0)

    k = sigmoid_steepness
    sig_x = sigmoid(ux, k)
    Wx = sig_x * w_px + (1 - sig_x) * w_mx
    sig_y = sigmoid(uy, k)
    Wy = sig_y * w_py + (1 - sig_y) * w_my
    sig_z = sigmoid(uz, k)
    Wz = sig_z * w_pz + (1 - sig_z) * w_mz

    p = blend_exponent
    abs_ux = np.abs(ux) ** p
    abs_uy = np.abs(uy) ** p
    abs_uz = np.abs(uz) ** p
    denom = abs_ux + abs_uy + abs_uz + eps
    pi_x = abs_ux / denom
    pi_y = abs_uy / denom
    pi_z = abs_uz / denom

    W_hazard = pi_x * Wx + pi_y * Wy + pi_z * Wz
    return W_hazard
