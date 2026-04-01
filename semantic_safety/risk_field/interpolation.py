"""
Directional interpolation: 6 LLM axis weights → continuous 3D multiplier W(x).

Uses centroid-relative directions (stable for non-convex shapes); pair with FMM distance separately.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple

def compute_directional_weights(
    X: np.ndarray, 
    Y: np.ndarray, 
    Z: np.ndarray, 
    bbox: Tuple[float, float, float, float, float, float], 
    weights_dict: Dict[str, float]
) -> np.ndarray:
    """
    Interpolates 6 discrete LLM weights into a continuous 3D spatial multiplier W(x).
    Uses AABB for exterior conformity and Centroid-blending for interior continuity.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = bbox
    
    # 1. Vector to the nearest AABB surface (Used for exterior voxels)
    dx_out = np.maximum(0.0, X - xmax) - np.maximum(0.0, xmin - X)
    dy_out = np.maximum(0.0, Y - ymax) - np.maximum(0.0, ymin - Y)
    dz_out = np.maximum(0.0, Z - zmax) - np.maximum(0.0, zmin - Z)
    
    # Check which voxels are physically inside the bounding box
    is_inside = (np.abs(dx_out) + np.abs(dy_out) + np.abs(dz_out)) < 1e-8
    
    # 2. Vector from the centroid (Used ONLY for interior voxels)
    cx, cy, cz = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0
    
    # 3. Seamlessly blend the vector fields
    dx = np.where(is_inside, X - cx, dx_out)
    dy = np.where(is_inside, Y - cy, dy_out)
    dz = np.where(is_inside, Z - cz, dz_out)
    
    # 4. Standard L1 normalization
    l1_norm = np.abs(dx) + np.abs(dy) + np.abs(dz)
    l1_norm = np.maximum(l1_norm, 1e-8) # Prevent division by zero
    
    # 5. Calculate lambdas and apply weights
    W_x = (np.maximum(0.0, dx)/l1_norm)*weights_dict.get('w_+x', 0.0) + (np.maximum(0.0, -dx)/l1_norm)*weights_dict.get('w_-x', 0.0)
    W_y = (np.maximum(0.0, dy)/l1_norm)*weights_dict.get('w_+y', 0.0) + (np.maximum(0.0, -dy)/l1_norm)*weights_dict.get('w_-y', 0.0)
    W_z = (np.maximum(0.0, dz)/l1_norm)*weights_dict.get('w_+z', 0.0) + (np.maximum(0.0, -dz)/l1_norm)*weights_dict.get('w_-z', 0.0)
    
    return W_x + W_y + W_z

def compute_hazard_field(
    W_field: np.ndarray, 
    distance_field: np.ndarray, 
    gamma: float, 
    alpha: float = 5.0
) -> np.ndarray:
    """Computes the final individual risk field V_i(x)."""
    return gamma * W_field * np.exp(-alpha * distance_field)