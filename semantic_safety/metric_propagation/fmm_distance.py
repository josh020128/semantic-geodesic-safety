"""
Fast Marching / Eikonal distances on a 3D voxel workspace.

- ``WorkspaceGrid``: fixed-axis grid; uses ``scikit-fmm`` for boundary-conformal distance.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np

try:
    import skfmm
except ImportError as e:
    skfmm = None 
    _SKFMM_IMPORT_ERROR = e

def _require_skfmm() -> None:
    if skfmm is None:
        raise ImportError(
            "scikit-fmm is required for WorkspaceGrid. Install with: pip install scikit-fmm"
        ) from _SKFMM_IMPORT_ERROR

class WorkspaceGrid:
    """
    3D voxel workspace aligned with world axes. Uses the Eikonal equation (FMM via skfmm)
    for exact distance from the physical hazard boundary ∂Ω.
    """

    def __init__(
        self,
        bounds: Tuple[float, float, float, float, float, float] = (-1.0, 1.0, -1.0, 1.0, 0.0, 1.0),
        resolution: float = 0.02
    ):
        _require_skfmm()
        if resolution <= 0:
            raise ValueError("resolution must be positive.")

        self.res = float(resolution)
        self.bounds = bounds
        xmin, xmax, ymin, ymax, zmin, zmax = bounds

        # Create coordinate arrays
        self.x = np.arange(xmin, xmax, self.res, dtype=np.float64)
        self.y = np.arange(ymin, ymax, self.res, dtype=np.float64)
        self.z = np.arange(zmin, zmax, self.res, dtype=np.float64)
        
        # 'ij' indexing ensures shape matches (nx, ny, nz)
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing="ij")
        self.shape = self.X.shape
        self.origin = np.array([xmin, ymin, zmin], dtype=np.float64)

    def _phi_from_object_mask(self, object_mask: np.ndarray) -> np.ndarray:
        """
        Creates a level-set field. -1 inside the object, 1 outside.
        This forces scikit-fmm to calculate distance exactly from the boundary ∂Ω (phi=0).
        """
        object_mask = np.asarray(object_mask, dtype=bool)
        if object_mask.shape != self.shape:
            raise ValueError(
                f"object_mask shape {object_mask.shape} does not match grid shape {self.shape}."
            )
        return np.where(object_mask, -1.0, 1.0).astype(np.float64)

    def compute_euclidean_distance(self, object_mask: np.ndarray) -> np.ndarray:
        """
        Clamped non-negative distance from ∂Ω: inside the hazard (signed distance < 0) → 0 so
        exp(-α d) stays at full risk; outside, d is the usual Eikonal distance (unobstructed).
        """
        _require_skfmm()
        object_mask = np.asarray(object_mask, dtype=bool)
        if object_mask.shape != self.shape:
            raise ValueError(
                f"object_mask shape {object_mask.shape} does not match grid shape {self.shape}."
            )
        if not np.any(object_mask):
            return np.full(self.shape, np.inf, dtype=np.float64)

        phi = self._phi_from_object_mask(object_mask)
        dist = skfmm.distance(phi, dx=self.res)
        return np.maximum(0.0, np.asarray(dist, dtype=np.float64))

    def compute_geodesic_distance(self, object_mask: np.ndarray, occupancy_grid: np.ndarray) -> np.ndarray:
        """
        Obstacle-aware geodesic distance from ∂Ω (clamped SDF: inside object → 0, outside ≥ 0).
        occupancy_grid: bool array (True = free, False = obstacle).
        """
        _require_skfmm()
        object_mask = np.asarray(object_mask, dtype=bool)
        if object_mask.shape != self.shape:
            raise ValueError(
                f"object_mask shape {object_mask.shape} does not match grid shape {self.shape}."
            )
        if not np.any(object_mask):
            return np.full(self.shape, np.inf, dtype=np.float64)

        free = np.asarray(occupancy_grid, dtype=bool)
        if free.shape != self.shape:
            raise ValueError(
                f"occupancy_grid shape {free.shape} does not match grid shape {self.shape}."
            )

        phi = self._phi_from_object_mask(object_mask)
        
        # THE FIX: Mask out obstacles, BUT guarantee the object itself is unmasked 
        # so the zero-level set (boundary) is visible to the FMM solver.
        obstacle_mask = (~free) & (~object_mask)
        
        phi_masked = np.ma.MaskedArray(phi, mask=obstacle_mask)

        dist = skfmm.distance(phi_masked, dx=self.res)
        
        # Safely handle both MaskedArrays and standard ndarrays
        dist_filled = np.ma.filled(dist, fill_value=np.inf)
        return np.maximum(0.0, np.asarray(dist_filled, dtype=np.float64))
