"""
Fast Marching / Eikonal distances on a 3D voxel workspace.

Core concepts
-------------
- object_mask : occupied voxels belonging to the hazard object Ω
- seed_mask   : free-space voxels immediately adjacent to Ω (boundary seeds S)
- occupancy   : True = free, False = obstacle

Recommended Loop 1 usage
------------------------
1) Build object_mask and occupancy_grid
2) Compute boundary-adjacent seed_mask
3) Compute:
      d_euc = Euclidean distance from seed_mask
      d_geo = obstacle-aware geodesic distance from seed_mask
4) Use shielding_ratio(d_geo, d_euc) later in the risk field

Notes
-----
- Boundary-seeded propagation is preferred over propagating directly from inside
  the object volume, because the risk field should emanate from free space
  adjacent to the hazard geometry.
- We keep the original object-mask-based APIs for compatibility and debugging.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np
from scipy import ndimage

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


def _validate_mask(mask: np.ndarray, shape: Tuple[int, int, int], name: str) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != shape:
        raise ValueError(f"{name} shape {mask.shape} does not match grid shape {shape}.")
    return mask


class WorkspaceGrid:
    """
    3D voxel workspace aligned with world axes.

    Provides:
    - object-boundary Euclidean distance
    - object-boundary geodesic distance
    - boundary-seeded Euclidean distance
    - boundary-seeded geodesic distance
    """

    def __init__(
        self,
        bounds: Tuple[float, float, float, float, float, float] = (-1.0, 1.0, -1.0, 1.0, 0.0, 1.0),
        resolution: float = 0.02,
    ):
        _require_skfmm()
        if resolution <= 0:
            raise ValueError("resolution must be positive.")

        self.res = float(resolution)
        self.bounds = bounds
        xmin, xmax, ymin, ymax, zmin, zmax = bounds

        self.x = np.arange(xmin, xmax, self.res, dtype=np.float64)
        self.y = np.arange(ymin, ymax, self.res, dtype=np.float64)
        self.z = np.arange(zmin, zmax, self.res, dtype=np.float64)

        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing="ij")
        self.shape = self.X.shape
        self.origin = np.array([xmin, ymin, zmin], dtype=np.float64)

    # ---------------------------------------------------------------------
    # Utility helpers
    # ---------------------------------------------------------------------

    def world_to_grid(self, points_xyz: np.ndarray) -> np.ndarray:
        """
        Convert world xyz points to integer voxel indices (i,j,k), clipped to grid bounds.

        points_xyz: (N, 3)
        returns:   (N, 3) int
        """
        pts = np.asarray(points_xyz, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("points_xyz must have shape (N, 3).")

        idx = np.floor((pts - self.origin[None, :]) / self.res).astype(np.int32)
        idx = np.clip(idx, 0, np.array(self.shape, dtype=np.int32) - 1)
        return idx

    def indices_to_mask(self, indices: np.ndarray) -> np.ndarray:
        """
        Convert voxel indices (N,3) into a boolean mask of shape self.shape.
        """
        mask = np.zeros(self.shape, dtype=bool)
        idx = np.asarray(indices, dtype=np.int32)
        if idx.size == 0:
            return mask

        if idx.ndim != 2 or idx.shape[1] != 3:
            raise ValueError("indices must have shape (N, 3).")

        valid = (
            (idx[:, 0] >= 0) & (idx[:, 0] < self.shape[0]) &
            (idx[:, 1] >= 0) & (idx[:, 1] < self.shape[1]) &
            (idx[:, 2] >= 0) & (idx[:, 2] < self.shape[2])
        )
        idx = idx[valid]
        mask[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        return mask

    def empty_mask(self) -> np.ndarray:
        return np.zeros(self.shape, dtype=bool)

    def _phi_from_object_mask(self, object_mask: np.ndarray) -> np.ndarray:
        """
        Signed level set:
          -1 inside object
          +1 outside object
        """
        object_mask = _validate_mask(object_mask, self.shape, "object_mask")
        return np.where(object_mask, -1.0, 1.0).astype(np.float64)

    def _phi_from_seed_mask(self, seed_mask: np.ndarray) -> np.ndarray:
        """
        Zero-set field for seed-based propagation:
          0 at seeds
          1 elsewhere

        This makes the seed voxels the distance source set.
        """
        seed_mask = _validate_mask(seed_mask, self.shape, "seed_mask")
        phi = np.ones(self.shape, dtype=np.float64)
        phi[seed_mask] = 0.0
        return phi

    # ---------------------------------------------------------------------
    # Boundary seed construction
    # ---------------------------------------------------------------------

    def compute_boundary_seed_mask(
        self,
        object_mask: np.ndarray,
        occupancy_grid: np.ndarray,
        connectivity: int = 1,
    ) -> np.ndarray:
        """
        Build free-space boundary seeds S:
        free voxels immediately adjacent to the object volume.

        object_mask    : True on hazard object voxels Ω
        occupancy_grid : True = free, False = obstacle
        connectivity   : 1 => 6-neighborhood, 2 => 18, 3 => 26
        """
        object_mask = _validate_mask(object_mask, self.shape, "object_mask")
        free = _validate_mask(occupancy_grid, self.shape, "occupancy_grid")

        if not np.any(object_mask):
            return np.zeros(self.shape, dtype=bool)

        struct = ndimage.generate_binary_structure(rank=3, connectivity=connectivity)
        dilated = ndimage.binary_dilation(object_mask, structure=struct)

        # Seeds must be:
        # - adjacent to object
        # - not inside object
        # - in free space
        seed_mask = dilated & (~object_mask) & free
        return seed_mask

    # ---------------------------------------------------------------------
    # Object-mask-based distances (kept for compatibility / debugging)
    # ---------------------------------------------------------------------

    def compute_euclidean_distance(self, object_mask: np.ndarray) -> np.ndarray:
        """
        Euclidean distance from the physical object boundary ∂Ω.
        Inside the object, distance is clamped to 0.
        """
        _require_skfmm()
        object_mask = _validate_mask(object_mask, self.shape, "object_mask")

        if not np.any(object_mask):
            return np.full(self.shape, np.inf, dtype=np.float64)

        phi = self._phi_from_object_mask(object_mask)
        dist = skfmm.distance(phi, dx=self.res)
        return np.maximum(0.0, np.asarray(dist, dtype=np.float64))

    def compute_geodesic_distance(
        self,
        object_mask: np.ndarray,
        occupancy_grid: np.ndarray,
    ) -> np.ndarray:
        """
        Obstacle-aware geodesic distance from the physical object boundary ∂Ω.
        Inside the object, distance is clamped to 0.

        occupancy_grid: True = free, False = obstacle
        """
        _require_skfmm()
        object_mask = _validate_mask(object_mask, self.shape, "object_mask")
        free = _validate_mask(occupancy_grid, self.shape, "occupancy_grid")

        if not np.any(object_mask):
            return np.full(self.shape, np.inf, dtype=np.float64)

        phi = self._phi_from_object_mask(object_mask)

        # Obstacles are masked out, but the object itself must remain visible
        # to the FMM solver so that ∂Ω exists as a source boundary.
        obstacle_mask = (~free) & (~object_mask)
        phi_masked = np.ma.MaskedArray(phi, mask=obstacle_mask)

        dist = skfmm.distance(phi_masked, dx=self.res)
        dist_filled = np.ma.filled(dist, fill_value=np.inf)
        return np.maximum(0.0, np.asarray(dist_filled, dtype=np.float64))

    # ---------------------------------------------------------------------
    # Preferred seed-based distances for Loop 1
    # ---------------------------------------------------------------------

    def compute_seeded_euclidean_distance(self, seed_mask: np.ndarray) -> np.ndarray:
        """
        Euclidean distance to the nearest boundary seed voxel.

        Uses exact Euclidean distance transform on the grid.
        """
        seed_mask = _validate_mask(seed_mask, self.shape, "seed_mask")

        if not np.any(seed_mask):
            return np.full(self.shape, np.inf, dtype=np.float64)

        # distance_transform_edt computes distance to the nearest zero.
        # So we pass ~seed_mask, making seed voxels the zeros.
        dist = ndimage.distance_transform_edt(~seed_mask, sampling=(self.res, self.res, self.res))
        return dist.astype(np.float64)

    def compute_seeded_geodesic_distance(
        self,
        seed_mask: np.ndarray,
        occupancy_grid: np.ndarray,
    ) -> np.ndarray:
        """
        Obstacle-aware geodesic distance from free-space boundary seeds.

        seed_mask      : True on source voxels S (must lie in free space)
        occupancy_grid : True = free, False = obstacle

        Unreachable voxels are filled with +inf.
        """
        _require_skfmm()
        seed_mask = _validate_mask(seed_mask, self.shape, "seed_mask")
        free = _validate_mask(occupancy_grid, self.shape, "occupancy_grid")

        if not np.any(seed_mask):
            return np.full(self.shape, np.inf, dtype=np.float64)

        # Enforce seeds in free space
        if np.any(seed_mask & (~free)):
            raise ValueError("seed_mask contains voxels that are not free in occupancy_grid.")

        phi = self._phi_from_seed_mask(seed_mask)

        # Obstacles masked out, seeds must remain unmasked
        obstacle_mask = (~free) & (~seed_mask)
        phi_masked = np.ma.MaskedArray(phi, mask=obstacle_mask)

        dist = skfmm.distance(phi_masked, dx=self.res)
        dist_filled = np.ma.filled(dist, fill_value=np.inf)

        # Numerical guard: seed voxels should be exactly zero
        dist_filled[seed_mask] = 0.0
        return np.maximum(0.0, np.asarray(dist_filled, dtype=np.float64))

    def compute_boundary_seeded_distances(
        self,
        object_mask: np.ndarray,
        occupancy_grid: np.ndarray,
        connectivity: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Preferred Loop 1 API.

        Given:
          - object_mask Ω
          - occupancy_grid (True free / False obstacle)

        Returns:
          - d_euc  : Euclidean distance from boundary-adjacent free-space seeds S
          - d_geo  : obstacle-aware geodesic distance from S
          - seeds  : boolean seed mask S
        """
        seed_mask = self.compute_boundary_seed_mask(
            object_mask=object_mask,
            occupancy_grid=occupancy_grid,
            connectivity=connectivity,
        )

        d_euc = self.compute_seeded_euclidean_distance(seed_mask)
        d_geo = self.compute_seeded_geodesic_distance(seed_mask, occupancy_grid)
        return d_euc, d_geo, seed_mask