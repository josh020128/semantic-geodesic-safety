from __future__ import annotations

from typing import Dict, Optional, Set, Tuple

import numpy as np

from semantic_safety.metric_propagation.fmm_distance import WorkspaceGrid


BBox = Tuple[float, float, float, float, float, float]

AXIS_KEYS = ("w_+x", "w_-x", "w_+y", "w_-y", "w_+z", "w_-z")


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def _empty_field(grid: WorkspaceGrid, dtype=np.float64) -> np.ndarray:
    return np.zeros(grid.shape, dtype=dtype)


def _default_weights(weights_dict: Optional[Dict]) -> Dict:
    if not weights_dict:
        return {k: 0.0 for k in AXIS_KEYS}
    out = {k: weights_dict.get(k, 0.0) for k in AXIS_KEYS}
    return out


def split_directional_weights(weights_dict: Optional[Dict]) -> tuple[Dict[str, float], Set[str]]:
    """
    Split JSON weights into:
      - finite_weights: numeric weights in [0, 1]
      - inf_dirs: directions marked as "inf"

    Assumes the caller has already validated schema as needed.
    """
    weights_dict = _default_weights(weights_dict)

    finite_weights: Dict[str, float] = {}
    inf_dirs: Set[str] = set()

    for k in AXIS_KEYS:
        v = weights_dict.get(k, 0.0)

        if isinstance(v, str):
            if v.lower() == "inf":
                inf_dirs.add(k)
                finite_weights[k] = 0.0
            else:
                raise ValueError(f"Unsupported string weight for {k}: {v}")
        elif np.isinf(v):
            inf_dirs.add(k)
            finite_weights[k] = 0.0
        else:
            finite_weights[k] = float(np.clip(v, 0.0, 1.0))

    return finite_weights, inf_dirs


def has_any_active_semantic_direction(weights_dict: Optional[Dict]) -> bool:
    finite_weights, inf_dirs = split_directional_weights(weights_dict)
    if inf_dirs:
        return True
    return any(v > 1e-8 for v in finite_weights.values())


def _bbox_center_xyz(bbox: BBox) -> np.ndarray:
    xmin, xmax, ymin, ymax, zmin, zmax = bbox
    return np.array(
        [
            0.5 * (xmin + xmax),
            0.5 * (ymin + ymax),
            0.5 * (zmin + zmax),
        ],
        dtype=np.float64,
    )


def _surface_center_xyz(grid: WorkspaceGrid, surface_mask: np.ndarray) -> Optional[np.ndarray]:
    if surface_mask is None or not np.any(surface_mask):
        return None

    idx = np.argwhere(surface_mask)
    if len(idx) == 0:
        return None

    return np.array(
        [
            float(grid.x[idx[:, 0]].mean()),
            float(grid.y[idx[:, 1]].mean()),
            float(grid.z[idx[:, 2]].mean()),
        ],
        dtype=np.float64,
    )


def _choose_source_anchor_xyz(
    grid: WorkspaceGrid,
    bbox: Optional[BBox],
    surface_mask: Optional[np.ndarray],
    source_mask: Optional[np.ndarray],
) -> np.ndarray:
    """
    Simple fallback anchor for debug / optional use.
    The main directional logic below uses nearest-source-point style
    approximations, so this anchor is not the core of the method.
    """
    c = None

    if surface_mask is not None and np.any(surface_mask):
        c = _surface_center_xyz(grid, surface_mask)
    elif source_mask is not None and np.any(source_mask):
        c = _surface_center_xyz(grid, source_mask)

    if c is not None:
        return c

    if bbox is not None:
        return _bbox_center_xyz(bbox)

    return np.zeros(3, dtype=np.float64)


def _get_sigma_m(risk_params: Dict) -> float:
    """
    Current schema uses sigma_m only.
    If absent, fall back to a conservative default for robustness.
    """
    sigma_m = float(risk_params.get("sigma_m", 0.10))
    return max(sigma_m, 1e-6)


# ---------------------------------------------------------------------
# Source geometry / directional decomposition
# ---------------------------------------------------------------------

def _build_source_points_world(
    grid: WorkspaceGrid,
    source_mask: Optional[np.ndarray],
    surface_mask: Optional[np.ndarray],
) -> np.ndarray:
    """
    Return world coordinates of source voxels used for directional reference.

    Preference:
      1) surface_mask if available
      2) source_mask otherwise

    Output shape: (M, 3)
    """
    mask = None
    if surface_mask is not None and np.any(surface_mask):
        mask = surface_mask
    elif source_mask is not None and np.any(source_mask):
        mask = source_mask

    if mask is None or not np.any(mask):
        return np.zeros((0, 3), dtype=np.float64)

    idx = np.argwhere(mask)
    pts = np.stack(
        [
            grid.x[idx[:, 0]],
            grid.y[idx[:, 1]],
            grid.z[idx[:, 2]],
        ],
        axis=1,
    ).astype(np.float64)
    return pts


def _nearest_source_offset_fields(
    grid: WorkspaceGrid,
    source_mask: Optional[np.ndarray],
    surface_mask: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Approximate nearest-source-point offsets for every voxel.

    Returns:
      dx, dy, dz  with same shape as grid

    Implementation note:
    - To keep the skeleton simple and robust, this version uses a Euclidean
      nearest-source assignment via distance transform indices on the chosen
      source mask.
    - That is good enough for directional decomposition; geodesic effects are
      already handled separately through d_geo and A_field.
    """
    chosen_mask = None
    if surface_mask is not None and np.any(surface_mask):
        chosen_mask = surface_mask
    elif source_mask is not None and np.any(source_mask):
        chosen_mask = source_mask

    if chosen_mask is None or not np.any(chosen_mask):
        return (
            np.zeros(grid.shape, dtype=np.float64),
            np.zeros(grid.shape, dtype=np.float64),
            np.zeros(grid.shape, dtype=np.float64),
        )

    # distance transform over non-source voxels returns nearest source indices
    _, nearest_idx = _distance_transform_to_source(chosen_mask)

    ni = nearest_idx[0]
    nj = nearest_idx[1]
    nk = nearest_idx[2]

    nearest_x = grid.x[ni]
    nearest_y = grid.y[nj]
    nearest_z = grid.z[nk]

    dx = grid.X - nearest_x
    dy = grid.Y - nearest_y
    dz = grid.Z - nearest_z
    return dx, dy, dz


def _distance_transform_to_source(source_mask: np.ndarray):
    """
    Helper wrapper so the main logic reads clearly.
    """
    from scipy import ndimage

    return ndimage.distance_transform_edt(~source_mask, return_indices=True)


def compute_directional_coefficients(
    grid: WorkspaceGrid,
    source_mask: Optional[np.ndarray],
    surface_mask: Optional[np.ndarray],
    eps: float = 1e-6,
) -> Dict[str, np.ndarray]:
    """
    Build the six local directional coefficient fields:
      a_{+x}, a_{-x}, a_{+y}, a_{-y}, a_{+z}, a_{-z}

    These sum approximately to 1 away from the source and are all zero on
    source voxels (up to eps behavior).
    """
    dx, dy, dz = _nearest_source_offset_fields(
        grid=grid,
        source_mask=source_mask,
        surface_mask=surface_mask,
    )

    l_px = np.maximum(dx, 0.0)
    l_nx = np.maximum(-dx, 0.0)
    l_py = np.maximum(dy, 0.0)
    l_ny = np.maximum(-dy, 0.0)
    l_pz = np.maximum(dz, 0.0)
    l_nz = np.maximum(-dz, 0.0)

    denom = l_px + l_nx + l_py + l_ny + l_pz + l_nz
    denom = np.maximum(denom, eps)

    coeffs = {
        "w_+x": l_px / denom,
        "w_-x": l_nx / denom,
        "w_+y": l_py / denom,
        "w_-y": l_ny / denom,
        "w_+z": l_pz / denom,
        "w_-z": l_nz / denom,
    }

    return coeffs


def build_directional_weight_field(
    directional_coeffs: Dict[str, np.ndarray],
    finite_weights: Dict[str, float],
) -> np.ndarray:
    """
    W_i(x) = sum_d a_{i,d}(x) * w_{i,d}, using only finite weights.
    """
    W = np.zeros_like(next(iter(directional_coeffs.values())), dtype=np.float64)
    for k in AXIS_KEYS:
        W += directional_coeffs[k] * float(finite_weights.get(k, 0.0))
    W[~np.isfinite(W)] = 0.0
    return W


# ---------------------------------------------------------------------
# Gaussian decay / hard semantic exclusion
# ---------------------------------------------------------------------

def build_geodesic_gaussian_decay(
    d_geo: np.ndarray,
    sigma_m: float,
) -> np.ndarray:
    """
    exp( - d_geo^2 / (2 sigma^2) )
    """
    sigma = max(float(sigma_m), 1e-6)
    d = np.clip(d_geo.astype(np.float64), 0.0, None)
    G = np.exp(-(d ** 2) / (2.0 * sigma ** 2))
    G[~np.isfinite(G)] = 0.0
    return G

# ---------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------

def build_risk_field_from_params(
    grid: WorkspaceGrid,
    risk_params: Dict,
    d_geo: np.ndarray,
    A_field: np.ndarray,
    bbox: Optional[BBox] = None,
    source_mask: Optional[np.ndarray] = None,
    surface_mask: Optional[np.ndarray] = None,
    hard_tau: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Current minimal runtime builder.

    Input schema:
      risk_params = {
        "weights": {...},
        "sigma_m": float,
      }

    Returns:
      V_soft: float field
      H_inf: bool hard semantic exclusion mask
      debug: optional intermediate fields for debugging / visualization
    """
    weights_dict = _default_weights(risk_params.get("weights", {}))
    sigma_m = _get_sigma_m(risk_params)

    finite_weights, inf_dirs = split_directional_weights(weights_dict)

    directional_coeffs = compute_directional_coefficients(
        grid=grid,
        source_mask=source_mask,
        surface_mask=surface_mask,
    )

    W_field = build_directional_weight_field(
        directional_coeffs=directional_coeffs,
        finite_weights=finite_weights,
    )

    gaussian_decay = build_geodesic_gaussian_decay(
        d_geo=d_geo,
        sigma_m=sigma_m,
    )

    V_soft = W_field * A_field * gaussian_decay
    V_soft[~np.isfinite(V_soft)] = 0.0

    # Hard semantic mask for any "inf" direction.
    # Interpretation: if a voxel is sufficiently aligned with an inf direction,
    # it is considered semantically forbidden (no-go).
    H_inf = np.zeros(grid.shape, dtype=bool)
    if inf_dirs:
        tau = float(np.clip(hard_tau, 0.0, 1.0))
        for k in inf_dirs:
            H_inf |= directional_coeffs[k] >= tau

    debug = {
        "directional_coeffs": directional_coeffs,
        "W_field": W_field,
        "gaussian_decay": gaussian_decay,
        "finite_weights": finite_weights,
        "inf_dirs": np.array(sorted(list(inf_dirs)), dtype=object),
        "source_anchor_xyz": _choose_source_anchor_xyz(
            grid=grid,
            bbox=bbox,
            surface_mask=surface_mask,
            source_mask=source_mask,
        ),
    }

    return V_soft, H_inf, debug