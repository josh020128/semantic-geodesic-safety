from __future__ import annotations

from typing import Dict, Tuple
from scipy import ndimage
import numpy as np

from semantic_safety.metric_propagation.fmm_distance import WorkspaceGrid
from semantic_safety.risk_field.interpolation import compute_directional_weights
from semantic_safety.risk_field.superposition import risk_cost_field


BBox = Tuple[float, float, float, float, float, float]


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def surface_mask_to_footprint(surface_mask: np.ndarray) -> np.ndarray:
    """
    Convert a 3D surface mask to a 2D (x, y) footprint.
    """
    if surface_mask is None:
        return None
    return np.any(surface_mask, axis=2)


def bbox_to_footprint_mask(grid, bbox):
    """
    Fallback 2D footprint mask from bbox.
    """
    xmin, xmax, ymin, ymax, _, _ = bbox
    return (
        (grid.X[:, :, 0] >= xmin) & (grid.X[:, :, 0] <= xmax) &
        (grid.Y[:, :, 0] >= ymin) & (grid.Y[:, :, 0] <= ymax)
    )


def gravity_column_from_surface_mask(
    grid,
    top_surface_mask: np.ndarray,
    A_field: np.ndarray,
    base_risk: float,
    w_plus_z: float,
    lateral_decay: str = "moderate",
) -> np.ndarray:
    """
    Gravity-aware upward field from the full top surface footprint.

    - Risk stays high anywhere above the object's top surface footprint
    - No artificial center-only +z column
    - Lateral decay happens in x/y away from the footprint
    """
    if top_surface_mask is None or not np.any(top_surface_mask):
        return np.zeros(grid.shape, dtype=np.float64)

    footprint = np.any(top_surface_mask, axis=2)
    if not np.any(footprint):
        return np.zeros(grid.shape, dtype=np.float64)

    # 2D xy distance to nearest footprint cell
    d_xy = ndimage.distance_transform_edt(~footprint) * grid.res

    # top z value for each (x, y)
    z_top_field = np.full(grid.shape[:2], -np.inf, dtype=np.float64)
    nx, ny, _ = top_surface_mask.shape

    for i in range(nx):
        for j in range(ny):
            z_idx = np.where(top_surface_mask[i, j, :])[0]
            if len(z_idx) > 0:
                z_top_field[i, j] = grid.z[int(z_idx.max())]

    z_top_3d = z_top_field[:, :, None]
    above_mask = footprint[:, :, None] & (grid.Z >= z_top_3d)

    lateral_alpha_map = {
        "narrow": 18.0,
        "moderate": 10.0,
        "wide": 5.0,
    }
    alpha_xy = lateral_alpha_map.get(lateral_decay, 10.0)

    d_xy_3d = d_xy[:, :, None]
    V = base_risk * float(w_plus_z) * A_field * np.exp(-alpha_xy * d_xy_3d)
    V *= above_mask.astype(np.float64)
    V[np.isnan(V)] = 0.0

    return V

def _default_weights(weights_dict: Dict[str, float] | None) -> Dict[str, float]:
    if not weights_dict:
        return {
            "w_+x": 1.0,
            "w_-x": 1.0,
            "w_+y": 1.0,
            "w_-y": 1.0,
            "w_+z": 1.0,
            "w_-z": 1.0,
        }
    return weights_dict


def _get_lateral_alpha(lateral_decay: str) -> float:
    return {
        "narrow": 22.0,
        "moderate": 14.0,
        "wide": 7.0,
    }.get(lateral_decay, 14.0)


def _get_standard_alpha(topology_template: str, risk_params: Dict) -> float:
    if "alpha" in risk_params:
        return float(risk_params["alpha"])

    defaults = {
        "upward_vertical_cone": 14.0,
        "isotropic_sphere": 8.0,
        "forward_directional_cone": 14.0,
        "planar_half_space": 6.0,
    }
    return defaults.get(topology_template, 10.0)


def _get_vertical_extent_m(risk_params: Dict) -> float:
    """
    Max useful height for gravity-dominant vertical hazard.
    Tighter default than before so the field does not fill too much of the volume.
    """
    return float(risk_params.get("vertical_extent_m", 0.22))


def _smooth_plateau_taper(z_rel: np.ndarray, extent_m: float, plateau_frac: float = 0.45) -> np.ndarray:
    """
    Returns a vertical gate in [0, 1]:
    - 0 below the object top
    - 1 for a short plateau above it
    - smooth cosine taper to 0 by extent_m
    """
    gate = np.zeros_like(z_rel, dtype=np.float64)

    if extent_m <= 1e-8:
        return gate

    plateau_m = plateau_frac * extent_m
    taper_m = max(extent_m - plateau_m, 1e-8)

    plateau_mask = (z_rel >= 0.0) & (z_rel <= plateau_m)
    gate[plateau_mask] = 1.0

    taper_mask = (z_rel > plateau_m) & (z_rel <= extent_m)
    t = (z_rel[taper_mask] - plateau_m) / taper_m
    gate[taper_mask] = 0.5 * (1.0 + np.cos(np.pi * t))

    return gate


def _bbox_footprint_distance_xy(grid: WorkspaceGrid, bbox: BBox) -> np.ndarray:
    xmin, xmax, ymin, ymax, _, _ = bbox

    dx = np.maximum(0.0, np.maximum(xmin - grid.X, grid.X - xmax))
    dy = np.maximum(0.0, np.maximum(ymin - grid.Y, grid.Y - ymax))
    return np.sqrt(dx * dx + dy * dy)


def _dominant_halfspace_signed_distance(grid: WorkspaceGrid, bbox: BBox, weights_dict: Dict[str, float]) -> np.ndarray:
    """
    Returns signed distance from the dominant bbox face.
    Positive means 'on the risky side' of that plane.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = bbox

    candidates = {
        "w_+x": grid.X - xmax,
        "w_-x": xmin - grid.X,
        "w_+y": grid.Y - ymax,
        "w_-y": ymin - grid.Y,
        "w_+z": grid.Z - zmax,
        "w_-z": zmin - grid.Z,
    }

    dominant_key = max(candidates.keys(), key=lambda k: float(weights_dict.get(k, 0.0)))
    return candidates[dominant_key]


# ---------------------------------------------------------------------
# Core field components
# ---------------------------------------------------------------------

def build_standard_decay_component(
    grid: WorkspaceGrid,
    bbox: BBox,
    weights_dict: Dict[str, float],
    d_geo: np.ndarray,
    A_field: np.ndarray,
    base_risk: float,
    alpha: float,
) -> np.ndarray:
    """
    Generic directionally weighted exponential decay field.
    """
    W_field = compute_directional_weights(grid.X, grid.Y, grid.Z, bbox, weights_dict)
    return risk_cost_field(
        W_hazard=W_field,
        A=A_field,
        d_geo=d_geo,
        alpha=alpha,
        base_risk=base_risk,
    )


def build_gravity_column_component(
    grid: WorkspaceGrid,
    bbox: BBox,
    A_field: np.ndarray,
    base_risk: float,
    w_plus_z: float,
    lateral_decay: str = "moderate",
    vertical_extent_m: float = 0.22,
    plateau_frac: float = 0.45,
) -> np.ndarray:
    """
    Gravity-aware upward field.

    Properties:
    - Only active above the object top face
    - Strong near the object's x/y footprint
    - Little/no vertical decay at first
    - Soft taper to zero by vertical_extent_m
    """
    _, _, _, _, _, zmax = bbox

    d_xy = _bbox_footprint_distance_xy(grid, bbox)
    z_rel = grid.Z - zmax

    alpha_xy = _get_lateral_alpha(lateral_decay)
    lateral_term = np.exp(-alpha_xy * d_xy)

    vertical_gate = _smooth_plateau_taper(
        z_rel=z_rel,
        extent_m=vertical_extent_m,
        plateau_frac=plateau_frac,
    )

    V_gravity = base_risk * float(w_plus_z) * A_field * lateral_term * vertical_gate
    return V_gravity


def build_planar_half_space_component(
    grid: WorkspaceGrid,
    bbox: BBox,
    weights_dict: Dict[str, float],
    A_field: np.ndarray,
    base_risk: float,
    risk_params: Dict,
) -> np.ndarray:
    """
    One-sided field extending from the dominant bbox face.
    Useful for surface-based restricted zones or side-specific hazards.
    """
    signed_dist = _dominant_halfspace_signed_distance(grid, bbox, weights_dict)
    side_gate = (signed_dist >= 0.0).astype(np.float64)

    alpha_plane = float(risk_params.get("planar_alpha", 4.0))
    normal_decay = np.exp(-alpha_plane * np.maximum(0.0, signed_dist))

    dominant_weight = max(float(v) for v in weights_dict.values()) if weights_dict else 1.0

    V_halfspace = base_risk * dominant_weight * A_field * normal_decay * side_gate
    return V_halfspace


# ---------------------------------------------------------------------
# Template builders
# ---------------------------------------------------------------------

def build_upward_vertical_cone_field(
    grid,
    bbox,
    object_mask,
    surface_mask,
    footprint_mask,
    risk_params,
    d_geo,
    A_field,
    base_risk,
):
    weights = risk_params.get("weights", {})
    vertical_rule = risk_params.get("vertical_rule", "standard_decay")
    lateral_decay = risk_params.get("lateral_decay", "moderate")

    # Standard directional field
    W_field = compute_directional_weights(
        X=grid.X,
        Y=grid.Y,
        Z=grid.Z,
        bbox=bbox,
        weights_dict=weights,
    )

    alpha_map = {"narrow": 14.0, "moderate": 10.0, "wide": 6.0}
    alpha = alpha_map.get(lateral_decay, 10.0)

    V = base_risk * W_field * A_field * np.exp(-alpha * np.clip(d_geo, 0, None))

    # Special gravity-aware vertical behavior
    if vertical_rule == "gravity_column":
        if footprint_mask is None and surface_mask is not None:
            footprint_mask = surface_mask_to_footprint(surface_mask)

        if surface_mask is None and object_mask is not None:
            # fallback: if top surface wasn't explicitly given, use object mask itself
            surface_mask = object_mask

        if surface_mask is not None and np.any(surface_mask):
            V_gravity = gravity_column_from_surface_mask(
                grid=grid,
                top_surface_mask=surface_mask,
                A_field=A_field,
                base_risk=base_risk,
                w_plus_z=float(weights.get("w_+z", 0.0)),
                lateral_decay=lateral_decay,
            )
            V = np.maximum(V, V_gravity)

    return V

def build_isotropic_sphere_field(
    grid: WorkspaceGrid,
    bbox: BBox,
    risk_params: Dict,
    d_geo: np.ndarray,
    A_field: np.ndarray,
    base_risk: float,
    object_mask=None,
    surface_mask=None,
    footprint_mask=None,
) -> np.ndarray:
    weights_dict = _default_weights(risk_params.get("weights", {}))
    alpha = _get_standard_alpha("isotropic_sphere", risk_params)

    return build_standard_decay_component(
        grid=grid,
        bbox=bbox,
        weights_dict=weights_dict,
        d_geo=d_geo,
        A_field=A_field,
        base_risk=base_risk,
        alpha=alpha,
    )


def build_forward_directional_cone_field(
    grid: WorkspaceGrid,
    bbox: BBox,
    risk_params: Dict,
    d_geo: np.ndarray,
    A_field: np.ndarray,
    base_risk: float,
    object_mask=None,
    surface_mask=None,
    footprint_mask=None,
) -> np.ndarray:
    """
    Directional cone:
    - uses directional interpolation
    - sharpens the directional multiplier to narrow the cone
    """
    weights_dict = _default_weights(risk_params.get("weights", {}))
    alpha = _get_standard_alpha("forward_directional_cone", risk_params)
    cone_power = float(risk_params.get("directional_power", 1.8))

    W_field = compute_directional_weights(grid.X, grid.Y, grid.Z, bbox, weights_dict)
    W_field = np.power(np.clip(W_field, 0.0, 1.0), cone_power)

    return risk_cost_field(
        W_hazard=W_field,
        A=A_field,
        d_geo=d_geo,
        alpha=alpha,
        base_risk=base_risk,
    )


def build_planar_half_space_field(
    grid: WorkspaceGrid,
    bbox: BBox,
    risk_params: Dict,
    d_geo: np.ndarray,
    A_field: np.ndarray,
    base_risk: float,
    object_mask=None,
    surface_mask=None,
    footprint_mask=None,
) -> np.ndarray:
    """
    Planar half-space field:
    - standard decay term
    - plus one-sided half-space component

    Internal combination uses max(), not LogSumExp, to avoid a nonzero baseline.
    """
    weights_dict = _default_weights(risk_params.get("weights", {}))
    alpha = _get_standard_alpha("planar_half_space", risk_params)

    V_decay = build_standard_decay_component(
        grid=grid,
        bbox=bbox,
        weights_dict=weights_dict,
        d_geo=d_geo,
        A_field=A_field,
        base_risk=base_risk,
        alpha=alpha,
    )

    V_halfspace = build_planar_half_space_component(
        grid=grid,
        bbox=bbox,
        weights_dict=weights_dict,
        A_field=A_field,
        base_risk=base_risk,
        risk_params=risk_params,
    )

    return np.maximum(V_decay, V_halfspace)


# ---------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------

def build_risk_field_from_params(
    grid,
    risk_params,
    d_geo,
    A_field,
    base_risk,
    bbox=None,
    object_mask=None,
    surface_mask=None,
    footprint_mask=None,
):
    topology = risk_params.get("topology_template", "isotropic_sphere")

    if topology == "upward_vertical_cone":
        return build_upward_vertical_cone_field(
            grid=grid,
            bbox=bbox,
            object_mask=object_mask,
            surface_mask=surface_mask,
            footprint_mask=footprint_mask,
            risk_params=risk_params,
            d_geo=d_geo,
            A_field=A_field,
            base_risk=base_risk,
        )

    if topology == "forward_directional_cone":
        return build_forward_directional_cone_field(
            grid=grid,
            bbox=bbox,
            object_mask=object_mask,
            surface_mask=surface_mask,
            footprint_mask=footprint_mask,
            risk_params=risk_params,
            d_geo=d_geo,
            A_field=A_field,
            base_risk=base_risk,
        )

    if topology == "planar_half_space":
        return build_planar_half_space_field(
            grid=grid,
            bbox=bbox,
            object_mask=object_mask,
            surface_mask=surface_mask,
            footprint_mask=footprint_mask,
            risk_params=risk_params,
            d_geo=d_geo,
            A_field=A_field,
            base_risk=base_risk,
        )

    return build_isotropic_sphere_field(
        grid=grid,
        bbox=bbox,
        object_mask=object_mask,
        surface_mask=surface_mask,
        footprint_mask=footprint_mask,
        risk_params=risk_params,
        d_geo=d_geo,
        A_field=A_field,
        base_risk=base_risk,
    )