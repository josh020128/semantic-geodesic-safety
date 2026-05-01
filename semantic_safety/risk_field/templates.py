from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy import ndimage

from semantic_safety.metric_propagation.fmm_distance import WorkspaceGrid


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
        "upward_vertical_cone": 8.0,
        "isotropic_sphere": 7.0,
        "forward_directional_cone": 8.0,
        "planar_half_space": 6.0,
    }
    return defaults.get(topology_template, 8.0)


def _get_vertical_extent_m(risk_params: Dict) -> float:
    """
    Max useful height for gravity-dominant vertical hazard.
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
# New support-shape helpers for general standard decay
# ---------------------------------------------------------------------

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


def _surface_center_xyz(grid: WorkspaceGrid, surface_mask: np.ndarray) -> np.ndarray | None:
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


def _surface_top_center_xyz(
    grid: WorkspaceGrid,
    surface_mask: np.ndarray,
    bbox: BBox,
) -> np.ndarray:
    c = _surface_center_xyz(grid, surface_mask)
    if c is None:
        c = _bbox_center_xyz(bbox)
        c[2] = bbox[5]
        return c

    idx = np.argwhere(surface_mask)
    c[2] = float(grid.z[idx[:, 2]].max())
    return c


def _choose_standard_decay_anchor(
    grid: WorkspaceGrid,
    bbox: BBox,
    surface_mask: np.ndarray,
    risk_params: Dict,
    topology_template: str,
) -> np.ndarray:
    anchor_mode = str(risk_params.get("anchor_mode", "auto")).lower()

    if anchor_mode == "bbox_center":
        return _bbox_center_xyz(bbox)

    if anchor_mode == "top_surface":
        return _surface_top_center_xyz(grid, surface_mask, bbox)

    # auto
    if topology_template == "upward_vertical_cone":
        return _surface_top_center_xyz(grid, surface_mask, bbox)

    return _bbox_center_xyz(bbox)


def _get_standard_radius_m(
    bbox: BBox,
    topology_template: str,
    risk_params: Dict,
) -> float:
    if "radius_m" in risk_params:
        return float(risk_params["radius_m"])

    xmin, xmax, ymin, ymax, zmin, zmax = bbox
    dx = float(xmax - xmin)
    dy = float(ymax - ymin)
    dz = float(zmax - zmin)

    horiz_half = 0.5 * max(dx, dy)
    diag_half = 0.5 * np.sqrt(dx * dx + dy * dy + dz * dz)

    defaults = {
        "isotropic_sphere": max(0.06, 0.90 * diag_half),
        "upward_vertical_cone": max(0.08, 0.95 * horiz_half),
        "forward_directional_cone": max(0.10, 1.05 * horiz_half),
        "planar_half_space": max(0.08, 0.95 * horiz_half),
    }
    return float(defaults.get(topology_template, max(0.10, 1.20 * diag_half)))


def _get_directional_gamma(
    topology_template: str,
    risk_params: Dict,
) -> float:
    if "directional_gamma" in risk_params:
        return float(risk_params["directional_gamma"])

    defaults = {
        "isotropic_sphere": 1.0,
        "upward_vertical_cone": 1.1,
        "forward_directional_cone": 1.8,
        "planar_half_space": 1.1,
    }
    return float(defaults.get(topology_template, 1.0))


def _directional_radius_multiplier(
    dx: np.ndarray,
    dy: np.ndarray,
    dz: np.ndarray,
    weights_dict: Dict[str, float],
    gamma: float,
) -> np.ndarray:
    l_px = np.maximum(dx, 0.0)
    l_nx = np.maximum(-dx, 0.0)
    l_py = np.maximum(dy, 0.0)
    l_ny = np.maximum(-dy, 0.0)
    l_pz = np.maximum(dz, 0.0)
    l_nz = np.maximum(-dz, 0.0)

    denom = l_px + l_nx + l_py + l_ny + l_pz + l_nz
    denom = np.maximum(denom, 1e-8)

    mix = (
        (l_px / denom) * float(weights_dict.get("w_+x", 0.0)) +
        (l_nx / denom) * float(weights_dict.get("w_-x", 0.0)) +
        (l_py / denom) * float(weights_dict.get("w_+y", 0.0)) +
        (l_ny / denom) * float(weights_dict.get("w_-y", 0.0)) +
        (l_pz / denom) * float(weights_dict.get("w_+z", 0.0)) +
        (l_nz / denom) * float(weights_dict.get("w_-z", 0.0))
    )

    mix = np.clip(mix, 0.0, None)
    if abs(gamma - 1.0) > 1e-8:
        mix = np.power(mix, gamma)

    return mix


def _directional_support_distance(
    grid: WorkspaceGrid,
    anchor_xyz: np.ndarray,
    weights_dict: Dict[str, float],
    radius_m: float,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    dx = grid.X - float(anchor_xyz[0])
    dy = grid.Y - float(anchor_xyz[1])
    dz = grid.Z - float(anchor_xyz[2])

    r = np.sqrt(dx * dx + dy * dy + dz * dz)
    radius_multiplier = _directional_radius_multiplier(dx, dy, dz, weights_dict, gamma)
    R = float(radius_m) * radius_multiplier

    d_support = np.maximum(0.0, r - R)
    return d_support, R


def _standard_decay_vertical_gate(
    grid: WorkspaceGrid,
    anchor_xyz: np.ndarray,
    weights_dict: Dict[str, float],
    risk_params: Dict,
) -> np.ndarray:
    gate_mode = str(risk_params.get("vertical_gate", "auto")).lower()

    if gate_mode == "none":
        return np.ones(grid.shape, dtype=np.float64)

    if gate_mode == "above_anchor":
        return (grid.Z >= float(anchor_xyz[2])).astype(np.float64)

    # auto:
    # if downward weight is effectively zero and upward weight is present,
    # use an upward hemisphere/dome gate
    if float(weights_dict.get("w_-z", 0.0)) <= 1e-8 and float(weights_dict.get("w_+z", 0.0)) > 0.0:
        return (grid.Z >= float(anchor_xyz[2])).astype(np.float64)

    return np.ones(grid.shape, dtype=np.float64)

def _get_radius_tail_fraction(risk_params: Dict) -> float:
    """
    At r = directional radius R, what fraction of peak should remain?
    Example:
      0.05 -> 5% remains at r = R
      0.02 -> 2% remains at r = R (faster decay)
    """
    tau = float(risk_params.get("radius_tail_fraction", 0.05))
    return float(np.clip(tau, 1e-4, 0.5))

def _get_radius_decay_k(risk_params: Dict) -> float:
    """
    If alpha is explicitly provided, respect it.
    Otherwise derive a dimensionless decay constant from radius_tail_fraction.
    """
    if "alpha" in risk_params:
        return float(risk_params["alpha"])

    tau = _get_radius_tail_fraction(risk_params)
    return float(-np.log(tau))

def _get_radius_cutoff_multiple(risk_params: Dict) -> float:
    """
    Optional soft/hard cutoff beyond the directional radius.
    Example:
      1.5 -> show until ~1.5 * R
      2.0 -> show until ~2.0 * R
    """
    return float(risk_params.get("radius_cutoff_multiple", 1.5))
# ---------------------------------------------------------------------
# Core field components
# ---------------------------------------------------------------------

def build_standard_decay_component(
    grid,
    bbox,
    weights_dict,
    d_geo,
    A_field,
    base_risk,
    alpha,
    topology_template,
    risk_params,
    object_mask=None,
    surface_mask=None,
    footprint_mask=None,
):
    weights_dict = _default_weights(weights_dict)

    anchor_xyz = _choose_standard_decay_anchor(
        grid=grid,
        bbox=bbox,
        surface_mask=surface_mask,
        risk_params=risk_params,
        topology_template=topology_template,
    )

    radius_m = _get_standard_radius_m(
        bbox=bbox,
        topology_template=topology_template,
        risk_params=risk_params,
    )

    gamma = _get_directional_gamma(
        topology_template=topology_template,
        risk_params=risk_params,
    )

    dx = grid.X - float(anchor_xyz[0])
    dy = grid.Y - float(anchor_xyz[1])
    dz = grid.Z - float(anchor_xyz[2])

    r = np.sqrt(dx * dx + dy * dy + dz * dz)

    radius_multiplier = _directional_radius_multiplier(
        dx=dx,
        dy=dy,
        dz=dz,
        weights_dict=weights_dict,
        gamma=gamma,
    )

    # 방향별 plateau radius
    R = np.maximum(float(radius_m) * radius_multiplier, 1e-6)

    # plateau 바깥으로 얼마나 나갔는지
    d_out = np.maximum(0.0, r - R)

    # tail 길이 (JSON에 없으면 내부 default)
    tail_m = float(risk_params.get("tail_m", max(0.04, 0.50 * float(radius_m))))
    tail_m = max(tail_m, 1e-6)

    # 빠른 decay
    # d_out = tail_m 일 때 exp(-4) ≈ 0.018
    d_out = np.maximum(0.0, r - R)
    V = base_risk * A_field * np.exp(-1.5 * d_out / tail_m)

    vertical_gate = _standard_decay_vertical_gate(
        grid=grid,
        anchor_xyz=anchor_xyz,
        weights_dict=weights_dict,
        risk_params=risk_params,
    )
    V *= vertical_gate

    # 너무 긴 tail 제거
    cutoff_m = float(risk_params.get("cutoff_m", radius_m + 2.5 * tail_m))
    V *= (r <= cutoff_m).astype(np.float64)

    V[np.isnan(V)] = 0.0
    return V


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
    weights = _default_weights(risk_params.get("weights", {}))
    vertical_rule = risk_params.get("vertical_rule", "standard_decay")
    alpha = _get_standard_alpha("upward_vertical_cone", risk_params)

    V = build_standard_decay_component(
        grid=grid,
        bbox=bbox,
        weights_dict=weights,
        d_geo=d_geo,
        A_field=A_field,
        base_risk=base_risk,
        alpha=alpha,
        topology_template="upward_vertical_cone",
        risk_params=risk_params,
        object_mask=object_mask,
        surface_mask=surface_mask,
        footprint_mask=footprint_mask,
    )

    if vertical_rule == "gravity_column":
        if footprint_mask is None and surface_mask is not None:
            footprint_mask = surface_mask_to_footprint(surface_mask)

        if surface_mask is None and object_mask is not None:
            surface_mask = object_mask

        if surface_mask is not None and np.any(surface_mask):
            V_gravity = gravity_column_from_surface_mask(
                grid=grid,
                top_surface_mask=surface_mask,
                A_field=A_field,
                base_risk=base_risk,
                w_plus_z=float(weights.get("w_+z", 0.0)),
                lateral_decay=risk_params.get("lateral_decay", "moderate"),
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
        topology_template="isotropic_sphere",
        risk_params=risk_params,
        object_mask=object_mask,
        surface_mask=surface_mask,
        footprint_mask=footprint_mask,
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
    weights_dict = _default_weights(risk_params.get("weights", {}))
    alpha = _get_standard_alpha("forward_directional_cone", risk_params)

    return build_standard_decay_component(
        grid=grid,
        bbox=bbox,
        weights_dict=weights_dict,
        d_geo=d_geo,
        A_field=A_field,
        base_risk=base_risk,
        alpha=alpha,
        topology_template="forward_directional_cone",
        risk_params=risk_params,
        object_mask=object_mask,
        surface_mask=surface_mask,
        footprint_mask=footprint_mask,
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
        topology_template="planar_half_space",
        risk_params=risk_params,
        object_mask=object_mask,
        surface_mask=surface_mask,
        footprint_mask=footprint_mask,
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
        risk_params=risk_params,
        d_geo=d_geo,
        A_field=A_field,
        base_risk=base_risk,
        object_mask=object_mask,
        surface_mask=surface_mask,
        footprint_mask=footprint_mask,
    )