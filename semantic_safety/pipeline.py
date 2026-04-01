"""
Main pipeline: Phase 0 (LLM prior) → Loop 1 (risk field) → Loop 2 (optimization).
Orchestrates perception outputs, occupancy, FMM, directional interpolation, and final V_risk.
"""

from typing import Any

import numpy as np

from .config import load_config
from .metric_propagation import WorkspaceGrid
from .metric_propagation.occupancy_grid import build_occupancy_grid
from .phase0_dataset import LLMPrior, RiskPrior
from .risk_field import compute_directional_weights, risk_cost_field, shielding_ratio


def run_phase0(
    manipulated: str,
    scene: str,
    config: dict[str, Any] | None = None,
) -> RiskPrior:
    """Phase 0: LLM → base risk score + 6-directional decay weights."""
    cfg = (config or {}).get("llm", {})
    prior = LLMPrior(
        provider=cfg.get("provider", "openai"),
        model=cfg.get("model", "gpt-4o"),
        temperature=cfg.get("temperature", 0.0),
        return_fallback_on_error=cfg.get("return_fallback_on_error", False),
    )
    return prior.get_risk_prior(manipulated, scene)


def run_phase1(
    point_cloud: dict[str, np.ndarray],
    hazard_label: int,
    centroid: np.ndarray,
    risk_prior: RiskPrior,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Loop 1: semantic labels on points → occupancy → FMM → directional + shielding → V_risk.
    point_cloud: coord (N,3), color (N,3), normal (N,3); segment (N,) required for meaningful hazards
    (populate via perception_2d3d when implemented).
    hazard_label: semantic label id for the hazard object.
    centroid: (3,) world coords of hazard centroid (used for directional W_hazard only).
    risk_prior: from Phase 0.
    Returns dict with V_risk, d_geo, d_euc, grid, origin, shape, resolution.
    Distances d_geo / d_euc are boundary-conformal Eikonal fields from ∂Ω via ``WorkspaceGrid``
    (geodesic with occupancy; Euclidean baseline without obstacle masking for shielding).
    """
    cfg = config or {}
    occ_cfg = cfg.get("occupancy", {})
    risk_cfg = cfg.get("risk_field", {})

    resolution = occ_cfg.get("grid_resolution", 0.02)
    coords = point_cloud["coord"]
    segment = point_cloud.get("segment")
    if segment is None:
        segment = np.zeros(len(coords), dtype=np.int32)

    grid, origin, shape, hazard_voxels = build_occupancy_grid(
        coords,
        segment,
        hazard_label=hazard_label,
        resolution=resolution,
        padding=0.1,
    )
    traversable = grid
    if hazard_voxels.size == 0:
        hazard_voxels = np.argwhere(~traversable)

    nx, ny, nz = int(shape[0]), int(shape[1]), int(shape[2])
    ox, oy, oz = float(origin[0]), float(origin[1]), float(origin[2])
    res = float(resolution)
    bounds = (
        ox,
        ox + nx * res,
        oy,
        oy + ny * res,
        oz,
        oz + nz * res,
    )
    workspace = WorkspaceGrid(bounds=bounds, resolution=res)
    if workspace.shape != (nx, ny, nz):
        raise RuntimeError(
            f"WorkspaceGrid shape {workspace.shape} != occupancy shape {(nx, ny, nz)}; "
            "check bounds/resolution consistency."
        )

    object_mask = np.zeros((nx, ny, nz), dtype=bool)
    for idx in hazard_voxels:
        i, j, k = int(idx[0]), int(idx[1]), int(idx[2])
        if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
            object_mask[i, j, k] = True

    d_geo = workspace.compute_geodesic_distance(object_mask, traversable)
    d_euc = workspace.compute_euclidean_distance(object_mask)

    weights_dict = {
        "w_+x": float(risk_prior.w_plus_x),
        "w_-x": float(risk_prior.w_minus_x),
        "w_+y": float(risk_prior.w_plus_y),
        "w_-y": float(risk_prior.w_minus_y),
        "w_+z": float(risk_prior.w_plus_z),
        "w_-z": float(risk_prior.w_minus_z),
    }
    W_hazard = compute_directional_weights(
        workspace.X,
        workspace.Y,
        workspace.Z,
        np.asarray(centroid, dtype=np.float64),
        weights_dict,
    )
    A = shielding_ratio(
        d_geo,
        d_euc,
        kappa=risk_cfg.get("shielding_kappa", 2.0),
        eps=risk_cfg.get("shielding_eps", 1e-6),
    )
    V_risk = risk_cost_field(
        W_hazard,
        A,
        d_geo,
        alpha=risk_cfg.get("decay_alpha", 1.0),
        base_risk=risk_prior.base_risk,
    )
    # Mask out unreachable / obstacle voxels for cleaner output
    V_risk = np.where(np.isfinite(d_geo) & (d_geo < 1e10), V_risk, 0.0)

    return {
        "V_risk": V_risk,
        "d_geo": d_geo,
        "d_euc": d_euc,
        "grid": grid,
        "origin": origin,
        "shape": shape,
        "resolution": resolution,
    }


def run_pipeline(
    point_cloud: dict[str, np.ndarray],
    manipulated_object: str,
    scene_object: str,
    hazard_label: int,
    centroid: np.ndarray,
    config_path: str | None = None,
) -> dict[str, Any]:
    """
    Full pipeline: Phase 0 → Loop 1. Loop 2 hooks live in phase2_control.
    """
    config = load_config(config_path)
    prior = run_phase0(manipulated_object, scene_object, config)
    phase1_out = run_phase1(point_cloud, hazard_label, centroid, prior, config)
    phase1_out["risk_prior"] = prior
    return phase1_out
