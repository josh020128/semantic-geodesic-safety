"""
Main pipeline: Phase 0 (LLM prior) → Loop 1 (risk field) → Loop 2 (optimization).
Orchestrates perception outputs, occupancy, FMM, directional interpolation, and final V_risk.
"""

from typing import Any

import numpy as np

from .config import load_config
from .metric_propagation import (
    build_occupancy_grid,
    euclidean_distance_field,
    extract_boundary_seeds,
    geodesic_distance_field,
)
from .phase0_dataset import LLMPrior, RiskPrior
from .risk_field import directional_weight_grid, risk_cost_field, shielding_ratio


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
    centroid: (3,) world coords of hazard centroid.
    risk_prior: from Phase 0.
    Returns dict with V_risk, d_geo, d_euc, grid, origin, shape, resolution.
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
    seeds = extract_boundary_seeds(grid, hazard_voxels)

    d_geo = geodesic_distance_field(traversable, seeds, resolution=resolution)
    d_euc = euclidean_distance_field(shape, origin, centroid, resolution)

    weights_6 = risk_prior.to_weights_tuple()
    W_hazard = directional_weight_grid(
        shape,
        origin,
        centroid,
        resolution,
        weights_6,
        sigmoid_steepness=risk_cfg.get("sigmoid_steepness", 10),
        blend_exponent=risk_cfg.get("blend_exponent", 2),
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
