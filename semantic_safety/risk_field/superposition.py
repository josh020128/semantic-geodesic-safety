"""
Occlusion shielding, per-hazard cost composition, and multi-hazard LogSumExp superposition.
"""

from __future__ import annotations

from typing import List

import numpy as np
from scipy.special import logsumexp


def shielding_ratio(
    d_geo: np.ndarray,
    d_euc: np.ndarray,
    kappa: float = 2.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    A(x) = (d_geo / (d_euc + ε))^κ.
    When d_geo ≈ d_euc (line of sight), A ≈ 1. When d_geo >> d_euc (occluded), A >> 1 then we use
    min(A, 1) or 1/A in cost so that shielded risk is reduced. Proposal: risk scaled by smooth
    factor; "dimmer" so we want A in (0,1]. So use A = (d_euc / (d_geo + ε))^κ so that
    unshielded (d_geo ≈ d_euc) → A ≈ 1, shielded (d_geo >> d_euc) → A ≪ 1.
    """
    ratio = (d_euc + eps) / (d_geo + eps)
    A = np.clip(ratio, 0.0, 1.0) ** kappa
    return A


def risk_cost_field(
    W_hazard: np.ndarray,
    A: np.ndarray,
    d_geo: np.ndarray,
    alpha: float = 1.0,
    base_risk: float = 1.0,
) -> np.ndarray:
    """
    V_risk(x) = base_risk * W_hazard(x) * A(x) * exp(-α * d_geo).
    Scalar cost field: 0 = safe, higher = more penalty.
    """
    return base_risk * W_hazard * A * np.exp(-alpha * np.clip(d_geo, 0, None))


def compute_logsumexp_superposition(
    hazard_fields: List[np.ndarray],
    beta: float = 10.0,
    v_max: float = 10.0,
) -> np.ndarray:
    """
    Combines multiple individual risk fields into a single bounded cost map.
    Uses the LogSumExp smooth-maximum formulation to preserve compound risk
    without creating impassable infinite-cost walls.

    hazard_fields: List of 3D numpy arrays, each representing V_i(x) for one object.
    beta: Sharpness parameter.
          Higher beta approaches the pure Max() operator.
          Lower beta blends the fields more (closer to Summation).
    v_max: Absolute ceiling for the risk cost to prevent trajectory optimizer failure.
    """
    if not hazard_fields:
        raise ValueError("Cannot superpose an empty list of hazard fields.")

    stacked_fields = np.stack(hazard_fields, axis=0)

    scaled_fields = beta * stacked_fields

    lse_result = logsumexp(scaled_fields, axis=0)

    v_final = (1.0 / beta) * lse_result

    v_final_capped = np.clip(v_final, a_min=0.0, a_max=v_max)

    return v_final_capped
