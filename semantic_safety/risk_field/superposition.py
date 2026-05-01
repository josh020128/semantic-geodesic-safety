"""
Occlusion shielding, per-hazard cost composition, and multi-hazard LogSumExp superposition.
"""

from __future__ import annotations

from typing import List

import numpy as np
from scipy.special import logsumexp


def shielding_ratio(d_geo: np.ndarray, d_euc: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    num = d_euc + eps
    den = d_geo + eps

    ratio = np.divide(
        num,
        den,
        out=np.zeros_like(d_euc, dtype=np.float64),
        where=np.isfinite(num) & np.isfinite(den),
    )

    ratio = np.clip(ratio, 0.0, 1.0)
    ratio[~np.isfinite(ratio)] = 0.0
    return ratio


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

def compute_hybrid_superposition(
    hazard_fields: list[np.ndarray],
    beta: float = 6.0,
    v_max: float = 120.0,
    additive_scale: float = 0.15,
) -> np.ndarray:
    if not hazard_fields:
        raise ValueError("Cannot superpose an empty list of hazard fields.")

    stacked = np.stack(hazard_fields, axis=0)

    # smooth-max core
    lse = (1.0 / beta) * logsumexp(beta * stacked, axis=0)

    # small overlap bonus above the max
    max_field = np.max(stacked, axis=0)
    overlap_bonus = additive_scale * np.clip(np.sum(stacked, axis=0) - max_field, 0.0, None)

    v_final = lse + overlap_bonus
    return np.clip(v_final, 0.0, v_max)

def compute_sum_superposition(
    hazard_fields: list[np.ndarray],
    v_max: float = 300.0,
) -> np.ndarray:
    if not hazard_fields:
        raise ValueError("Cannot superpose an empty list of hazard fields.")

    stacked = np.stack(hazard_fields, axis=0)
    v_final = np.sum(stacked, axis=0)
    return np.clip(v_final, 0.0, v_max)