"""
Occlusion shielding, soft-field superposition, and hard semantic mask union.
"""

from __future__ import annotations

from typing import List

import numpy as np
from scipy.special import logsumexp


def shielding_ratio(d_geo: np.ndarray, d_euc: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    A(x) = clip((d_euc + eps) / (d_geo + eps), 0, 1)

    - near 1 when Euclidean and geodesic distances are similar
    - near 0 when geodesic distance is much larger (strong occlusion / shielding)
    """
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


def compute_sum_superposition(
    hazard_fields: List[np.ndarray],
    v_max: float = 8.0,
) -> np.ndarray:
    """
    Combine per-object volumes ``V_object`` after each has been built separately.

    **Between objects:** voxel-wise sum so overlapping fields add; clip to
    ``v_max`` so the combined field cannot grow without bound.

    **Within a single object** (caller responsibility): merge soft Gaussian risk
    with the persistent ``w_+z = inf`` upward column using ``np.maximum``, not
    sum—this function only handles the cross-object stage.
    """
    if not hazard_fields:
        raise ValueError("Cannot superpose an empty list of hazard fields.")

    stacked = np.stack(hazard_fields, axis=0).astype(np.float64)
    v_final = np.sum(stacked, axis=0)
    return np.clip(v_final, 0.0, v_max)


def compute_logsumexp_superposition(
    hazard_fields: List[np.ndarray],
    beta: float = 8.0,
    v_max: float = 8.0,
) -> np.ndarray:
    """
    Optional smooth-max style superposition for soft semantic fields.
    """
    if not hazard_fields:
        raise ValueError("Cannot superpose an empty list of hazard fields.")

    stacked_fields = np.stack(hazard_fields, axis=0).astype(np.float64)
    scaled_fields = beta * stacked_fields
    lse_result = logsumexp(scaled_fields, axis=0)
    v_final = (1.0 / beta) * lse_result
    return np.clip(v_final, a_min=0.0, a_max=v_max)


def compute_hard_mask_union(
    hard_masks: List[np.ndarray],
) -> np.ndarray:
    """
    Union of per-object hard semantic exclusion masks.

    Returns:
      bool array, True = semantically forbidden
    """
    if not hard_masks:
        raise ValueError("Cannot union an empty list of hard semantic masks.")

    stacked = np.stack(hard_masks, axis=0).astype(bool)
    return np.any(stacked, axis=0)