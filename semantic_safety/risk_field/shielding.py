"""
Occlusion shielding ratio A(x) = (d_geo / (d_euc + ε))^κ.
Smooth attenuation: unshielded → A≈1, heavily shielded → A≪1.
"""

import numpy as np


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
