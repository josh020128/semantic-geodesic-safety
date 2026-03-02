"""
Final semantic cost: V_risk(x) = base_risk * W_hazard(x) * A(x) * exp(-α * d_geo(x, S)).
"""

import numpy as np


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
