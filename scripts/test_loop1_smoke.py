import os
import numpy as np
from test_full_pipeline import run_pipeline


def test_loop1_smoke():
    if os.path.exists("loop1_risk_field.npz"):
        os.remove("loop1_risk_field.npz")

    run_pipeline(
        manipulated_obj="cup of water",
        camera_type="Mujoco",
        target_label="power drill",
    )

    assert os.path.exists("loop1_risk_field.npz")

    data = np.load("loop1_risk_field.npz")
    V = data["risk_field"]
    x = data["x"]
    y = data["y"]
    z = data["z"]

    assert V.ndim == 3
    assert V.shape == (len(x), len(y), len(z))
    assert np.max(V) > 0.0


if __name__ == "__main__":
    test_loop1_smoke()
    print("Loop1 smoke test passed.")