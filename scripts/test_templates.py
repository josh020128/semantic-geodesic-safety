import numpy as np

from semantic_safety.metric_propagation.fmm_distance import WorkspaceGrid
from semantic_safety.risk_field.templates import (
    build_upward_vertical_cone_field,
    build_isotropic_sphere_field,
    build_forward_directional_cone_field,
    build_planar_half_space_field,
)


def make_bbox_mask(grid, bbox):
    xmin, xmax, ymin, ymax, zmin, zmax = bbox
    return (
        (grid.X >= xmin) & (grid.X <= xmax) &
        (grid.Y >= ymin) & (grid.Y <= ymax) &
        (grid.Z >= zmin) & (grid.Z <= zmax)
    )


def make_test_grid():
    grid = WorkspaceGrid(bounds=(-0.25, 0.25, -0.25, 0.25, 0.0, 0.5), resolution=0.02)
    bbox = (-0.03, 0.03, -0.03, 0.03, 0.08, 0.14)

    object_mask = make_bbox_mask(grid, bbox)
    occupancy = np.ones(grid.shape, dtype=bool)
    occupancy[object_mask] = False

    d_euc, d_geo, seed_mask = grid.compute_boundary_seeded_distances(object_mask, occupancy)
    A_field = np.ones_like(d_geo, dtype=np.float64)  # isolate template behavior

    return grid, bbox, d_geo, A_field


def test_upward_vertical_cone_gravity_column():
    grid, bbox, d_geo, A_field = make_test_grid()

    risk_params = {
        "topology_template": "upward_vertical_cone",
        "weights": {
            "w_+x": 0.2,
            "w_-x": 0.2,
            "w_+y": 0.2,
            "w_-y": 0.2,
            "w_+z": 0.9,
            "w_-z": 0.0,
        },
        "vertical_rule": "gravity_column",
        "lateral_decay": "moderate",
        "vertical_extent_m": 0.30,
        "receptacle_attenuation": 1.0,
    }

    V = build_upward_vertical_cone_field(
        grid=grid,
        bbox=bbox,
        risk_params=risk_params,
        d_geo=d_geo,
        A_field=A_field,
        base_risk=100.0,
    )

    x0 = np.argmin(np.abs(grid.x - 0.0))
    y0 = np.argmin(np.abs(grid.y - 0.0))

    z_above = np.argmin(np.abs(grid.z - 0.20))
    z_below = np.argmin(np.abs(grid.z - 0.04))
    z_far_above = np.argmin(np.abs(grid.z - 0.48))

    center_above = V[x0, y0, z_above]
    center_below = V[x0, y0, z_below]
    center_far_above = V[x0, y0, z_far_above]

    assert center_above > center_below, "Above-object risk should exceed below-object risk"
    assert center_far_above < center_above, "Risk should taper by vertical extent"


def test_isotropic_sphere_symmetry():
    grid, bbox, d_geo, A_field = make_test_grid()

    risk_params = {
        "topology_template": "isotropic_sphere",
        "weights": {
            "w_+x": 1.0, "w_-x": 1.0,
            "w_+y": 1.0, "w_-y": 1.0,
            "w_+z": 1.0, "w_-z": 1.0,
        },
        "receptacle_attenuation": 1.0,
    }

    V = build_isotropic_sphere_field(
        grid=grid,
        bbox=bbox,
        risk_params=risk_params,
        d_geo=d_geo,
        A_field=A_field,
        base_risk=100.0,
    )

    xp = np.argmin(np.abs(grid.x - 0.12))
    xn = np.argmin(np.abs(grid.x + 0.12))
    y0 = np.argmin(np.abs(grid.y - 0.0))
    z0 = np.argmin(np.abs(grid.z - 0.12))

    vp = V[xp, y0, z0]
    vn = V[xn, y0, z0]

    assert abs(vp - vn) < 5.0, "Isotropic field should be roughly symmetric"


def test_forward_directional_cone_bias():
    grid, bbox, d_geo, A_field = make_test_grid()

    risk_params = {
        "topology_template": "forward_directional_cone",
        "weights": {
            "w_+x": 1.0,
            "w_-x": 0.1,
            "w_+y": 0.1,
            "w_-y": 0.1,
            "w_+z": 0.1,
            "w_-z": 0.1,
        },
        "directional_power": 2.0,
        "receptacle_attenuation": 1.0,
    }

    V = build_forward_directional_cone_field(
        grid=grid,
        bbox=bbox,
        risk_params=risk_params,
        d_geo=d_geo,
        A_field=A_field,
        base_risk=100.0,
    )

    xp = np.argmin(np.abs(grid.x - 0.12))
    xn = np.argmin(np.abs(grid.x + 0.12))
    y0 = np.argmin(np.abs(grid.y - 0.0))
    z0 = np.argmin(np.abs(grid.z - 0.12))

    assert V[xp, y0, z0] > V[xn, y0, z0], "Forward cone should prefer +x side"


def test_planar_half_space_one_sided():
    grid, bbox, d_geo, A_field = make_test_grid()

    risk_params = {
        "topology_template": "planar_half_space",
        "weights": {
            "w_+x": 1.0,
            "w_-x": 0.0,
            "w_+y": 0.0,
            "w_-y": 0.0,
            "w_+z": 0.0,
            "w_-z": 0.0,
        },
        "planar_alpha": 4.0,
        "receptacle_attenuation": 1.0,
    }

    V = build_planar_half_space_field(
        grid=grid,
        bbox=bbox,
        risk_params=risk_params,
        d_geo=d_geo,
        A_field=A_field,
        base_risk=100.0,
    )

    xp = np.argmin(np.abs(grid.x - 0.15))
    xn = np.argmin(np.abs(grid.x + 0.15))
    y0 = np.argmin(np.abs(grid.y - 0.0))
    z0 = np.argmin(np.abs(grid.z - 0.12))

    assert V[xp, y0, z0] > V[xn, y0, z0], "Half-space should be stronger on risky side"


if __name__ == "__main__":
    test_upward_vertical_cone_gravity_column()
    test_isotropic_sphere_symmetry()
    test_forward_directional_cone_bias()
    test_planar_half_space_one_sided()
    print("Template tests passed.")