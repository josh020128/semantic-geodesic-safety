import numpy as np
from semantic_safety.metric_propagation.fmm_distance import WorkspaceGrid


def make_box_mask(grid, xmin, xmax, ymin, ymax, zmin, zmax):
    return (
        (grid.X >= xmin) & (grid.X <= xmax) &
        (grid.Y >= ymin) & (grid.Y <= ymax) &
        (grid.Z >= zmin) & (grid.Z <= zmax)
    )


def test_boundary_seeded_distances_open_space():
    grid = WorkspaceGrid(bounds=(-0.2, 0.2, -0.2, 0.2, 0.0, 0.3), resolution=0.02)

    object_mask = make_box_mask(grid, -0.03, 0.03, -0.03, 0.03, 0.08, 0.14)
    occupancy = np.ones(grid.shape, dtype=bool)
    occupancy[object_mask] = False

    d_euc, d_geo, seed_mask = grid.compute_boundary_seeded_distances(object_mask, occupancy)

    assert seed_mask.shape == grid.shape
    assert np.any(seed_mask)
    assert d_euc.shape == grid.shape
    assert d_geo.shape == grid.shape
    assert np.isfinite(d_euc).any()
    assert np.isfinite(d_geo).any()

    # open space: d_geo should be close to d_euc in many voxels
    diff = np.abs(d_geo - d_euc)
    valid = np.isfinite(diff)
    assert np.mean(diff[valid]) < 0.05


def test_boundary_seeded_distances_with_wall():
    grid = WorkspaceGrid(bounds=(-0.2, 0.2, -0.2, 0.2, 0.0, 0.3), resolution=0.02)

    object_mask = make_box_mask(grid, -0.03, 0.03, -0.03, 0.03, 0.08, 0.14)
    occupancy = np.ones(grid.shape, dtype=bool)
    occupancy[object_mask] = False

    # wall above object, spanning in x
    wall_mask = (
        (grid.X >= -0.15) & (grid.X <= 0.15) &
        (grid.Y >= 0.04) & (grid.Y <= 0.06) &
        (grid.Z >= 0.05) & (grid.Z <= 0.25)
    )
    occupancy[wall_mask] = False

    d_euc, d_geo, seed_mask = grid.compute_boundary_seeded_distances(object_mask, occupancy)

    # choose a point behind the wall
    ix = np.argmin(np.abs(grid.x - 0.0))
    iy = np.argmin(np.abs(grid.y - 0.12))
    iz = np.argmin(np.abs(grid.z - 0.12))

    assert np.isfinite(d_euc[ix, iy, iz])
    # geodesic should be larger or unreachable
    assert (d_geo[ix, iy, iz] > d_euc[ix, iy, iz]) or np.isinf(d_geo[ix, iy, iz])


if __name__ == "__main__":
    test_boundary_seeded_distances_open_space()
    test_boundary_seeded_distances_with_wall()
    print("FMM tests passed.")