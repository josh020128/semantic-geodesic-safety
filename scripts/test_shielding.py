import argparse
import os

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from semantic_safety.metric_propagation.fmm_distance import WorkspaceGrid
from semantic_safety.risk_field.superposition import shielding_ratio


def get_body_pos(model, data, body_name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body '{body_name}' not found.")
    return data.xpos[body_id].copy()


def get_box_geom_world(model, data, geom_name: str):
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if geom_id < 0:
        return None
    pos = data.geom_xpos[geom_id].copy()
    size = model.geom_size[geom_id].copy()
    return pos, size


def box_to_mask(grid: WorkspaceGrid, pos: np.ndarray, size: np.ndarray) -> np.ndarray:
    return (
        (np.abs(grid.X - pos[0]) <= size[0]) &
        (np.abs(grid.Y - pos[1]) <= size[1]) &
        (np.abs(grid.Z - pos[2]) <= size[2])
    )


def sample_field_nearest(grid: WorkspaceGrid, field: np.ndarray, point_xyz: np.ndarray) -> float:
    ix = int(np.argmin(np.abs(grid.x - point_xyz[0])))
    iy = int(np.argmin(np.abs(grid.y - point_xyz[1])))
    iz = int(np.argmin(np.abs(grid.z - point_xyz[2])))
    return float(field[ix, iy, iz])


def save_slice_image(field, x, y, z, axis: str, index: int, title: str, outpath: str):
    plt.figure(figsize=(6, 5))

    if axis == "x":
        img = field[index, :, :].T
        extent = [y.min(), y.max(), z.min(), z.max()]
        xlabel, ylabel = "y", "z"
    elif axis == "y":
        img = field[:, index, :].T
        extent = [x.min(), x.max(), z.min(), z.max()]
        xlabel, ylabel = "x", "z"
    elif axis == "z":
        img = field[:, :, index].T
        extent = [x.min(), x.max(), y.min(), y.max()]
        xlabel, ylabel = "x", "y"
    else:
        raise ValueError("axis must be one of {'x', 'y', 'z'}")

    plt.imshow(img, origin="lower", extent=extent, aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def build_workspace_and_fields(xml_path: str):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    for _ in range(100):
        mujoco.mj_step(model, data)

    drill_pos = get_body_pos(model, data, "power_drill")
    print("drill_pos:", drill_pos)

    table_info = get_box_geom_world(model, data, "table")
    if table_info is None:
        raise ValueError("Scene must contain geom 'table'.")
    table_pos, table_size = table_info
    table_top_z = float(table_pos[2] + table_size[2])

    # Local grid centered around drill
    bounds = (
        drill_pos[0] - 0.35, drill_pos[0] + 0.35,
        drill_pos[1] - 0.35, drill_pos[1] + 0.35,
        max(0.0, drill_pos[2] - 0.10), drill_pos[2] + 0.45,
    )
    grid = WorkspaceGrid(bounds=bounds, resolution=0.01)

    # Approximate drill as a compact bbox source
    drill_bbox_half = np.array([0.07, 0.04, 0.05], dtype=np.float64)
    object_mask = (
        (np.abs(grid.X - drill_pos[0]) <= drill_bbox_half[0]) &
        (np.abs(grid.Y - drill_pos[1]) <= drill_bbox_half[1]) &
        (np.abs(grid.Z - drill_pos[2]) <= drill_bbox_half[2])
    )

    free = np.ones(grid.shape, dtype=bool)

    # Table blocks anything below its top plane
    free[grid.Z < table_top_z] = False

    # Object itself is occupied
    free[object_mask] = False

    blocker_names = []
    for name in ["shield_wall", "shield_shelf"]:
        info = get_box_geom_world(model, data, name)
        if info is not None:
            pos, size = info
            free[box_to_mask(grid, pos, size)] = False
            blocker_names.append(name)

    print("blockers found:", blocker_names)

    d_euc, d_geo, seed_mask = grid.compute_boundary_seeded_distances(
        object_mask=object_mask,
        occupancy_grid=free,
        connectivity=1,
    )
    A = shielding_ratio(d_geo, d_euc)

    return model, data, grid, drill_pos, table_top_z, object_mask, free, d_euc, d_geo, A


def run_wall_probe(grid: WorkspaceGrid, drill_pos: np.ndarray, d_euc, d_geo, A):
    """
    Probe a point behind the wall (+y side of drill) and a side control point.
    """
    blocked_pt = np.array([drill_pos[0], drill_pos[1] + 0.28, drill_pos[2] + 0.08])
    control_pt = np.array([drill_pos[0] + 0.18, drill_pos[1], drill_pos[2] + 0.08])

    print("\n[Wall Scene Probes]")
    for name, pt in [("blocked_pt", blocked_pt), ("control_pt", control_pt)]:
        ve = sample_field_nearest(grid, d_euc, pt)
        vg = sample_field_nearest(grid, d_geo, pt)
        va = sample_field_nearest(grid, A, pt)
        print(f"{name}: point={pt}, d_euc={ve:.3f}, d_geo={vg:.3f}, A={va:.4f}")

    return blocked_pt, control_pt


def run_shelf_probe(grid: WorkspaceGrid, drill_pos: np.ndarray, d_euc, d_geo, A):
    """
    Probe a point directly above the shelf and a side-above control point.
    """
    blocked_pt = np.array([drill_pos[0], drill_pos[1], drill_pos[2] + 0.32])
    control_pt = np.array([drill_pos[0] + 0.18, drill_pos[1], drill_pos[2] + 0.32])

    print("\n[Shelf Scene Probes]")
    for name, pt in [("blocked_pt", blocked_pt), ("control_pt", control_pt)]:
        ve = sample_field_nearest(grid, d_euc, pt)
        vg = sample_field_nearest(grid, d_geo, pt)
        va = sample_field_nearest(grid, A, pt)
        print(f"{name}: point={pt}, d_euc={ve:.3f}, d_geo={vg:.3f}, A={va:.4f}")

    return blocked_pt, control_pt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, required=True,
                        help="Path to MuJoCo XML scene (e.g. tabletop_wall.xml or tabletop_shelf.xml)")
    parser.add_argument("--outdir", type=str, default="shielding_debug")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    model, data, grid, drill_pos, table_top_z, object_mask, free, d_euc, d_geo, A = build_workspace_and_fields(args.xml)

    # Save diagnostic arrays
    np.savez_compressed(
        os.path.join(args.outdir, "shielding_debug.npz"),
        x=grid.x.astype(np.float32),
        y=grid.y.astype(np.float32),
        z=grid.z.astype(np.float32),
        d_euc=d_euc.astype(np.float32),
        d_geo=d_geo.astype(np.float32),
        A=A.astype(np.float32),
        object_mask=object_mask.astype(np.uint8),
        free=free.astype(np.uint8),
        drill_pos=drill_pos.astype(np.float32),
        table_top_z=np.array([table_top_z], dtype=np.float32),
    )

    # Slice indices through drill center
    ix = int(np.argmin(np.abs(grid.x - drill_pos[0])))
    iy = int(np.argmin(np.abs(grid.y - drill_pos[1])))
    iz = int(np.argmin(np.abs(grid.z - drill_pos[2] + 0.05)))

    save_slice_image(A, grid.x, grid.y, grid.z, axis="x", index=ix,
                     title="A(x): yz-slice through drill center x",
                     outpath=os.path.join(args.outdir, "A_yz_slice.png"))
    save_slice_image(A, grid.x, grid.y, grid.z, axis="y", index=iy,
                     title="A(x): xz-slice through drill center y",
                     outpath=os.path.join(args.outdir, "A_xz_slice.png"))
    save_slice_image(A, grid.x, grid.y, grid.z, axis="z", index=iz,
                     title="A(x): xy-slice near drill height",
                     outpath=os.path.join(args.outdir, "A_xy_slice.png"))

    save_slice_image(d_euc, grid.x, grid.y, grid.z, axis="y", index=iy,
                     title="d_euc: xz-slice",
                     outpath=os.path.join(args.outdir, "d_euc_xz_slice.png"))
    save_slice_image(d_geo, grid.x, grid.y, grid.z, axis="y", index=iy,
                     title="d_geo: xz-slice",
                     outpath=os.path.join(args.outdir, "d_geo_xz_slice.png"))

    if "wall" in os.path.basename(args.xml):
        run_wall_probe(grid, drill_pos, d_euc, d_geo, A)
    if "shelf" in os.path.basename(args.xml):
        run_shelf_probe(grid, drill_pos, d_euc, d_geo, A)

    print(f"\nSaved shielding diagnostics to: {args.outdir}")


if __name__ == "__main__":
    main()