from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from semantic_safety.ur5_experiment.risk_volume_query import RiskVolumeQuery


def load_scene_objects(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def draw_bbox_xy(ax, bbox, color="cyan", linewidth=1.5, label=None):
    xmin, xmax, ymin, ymax, zmin, zmax = bbox
    xs = [xmin, xmax, xmax, xmin, xmin]
    ys = [ymin, ymin, ymax, ymax, ymin]
    ax.plot(xs, ys, color=color, linewidth=linewidth)
    if label is not None:
        ax.text(xmin, ymax, label, color=color, fontsize=9)


def draw_bbox_xz(ax, bbox, color="cyan", linewidth=1.5, label=None):
    xmin, xmax, ymin, ymax, zmin, zmax = bbox
    xs = [xmin, xmax, xmax, xmin, xmin]
    zs = [zmin, zmin, zmax, zmax, zmin]
    ax.plot(xs, zs, color=color, linewidth=linewidth)
    if label is not None:
        ax.text(xmin, zmax, label, color=color, fontsize=9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--risk-npz", type=str, default="loop1_risk_field.npz")
    parser.add_argument("--scene-json", type=str, default="loop1_scene_objects.json")
    parser.add_argument("--collision-path", type=str, default="collision_only_path.npy")
    parser.add_argument("--risk-aware-path", type=str, default="risk_aware_path.npy")
    parser.add_argument("--out-dir", type=str, default="out/ur5_viz")
    parser.add_argument(
        "--z-mode",
        type=str,
        default="path_mean",
        choices=["path_mean", "table_top", "mid"],
        help="Which z slice to use for XY visualization",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rv = RiskVolumeQuery.from_npz(args.risk_npz)
    scene_objects = load_scene_objects(args.scene_json)

    collision_path = np.load(args.collision_path)
    risk_aware_path = np.load(args.risk_aware_path)

    # ------------------------------------------------------------
    # Choose z slice for XY visualization
    # ------------------------------------------------------------
    if args.z_mode == "path_mean":
        all_z = []
        if len(collision_path) > 0:
            all_z.append(collision_path[:, 2])
        if len(risk_aware_path) > 0:
            all_z.append(risk_aware_path[:, 2])
        if len(all_z) == 0:
            z_slice = rv.z[len(rv.z) // 2]
        else:
            z_slice = float(np.mean(np.concatenate(all_z)))
    elif args.z_mode == "table_top" and rv.table_top_z is not None:
        z_slice = float(rv.table_top_z + 0.10)
    else:
        z_slice = float(rv.z[len(rv.z) // 2])

    iz = int(np.argmin(np.abs(rv.z - z_slice)))

    # XY image = risk slice at chosen z
    risk_xy = rv.risk_field[:, :, iz].T

    # XZ image = max over y
    risk_xz = np.max(rv.risk_field, axis=1).T

    # ------------------------------------------------------------
    # Figure 1: XY top-down
    # ------------------------------------------------------------
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111)

    extent_xy = [rv.x[0], rv.x[-1], rv.y[0], rv.y[-1]]
    im1 = ax1.imshow(
        risk_xy,
        origin="lower",
        extent=extent_xy,
        aspect="auto",
    )
    plt.colorbar(im1, ax=ax1, label=f"Risk at z={rv.z[iz]:.3f} m")

    if len(collision_path) > 0:
        ax1.plot(
            collision_path[:, 0],
            collision_path[:, 1],
            linewidth=2,
            label="collision-only",
        )
        ax1.scatter(collision_path[0, 0], collision_path[0, 1], marker="o", s=50, label="start")
        ax1.scatter(collision_path[-1, 0], collision_path[-1, 1], marker="x", s=60, label="goal")

    if len(risk_aware_path) > 0:
        ax1.plot(
            risk_aware_path[:, 0],
            risk_aware_path[:, 1],
            linewidth=2,
            label="risk-aware",
        )

    for obj in scene_objects:
        bbox = obj.get("bbox_3d", None)
        label = obj.get("label", "obj")
        if bbox is not None:
            draw_bbox_xy(ax1, bbox, color="cyan", linewidth=1.5, label=label)

    ax1.set_title("Top-down XY path overlay")
    ax1.set_xlabel("x (world m)")
    ax1.set_ylabel("y (world m)")
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(out_dir / "paths_xy.png", dpi=200)
    plt.close(fig1)

    # ------------------------------------------------------------
    # Figure 2: XZ side view
    # ------------------------------------------------------------
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111)

    extent_xz = [rv.x[0], rv.x[-1], rv.z[0], rv.z[-1]]
    im2 = ax2.imshow(
        risk_xz,
        origin="lower",
        extent=extent_xz,
        aspect="auto",
    )
    plt.colorbar(im2, ax=ax2, label="Risk max-projection over y")

    if len(collision_path) > 0:
        ax2.plot(
            collision_path[:, 0],
            collision_path[:, 2],
            linewidth=2,
            label="collision-only",
        )
        ax2.scatter(collision_path[0, 0], collision_path[0, 2], marker="o", s=50, label="start")
        ax2.scatter(collision_path[-1, 0], collision_path[-1, 2], marker="x", s=60, label="goal")

    if len(risk_aware_path) > 0:
        ax2.plot(
            risk_aware_path[:, 0],
            risk_aware_path[:, 2],
            linewidth=2,
            label="risk-aware",
        )

    for obj in scene_objects:
        bbox = obj.get("bbox_3d", None)
        label = obj.get("label", "obj")
        if bbox is not None:
            draw_bbox_xz(ax2, bbox, color="cyan", linewidth=1.5, label=label)

    ax2.set_title("Side XZ path overlay")
    ax2.set_xlabel("x (world m)")
    ax2.set_ylabel("z (world m)")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(out_dir / "paths_xz.png", dpi=200)
    plt.close(fig2)

    # ------------------------------------------------------------
    # Figure 3: simple 3D path plot
    # ------------------------------------------------------------
    fig3 = plt.figure(figsize=(8, 6))
    ax3 = fig3.add_subplot(111, projection="3d")

    if len(collision_path) > 0:
        ax3.plot(
            collision_path[:, 0],
            collision_path[:, 1],
            collision_path[:, 2],
            linewidth=2,
            label="collision-only",
        )
        ax3.scatter(collision_path[0, 0], collision_path[0, 1], collision_path[0, 2], s=40)
        ax3.scatter(collision_path[-1, 0], collision_path[-1, 1], collision_path[-1, 2], s=40)

    if len(risk_aware_path) > 0:
        ax3.plot(
            risk_aware_path[:, 0],
            risk_aware_path[:, 1],
            risk_aware_path[:, 2],
            linewidth=2,
            label="risk-aware",
        )

    for obj in scene_objects:
        bbox = obj.get("bbox_3d", None)
        if bbox is None:
            continue
        xmin, xmax, ymin, ymax, zmin, zmax = bbox
        corners = np.array([
            [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin],
            [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax],
        ])
        ax3.scatter(corners[:, 0], corners[:, 1], corners[:, 2], s=10)

    ax3.set_title("3D path view")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(out_dir / "paths_3d.png", dpi=200)
    plt.close(fig3)

    print(f"Saved visualizations to: {out_dir}")
    print(f"  - {out_dir / 'paths_xy.png'}")
    print(f"  - {out_dir / 'paths_xz.png'}")
    print(f"  - {out_dir / 'paths_3d.png'}")


if __name__ == "__main__":
    main()