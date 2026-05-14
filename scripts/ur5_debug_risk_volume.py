from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from semantic_safety.ur5_experiment.risk_volume_query import RiskVolumeQuery


def load_scene_objects(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--risk-npz", type=str, default="loop1_risk_field.npz")
    parser.add_argument("--scene-json", type=str, default="loop1_scene_objects.json")
    parser.add_argument(
        "--probe",
        type=float,
        nargs=3,
        action="append",
        default=None,
        help="Optional world point probe: --probe x y z",
    )
    parser.add_argument(
        "--margin-m",
        type=float,
        default=0.03,
        help="Margin for conservative free-space query.",
    )
    args = parser.parse_args()

    rv = RiskVolumeQuery.from_npz(args.risk_npz)
    print(rv.summary())

    objects = load_scene_objects(args.scene_json)
    if objects:
        print("\n=== Scene object probes ===")
        for obj in objects:
            label = obj.get("label", "unknown")
            world_pt = np.asarray(obj.get("world_pt", [np.nan, np.nan, np.nan]), dtype=np.float64)
            bbox = obj.get("bbox_3d", None)

            risk_nn = rv.sample_risk_nearest(world_pt, outside_value=np.nan)
            risk_tri = rv.sample_risk_trilinear(world_pt, outside_value=np.nan)
            free = rv.is_free(world_pt)
            free_margin = rv.is_free_with_margin(world_pt, args.margin_m)

            print(f"\n[{label}]")
            print(f"  world_pt          : {np.round(world_pt, 4)}")
            print(f"  bbox_3d           : {bbox}")
            print(f"  nearest risk      : {risk_nn:.4f}")
            print(f"  trilinear risk    : {risk_tri:.4f}")
            print(f"  is_free           : {free}")
            print(f"  is_free_margin    : {free_margin}  margin={args.margin_m:.3f} m")

            if bbox is not None and len(bbox) == 6:
                bbox_center = np.array(
                    [
                        0.5 * (bbox[0] + bbox[1]),
                        0.5 * (bbox[2] + bbox[3]),
                        0.5 * (bbox[4] + bbox[5]),
                    ],
                    dtype=np.float64,
                )
                print(f"  bbox_center       : {np.round(bbox_center, 4)}")
                print(f"  bbox center risk  : {rv.sample_risk_nearest(bbox_center, outside_value=np.nan):.4f}")
                print(f"  bbox center free  : {rv.is_free(bbox_center)}")

    if args.probe:
        print("\n=== User probes ===")
        for p in args.probe:
            world_pt = np.asarray(p, dtype=np.float64)
            inside = rv.is_inside_bounds(world_pt)
            risk_nn = rv.sample_risk_nearest(world_pt, outside_value=np.nan)
            risk_tri = rv.sample_risk_trilinear(world_pt, outside_value=np.nan)
            free = rv.is_free(world_pt)
            free_margin = rv.is_free_with_margin(world_pt, args.margin_m)

            print(f"\nprobe {np.round(world_pt, 4)}")
            print(f"  inside            : {inside}")
            if inside:
                idx = rv.world_to_grid_idx(world_pt, clip=True)
                print(f"  idx               : ({idx.ix}, {idx.iy}, {idx.iz})")
                print(f"  idx world         : {np.round(rv.grid_idx_to_world(idx), 4)}")
            print(f"  nearest risk      : {risk_nn:.4f}")
            print(f"  trilinear risk    : {risk_tri:.4f}")
            print(f"  is_free           : {free}")
            print(f"  is_free_margin    : {free_margin}  margin={args.margin_m:.3f} m")


if __name__ == "__main__":
    main()