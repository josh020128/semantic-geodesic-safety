from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from semantic_safety.ur5_experiment.risk_volume_query import RiskVolumeQuery
from semantic_safety.ur5_experiment.workspace_astar import WorkspaceAStar, AStarConfig


def load_scene_objects(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        print(f"Warning: scene JSON not found: {path}")
        return []

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def result_to_dict(result) -> dict:
    return {
        "success": bool(result.success),
        "message": result.message,
        "total_cost": float(result.total_cost),
        "path_length_m": float(result.path_length_m),
        "integrated_risk": float(result.integrated_risk),
        "max_risk": float(result.max_risk),
        "num_expanded": int(result.num_expanded),
        "risk_weight": float(result.risk_weight),
        "stride": int(result.stride),
        "num_waypoints": int(len(result.world_path)),
    }


def print_result(name: str, result) -> None:
    print(f"\n{name}")
    print("  success        :", result.success)
    print("  message        :", result.message)
    print("  path length    :", result.path_length_m)
    print("  integrated risk:", result.integrated_risk)
    print("  max risk       :", result.max_risk)
    print("  num expanded   :", result.num_expanded)
    print("  num waypoints  :", len(result.world_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan collision-only and semantic-risk-aware Cartesian EE paths."
    )

    parser.add_argument("--risk-npz", type=str, default="loop1_risk_field.npz")
    parser.add_argument("--scene-json", type=str, default="loop1_scene_objects.json")
    parser.add_argument("--out-dir", type=str, default="out/ur5_planning")

    parser.add_argument(
        "--start",
        type=float,
        nargs=3,
        default=[-0.30, -0.20, 0.18],
        help="Start world point: x y z",
    )
    parser.add_argument(
        "--goal",
        type=float,
        nargs=3,
        default=[0.25, 0.20, 0.18],
        help="Goal world point: x y z",
    )

    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--risk-weight", type=float, default=5.0)
    parser.add_argument(
        "--risk-reduce",
        type=str,
        default="mean",
        choices=["center", "mean", "max"],
    )
    parser.add_argument("--risk-power", type=float, default=1.0)
    parser.add_argument("--max-expansions", type=int, default=300000)
    parser.add_argument("--nearest-free-search-radius", type=int, default=10)

    parser.add_argument("--bbox-pad-xy-m", type=float, default=0.02)
    parser.add_argument("--bbox-pad-z-m", type=float, default=0.01)

    parser.add_argument(
        "--table-clearance-m",
        type=float,
        default=0.12,
        help="Added to loaded table_top_z for min-z constraint when > 0.",
    )
    parser.add_argument(
        "--planning-min-z-floor-m",
        type=float,
        default=None,
        help="Optional absolute min EE z (m); combined with table clearance via max().",
    )
    parser.add_argument(
        "--ee-radius-m",
        type=float,
        default=0.05,
        help="Inflate EE as a sphere for validity; 0 disables.",
    )

    parser.add_argument(
        "--no-bbox-obstacles",
        action="store_true",
        help="Use saved occupancy_free directly instead of solid bbox planner obstacles.",
    )

    parser.add_argument(
        "--risk-block-threshold",
        type=float,
        default=None,
        help="Optional: treat voxels above this raw risk as blocked.",
    )
    parser.add_argument("--edge-check-spacing-m", type=float, default=0.01)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rv = RiskVolumeQuery.from_npz(args.risk_npz)
    print(rv.summary())

    scene_objects = load_scene_objects(args.scene_json)

    if args.no_bbox_obstacles:
        planner_free = rv.occupancy_free.copy()
        debug_rows = []
        print("\nUsing saved occupancy_free directly for planner.")
    else:
        planner_free, debug_rows = rv.build_planner_free_mask_from_bboxes(
            scene_objects,
            pad_xy_m=args.bbox_pad_xy_m,
            pad_z_m=args.bbox_pad_z_m,
            return_debug=True,
        )
        rv.print_planner_mask_debug(planner_free, debug_rows)

    planner = WorkspaceAStar(rv, planner_free_mask=planner_free)

    start = np.asarray(args.start, dtype=np.float64)
    goal = np.asarray(args.goal, dtype=np.float64)

    print("\n=== Planning Request ===")
    print("start:", np.round(start, 4))
    print("goal :", np.round(goal, 4))
    print("start risk:", rv.sample_risk_trilinear(start, outside_value=np.nan))
    print("goal risk :", rv.sample_risk_trilinear(goal, outside_value=np.nan))
    print("start free:", rv.is_free(start))
    print("goal free :", rv.is_free(goal))

    common_cfg = dict(
        stride=args.stride,
        risk_reduce=args.risk_reduce,
        risk_power=args.risk_power,
        allow_diagonal=True,
        max_expansions=args.max_expansions,
        nearest_free_search_radius=args.nearest_free_search_radius,
        risk_block_threshold=args.risk_block_threshold,
        table_clearance_m=args.table_clearance_m,
        planning_min_z_floor_m=args.planning_min_z_floor_m,
        ee_radius_m=args.ee_radius_m,
        edge_check_spacing_m=args.edge_check_spacing_m,
    )

    collision_only = planner.plan(
        start,
        goal,
        AStarConfig(
            risk_weight=0.0,
            **common_cfg,
        ),
    )

    risk_aware = planner.plan(
        start,
        goal,
        AStarConfig(
            risk_weight=args.risk_weight,
            **common_cfg,
        ),
    )

    print_result("collision_only", collision_only)
    print_result("risk_aware", risk_aware)

    collision_path_path = out_dir / "collision_only_path.npy"
    risk_aware_path_path = out_dir / "risk_aware_path.npy"
    metrics_path = out_dir / "planning_metrics.json"
    debug_path = out_dir / "planner_mask_debug.json"

    np.save(collision_path_path, collision_only.world_path)
    np.save(risk_aware_path_path, risk_aware.world_path)

    metrics = {
        "risk_npz": args.risk_npz,
        "scene_json": args.scene_json,
        "start": start.tolist(),
        "goal": goal.tolist(),
        "planner_config": {
            "stride": args.stride,
            "risk_weight": args.risk_weight,
            "risk_reduce": args.risk_reduce,
            "risk_power": args.risk_power,
            "max_expansions": args.max_expansions,
            "nearest_free_search_radius": args.nearest_free_search_radius,
            "bbox_pad_xy_m": args.bbox_pad_xy_m,
            "bbox_pad_z_m": args.bbox_pad_z_m,
            "table_clearance_m": args.table_clearance_m,
            "planning_min_z_floor_m": args.planning_min_z_floor_m,
            "ee_radius_m": args.ee_radius_m,
            "risk_block_threshold": args.risk_block_threshold,
            "no_bbox_obstacles": bool(args.no_bbox_obstacles),
        },
        "collision_only": result_to_dict(collision_only),
        "risk_aware": result_to_dict(risk_aware),
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with debug_path.open("w", encoding="utf-8") as f:
        json.dump(debug_rows, f, indent=2)

    print("\nSaved outputs:")
    print(f"  collision path : {collision_path_path}")
    print(f"  risk-aware path: {risk_aware_path_path}")
    print(f"  metrics        : {metrics_path}")
    print(f"  mask debug     : {debug_path}")

    print("\nTo visualize:")
    print(
        "  python scripts/ur5_visualize_paths.py "
        f"--risk-npz {args.risk_npz} "
        f"--scene-json {args.scene_json} "
        f"--collision-path {collision_path_path} "
        f"--risk-aware-path {risk_aware_path_path}"
    )


if __name__ == "__main__":
    main()