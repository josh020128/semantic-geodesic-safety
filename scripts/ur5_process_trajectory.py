from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from semantic_safety.ur5_experiment.risk_volume_query import RiskVolumeQuery
from semantic_safety.ur5_experiment.trajectory import (
    TrajectoryProcessor,
    TrajectoryProcessingConfig,
    compute_path_length,
    make_pose_waypoints,
    rotation_matrix_from_rpy,
)


def load_scene_objects(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        print(f"Warning: scene JSON not found: {path}")
        return []

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_point_valid_fn(
    rv: RiskVolumeQuery,
    planner_free_mask: np.ndarray,
    margin_m: float,
):
    """
    Conservative point validity against the planner_free_mask.

    This uses the bbox-solid planner obstacle mask, not only the raw
    occupancy_free saved by Loop 1.
    """

    planner_free_mask = np.asarray(planner_free_mask, dtype=bool)

    if planner_free_mask.shape != rv.shape:
        raise ValueError(
            f"planner_free_mask shape {planner_free_mask.shape} does not match rv shape {rv.shape}"
        )

    margin_voxels = int(np.ceil(margin_m / float(np.min(rv.spacing))))

    def point_is_valid(p: np.ndarray) -> bool:
        p = np.asarray(p, dtype=np.float64)

        if not rv.is_inside_bounds(p):
            return False

        idx = rv.world_to_grid_idx(p, clip=True, nearest=True)

        ix0 = max(0, idx.ix - margin_voxels)
        ix1 = min(rv.shape[0], idx.ix + margin_voxels + 1)
        iy0 = max(0, idx.iy - margin_voxels)
        iy1 = min(rv.shape[1], idx.iy + margin_voxels + 1)
        iz0 = max(0, idx.iz - margin_voxels)
        iz1 = min(rv.shape[2], idx.iz + margin_voxels + 1)

        local = planner_free_mask[ix0:ix1, iy0:iy1, iz0:iz1]
        if local.size == 0:
            return False

        return bool(np.all(local))

    return point_is_valid


def build_segment_valid_fn(point_is_valid, step_m: float):
    def segment_is_valid(p0: np.ndarray, p1: np.ndarray) -> bool:
        p0 = np.asarray(p0, dtype=np.float64)
        p1 = np.asarray(p1, dtype=np.float64)

        dist = float(np.linalg.norm(p1 - p0))
        if dist <= 1e-12:
            return bool(point_is_valid(p0))

        n = max(1, int(np.ceil(dist / step_m)))
        for t in np.linspace(0.0, 1.0, n + 1):
            p = (1.0 - t) * p0 + t * p1
            if not point_is_valid(p):
                return False

        return True

    return segment_is_valid


def compute_integrated_risk(rv: RiskVolumeQuery, path: np.ndarray) -> float:
    path = np.asarray(path, dtype=np.float64)

    if len(path) < 2:
        return 0.0

    risks = np.asarray(
        [rv.sample_risk_trilinear(p, outside_value=0.0) for p in path],
        dtype=np.float64,
    )

    dists = np.linalg.norm(np.diff(path, axis=0), axis=1)
    avg_risks = 0.5 * (risks[:-1] + risks[1:])

    return float(np.sum(avg_risks * dists))


def compute_max_risk(rv: RiskVolumeQuery, path: np.ndarray) -> float:
    path = np.asarray(path, dtype=np.float64)

    if len(path) == 0:
        return 0.0

    risks = [rv.sample_risk_trilinear(p, outside_value=0.0) for p in path]
    return float(np.max(risks))


def count_invalid_points(point_is_valid, path: np.ndarray) -> int:
    path = np.asarray(path, dtype=np.float64)
    return int(sum(not bool(point_is_valid(p)) for p in path))


def count_invalid_segments(segment_is_valid, path: np.ndarray) -> int:
    path = np.asarray(path, dtype=np.float64)

    if len(path) < 2:
        return 0

    invalid = 0
    for p0, p1 in zip(path[:-1], path[1:]):
        if not bool(segment_is_valid(p0, p1)):
            invalid += 1

    return int(invalid)


def process_one_path(
    *,
    name: str,
    raw_path: np.ndarray,
    rv: RiskVolumeQuery,
    point_is_valid,
    segment_is_valid,
    fixed_rotation: np.ndarray,
    enable_shortcut: bool,
    enable_smoothing: bool,
    smoothing_window: int,
    shortcut_step_m: float,
    resample_spacing_m: float,
) -> tuple[dict, dict[str, np.ndarray]]:
    config = TrajectoryProcessingConfig(
        enable_shortcut=enable_shortcut,
        shortcut_step_m=shortcut_step_m,
        resample_spacing_m=resample_spacing_m,
        enable_smoothing=enable_smoothing,
        smoothing_window=smoothing_window,
        validate_smoothed_points=True,
    )

    processor = TrajectoryProcessor(config)

    processed = processor.process(
        raw_path,
        point_is_valid=point_is_valid,
        segment_is_valid=segment_is_valid,
        fixed_rotation=fixed_rotation,
    )

    pose_dict = make_pose_waypoints(
        processed.final_path,
        fixed_rotation,
    )

    raw_integrated_risk = compute_integrated_risk(rv, processed.raw_path)
    final_integrated_risk = compute_integrated_risk(rv, processed.final_path)

    raw_max_risk = compute_max_risk(rv, processed.raw_path)
    final_max_risk = compute_max_risk(rv, processed.final_path)

    invalid_points = count_invalid_points(point_is_valid, processed.final_path)
    invalid_segments = count_invalid_segments(segment_is_valid, processed.final_path)

    metrics = {
        "name": name,
        "raw_points": int(len(processed.raw_path)),
        "simplified_points": int(len(processed.simplified_path)),
        "resampled_points": int(len(processed.resampled_path)),
        "smoothed_points": int(len(processed.smoothed_path)),
        "final_points": int(len(processed.final_path)),
        "path_length_raw": float(compute_path_length(processed.raw_path)),
        "path_length_final": float(compute_path_length(processed.final_path)),
        "integrated_risk_raw": float(raw_integrated_risk),
        "integrated_risk_final": float(final_integrated_risk),
        "max_risk_raw": float(raw_max_risk),
        "max_risk_final": float(final_max_risk),
        "invalid_final_points": int(invalid_points),
        "invalid_final_segments": int(invalid_segments),
        "enable_shortcut": bool(enable_shortcut),
        "enable_smoothing": bool(enable_smoothing),
        "resample_spacing_m": float(resample_spacing_m),
        "shortcut_step_m": float(shortcut_step_m),
        "smoothing_window": int(smoothing_window),
    }

    arrays = {
        "raw_path": processed.raw_path,
        "simplified_path": processed.simplified_path,
        "resampled_path": processed.resampled_path,
        "smoothed_path": processed.smoothed_path,
        "final_path": processed.final_path,
        "poses": pose_dict["poses"],
        "positions": pose_dict["positions"],
        "rotations": pose_dict["rotations"],
        "quaternions_xyzw": pose_dict["quaternions_xyzw"],
    }

    return metrics, arrays


def print_metrics(metrics: dict) -> None:
    print(f"\n=== {metrics['name']} trajectory processing ===")
    print("raw points             :", metrics["raw_points"])
    print("simplified points      :", metrics["simplified_points"])
    print("resampled points       :", metrics["resampled_points"])
    print("final points           :", metrics["final_points"])
    print("path length raw        :", metrics["path_length_raw"])
    print("path length final      :", metrics["path_length_final"])
    print("integrated risk raw    :", metrics["integrated_risk_raw"])
    print("integrated risk final  :", metrics["integrated_risk_final"])
    print("max risk raw           :", metrics["max_risk_raw"])
    print("max risk final         :", metrics["max_risk_final"])
    print("invalid final points   :", metrics["invalid_final_points"])
    print("invalid final segments :", metrics["invalid_final_segments"])
    print("shortcut enabled       :", metrics["enable_shortcut"])
    print("smoothing enabled      :", metrics["enable_smoothing"])


def save_arrays(out_dir: Path, prefix: str, arrays: dict[str, np.ndarray]) -> None:
    for key, arr in arrays.items():
        np.save(out_dir / f"{prefix}_{key}.npy", arr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process raw A* Cartesian paths into IK/PyRoki-friendly pose waypoints."
    )

    parser.add_argument("--risk-npz", type=str, default="loop1_risk_field.npz")
    parser.add_argument("--scene-json", type=str, default="loop1_scene_objects.json")

    parser.add_argument(
        "--collision-path",
        type=str,
        default="out/ur5_planning/collision_only_path.npy",
    )
    parser.add_argument(
        "--risk-aware-path",
        type=str,
        default="out/ur5_planning/risk_aware_path.npy",
    )

    parser.add_argument("--out-dir", type=str, default="out/ur5_trajectory")

    parser.add_argument("--bbox-pad-xy-m", type=float, default=0.02)
    parser.add_argument("--bbox-pad-z-m", type=float, default=0.01)
    parser.add_argument("--valid-margin-m", type=float, default=0.02)

    parser.add_argument("--resample-spacing-m", type=float, default=0.03)
    parser.add_argument("--shortcut-step-m", type=float, default=0.01)
    parser.add_argument("--segment-check-step-m", type=float, default=0.01)

    parser.add_argument(
        "--collision-shortcut",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable shortcutting for collision-only path.",
    )
    parser.add_argument(
        "--risk-aware-shortcut",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable shortcutting for risk-aware path. Default false to preserve semantic detour.",
    )

    parser.add_argument(
        "--smoothing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable moving-average smoothing.",
    )
    parser.add_argument("--smoothing-window", type=int, default=3)

    parser.add_argument(
        "--fixed-rpy",
        type=float,
        nargs=3,
        default=[0.0, np.pi, 0.0],
        help="Fixed end-effector orientation as roll pitch yaw in radians.",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rv = RiskVolumeQuery.from_npz(args.risk_npz)
    print(rv.summary())

    scene_objects = load_scene_objects(args.scene_json)

    planner_free, debug_rows = rv.build_planner_free_mask_from_bboxes(
        scene_objects,
        pad_xy_m=args.bbox_pad_xy_m,
        pad_z_m=args.bbox_pad_z_m,
        return_debug=True,
    )

    rv.print_planner_mask_debug(planner_free, debug_rows)

    point_is_valid = build_point_valid_fn(
        rv=rv,
        planner_free_mask=planner_free,
        margin_m=args.valid_margin_m,
    )

    segment_is_valid = build_segment_valid_fn(
        point_is_valid,
        step_m=args.segment_check_step_m,
    )

    collision_raw = np.load(args.collision_path)
    risk_aware_raw = np.load(args.risk_aware_path)

    fixed_rotation = rotation_matrix_from_rpy(
        roll=float(args.fixed_rpy[0]),
        pitch=float(args.fixed_rpy[1]),
        yaw=float(args.fixed_rpy[2]),
    )

    collision_metrics, collision_arrays = process_one_path(
        name="collision_only",
        raw_path=collision_raw,
        rv=rv,
        point_is_valid=point_is_valid,
        segment_is_valid=segment_is_valid,
        fixed_rotation=fixed_rotation,
        enable_shortcut=bool(args.collision_shortcut),
        enable_smoothing=bool(args.smoothing),
        smoothing_window=int(args.smoothing_window),
        shortcut_step_m=float(args.shortcut_step_m),
        resample_spacing_m=float(args.resample_spacing_m),
    )

    risk_metrics, risk_arrays = process_one_path(
        name="risk_aware",
        raw_path=risk_aware_raw,
        rv=rv,
        point_is_valid=point_is_valid,
        segment_is_valid=segment_is_valid,
        fixed_rotation=fixed_rotation,
        enable_shortcut=bool(args.risk_aware_shortcut),
        enable_smoothing=bool(args.smoothing),
        smoothing_window=int(args.smoothing_window),
        shortcut_step_m=float(args.shortcut_step_m),
        resample_spacing_m=float(args.resample_spacing_m),
    )

    print_metrics(collision_metrics)
    print_metrics(risk_metrics)

    save_arrays(out_dir, "collision_only", collision_arrays)
    save_arrays(out_dir, "risk_aware", risk_arrays)

    metrics = {
        "risk_npz": args.risk_npz,
        "scene_json": args.scene_json,
        "collision_path": args.collision_path,
        "risk_aware_path": args.risk_aware_path,
        "config": {
            "bbox_pad_xy_m": float(args.bbox_pad_xy_m),
            "bbox_pad_z_m": float(args.bbox_pad_z_m),
            "valid_margin_m": float(args.valid_margin_m),
            "resample_spacing_m": float(args.resample_spacing_m),
            "shortcut_step_m": float(args.shortcut_step_m),
            "segment_check_step_m": float(args.segment_check_step_m),
            "collision_shortcut": bool(args.collision_shortcut),
            "risk_aware_shortcut": bool(args.risk_aware_shortcut),
            "smoothing": bool(args.smoothing),
            "smoothing_window": int(args.smoothing_window),
            "fixed_rpy": [float(x) for x in args.fixed_rpy],
        },
        "collision_only": collision_metrics,
        "risk_aware": risk_metrics,
        "planner_mask_debug": debug_rows,
    }

    metrics_path = out_dir / "trajectory_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved processed trajectory outputs:")
    print(f"  out dir: {out_dir}")
    print(f"  metrics: {metrics_path}")

    print("\nImportant files:")
    print(f"  {out_dir / 'collision_only_final_path.npy'}")
    print(f"  {out_dir / 'risk_aware_final_path.npy'}")
    print(f"  {out_dir / 'collision_only_poses.npy'}")
    print(f"  {out_dir / 'risk_aware_poses.npy'}")


if __name__ == "__main__":
    main()