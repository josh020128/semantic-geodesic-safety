from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from semantic_safety.ur5_experiment.pyroki_solver import (
    PyRokiIKConfig,
    PyRokiUR5IKSolver,
)


def make_json_safe(value: Any):
    """
    Convert numpy / inf / nan values into JSON-safe Python objects.
    """
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}

    if isinstance(value, list):
        return [make_json_safe(v) for v in value]

    if isinstance(value, tuple):
        return [make_json_safe(v) for v in value]

    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())

    if isinstance(value, (np.float32, np.float64, float)):
        v = float(value)
        if not np.isfinite(v):
            return None
        return v

    if isinstance(value, (np.int32, np.int64, int)):
        return int(value)

    if isinstance(value, (np.bool_, bool)):
        return bool(value)

    return value


def compute_joint_path_length(q_traj: np.ndarray) -> float:
    q_traj = np.asarray(q_traj, dtype=np.float64)

    if len(q_traj) < 2:
        return 0.0

    q_unwrapped = np.unwrap(q_traj, axis=0)
    return float(np.sum(np.linalg.norm(np.diff(q_unwrapped, axis=0), axis=1)))


def compute_joint_smoothness(q_traj: np.ndarray) -> float:
    """
    Acceleration-like smoothness score:
        sum ||q[i+1] - 2q[i] + q[i-1]||^2
    """
    q_traj = np.asarray(q_traj, dtype=np.float64)

    if len(q_traj) < 3:
        return 0.0

    q_unwrapped = np.unwrap(q_traj, axis=0)
    acc = q_unwrapped[2:] - 2.0 * q_unwrapped[1:-1] + q_unwrapped[:-2]
    return float(np.sum(np.linalg.norm(acc, axis=1) ** 2))


def per_waypoint_debug(seq_result) -> dict[str, np.ndarray]:
    pos_errors = np.asarray(
        [r.pos_error_m for r in seq_result.waypoint_results],
        dtype=np.float64,
    )
    rot_errors = np.asarray(
        [r.rot_error_rad for r in seq_result.waypoint_results],
        dtype=np.float64,
    )
    success = np.asarray(
        [r.success for r in seq_result.waypoint_results],
        dtype=bool,
    )

    target_positions_pyroki = np.asarray(
        [r.target_position_pyroki for r in seq_result.waypoint_results],
        dtype=np.float64,
    )
    target_wxyzs_pyroki = np.asarray(
        [r.target_wxyz_pyroki for r in seq_result.waypoint_results],
        dtype=np.float64,
    )

    fk_positions = []
    fk_wxyzs = []
    for r in seq_result.waypoint_results:
        if r.fk_position_pyroki is None:
            fk_positions.append(np.full(3, np.nan))
        else:
            fk_positions.append(r.fk_position_pyroki)

        if r.fk_wxyz_pyroki is None:
            fk_wxyzs.append(np.full(4, np.nan))
        else:
            fk_wxyzs.append(r.fk_wxyz_pyroki)

    fk_positions_pyroki = np.asarray(fk_positions, dtype=np.float64)
    fk_wxyzs_pyroki = np.asarray(fk_wxyzs, dtype=np.float64)

    return {
        "pos_errors": pos_errors,
        "rot_errors": rot_errors,
        "success": success,
        "target_positions_pyroki": target_positions_pyroki,
        "target_wxyzs_pyroki": target_wxyzs_pyroki,
        "fk_positions_pyroki": fk_positions_pyroki,
        "fk_wxyzs_pyroki": fk_wxyzs_pyroki,
    }


def save_waypoint_debug(out_dir: Path, prefix: str, seq_result) -> dict[str, str]:
    debug = per_waypoint_debug(seq_result)

    paths = {}
    for key, arr in debug.items():
        path = out_dir / f"{prefix}_{key}.npy"
        np.save(path, arr)
        paths[key] = str(path)

    return paths


def print_summary(name: str, summary: dict) -> None:
    print(f"\n=== {name} PyRoki IK summary ===")
    for k, v in summary.items():
        if k == "messages":
            continue
        print(f"{k}: {v}")

    messages = summary.get("messages", [])
    if messages:
        unique_messages = {}
        for msg in messages:
            unique_messages[msg] = unique_messages.get(msg, 0) + 1

        print("messages:")
        for msg, count in unique_messages.items():
            print(f"  {msg}: {count}")


def solve_one_path(
    *,
    name: str,
    poses_path: Path,
    solver: PyRokiUR5IKSolver,
    q_seed: np.ndarray | None,
    out_dir: Path,
) -> dict:
    poses = np.load(poses_path)

    print(f"\nLoaded {name} poses:")
    print(f"  path : {poses_path}")
    print(f"  shape: {poses.shape}")

    seq_result = solver.solve_pose_sequence(
        poses,
        q_seed_mujoco=q_seed,
    )

    q_traj = seq_result.q_traj_mujoco_order

    q_path = out_dir / f"{name}_q_traj.npy"
    np.save(q_path, q_traj)

    waypoint_debug_paths = save_waypoint_debug(
        out_dir=out_dir,
        prefix=name,
        seq_result=seq_result,
    )

    summary = solver.summarize_sequence_result(seq_result)
    summary["q_traj_path"] = str(q_path)
    summary["poses_path"] = str(poses_path)
    summary["joint_path_length_rad"] = compute_joint_path_length(q_traj)
    summary["joint_smoothness"] = compute_joint_smoothness(q_traj)
    summary["waypoint_debug_paths"] = waypoint_debug_paths

    print_summary(name, summary)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve UR5 IK using PyRoki for processed Cartesian pose waypoints."
    )

    parser.add_argument(
        "--collision-poses",
        type=str,
        default="out/ur5_trajectory/collision_only_poses.npy",
        help="Processed collision-only pose waypoints, shape (N, 4, 4).",
    )
    parser.add_argument(
        "--risk-aware-poses",
        type=str,
        default="out/ur5_trajectory/risk_aware_poses.npy",
        help="Processed risk-aware pose waypoints, shape (N, 4, 4).",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="out/ur5_pyroki_ik",
    )

    parser.add_argument(
        "--robot-description",
        type=str,
        default="ur5e_description",
        help="robot_descriptions name used by PyRoki.",
    )
    parser.add_argument(
        "--target-link-name",
        type=str,
        default="tool0",
        help="PyRoki target link. Usually one of: tool0, flange, wrist_3_link.",
    )
    parser.add_argument(
        "--pyroki-examples-path",
        type=str,
        default="/home/gl34/research/pyroki/examples",
        help="Path to pyroki/examples so pyroki_snippets can be imported.",
    )

    parser.add_argument(
        "--frame-alignment",
        type=str,
        default="mujoco_to_pyroki_z180",
        choices=["mujoco_to_pyroki_z180", "identity"],
        help=(
            "Transform from MuJoCo world pose to PyRoki URDF world pose. "
            "Observed UR5e setup uses diag(-1,-1,1), i.e. z180."
        ),
    )

    parser.add_argument(
        "--q-seed",
        type=float,
        nargs=6,
        default=[3.1416, -1.2, 1.6, -1.9, -1.5708, 0.0],
        help=(
            "Optional seed posture in MuJoCo joint order. "
            "Current PyRoki solve_ik may not warm-start internally, but this is used "
            "for continuity of equivalent 2pi joint solutions."
        ),
    )

    parser.add_argument("--pos-tol-m", type=float, default=1e-2)
    parser.add_argument("--rot-tol-rad", type=float, default=0.15)

    parser.add_argument(
        "--use-manipulability",
        action="store_true",
        help="Use pyroki_snippets.solve_ik_with_manipulability instead of solve_ik.",
    )
    parser.add_argument(
        "--manipulability-weight",
        type=float,
        default=0.0,
    )

    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    q_seed = None
    if args.q_seed is not None:
        q_seed = np.asarray(args.q_seed, dtype=np.float64)

    config = PyRokiIKConfig(
        robot_description=args.robot_description,
        target_link_name=args.target_link_name,
        pyroki_examples_path=args.pyroki_examples_path,
        frame_alignment=args.frame_alignment,
        use_manipulability=bool(args.use_manipulability),
        manipulability_weight=float(args.manipulability_weight),
        pos_tol_m=float(args.pos_tol_m),
        rot_tol_rad=float(args.rot_tol_rad),
        verbose=bool(args.verbose),
    )

    print("\n=== PyRoki solver config ===")
    print(config)

    solver = PyRokiUR5IKSolver(config)

    print("\n=== PyRoki robot info ===")
    print("actuated joints:", solver.joint_names)
    print("links:", solver.link_names)
    print("target link:", args.target_link_name)
    print("frame alignment:", args.frame_alignment)
    print("q seed:", None if q_seed is None else np.round(q_seed, 5))

    metrics = {
        "config": {
            "robot_description": args.robot_description,
            "target_link_name": args.target_link_name,
            "pyroki_examples_path": args.pyroki_examples_path,
            "frame_alignment": args.frame_alignment,
            "pos_tol_m": float(args.pos_tol_m),
            "rot_tol_rad": float(args.rot_tol_rad),
            "use_manipulability": bool(args.use_manipulability),
            "manipulability_weight": float(args.manipulability_weight),
            "q_seed": None if q_seed is None else q_seed.tolist(),
        },
        "robot": {
            "actuated_joints": solver.joint_names,
            "links": solver.link_names,
        },
    }

    collision_summary = solve_one_path(
        name="collision_only",
        poses_path=Path(args.collision_poses),
        solver=solver,
        q_seed=q_seed,
        out_dir=out_dir,
    )

    risk_summary = solve_one_path(
        name="risk_aware",
        poses_path=Path(args.risk_aware_poses),
        solver=solver,
        q_seed=q_seed,
        out_dir=out_dir,
    )

    metrics["collision_only"] = collision_summary
    metrics["risk_aware"] = risk_summary

    metrics_path = out_dir / "pyroki_ik_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(make_json_safe(metrics), f, indent=2)

    print("\nSaved PyRoki IK outputs:")
    print(f"  {out_dir / 'collision_only_q_traj.npy'}")
    print(f"  {out_dir / 'risk_aware_q_traj.npy'}")
    print(f"  {metrics_path}")

    print("\nTo replay PyRoki result in MuJoCo:")
    print(
        "  python scripts/ur5_replay_trajectory.py "
        "--xml-path data/assets/universal_robots_ur5e/ur5_power_drill_center.xml "
        "--ee-site-name attachment_site "
        "--camera-name replay_cam "
        f"--collision-q-traj {out_dir / 'collision_only_q_traj.npy'} "
        f"--risk-aware-q-traj {out_dir / 'risk_aware_q_traj.npy'} "
        f"--out-dir {out_dir / 'replay'} "
        "--width 640 --height 480"
    )


if __name__ == "__main__":
    main()