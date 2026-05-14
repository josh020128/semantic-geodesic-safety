from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from semantic_safety.ur5_experiment.ik_solver import (
    IKConfig,
    MujocoDampedLeastSquaresIK,
)
from semantic_safety.ur5_experiment.mujoco_ur5_env import (
    MujocoUR5Env,
    UR5EnvConfig,
)


def result_summary(seq_result) -> dict:
    pos_errors = [float(r.final_pos_error_m) for r in seq_result.waypoint_results]
    rot_errors = [float(r.final_rot_error_rad) for r in seq_result.waypoint_results]
    iters = [int(r.num_iters) for r in seq_result.waypoint_results]

    if len(pos_errors) == 0:
        return {
            "success": False,
            "success_rate": 0.0,
            "num_waypoints": 0,
            "num_failed": 0,
        }

    return {
        "success": bool(seq_result.success),
        "success_rate": float(seq_result.success_rate),
        "num_waypoints": int(len(seq_result.waypoint_results)),
        "num_failed": int(len(seq_result.failed_indices)),
        "failed_indices": [int(i) for i in seq_result.failed_indices],
        "mean_pos_error_m": float(np.mean(pos_errors)),
        "max_pos_error_m": float(np.max(pos_errors)),
        "mean_rot_error_rad": float(np.mean(rot_errors)),
        "max_rot_error_rad": float(np.max(rot_errors)),
        "mean_iters": float(np.mean(iters)),
        "max_iters": int(np.max(iters)),
    }


def print_summary(name: str, summary: dict) -> None:
    print(f"\n=== {name} IK summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")


def solve_one(
    *,
    name: str,
    poses_path: Path,
    env: MujocoUR5Env,
    q_seed: np.ndarray,
    config: IKConfig,
    out_dir: Path,
    stop_on_failure: bool,
):
    poses = np.load(poses_path)
    print(f"\nLoaded {name} poses: {poses_path}, shape={poses.shape}")

    solver = MujocoDampedLeastSquaresIK(env, config)

    # Reset to seed before each path so comparison is fair.
    env.set_qpos(q_seed, zero_velocity=True, forward=True)

    seq_result = solver.solve_pose_sequence(
        poses,
        q_seed=q_seed,
        stop_on_failure=stop_on_failure,
    )

    q_path = out_dir / f"{name}_q_traj.npy"
    np.save(q_path, seq_result.q_traj)

    summary = result_summary(seq_result)
    summary["q_traj_path"] = str(q_path)
    summary["poses_path"] = str(poses_path)

    print_summary(name, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve UR5 IK for processed Cartesian pose waypoints."
    )

    parser.add_argument(
        "--xml-path",
        type=str,
        default="data/assets/universal_robots_ur5e/ur5_power_drill_center.xml",
    )
    parser.add_argument("--ee-site-name", type=str, default="attachment_site")
    parser.add_argument("--camera-name", type=str, default="main_cam")

    parser.add_argument(
        "--collision-poses",
        type=str,
        default="out/ur5_trajectory/collision_only_poses.npy",
    )
    parser.add_argument(
        "--risk-aware-poses",
        type=str,
        default="out/ur5_trajectory/risk_aware_poses.npy",
    )
    parser.add_argument("--out-dir", type=str, default="out/ur5_ik")

    parser.add_argument(
        "--q-seed",
        type=float,
        nargs=6,
        default=[3.1416, -1.2, 1.6, -1.9, -1.5708, 0.0],
        help="Initial UR5 seed posture in radians.",
    )

    parser.add_argument("--max-iters", type=int, default=250)
    parser.add_argument("--pos-tol-m", type=float, default=3e-3)
    parser.add_argument("--rot-tol-rad", type=float, default=8e-2)
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--max-delta-q-norm", type=float, default=0.15)

    parser.add_argument(
        "--position-only",
        action="store_true",
        help="Ignore orientation during IK. Useful for first feasibility debug.",
    )
    parser.add_argument("--orientation-weight", type=float, default=0.25)
    parser.add_argument("--nominal-weight", type=float, default=1e-3)

    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    q_seed = np.asarray(args.q_seed, dtype=np.float64)

    env = MujocoUR5Env(
        UR5EnvConfig(
            xml_path=args.xml_path,
            ee_site_name=args.ee_site_name,
            camera_name=args.camera_name,
            settle_steps=0,
        )
    )

    env.set_qpos(q_seed, zero_velocity=True, forward=True)

    print("\n=== IK Environment ===")
    env.print_debug_summary()
    print("\nUsing q_seed:", np.round(q_seed, 5))
    print("Seed EE position:", np.round(env.get_ee_position(), 5))

    ik_config = IKConfig(
        max_iters=int(args.max_iters),
        pos_tol_m=float(args.pos_tol_m),
        rot_tol_rad=float(args.rot_tol_rad),
        damping=float(args.damping),
        step_size=float(args.step_size),
        max_delta_q_norm=float(args.max_delta_q_norm),
        use_orientation=not bool(args.position_only),
        orientation_weight=float(args.orientation_weight),
        nominal_weight=float(args.nominal_weight),
        q_nominal=tuple(float(x) for x in q_seed),
        verbose=bool(args.verbose),
    )

    metrics = {
        "xml_path": args.xml_path,
        "ee_site_name": args.ee_site_name,
        "q_seed": q_seed.tolist(),
        "ik_config": {
            "max_iters": ik_config.max_iters,
            "pos_tol_m": ik_config.pos_tol_m,
            "rot_tol_rad": ik_config.rot_tol_rad,
            "damping": ik_config.damping,
            "step_size": ik_config.step_size,
            "max_delta_q_norm": ik_config.max_delta_q_norm,
            "use_orientation": ik_config.use_orientation,
            "orientation_weight": ik_config.orientation_weight,
            "nominal_weight": ik_config.nominal_weight,
        },
    }

    collision_summary = solve_one(
        name="collision_only",
        poses_path=Path(args.collision_poses),
        env=env,
        q_seed=q_seed,
        config=ik_config,
        out_dir=out_dir,
        stop_on_failure=bool(args.stop_on_failure),
    )

    risk_summary = solve_one(
        name="risk_aware",
        poses_path=Path(args.risk_aware_poses),
        env=env,
        q_seed=q_seed,
        config=ik_config,
        out_dir=out_dir,
        stop_on_failure=bool(args.stop_on_failure),
    )

    metrics["collision_only"] = collision_summary
    metrics["risk_aware"] = risk_summary

    metrics_path = out_dir / "ik_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved:")
    print(f"  {out_dir / 'collision_only_q_traj.npy'}")
    print(f"  {out_dir / 'risk_aware_q_traj.npy'}")
    print(f"  {metrics_path}")


if __name__ == "__main__":
    main()