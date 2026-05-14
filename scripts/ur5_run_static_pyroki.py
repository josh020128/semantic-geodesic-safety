from __future__ import annotations

import argparse
import sys
from pathlib import Path


_THIS_FILE = Path(__file__).resolve()
_SCRIPT_DIR = _THIS_FILE.parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from ur5_run_static_ik import (  # noqa: E402
    PYTHON,
    add_common_args,
    build_run_dir,
    launch_viewer,
    run_cmd,
    run_common_until_trajectory,
    run_replay_and_executed_visualization,
)


def run_pyroki_ik(args: argparse.Namespace, run_dir: Path) -> Path:
    ik_dir = run_dir / "ik_pyroki"

    cmd = [
        PYTHON,
        "scripts/ur5_solve_pyroki.py",
        "--collision-poses",
        str(run_dir / "trajectory" / "collision_only_poses.npy"),
        "--risk-aware-poses",
        str(run_dir / "trajectory" / "risk_aware_poses.npy"),
        "--robot-description",
        args.robot_description,
        "--target-link-name",
        args.target_link_name,
        "--frame-alignment",
        args.frame_alignment,
        "--q-seed",
        *(str(x) for x in args.q_seed),
        "--out-dir",
        str(ik_dir),
    ]

    if args.pyroki_examples_path:
        cmd.extend(["--pyroki-examples-path", args.pyroki_examples_path])

    if args.use_manipulability:
        cmd.append("--use-manipulability")
        cmd.extend(["--manipulability-weight", str(args.manipulability_weight)])

    run_cmd(cmd)
    return ik_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run full static UR5 pipeline using PyRoki warm-start IK."
    )

    add_common_args(parser)

    parser.add_argument("--robot-description", type=str, default="ur5e_description")
    parser.add_argument("--target-link-name", type=str, default="tool0")
    parser.add_argument(
        "--frame-alignment",
        type=str,
        default="mujoco_to_pyroki_z180",
        choices=["mujoco_to_pyroki_z180", "identity"],
    )
    parser.add_argument(
        "--pyroki-examples-path",
        type=str,
        default="/home/gl34/research/pyroki/examples",
    )
    parser.add_argument("--use-manipulability", action="store_true")
    parser.add_argument("--manipulability-weight", type=float, default=0.0)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    args.solver_name = "pyroki"
    args.xml_path = str(Path(args.xml_path).resolve())
    args.prior_json_path = str(Path(args.prior_json_path).resolve())

    run_dir = build_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== UR5 Static Experiment: PyRoki Warm-start IK ===")
    print(f"run_dir: {run_dir}")
    print(f"xml    : {args.xml_path}")

    run_common_until_trajectory(args, run_dir)

    solver_dir = run_pyroki_ik(args, run_dir)

    run_replay_and_executed_visualization(
        args=args,
        run_dir=run_dir,
        solver_dir=solver_dir,
        replay_subdir="replay_pyroki",
        viz_subdir="viz_pyroki_executed",
    )

    if args.launch_viewer:
        launch_viewer(args, solver_dir)

    print("\nDONE.")
    print(f"Outputs saved under: {run_dir}")


if __name__ == "__main__":
    main()