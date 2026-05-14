from __future__ import annotations

import argparse
import json
import re
import shutil
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

DEFAULT_PRIOR_JSON = PROJECT_ROOT / "data" / "semantic_risk_demo_claude.json"

DEFAULT_START = [0.30, -0.25, 0.22]
DEFAULT_GOAL = [0.80, 0.25, 0.22]
DEFAULT_Q_SEED = [3.1416, -1.2, 1.6, -1.9, -1.5708, 0.0]


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def xml_situation_slug(xml_path: str | Path) -> str:
    stem = Path(xml_path).stem

    for prefix in ["ur5_", "tabletop_", "semantic_safety_"]:
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break

    return slugify(stem)


def build_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir is not None:
        return Path(args.run_dir)

    manipulated_slug = slugify(args.manipulated)
    env_slug = xml_situation_slug(args.xml_path)

    return Path(args.out_root) / f"ur5_{manipulated_slug}_{env_slug}"


def run_cmd(cmd: list[str], *, cwd: Path = PROJECT_ROOT) -> None:
    print("\n" + "=" * 100)
    print("RUN:", shlex.join(str(x) for x in cmd))
    print("=" * 100)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def copy_output_if_exists(src: Path, dst_dir: Path) -> None:
    if not src.exists():
        return

    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name

    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def save_run_config(args: argparse.Namespace, run_dir: Path, solver_name: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "solver": solver_name,
        "manipulated": args.manipulated,
        "xml_path": str(Path(args.xml_path).resolve()),
        "candidate_labels": args.candidate_labels,
        "scene_label": args.scene_label,
        "prior_json_path": str(Path(args.prior_json_path).resolve()),
        "start": args.start,
        "goal": args.goal,
        "stride": args.stride,
        "risk_weight": args.risk_weight,
        "q_seed": args.q_seed,
        "save_video": args.save_video,
        "launch_viewer": args.launch_viewer,
        "run_dir": str(run_dir),
    }

    with (run_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def run_risk_generation(args: argparse.Namespace, run_dir: Path) -> None:
    risk_dir = run_dir / "risk"
    risk_dir.mkdir(parents=True, exist_ok=True)

    scene_label = args.scene_label
    if scene_label is None:
        scene_label = args.candidate_labels[0]

    cmd = [
        PYTHON,
        "scripts/test_full_siglip2_pipeline.py",
        "--xml-path",
        str(args.xml_path),
        "--manipulated",
        args.manipulated,
        "--scene-label",
        scene_label,
        "--candidate-labels",
        *args.candidate_labels,
        "--prior-json-path",
        str(args.prior_json_path),
    ]

    # Important:
    # Do NOT add --strict-candidate-labels.
    run_cmd(cmd)

    outputs_to_copy = [
        "loop1_risk_field.npz",
        "loop1_scene_objects.json",
        "continuous_risk_overlay.png",
        "segmentation_attenuation_overlay.png",
        "test_rgb.png",
        "test_depth_debug.png",
        "perception_debug",
    ]

    for name in outputs_to_copy:
        copy_output_if_exists(PROJECT_ROOT / name, risk_dir)


def run_risk_debug(args: argparse.Namespace, run_dir: Path) -> None:
    cmd = [
        PYTHON,
        "scripts/ur5_debug_risk_volume.py",
        "--risk-npz",
        str(run_dir / "risk" / "loop1_risk_field.npz"),
        "--scene-json",
        str(run_dir / "risk" / "loop1_scene_objects.json"),
        "--probe",
        *(str(x) for x in args.start),
        "--probe",
        *(str(x) for x in args.goal),
    ]

    run_cmd(cmd)


def run_cartesian_planning(args: argparse.Namespace, run_dir: Path) -> None:
    cmd = [
        PYTHON,
        "scripts/ur5_plan_cartesian.py",
        "--risk-npz",
        str(run_dir / "risk" / "loop1_risk_field.npz"),
        "--scene-json",
        str(run_dir / "risk" / "loop1_scene_objects.json"),
        "--start",
        *(str(x) for x in args.start),
        "--goal",
        *(str(x) for x in args.goal),
        "--stride",
        str(args.stride),
        "--risk-weight",
        str(args.risk_weight),
        "--out-dir",
        str(run_dir / "planning"),
    ]

    run_cmd(cmd)


def run_path_visualization(
    *,
    run_dir: Path,
    collision_path: Path,
    risk_aware_path: Path,
    out_subdir: str,
) -> None:
    cmd = [
        PYTHON,
        "scripts/ur5_visualize_paths.py",
        "--risk-npz",
        str(run_dir / "risk" / "loop1_risk_field.npz"),
        "--scene-json",
        str(run_dir / "risk" / "loop1_scene_objects.json"),
        "--collision-path",
        str(collision_path),
        "--risk-aware-path",
        str(risk_aware_path),
        "--out-dir",
        str(run_dir / out_subdir),
    ]

    run_cmd(cmd)


def run_trajectory_processing(args: argparse.Namespace, run_dir: Path) -> None:
    cmd = [
        PYTHON,
        "scripts/ur5_process_trajectory.py",
        "--risk-npz",
        str(run_dir / "risk" / "loop1_risk_field.npz"),
        "--scene-json",
        str(run_dir / "risk" / "loop1_scene_objects.json"),
        "--collision-path",
        str(run_dir / "planning" / "collision_only_path.npy"),
        "--risk-aware-path",
        str(run_dir / "planning" / "risk_aware_path.npy"),
        "--out-dir",
        str(run_dir / "trajectory"),
        "--no-risk-aware-shortcut",
    ]

    run_cmd(cmd)


def run_env_debug(args: argparse.Namespace, run_dir: Path) -> None:
    cmd = [
        PYTHON,
        "scripts/ur5_debug_env.py",
        "--xml-path",
        str(args.xml_path),
        "--ee-site-name",
        args.ee_site_name,
        "--camera-name",
        "main_cam",
        "--qpos",
        *(str(x) for x in args.q_seed),
        "--out-dir",
        str(run_dir / "debug_env"),
    ]

    run_cmd(cmd)


def run_common_until_trajectory(args: argparse.Namespace, run_dir: Path) -> None:
    save_run_config(args, run_dir, solver_name=args.solver_name)

    if not args.skip_risk:
        run_risk_generation(args, run_dir)

    if not args.skip_risk_debug:
        run_risk_debug(args, run_dir)

    run_cartesian_planning(args, run_dir)

    run_path_visualization(
        run_dir=run_dir,
        collision_path=run_dir / "planning" / "collision_only_path.npy",
        risk_aware_path=run_dir / "planning" / "risk_aware_path.npy",
        out_subdir="viz_raw",
    )

    run_trajectory_processing(args, run_dir)

    run_path_visualization(
        run_dir=run_dir,
        collision_path=run_dir / "trajectory" / "collision_only_final_path.npy",
        risk_aware_path=run_dir / "trajectory" / "risk_aware_final_path.npy",
        out_subdir="viz_processed",
    )

    if args.debug_env:
        run_env_debug(args, run_dir)


def run_mujoco_ik(args: argparse.Namespace, run_dir: Path) -> Path:
    ik_dir = run_dir / "ik_mujoco"

    cmd = [
        PYTHON,
        "scripts/ur5_solve_ik.py",
        "--xml-path",
        str(args.xml_path),
        "--ee-site-name",
        args.ee_site_name,
        "--collision-poses",
        str(run_dir / "trajectory" / "collision_only_poses.npy"),
        "--risk-aware-poses",
        str(run_dir / "trajectory" / "risk_aware_poses.npy"),
        "--q-seed",
        *(str(x) for x in args.q_seed),
        "--pos-tol-m",
        str(args.ik_pos_tol_m),
        "--rot-tol-rad",
        str(args.ik_rot_tol_rad),
        "--orientation-weight",
        str(args.ik_orientation_weight),
        "--max-iters",
        str(args.ik_max_iters),
        "--nominal-weight",
        str(args.ik_nominal_weight),
        "--out-dir",
        str(ik_dir),
    ]

    run_cmd(cmd)
    return ik_dir


def run_replay_and_executed_visualization(
    args: argparse.Namespace,
    run_dir: Path,
    solver_dir: Path,
    replay_subdir: str,
    viz_subdir: str,
) -> None:
    replay_dir = run_dir / replay_subdir

    if args.save_video:
        camera_name = args.replay_camera_name
        video_flag = "--save-video"
    else:
        # Empty camera name prevents offscreen renderer setup when no video is needed.
        camera_name = ""
        video_flag = "--no-save-video"

    cmd = [
        PYTHON,
        "scripts/ur5_replay_trajectory.py",
        "--xml-path",
        str(args.xml_path),
        "--ee-site-name",
        args.ee_site_name,
        "--camera-name",
        camera_name,
        "--collision-q-traj",
        str(solver_dir / "collision_only_q_traj.npy"),
        "--risk-aware-q-traj",
        str(solver_dir / "risk_aware_q_traj.npy"),
        "--out-dir",
        str(replay_dir),
        "--width",
        str(args.video_width),
        "--height",
        str(args.video_height),
        video_flag,
        "--no-save-frames",
    ]

    run_cmd(cmd)

    run_path_visualization(
        run_dir=run_dir,
        collision_path=replay_dir / "collision_only_ee_path.npy",
        risk_aware_path=replay_dir / "risk_aware_ee_path.npy",
        out_subdir=viz_subdir,
    )


def launch_viewer(args: argparse.Namespace, solver_dir: Path) -> None:
    cmd = [
        PYTHON,
        "scripts/ur5_viewer_replay.py",
        "--xml-path",
        str(args.xml_path),
        "--ee-site-name",
        args.ee_site_name,
        "--collision-q-traj",
        str(solver_dir / "collision_only_q_traj.npy"),
        "--risk-aware-q-traj",
        str(solver_dir / "risk_aware_q_traj.npy"),
        "--speed",
        str(args.viewer_speed),
    ]

    if args.viewer_loop:
        cmd.append("--loop")

    if args.viewer_fixed_camera:
        cmd.extend(["--fixed-camera", args.viewer_fixed_camera])

    run_cmd(cmd)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manipulated", type=str, required=True)
    parser.add_argument("--xml-path", type=str, required=True)
    parser.add_argument(
        "--candidate-labels",
        type=str,
        nargs="+",
        required=True,
        help='Example: --candidate-labels "power drill" "table"',
    )
    parser.add_argument(
        "--scene-label",
        type=str,
        default=None,
        help="Default: first candidate label.",
    )
    parser.add_argument(
        "--prior-json-path",
        type=str,
        default=str(DEFAULT_PRIOR_JSON),
    )

    parser.add_argument("--out-root", type=str, default="out")
    parser.add_argument("--run-dir", type=str, default=None)

    parser.add_argument("--start", type=float, nargs=3, default=DEFAULT_START)
    parser.add_argument("--goal", type=float, nargs=3, default=DEFAULT_GOAL)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--risk-weight", type=float, default=5.0)

    parser.add_argument("--q-seed", type=float, nargs=6, default=DEFAULT_Q_SEED)
    parser.add_argument("--ee-site-name", type=str, default="attachment_site")

    parser.add_argument(
        "--save-video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save replay AVI video.",
    )
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--replay-camera-name", type=str, default="replay_cam")

    parser.add_argument(
        "--launch-viewer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Launch interactive MuJoCo viewer at the end. Use on Ubuntu desktop.",
    )
    parser.add_argument("--viewer-speed", type=float, default=0.5)
    parser.add_argument("--viewer-loop", action="store_true")
    parser.add_argument("--viewer-fixed-camera", type=str, default="")

    parser.add_argument("--debug-env", action="store_true")
    parser.add_argument("--skip-risk", action="store_true")
    parser.add_argument("--skip-risk-debug", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run full static UR5 pipeline using MuJoCo Jacobian IK."
    )

    add_common_args(parser)

    parser.add_argument("--ik-pos-tol-m", type=float, default=0.008)
    parser.add_argument("--ik-rot-tol-rad", type=float, default=0.10)
    parser.add_argument("--ik-orientation-weight", type=float, default=0.10)
    parser.add_argument("--ik-max-iters", type=int, default=500)
    parser.add_argument("--ik-nominal-weight", type=float, default=0.0)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    args.solver_name = "mujoco_ik"
    args.xml_path = str(Path(args.xml_path).resolve())
    args.prior_json_path = str(Path(args.prior_json_path).resolve())

    run_dir = build_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== UR5 Static Experiment: MuJoCo Jacobian IK ===")
    print(f"run_dir: {run_dir}")
    print(f"xml    : {args.xml_path}")

    run_common_until_trajectory(args, run_dir)

    solver_dir = run_mujoco_ik(args, run_dir)

    run_replay_and_executed_visualization(
        args=args,
        run_dir=run_dir,
        solver_dir=solver_dir,
        replay_subdir="replay_mujoco_ik",
        viz_subdir="viz_mujoco_ik_executed",
    )

    if args.launch_viewer:
        launch_viewer(args, solver_dir)

    print("\nDONE.")
    print(f"Outputs saved under: {run_dir}")


if __name__ == "__main__":
    main()