from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from semantic_safety.ur5_experiment.mujoco_ur5_env import (
    MujocoUR5Env,
    UR5EnvConfig,
)


def unwrap_joint_trajectory(q_traj: np.ndarray) -> np.ndarray:
    """
    Make revolute joint trajectories continuous before interpolation.

    This avoids large visual jumps if one joint crosses +/- pi.
    """
    q_traj = np.asarray(q_traj, dtype=np.float64)

    if q_traj.ndim != 2:
        raise ValueError(f"q_traj must be 2D, got {q_traj.shape}")

    return np.unwrap(q_traj, axis=0)


def interpolate_joint_trajectory(
    q_traj: np.ndarray,
    steps_per_segment: int = 8,
) -> np.ndarray:
    """
    Linearly interpolate between IK waypoint configurations for smoother replay.
    """
    q_traj = np.asarray(q_traj, dtype=np.float64)

    if q_traj.ndim != 2:
        raise ValueError(f"q_traj must have shape (N, D), got {q_traj.shape}")

    if len(q_traj) <= 1:
        return q_traj.copy()

    steps_per_segment = max(1, int(steps_per_segment))

    q_unwrapped = unwrap_joint_trajectory(q_traj)

    frames = []
    for i in range(len(q_unwrapped) - 1):
        q0 = q_unwrapped[i]
        q1 = q_unwrapped[i + 1]

        for s in range(steps_per_segment):
            t = s / float(steps_per_segment)
            q = (1.0 - t) * q0 + t * q1
            frames.append(q)

    frames.append(q_unwrapped[-1])
    return np.asarray(frames, dtype=np.float64)


def compute_path_length(path: np.ndarray) -> float:
    path = np.asarray(path, dtype=np.float64)

    if len(path) < 2:
        return 0.0

    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def compute_joint_path_length(q_traj: np.ndarray) -> float:
    q_traj = np.asarray(q_traj, dtype=np.float64)

    if len(q_traj) < 2:
        return 0.0

    q_unwrapped = unwrap_joint_trajectory(q_traj)
    return float(np.sum(np.linalg.norm(np.diff(q_unwrapped, axis=0), axis=1)))


def compute_joint_smoothness(q_traj: np.ndarray) -> float:
    """
    Simple acceleration-like smoothness score:
        sum ||q[i+1] - 2q[i] + q[i-1]||^2

    Lower is smoother.
    """
    q_traj = np.asarray(q_traj, dtype=np.float64)

    if len(q_traj) < 3:
        return 0.0

    q_unwrapped = unwrap_joint_trajectory(q_traj)
    acc = q_unwrapped[2:] - 2.0 * q_unwrapped[1:-1] + q_unwrapped[:-2]
    return float(np.sum(np.linalg.norm(acc, axis=1) ** 2))

def save_video_avi_mjpg(
    frames_rgb: list[np.ndarray],
    out_path: Path,
    fps: int = 30,
) -> None:
    if len(frames_rgb) == 0:
        print(f"Warning: no frames to save for {out_path}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = frames_rgb[0].shape[:2]
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        float(fps),
        (w, h),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {out_path}")

    for frame_rgb in frames_rgb:
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()

def save_frame_images(
    frames_rgb: list[np.ndarray],
    out_dir: Path,
    prefix: str,
    every: int = 1,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    every = max(1, int(every))

    for i, frame_rgb in enumerate(frames_rgb):
        if i % every != 0:
            continue

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / f"{prefix}_{i:04d}.png"), frame_bgr)


def replay_one_trajectory(
    *,
    name: str,
    q_traj_path: Path,
    env: MujocoUR5Env,
    out_dir: Path,
    steps_per_segment: int,
    fps: int,
    save_video: bool,
    save_frames: bool,
    frame_stride: int,
) -> dict:
    q_waypoints = np.load(q_traj_path)
    q_replay = interpolate_joint_trajectory(
        q_waypoints,
        steps_per_segment=steps_per_segment,
    )

    print(f"\n=== Replaying {name} ===")
    print(f"q_waypoints path : {q_traj_path}")
    print(f"q_waypoints shape: {q_waypoints.shape}")
    print(f"q_replay shape   : {q_replay.shape}")

    frames: list[np.ndarray] = []
    ee_positions: list[np.ndarray] = []
    ee_rotations: list[np.ndarray] = []

    for i, q in enumerate(q_replay):
        env.set_qpos(q, zero_velocity=True, forward=True)

        ee_pos, ee_R = env.get_ee_pose()
        ee_positions.append(ee_pos.copy())
        ee_rotations.append(ee_R.copy())

        if save_video or save_frames:
            rgb, _, _ = env.capture_rgbd()
            frames.append(rgb.copy())

        if i % max(1, len(q_replay) // 10) == 0:
            print(f"  frame {i:04d}/{len(q_replay)} ee={np.round(ee_pos, 4)}")

    ee_positions_arr = np.asarray(ee_positions, dtype=np.float64)
    ee_rotations_arr = np.asarray(ee_rotations, dtype=np.float64)

    ee_path_path = out_dir / f"{name}_ee_path.npy"
    ee_rot_path = out_dir / f"{name}_ee_rotations.npy"
    q_replay_path = out_dir / f"{name}_q_replay_interp.npy"

    np.save(ee_path_path, ee_positions_arr)
    np.save(ee_rot_path, ee_rotations_arr)
    np.save(q_replay_path, q_replay)

    video_path = out_dir / f"{name}_replay.avi"
    if save_video:
        save_video_avi_mjpg(frames, video_path, fps=fps)

    if save_frames:
        save_frame_images(
            frames,
            out_dir / f"{name}_frames",
            prefix=name,
            every=frame_stride,
        )

    metrics = {
        "name": name,
        "q_traj_path": str(q_traj_path),
        "q_waypoints_shape": list(q_waypoints.shape),
        "q_replay_shape": list(q_replay.shape),
        "num_rendered_frames": int(len(frames)),
        "ee_path_path": str(ee_path_path),
        "ee_rotations_path": str(ee_rot_path),
        "q_replay_interp_path": str(q_replay_path),
        "video_path": str(video_path) if save_video else None,
        "ee_path_length_m": compute_path_length(ee_positions_arr),
        "joint_path_length_rad": compute_joint_path_length(q_waypoints),
        "joint_smoothness": compute_joint_smoothness(q_waypoints),
        "ee_start": ee_positions_arr[0].tolist() if len(ee_positions_arr) else None,
        "ee_goal": ee_positions_arr[-1].tolist() if len(ee_positions_arr) else None,
    }

    print(f"\n{name} replay summary:")
    print(f"  EE path length     : {metrics['ee_path_length_m']:.4f} m")
    print(f"  joint path length  : {metrics['joint_path_length_rad']:.4f} rad")
    print(f"  joint smoothness   : {metrics['joint_smoothness']:.6f}")
    print(f"  EE start           : {np.round(metrics['ee_start'], 4)}")
    print(f"  EE goal            : {np.round(metrics['ee_goal'], 4)}")

    if save_video:
        print(f"  video saved        : {video_path}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay UR5 joint trajectories in MuJoCo and save videos/EE paths."
    )

    parser.add_argument(
        "--xml-path",
        type=str,
        default="data/assets/universal_robots_ur5e/ur5_power_drill_center.xml",
    )
    parser.add_argument("--ee-site-name", type=str, default="attachment_site")

    # For replay video, replay_cam is usually better than main_cam.
    parser.add_argument("--camera-name", type=str, default="replay_cam")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)

    parser.add_argument(
        "--collision-q-traj",
        type=str,
        default="out/ur5_ik_oriented_tol8mm/collision_only_q_traj.npy",
    )
    parser.add_argument(
        "--risk-aware-q-traj",
        type=str,
        default="out/ur5_ik_oriented_tol8mm/risk_aware_q_traj.npy",
    )

    parser.add_argument("--out-dir", type=str, default="out/ur5_replay")

    parser.add_argument(
        "--steps-per-segment",
        type=int,
        default=8,
        help="Interpolation steps between IK waypoint configurations.",
    )
    parser.add_argument("--fps", type=int, default=30)

    parser.add_argument(
        "--save-video",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save-frames",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=5,
        help="Save every Nth frame if --save-frames is enabled.",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = MujocoUR5Env(
        UR5EnvConfig(
            xml_path=args.xml_path,
            ee_site_name=args.ee_site_name,
            camera_name=args.camera_name,
            width=args.width,
            height=args.height,
            settle_steps=0,
        )
    )

    print("\n=== Replay Environment ===")
    env.print_debug_summary()

    metrics = {
        "xml_path": args.xml_path,
        "ee_site_name": args.ee_site_name,
        "camera_name": args.camera_name,
        "width": int(args.width),
        "height": int(args.height),
        "steps_per_segment": int(args.steps_per_segment),
        "fps": int(args.fps),
    }

    collision_metrics = replay_one_trajectory(
        name="collision_only",
        q_traj_path=Path(args.collision_q_traj),
        env=env,
        out_dir=out_dir,
        steps_per_segment=int(args.steps_per_segment),
        fps=int(args.fps),
        save_video=bool(args.save_video),
        save_frames=bool(args.save_frames),
        frame_stride=int(args.frame_stride),
    )

    risk_metrics = replay_one_trajectory(
        name="risk_aware",
        q_traj_path=Path(args.risk_aware_q_traj),
        env=env,
        out_dir=out_dir,
        steps_per_segment=int(args.steps_per_segment),
        fps=int(args.fps),
        save_video=bool(args.save_video),
        save_frames=bool(args.save_frames),
        frame_stride=int(args.frame_stride),
    )

    metrics["collision_only"] = collision_metrics
    metrics["risk_aware"] = risk_metrics

    metrics_path = out_dir / "replay_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved replay outputs:")
    print(f"  {out_dir / 'collision_only_replay.mp4'}")
    print(f"  {out_dir / 'risk_aware_replay.mp4'}")
    print(f"  {out_dir / 'collision_only_ee_path.npy'}")
    print(f"  {out_dir / 'risk_aware_ee_path.npy'}")
    print(f"  {metrics_path}")


if __name__ == "__main__":
    main()