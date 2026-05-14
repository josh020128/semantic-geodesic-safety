from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
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
    Make revolute joint trajectories continuous for visualization.

    This avoids visual jumps if an equivalent joint angle crosses +/- pi.
    """
    q_traj = np.asarray(q_traj, dtype=np.float64)

    if q_traj.ndim != 2:
        raise ValueError(f"q_traj must be 2D, got shape {q_traj.shape}")

    return np.unwrap(q_traj, axis=0)


def interpolate_joint_trajectory(
    q_traj: np.ndarray,
    steps_per_segment: int = 8,
) -> np.ndarray:
    """
    Linearly interpolate between IK waypoint configurations for smoother viewer replay.
    """
    q_traj = np.asarray(q_traj, dtype=np.float64)

    if q_traj.ndim != 2 or q_traj.shape[1] != 6:
        raise ValueError(f"q_traj must have shape (N, 6), got {q_traj.shape}")

    if len(q_traj) <= 1:
        return q_traj.copy()

    steps_per_segment = max(1, int(steps_per_segment))
    q_unwrapped = unwrap_joint_trajectory(q_traj)

    q_replay = []

    for i in range(len(q_unwrapped) - 1):
        q0 = q_unwrapped[i]
        q1 = q_unwrapped[i + 1]

        for s in range(steps_per_segment):
            t = s / float(steps_per_segment)
            q = (1.0 - t) * q0 + t * q1
            q_replay.append(q)

    q_replay.append(q_unwrapped[-1])

    return np.asarray(q_replay, dtype=np.float64)


def configure_viewer_camera(viewer, model, camera_name: str | None) -> None:
    """
    Use a fixed MuJoCo camera if provided.
    If no camera is provided, use the viewer's free camera.
    """
    if camera_name is None or camera_name.strip() == "":
        print("Using free viewer camera.")
        return

    cam_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_CAMERA,
        camera_name,
    )

    if cam_id < 0:
        print(f"Warning: camera '{camera_name}' not found. Using free camera.")
        return

    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = cam_id

    print(f"Using fixed camera: {camera_name}")


def print_traj_summary(name: str, q_traj: np.ndarray) -> None:
    q_traj = np.asarray(q_traj, dtype=np.float64)

    print(f"\n=== Loaded trajectory: {name} ===")
    print(f"shape: {q_traj.shape}")
    print("first q:", np.round(q_traj[0], 5))
    print("last q :", np.round(q_traj[-1], 5))

    if len(q_traj) >= 2:
        q_unwrapped = unwrap_joint_trajectory(q_traj)
        diffs = np.diff(q_unwrapped, axis=0)
        step_norms = np.linalg.norm(diffs, axis=1)

        print("max step norm :", float(np.max(step_norms)))
        print("mean step norm:", float(np.mean(step_norms)))
        print("joint path length:", float(np.sum(step_norms)))


def replay_once(
    *,
    env: MujocoUR5Env,
    viewer,
    q_replay: np.ndarray,
    label: str,
    fps: float,
    speed: float,
    hold_start_s: float,
    hold_end_s: float,
) -> None:
    """
    Replay one interpolated trajectory once.
    """
    q_replay = np.asarray(q_replay, dtype=np.float64)

    dt = 1.0 / max(float(fps), 1e-6)
    sleep_dt = dt / max(float(speed), 1e-6)

    print(f"\nReplaying: {label}")
    print(f"frames: {len(q_replay)}, fps={fps}, speed={speed}")

    env.set_qpos(q_replay[0], zero_velocity=True, forward=True)
    viewer.sync()
    time.sleep(max(0.0, hold_start_s))

    for i, q in enumerate(q_replay):
        if not viewer.is_running():
            return

        env.set_qpos(q, zero_velocity=True, forward=True)

        if i % max(1, len(q_replay) // 20) == 0:
            ee_pos = env.get_ee_position()
            print(f"  frame {i:04d}/{len(q_replay)} ee={np.round(ee_pos, 4)}")

        viewer.sync()
        time.sleep(sleep_dt)

    env.set_qpos(q_replay[-1], zero_velocity=True, forward=True)
    viewer.sync()
    time.sleep(max(0.0, hold_end_s))


def load_and_prepare_trajectory(
    q_traj_path: Path,
    steps_per_segment: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not q_traj_path.exists():
        raise FileNotFoundError(f"q trajectory not found: {q_traj_path}")

    q_waypoints = np.load(q_traj_path)
    q_replay = interpolate_joint_trajectory(
        q_waypoints,
        steps_per_segment=steps_per_segment,
    )

    return q_waypoints, q_replay


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay UR5 q_traj.npy in the interactive MuJoCo viewer."
    )

    parser.add_argument(
        "--xml-path",
        type=str,
        default="data/assets/universal_robots_ur5e/ur5_power_drill_center.xml",
    )
    parser.add_argument("--ee-site-name", type=str, default="attachment_site")

    parser.add_argument(
        "--q-traj",
        type=str,
        default=None,
        help="Path to one q trajectory .npy file.",
    )

    parser.add_argument(
        "--collision-q-traj",
        type=str,
        default=None,
        help="Optional collision-only q trajectory .npy.",
    )
    parser.add_argument(
        "--risk-aware-q-traj",
        type=str,
        default=None,
        help="Optional risk-aware q trajectory .npy.",
    )

    parser.add_argument(
        "--fixed-camera",
        type=str,
        default="",
        help="Optional fixed MuJoCo camera name, e.g. replay_cam. Empty means free camera.",
    )

    parser.add_argument("--steps-per-segment", type=int, default=8)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier. 0.5 = slower, 2.0 = faster.",
    )

    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop replay until viewer is closed.",
    )
    parser.add_argument("--hold-start-s", type=float, default=1.0)
    parser.add_argument("--hold-end-s", type=float, default=1.0)

    parser.add_argument(
        "--pause-between-s",
        type=float,
        default=1.0,
        help="Pause between collision-only and risk-aware if both are provided.",
    )

    args = parser.parse_args()

    trajectory_items: list[tuple[str, Path]] = []

    if args.q_traj is not None:
        trajectory_items.append(("trajectory", Path(args.q_traj)))

    if args.collision_q_traj is not None:
        trajectory_items.append(("collision_only", Path(args.collision_q_traj)))

    if args.risk_aware_q_traj is not None:
        trajectory_items.append(("risk_aware", Path(args.risk_aware_q_traj)))

    if not trajectory_items:
        raise ValueError(
            "Provide either --q-traj, or --collision-q-traj / --risk-aware-q-traj."
        )

    prepared: list[tuple[str, np.ndarray, np.ndarray]] = []

    for name, path in trajectory_items:
        q_waypoints, q_replay = load_and_prepare_trajectory(
            path,
            steps_per_segment=args.steps_per_segment,
        )

        print_traj_summary(name, q_waypoints)
        print(f"interpolated replay shape: {q_replay.shape}")

        prepared.append((name, q_waypoints, q_replay))

    # Important:
    # camera_name="" prevents MujocoUR5Env from creating an offscreen Renderer.
    # The interactive viewer is handled by mujoco.viewer instead.
    env = MujocoUR5Env(
        UR5EnvConfig(
            xml_path=args.xml_path,
            ee_site_name=args.ee_site_name,
            camera_name="",
            settle_steps=0,
        )
    )

    first_q = prepared[0][2][0]
    env.set_qpos(first_q, zero_velocity=True, forward=True)

    print("\n=== Viewer Environment ===")
    env.print_debug_summary()

    print("\nOpening MuJoCo viewer...")
    print("Close the viewer window or press Ctrl+C in terminal to stop.")

    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            configure_viewer_camera(
                viewer=viewer,
                model=env.model,
                camera_name=args.fixed_camera,
            )

            if args.loop:
                while viewer.is_running():
                    for name, _, q_replay in prepared:
                        if not viewer.is_running():
                            break

                        replay_once(
                            env=env,
                            viewer=viewer,
                            q_replay=q_replay,
                            label=name,
                            fps=args.fps,
                            speed=args.speed,
                            hold_start_s=args.hold_start_s,
                            hold_end_s=args.hold_end_s,
                        )

                        time.sleep(max(0.0, args.pause_between_s))
            else:
                for name, _, q_replay in prepared:
                    if not viewer.is_running():
                        break

                    replay_once(
                        env=env,
                        viewer=viewer,
                        q_replay=q_replay,
                        label=name,
                        fps=args.fps,
                        speed=args.speed,
                        hold_start_s=args.hold_start_s,
                        hold_end_s=args.hold_end_s,
                    )

                    time.sleep(max(0.0, args.pause_between_s))

                print("\nReplay finished. Viewer will stay open until you close it.")
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.03)

    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    main()