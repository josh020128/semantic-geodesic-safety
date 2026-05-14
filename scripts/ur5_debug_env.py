from __future__ import annotations

import argparse
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


def save_depth_debug(depth: np.ndarray, path: Path) -> None:
    valid = np.isfinite(depth) & (depth > 0.0)

    if not np.any(valid):
        cv2.imwrite(str(path), np.zeros_like(depth, dtype=np.uint8))
        return

    vals = depth[valid]
    lo = float(np.percentile(vals, 2))
    hi = float(np.percentile(vals, 98))

    if hi <= lo:
        hi = lo + 1e-6

    clipped = np.clip(depth, lo, hi)
    vis = ((clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)
    vis[~valid] = 0

    color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    cv2.imwrite(str(path), color)


def parse_qpos(values: list[float] | None) -> np.ndarray | None:
    if values is None:
        return None

    q = np.asarray(values, dtype=np.float64)
    if q.shape != (6,):
        raise ValueError(f"--qpos expects 6 values, got {q.shape}")
    return q


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug-load a UR5 MuJoCo XML scene and render RGB-D."
    )

    parser.add_argument(
        "--xml-path",
        type=str,
        default="data/assets/universal_robots_ur5e/ur5_power_drill_center.xml",
    )
    parser.add_argument(
        "--ee-site-name",
        type=str,
        default="attachment_site",
    )
    parser.add_argument(
        "--camera-name",
        type=str,
        default="main_cam",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)

    parser.add_argument(
        "--qpos",
        type=float,
        nargs=6,
        default=None,
        help="Optional UR5 qpos: six joint values in radians.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="out/ur5_debug_env",
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
        )
    )

    q_cmd = parse_qpos(args.qpos)
    if q_cmd is not None:
        env.set_qpos(q_cmd)
        print("\nSet custom qpos:", np.round(q_cmd, 5))

    env.print_debug_summary()

    q = env.get_qpos()
    ee_pos, ee_rot = env.get_ee_pose()

    print("\n=== Quick Values ===")
    print("qpos:", np.round(q, 5))
    print("ee position:", np.round(ee_pos, 5))
    print("ee rotation:")
    print(np.round(ee_rot, 5))

    rgb, depth, intrinsics = env.capture_rgbd()

    rgb_path = out_dir / "rgb.png"
    depth_path = out_dir / "depth_debug.png"

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(rgb_path), bgr)
    save_depth_debug(depth, depth_path)

    valid_depth = np.isfinite(depth) & (depth > 0.0)
    print("\n=== Render Debug ===")
    print("rgb shape:", rgb.shape, "dtype:", rgb.dtype)
    print("depth shape:", depth.shape, "dtype:", depth.dtype)
    if np.any(valid_depth):
        vals = depth[valid_depth]
        print(
            "depth valid min/median/max:",
            float(vals.min()),
            float(np.median(vals)),
            float(vals.max()),
        )
    else:
        print("depth has no valid positive values")

    print("intrinsics:", intrinsics)

    print("\nSaved:")
    print(f"  {rgb_path}")
    print(f"  {depth_path}")

    # Optional useful object sanity checks.
    for body_name in ["power_drill", "base", "world"]:
        try:
            pos = env.get_body_position(body_name)
            print(f"body {body_name:<16s} pos: {np.round(pos, 5)}")
        except Exception:
            pass

    for geom_name in ["power_drill_geom", "table"]:
        try:
            pos = env.get_geom_position(geom_name)
            print(f"geom {geom_name:<16s} pos: {np.round(pos, 5)}")
        except Exception:
            pass


if __name__ == "__main__":
    main()