from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


SegmentValidityFn = Callable[[np.ndarray, np.ndarray], bool]
PointValidityFn = Callable[[np.ndarray], bool]


@dataclass
class TrajectoryProcessingConfig:
    """
    Config for converting raw A* waypoints into IK/PyRoki-friendly waypoints.
    """

    # Remove points closer than this.
    min_point_spacing_m: float = 1e-4

    # Greedy line-of-sight shortcut.
    enable_shortcut: bool = True
    shortcut_step_m: float = 0.01

    # Resample final path at approximately fixed Cartesian spacing.
    resample_spacing_m: float = 0.03

    # Optional moving-average smoothing.
    enable_smoothing: bool = True
    smoothing_window: int = 3

    # After smoothing, check each point is still valid if a validity fn is provided.
    validate_smoothed_points: bool = True


@dataclass
class ProcessedTrajectory:
    raw_path: np.ndarray
    simplified_path: np.ndarray
    resampled_path: np.ndarray
    smoothed_path: np.ndarray
    final_path: np.ndarray

    path_length_raw: float
    path_length_final: float

    pose_matrices: np.ndarray | None = None
    positions: np.ndarray | None = None
    rotations: np.ndarray | None = None


def as_path_array(path: np.ndarray) -> np.ndarray:
    path = np.asarray(path, dtype=np.float64)

    if path.ndim != 2 or path.shape[1] != 3:
        raise ValueError(f"path must have shape (N, 3), got {path.shape}")

    if len(path) == 0:
        return path.reshape(0, 3)

    if not np.all(np.isfinite(path)):
        raise ValueError("path contains non-finite values.")

    return path


def compute_path_length(path: np.ndarray) -> float:
    path = as_path_array(path)
    if len(path) < 2:
        return 0.0

    diffs = np.diff(path, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def remove_near_duplicate_points(
    path: np.ndarray,
    min_spacing_m: float = 1e-4,
) -> np.ndarray:
    """
    Remove consecutive points that are almost identical.
    """
    path = as_path_array(path)

    if len(path) <= 1:
        return path.copy()

    kept = [path[0]]

    for p in path[1:]:
        if np.linalg.norm(p - kept[-1]) >= min_spacing_m:
            kept.append(p)

    # Always preserve final goal.
    if np.linalg.norm(path[-1] - kept[-1]) > 0.0:
        kept.append(path[-1])

    return np.asarray(kept, dtype=np.float64)


def sample_segment(
    p0: np.ndarray,
    p1: np.ndarray,
    step_m: float = 0.01,
    include_end: bool = True,
) -> np.ndarray:
    """
    Sample a straight segment from p0 to p1.
    """
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)

    dist = float(np.linalg.norm(p1 - p0))
    if dist <= 1e-12:
        return p0.reshape(1, 3)

    n_steps = max(1, int(np.ceil(dist / step_m)))
    if include_end:
        ts = np.linspace(0.0, 1.0, n_steps + 1)
    else:
        ts = np.linspace(0.0, 1.0, n_steps, endpoint=False)

    return p0[None, :] * (1.0 - ts[:, None]) + p1[None, :] * ts[:, None]


def segment_is_valid_by_points(
    p0: np.ndarray,
    p1: np.ndarray,
    point_is_valid: PointValidityFn,
    step_m: float = 0.01,
) -> bool:
    """
    Check segment validity by sampling points along the segment.
    """
    pts = sample_segment(p0, p1, step_m=step_m, include_end=True)
    return all(bool(point_is_valid(p)) for p in pts)


def shortcut_path_greedy(
    path: np.ndarray,
    *,
    segment_is_valid: SegmentValidityFn | None = None,
    point_is_valid: PointValidityFn | None = None,
    step_m: float = 0.01,
) -> np.ndarray:
    """
    Greedy path shortcutting.

    If segment_is_valid is provided, it tries to skip intermediate waypoints
    whenever a straight segment is valid.

    If no validity function is provided, this function returns the original path
    after duplicate removal. This is intentional: we do not want to accidentally
    shortcut through obstacles.
    """
    path = as_path_array(path)

    if len(path) <= 2:
        return path.copy()

    if segment_is_valid is None and point_is_valid is not None:
        def _seg_valid(a: np.ndarray, b: np.ndarray) -> bool:
            return segment_is_valid_by_points(
                a,
                b,
                point_is_valid=point_is_valid,
                step_m=step_m,
            )
        segment_is_valid = _seg_valid

    if segment_is_valid is None:
        return path.copy()

    simplified = []
    i = 0
    n = len(path)

    while i < n - 1:
        simplified.append(path[i])

        # Try farthest reachable waypoint first.
        next_i = i + 1
        for j in range(n - 1, i, -1):
            if segment_is_valid(path[i], path[j]):
                next_i = j
                break

        i = next_i

    simplified.append(path[-1])
    return remove_near_duplicate_points(np.asarray(simplified), min_spacing_m=1e-4)


def resample_path_by_spacing(
    path: np.ndarray,
    spacing_m: float = 0.03,
) -> np.ndarray:
    """
    Resample a polyline path at approximately fixed Cartesian spacing.

    This is important before IK/PyRoki because raw A* paths are often voxel-staircase paths.
    """
    path = as_path_array(path)

    if len(path) <= 1:
        return path.copy()

    if spacing_m <= 0.0:
        raise ValueError("spacing_m must be positive.")

    seg_vecs = np.diff(path, axis=0)
    seg_lens = np.linalg.norm(seg_vecs, axis=1)
    total_len = float(np.sum(seg_lens))

    if total_len <= 1e-12:
        return path[:1].copy()

    num_samples = max(2, int(np.ceil(total_len / spacing_m)) + 1)
    target_s = np.linspace(0.0, total_len, num_samples)

    cumulative = np.concatenate([[0.0], np.cumsum(seg_lens)])

    out = []
    seg_idx = 0

    for s in target_s:
        while seg_idx < len(seg_lens) - 1 and cumulative[seg_idx + 1] < s:
            seg_idx += 1

        s0 = cumulative[seg_idx]
        s1 = cumulative[seg_idx + 1]
        denom = max(s1 - s0, 1e-12)
        t = (s - s0) / denom

        p = path[seg_idx] * (1.0 - t) + path[seg_idx + 1] * t
        out.append(p)

    out_arr = np.asarray(out, dtype=np.float64)
    out_arr[0] = path[0]
    out_arr[-1] = path[-1]
    return out_arr


def smooth_path_moving_average(
    path: np.ndarray,
    window: int = 3,
    *,
    keep_endpoints: bool = True,
) -> np.ndarray:
    """
    Simple moving-average path smoother.

    This does not guarantee collision-free output by itself.
    If needed, validate smoothed points afterward.
    """
    path = as_path_array(path)

    if len(path) <= 2:
        return path.copy()

    if window <= 1:
        return path.copy()

    if window % 2 == 0:
        raise ValueError("window must be odd.")

    half = window // 2
    smoothed = path.copy()

    for i in range(len(path)):
        if keep_endpoints and (i == 0 or i == len(path) - 1):
            continue

        lo = max(0, i - half)
        hi = min(len(path), i + half + 1)
        smoothed[i] = np.mean(path[lo:hi], axis=0)

    return smoothed


def validate_path_points(
    path: np.ndarray,
    point_is_valid: PointValidityFn,
) -> bool:
    path = as_path_array(path)
    return all(bool(point_is_valid(p)) for p in path)


def fallback_if_invalid(
    candidate_path: np.ndarray,
    fallback_path: np.ndarray,
    *,
    point_is_valid: PointValidityFn | None,
) -> np.ndarray:
    """
    Use candidate_path only if every point is valid.
    Otherwise return fallback_path.
    """
    if point_is_valid is None:
        return candidate_path

    if validate_path_points(candidate_path, point_is_valid):
        return candidate_path

    return fallback_path


def make_homogeneous_poses(
    positions: np.ndarray,
    rotation_matrix: np.ndarray,
) -> np.ndarray:
    """
    Convert positions + fixed rotation into SE(3) pose matrices.

    Returns:
        poses: (N, 4, 4)
    """
    positions = as_path_array(positions)
    R = np.asarray(rotation_matrix, dtype=np.float64)

    if R.shape != (3, 3):
        raise ValueError(f"rotation_matrix must have shape (3, 3), got {R.shape}")

    poses = np.tile(np.eye(4, dtype=np.float64), (len(positions), 1, 1))
    poses[:, :3, :3] = R[None, :, :]
    poses[:, :3, 3] = positions
    return poses


def rotation_matrix_from_rpy(
    roll: float,
    pitch: float,
    yaw: float,
) -> np.ndarray:
    """
    Create rotation matrix from roll, pitch, yaw.

    Convention:
        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr, cr],
        ],
        dtype=np.float64,
    )

    Ry = np.array(
        [
            [cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ],
        dtype=np.float64,
    )

    Rz = np.array(
        [
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    return Rz @ Ry @ Rx


def rotation_matrix_to_quaternion_xyzw(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to quaternion [x, y, z, w].

    Useful if later PyRoki or another solver expects quaternion poses.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"R must have shape (3, 3), got {R.shape}")

    tr = float(np.trace(R))

    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= max(np.linalg.norm(q), 1e-12)
    return q


def make_pose_waypoints(
    positions: np.ndarray,
    rotation_matrix: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Return pose waypoints in multiple convenient formats.

    Returns:
        {
          "positions": (N, 3),
          "rotations": (N, 3, 3),
          "poses": (N, 4, 4),
          "quaternions_xyzw": (N, 4)
        }
    """
    positions = as_path_array(positions)
    R = np.asarray(rotation_matrix, dtype=np.float64)

    poses = make_homogeneous_poses(positions, R)
    rotations = np.repeat(R[None, :, :], len(positions), axis=0)
    q = rotation_matrix_to_quaternion_xyzw(R)
    quats = np.repeat(q[None, :], len(positions), axis=0)

    return {
        "positions": positions,
        "rotations": rotations,
        "poses": poses,
        "quaternions_xyzw": quats,
    }


class TrajectoryProcessor:
    """
    Converts raw A* path into IK-friendly Cartesian waypoints.
    """

    def __init__(self, config: TrajectoryProcessingConfig):
        self.config = config

    def process(
        self,
        raw_path: np.ndarray,
        *,
        point_is_valid: PointValidityFn | None = None,
        segment_is_valid: SegmentValidityFn | None = None,
        fixed_rotation: np.ndarray | None = None,
    ) -> ProcessedTrajectory:
        raw_path = as_path_array(raw_path)

        if len(raw_path) == 0:
            return ProcessedTrajectory(
                raw_path=raw_path,
                simplified_path=raw_path,
                resampled_path=raw_path,
                smoothed_path=raw_path,
                final_path=raw_path,
                path_length_raw=0.0,
                path_length_final=0.0,
            )

        deduped = remove_near_duplicate_points(
            raw_path,
            min_spacing_m=self.config.min_point_spacing_m,
        )

        if self.config.enable_shortcut:
            simplified = shortcut_path_greedy(
                deduped,
                segment_is_valid=segment_is_valid,
                point_is_valid=point_is_valid,
                step_m=self.config.shortcut_step_m,
            )
        else:
            simplified = deduped

        resampled = resample_path_by_spacing(
            simplified,
            spacing_m=self.config.resample_spacing_m,
        )

        if self.config.enable_smoothing:
            smoothed_candidate = smooth_path_moving_average(
                resampled,
                window=self.config.smoothing_window,
                keep_endpoints=True,
            )

            if self.config.validate_smoothed_points:
                smoothed = fallback_if_invalid(
                    smoothed_candidate,
                    resampled,
                    point_is_valid=point_is_valid,
                )
            else:
                smoothed = smoothed_candidate
        else:
            smoothed = resampled

        final_path = remove_near_duplicate_points(
            smoothed,
            min_spacing_m=self.config.min_point_spacing_m,
        )

        pose_matrices = None
        positions = None
        rotations = None

        if fixed_rotation is not None:
            pose_dict = make_pose_waypoints(final_path, fixed_rotation)
            pose_matrices = pose_dict["poses"]
            positions = pose_dict["positions"]
            rotations = pose_dict["rotations"]

        return ProcessedTrajectory(
            raw_path=raw_path,
            simplified_path=simplified,
            resampled_path=resampled,
            smoothed_path=smoothed,
            final_path=final_path,
            path_length_raw=compute_path_length(raw_path),
            path_length_final=compute_path_length(final_path),
            pose_matrices=pose_matrices,
            positions=positions,
            rotations=rotations,
        )