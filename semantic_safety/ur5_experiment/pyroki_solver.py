from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import importlib
import os
import sys

import numpy as np


MUJOCO_UR5_JOINT_ORDER = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


@dataclass
class PyRokiIKConfig:
    """
    Config for PyRoki-based UR5 IK.

    This wrapper intentionally starts simple:
      - use pyroki_snippets.solve_ik for each pose waypoint
      - convert MuJoCo-world target poses into PyRoki URDF frame
      - convert returned q into MuJoCo joint order

    Later, we can add solve_trajopt / collision-aware PyRoki planning.
    """

    robot_description: str = "ur5e_description"
    target_link_name: str = "tool0"

    # Needed because pyroki_snippets may live under cloned pyroki/examples.
    pyroki_examples_path: str | None = "/home/gl34/research/pyroki/examples"

    # Frame alignment observed between MuJoCo Menagerie UR5e MJCF and
    # robot_descriptions ur5e URDF:
    #   p_pyroki ~= diag(-1, -1, 1) @ p_mujoco
    frame_alignment: str = "mujoco_to_pyroki_z180"

    # Whether to use solve_ik_with_manipulability instead of solve_ik.
    use_manipulability: bool = False
    manipulability_weight: float = 0.0

    # Sanity thresholds only used for reporting success.
    pos_tol_m: float = 1e-2
    rot_tol_rad: float = 0.15

    # If true, print per-waypoint debug info.
    verbose: bool = False


@dataclass
class PyRokiIKWaypointResult:
    success: bool
    q_mujoco_order: np.ndarray
    q_pyroki_order: np.ndarray
    target_position_pyroki: np.ndarray
    target_wxyz_pyroki: np.ndarray
    fk_position_pyroki: np.ndarray | None
    fk_wxyz_pyroki: np.ndarray | None
    pos_error_m: float
    rot_error_rad: float
    message: str


@dataclass
class PyRokiIKSequenceResult:
    success: bool
    q_traj_mujoco_order: np.ndarray
    waypoint_results: list[PyRokiIKWaypointResult]
    success_rate: float
    failed_indices: list[int]


def add_pyroki_examples_to_syspath(path: str | None) -> None:
    """
    Add pyroki/examples to sys.path so `import pyroki_snippets` works.

    Your local setup has:
        /home/gl34/research/pyroki/examples/pyroki_snippets
    """
    candidates: list[str] = []

    if path:
        candidates.append(path)

    env_path = os.environ.get("PYROKI_EXAMPLES_PATH", "")
    if env_path:
        candidates.append(env_path)

    candidates.extend(
        [
            "/home/gl34/research/pyroki/examples",
            str(Path.cwd().parent / "pyroki" / "examples"),
            str(Path.cwd() / "external" / "pyroki" / "examples"),
        ]
    )

    for p in candidates:
        pth = Path(p).expanduser().resolve()
        if pth.exists() and pth.is_dir():
            if str(pth) not in sys.path:
                sys.path.insert(0, str(pth))
            return


def import_pyroki_stack(config: PyRokiIKConfig):
    add_pyroki_examples_to_syspath(config.pyroki_examples_path)

    pk = importlib.import_module("pyroki")
    pks = importlib.import_module("pyroki_snippets")

    loader_mod = importlib.import_module("robot_descriptions.loaders.yourdfpy")
    load_robot_description = getattr(loader_mod, "load_robot_description")

    return pk, pks, load_robot_description


def load_pyroki_robot(config: PyRokiIKConfig):
    pk, pks, load_robot_description = import_pyroki_stack(config)

    urdf = load_robot_description(config.robot_description)
    robot = pk.Robot.from_urdf(urdf)

    return pk, pks, urdf, robot


def get_pyroki_actuated_joint_names(robot: Any) -> list[str]:
    joints = getattr(robot, "joints", None)
    if joints is None:
        return []

    if hasattr(joints, "actuated_names"):
        return list(getattr(joints, "actuated_names"))

    if hasattr(joints, "names"):
        names = list(getattr(joints, "names"))
        return [name for name in names if name in MUJOCO_UR5_JOINT_ORDER]

    return []


def get_pyroki_link_names(robot: Any) -> list[str]:
    links = getattr(robot, "links", None)
    if links is None:
        return []

    if hasattr(links, "names"):
        return list(getattr(links, "names"))

    return []


def build_pyroki_to_mujoco_index(robot: Any) -> list[int]:
    """
    Return mapping from MuJoCo q index -> PyRoki q index.

    For your current ur5e_description, this should be:
        [0, 1, 2, 3, 4, 5]
    """
    joint_names = get_pyroki_actuated_joint_names(robot)

    missing = [name for name in MUJOCO_UR5_JOINT_ORDER if name not in joint_names]
    if missing:
        raise ValueError(
            "PyRoki robot is missing MuJoCo UR5 joints: "
            f"{missing}. PyRoki actuated joints: {joint_names}"
        )

    return [joint_names.index(name) for name in MUJOCO_UR5_JOINT_ORDER]


def q_pyroki_to_mujoco_order(robot: Any, q_pyroki: np.ndarray) -> np.ndarray:
    """
    Convert PyRoki q order into MuJoCo UR5 joint order.
    """
    q_pyroki = np.asarray(q_pyroki, dtype=np.float64).reshape(-1)

    mapping = build_pyroki_to_mujoco_index(robot)

    q_mj = np.zeros(6, dtype=np.float64)
    for mj_i, pk_i in enumerate(mapping):
        q_mj[mj_i] = q_pyroki[pk_i]

    return q_mj


def q_mujoco_to_pyroki_order(robot: Any, q_mujoco: np.ndarray) -> np.ndarray:
    """
    Convert MuJoCo UR5 q order into PyRoki actuated q order.
    """
    q_mujoco = np.asarray(q_mujoco, dtype=np.float64).reshape(-1)
    if q_mujoco.shape != (6,):
        raise ValueError(f"q_mujoco must have shape (6,), got {q_mujoco.shape}")

    joint_names = get_pyroki_actuated_joint_names(robot)
    q_pk = np.zeros(len(joint_names), dtype=np.float64)

    for mj_i, joint_name in enumerate(MUJOCO_UR5_JOINT_ORDER):
        pk_i = joint_names.index(joint_name)
        q_pk[pk_i] = q_mujoco[mj_i]

    return q_pk


def mujoco_to_pyroki_alignment_matrix(config: PyRokiIKConfig) -> np.ndarray:
    """
    Return R such that:
        p_pyroki = R @ p_mujoco
        R_pyroki = R @ R_mujoco

    Current observed alignment:
        R = RotZ(pi) = diag(-1, -1, 1)
    """
    if config.frame_alignment == "identity":
        return np.eye(3, dtype=np.float64)

    if config.frame_alignment == "mujoco_to_pyroki_z180":
        return np.diag([-1.0, -1.0, 1.0]).astype(np.float64)

    raise ValueError(f"Unknown frame_alignment: {config.frame_alignment}")


def transform_pose_mujoco_to_pyroki(
    T_mujoco: np.ndarray,
    config: PyRokiIKConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a MuJoCo-world homogeneous pose to PyRoki URDF-world pose.

    Returns:
        position_pyroki: (3,)
        rotation_pyroki: (3, 3)
    """
    T_mujoco = np.asarray(T_mujoco, dtype=np.float64)

    if T_mujoco.shape != (4, 4):
        raise ValueError(f"T_mujoco must have shape (4, 4), got {T_mujoco.shape}")

    R_align = mujoco_to_pyroki_alignment_matrix(config)

    p_mj = T_mujoco[:3, 3]
    R_mj = T_mujoco[:3, :3]

    p_pk = R_align @ p_mj
    R_pk = R_align @ R_mj

    return p_pk, R_pk


def rotation_matrix_to_quaternion_wxyz(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to quaternion [w, x, y, z].
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

    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= max(np.linalg.norm(q), 1e-12)

    # Use a consistent sign to avoid unnecessary jumps.
    if q[0] < 0:
        q = -q

    return q


def quaternion_wxyz_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [w, x, y, z] to 3x3 rotation matrix.
    """
    q = np.asarray(q, dtype=np.float64).reshape(4)
    q = q / max(np.linalg.norm(q), 1e-12)

    w, x, y, z = q

    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quaternion_angular_distance_wxyz(q1: np.ndarray, q2: np.ndarray) -> float:
    """
    Quaternion angular distance in radians.
    """
    q1 = np.asarray(q1, dtype=np.float64).reshape(4)
    q2 = np.asarray(q2, dtype=np.float64).reshape(4)

    q1 /= max(np.linalg.norm(q1), 1e-12)
    q2 /= max(np.linalg.norm(q2), 1e-12)

    dot = float(abs(np.dot(q1, q2)))
    dot = float(np.clip(dot, -1.0, 1.0))

    return float(2.0 * np.arccos(dot))


def get_fk_link_pose7(
    robot: Any,
    q_pyroki: np.ndarray,
    link_name: str,
) -> np.ndarray:
    """
    Return PyRoki FK pose7 for a link.

    pose7 convention observed:
        [w, x, y, z, px, py, pz]
    """
    q_pyroki = np.asarray(q_pyroki, dtype=np.float64).reshape(-1)

    link_names = get_pyroki_link_names(robot)
    if link_name not in link_names:
        raise ValueError(f"link_name '{link_name}' not found. Available links: {link_names}")

    link_idx = link_names.index(link_name)

    fk = robot.forward_kinematics(q_pyroki)
    fk_np = np.asarray(fk, dtype=np.float64)

    if fk_np.ndim == 3:
        fk_np = fk_np[0]

    if fk_np.ndim != 2 or fk_np.shape[1] != 7:
        raise ValueError(f"Unexpected FK output shape: {fk_np.shape}")

    return fk_np[link_idx].copy()


def unwrap_joint_trajectory(q_traj: np.ndarray) -> np.ndarray:
    q_traj = np.asarray(q_traj, dtype=np.float64)
    if q_traj.ndim != 2:
        raise ValueError(f"q_traj must be 2D, got {q_traj.shape}")
    return np.unwrap(q_traj, axis=0)


def make_q_continuous_with_previous(q: np.ndarray, q_prev: np.ndarray | None) -> np.ndarray:
    """
    Shift each revolute joint by multiples of 2π so it is closest to previous q.

    PyRoki solve_ik may return equivalent joint angles with 2π jumps.
    """
    q = np.asarray(q, dtype=np.float64).reshape(-1).copy()

    if q_prev is None:
        return q

    q_prev = np.asarray(q_prev, dtype=np.float64).reshape(-1)

    out = q.copy()
    for i in range(len(out)):
        delta = out[i] - q_prev[i]
        out[i] -= 2.0 * np.pi * np.round(delta / (2.0 * np.pi))

    return out


class PyRokiUR5IKSolver:
    """
    PyRoki-based UR5 IK wrapper.

    First version:
      - Uses pyroki_snippets.solve_ik or solve_ik_with_manipulability.
      - Solves each pose waypoint independently.
      - Makes resulting q sequence continuous afterward.
      - Returns q in MuJoCo joint order so existing replay scripts can use it.

    This is not yet trajectory-level PyRoki optimization.
    That can be added later using solve_trajopt / solve_online_planning.
    """

    def __init__(self, config: PyRokiIKConfig):
        self.config = config
        self.pk, self.pks, self.urdf, self.robot = load_pyroki_robot(config)

        self.joint_names = get_pyroki_actuated_joint_names(self.robot)
        self.link_names = get_pyroki_link_names(self.robot)

        self._validate_robot()

    def _validate_robot(self) -> None:
        missing = [name for name in MUJOCO_UR5_JOINT_ORDER if name not in self.joint_names]
        if missing:
            raise ValueError(
                f"PyRoki robot missing MuJoCo UR5 joints: {missing}. "
                f"PyRoki actuated joints: {self.joint_names}"
            )

        if self.config.target_link_name not in self.link_names:
            raise ValueError(
                f"target_link_name '{self.config.target_link_name}' not in PyRoki links. "
                f"Available links: {self.link_names}"
            )

    def _make_robot_with_default_cfg(self, q_mujoco: np.ndarray | None):
        if q_mujoco is None:
            return self.robot

        q_pk = q_mujoco_to_pyroki_order(self.robot, q_mujoco)

        try:
            return self.pk.Robot.from_urdf(
                self.urdf,
                default_joint_cfg=q_pk,
            )
        except Exception as e:
            if self.config.verbose:
                print(f"[WARN] Could not create warm-start robot: {e}")
            return self.robot

    def solve_pose(self, T_mujoco: np.ndarray, q_prev_mujoco: np.ndarray | None = None) -> PyRokiIKWaypointResult:
        p_pk, R_pk = transform_pose_mujoco_to_pyroki(T_mujoco, self.config)
        q_wxyz_pk = rotation_matrix_to_quaternion_wxyz(R_pk)

        try:
            robot_for_ik = self._make_robot_with_default_cfg(q_prev_mujoco)
            if self.config.use_manipulability:
                q_pk_raw = self.pks.solve_ik_with_manipulability(
                    robot_for_ik,
                    self.config.target_link_name,
                    p_pk,
                    q_wxyz_pk,
                    manipulability_weight=float(self.config.manipulability_weight),
                )
            else:
                q_pk_raw = self.pks.solve_ik(
                    robot_for_ik,
                    self.config.target_link_name,
                    q_wxyz_pk,
                    p_pk,
                )

            q_pk_raw = np.asarray(q_pk_raw, dtype=np.float64).reshape(-1)
            q_mj_raw = q_pyroki_to_mujoco_order(self.robot, q_pk_raw)

            if q_prev_mujoco is not None:
                q_mj = make_q_continuous_with_previous(q_mj_raw, q_prev_mujoco)
            else:
                q_mj = q_mj_raw

            q_pk = q_mujoco_to_pyroki_order(self.robot, q_mj)

            fk_pose7 = get_fk_link_pose7(
                self.robot,
                q_pk,
                self.config.target_link_name,
            )
            fk_wxyz = fk_pose7[:4]
            fk_pos = fk_pose7[4:7]

            pos_err = float(np.linalg.norm(fk_pos - p_pk))
            rot_err = quaternion_angular_distance_wxyz(fk_wxyz, q_wxyz_pk)

            success = (
                pos_err <= float(self.config.pos_tol_m)
                and rot_err <= float(self.config.rot_tol_rad)
            )

            message = "success" if success else "tracking_error_above_threshold"

            return PyRokiIKWaypointResult(
                success=success,
                q_mujoco_order=q_mj,
                q_pyroki_order=q_pk,
                target_position_pyroki=p_pk,
                target_wxyz_pyroki=q_wxyz_pk,
                fk_position_pyroki=fk_pos,
                fk_wxyz_pyroki=fk_wxyz,
                pos_error_m=pos_err,
                rot_error_rad=rot_err,
                message=message,
            )

        except Exception as e:
            q_fallback = np.asarray(q_prev_mujoco, dtype=np.float64).reshape(6) if q_prev_mujoco is not None else np.zeros(6)

            return PyRokiIKWaypointResult(
                success=False,
                q_mujoco_order=q_fallback.copy(),
                q_pyroki_order=q_mujoco_to_pyroki_order(self.robot, q_fallback),
                target_position_pyroki=p_pk,
                target_wxyz_pyroki=q_wxyz_pk,
                fk_position_pyroki=None,
                fk_wxyz_pyroki=None,
                pos_error_m=np.inf,
                rot_error_rad=np.inf,
                message=f"exception: {type(e).__name__}: {e}",
            )

    def solve_pose_sequence(
        self,
        poses_mujoco: np.ndarray,
        *,
        q_seed_mujoco: Iterable[float] | None = None,
    ) -> PyRokiIKSequenceResult:
        poses_mujoco = np.asarray(poses_mujoco, dtype=np.float64)

        if poses_mujoco.ndim != 3 or poses_mujoco.shape[1:] != (4, 4):
            raise ValueError(f"poses_mujoco must have shape (N, 4, 4), got {poses_mujoco.shape}")

        q_prev: np.ndarray | None
        if q_seed_mujoco is None:
            q_prev = None
        else:
            q_prev = np.asarray(list(q_seed_mujoco), dtype=np.float64).reshape(6)

        results: list[PyRokiIKWaypointResult] = []
        q_list: list[np.ndarray] = []
        failed_indices: list[int] = []

        for i, T in enumerate(poses_mujoco):
            result = self.solve_pose(T, q_prev_mujoco=q_prev)

            results.append(result)
            q_list.append(result.q_mujoco_order.copy())

            if not result.success:
                failed_indices.append(i)

            q_prev = result.q_mujoco_order.copy()

            if self.config.verbose:
                status = "OK" if result.success else "FAIL"
                print(
                    f"[PyRoki IK] {i:03d}/{len(poses_mujoco)} {status} "
                    f"pos_err={result.pos_error_m:.5f} "
                    f"rot_err={result.rot_error_rad:.5f} "
                    f"msg={result.message}"
                )

        q_traj = np.asarray(q_list, dtype=np.float64)
        q_traj = unwrap_joint_trajectory(q_traj)

        success_rate = 0.0 if len(results) == 0 else float(sum(r.success for r in results) / len(results))

        return PyRokiIKSequenceResult(
            success=(len(failed_indices) == 0),
            q_traj_mujoco_order=q_traj,
            waypoint_results=results,
            success_rate=success_rate,
            failed_indices=failed_indices,
        )

    def summarize_sequence_result(self, result: PyRokiIKSequenceResult) -> dict[str, Any]:
        pos_errors = [float(r.pos_error_m) for r in result.waypoint_results]
        rot_errors = [float(r.rot_error_rad) for r in result.waypoint_results]
        messages = [r.message for r in result.waypoint_results]

        if len(result.waypoint_results) == 0:
            return {
                "success": False,
                "success_rate": 0.0,
                "num_waypoints": 0,
                "num_failed": 0,
                "failed_indices": [],
            }

        return {
            "success": bool(result.success),
            "success_rate": float(result.success_rate),
            "num_waypoints": int(len(result.waypoint_results)),
            "num_failed": int(len(result.failed_indices)),
            "failed_indices": [int(i) for i in result.failed_indices],
            "mean_pos_error_m": float(np.mean(pos_errors)),
            "max_pos_error_m": float(np.max(pos_errors)),
            "mean_rot_error_rad": float(np.mean(rot_errors)),
            "max_rot_error_rad": float(np.max(rot_errors)),
            "messages": messages,
        }