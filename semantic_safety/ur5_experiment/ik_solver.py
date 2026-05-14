from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import mujoco
import numpy as np

from semantic_safety.ur5_experiment.mujoco_ur5_env import MujocoUR5Env


@dataclass
class IKConfig:
    max_iters: int = 200
    pos_tol_m: float = 2e-3
    rot_tol_rad: float = 5e-2

    damping: float = 1e-3
    step_size: float = 1.0
    max_delta_q_norm: float = 0.15

    use_orientation: bool = True
    position_weight: float = 1.0
    orientation_weight: float = 0.25

    joint_limit_margin_rad: float = 1e-4

    # Small regularization toward a nominal posture.
    # Set weight=0.0 to disable.
    nominal_weight: float = 1e-3
    q_nominal: tuple[float, ...] | None = None

    verbose: bool = False


@dataclass
class IKResult:
    success: bool
    q: np.ndarray
    final_pos_error_m: float
    final_rot_error_rad: float
    num_iters: int
    message: str


@dataclass
class IKSequenceResult:
    success: bool
    q_traj: np.ndarray
    waypoint_results: list[IKResult]
    success_rate: float
    failed_indices: list[int]


def rotation_vector_from_matrix(R: np.ndarray) -> np.ndarray:
    """
    Convert a rotation matrix to axis-angle rotation vector.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"R must have shape (3, 3), got {R.shape}")

    cos_angle = (np.trace(R) - 1.0) * 0.5
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    angle = float(np.arccos(cos_angle))

    if angle < 1e-9:
        return np.zeros(3, dtype=np.float64)

    if np.pi - angle < 1e-5:
        # Robust fallback near 180 degrees.
        axis = np.array(
            [
                np.sqrt(max(0.0, (R[0, 0] + 1.0) / 2.0)),
                np.sqrt(max(0.0, (R[1, 1] + 1.0) / 2.0)),
                np.sqrt(max(0.0, (R[2, 2] + 1.0) / 2.0)),
            ],
            dtype=np.float64,
        )

        # Recover signs from off-diagonal terms.
        if R[2, 1] - R[1, 2] < 0:
            axis[0] = -axis[0]
        if R[0, 2] - R[2, 0] < 0:
            axis[1] = -axis[1]
        if R[1, 0] - R[0, 1] < 0:
            axis[2] = -axis[2]

        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            axis /= norm

        return axis * angle

    axis = np.array(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ],
        dtype=np.float64,
    ) / (2.0 * np.sin(angle))

    return axis * angle


def pose_error(
    current_pos: np.ndarray,
    current_R: np.ndarray,
    target_pos: np.ndarray,
    target_R: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Return position error and world-frame orientation rotation-vector error.
    """
    current_pos = np.asarray(current_pos, dtype=np.float64)
    target_pos = np.asarray(target_pos, dtype=np.float64)
    current_R = np.asarray(current_R, dtype=np.float64)
    target_R = np.asarray(target_R, dtype=np.float64)

    e_pos = target_pos - current_pos

    # Left-multiplicative world-frame rotation:
    # target_R = R_err @ current_R
    R_err = target_R @ current_R.T
    e_rot = rotation_vector_from_matrix(R_err)

    return e_pos, e_rot, float(np.linalg.norm(e_pos)), float(np.linalg.norm(e_rot))


class MujocoDampedLeastSquaresIK:
    """
    Damped least-squares IK using MuJoCo site Jacobians.

    This solver is meant as a simple, reliable fallback/debug tool before PyRoki.
    It solves one pose at a time and uses warm-starting for pose sequences.
    """

    def __init__(self, env: MujocoUR5Env, config: IKConfig):
        self.env = env
        self.config = config

        if config.q_nominal is None:
            self.q_nominal = env.get_qpos().copy()
        else:
            q_nom = np.asarray(config.q_nominal, dtype=np.float64)
            if q_nom.shape != (env.dof,):
                raise ValueError(f"q_nominal must have shape {(env.dof,)}, got {q_nom.shape}")
            self.q_nominal = q_nom

    def solve_pose(
        self,
        target_pos: np.ndarray,
        target_R: np.ndarray | None = None,
        *,
        q_init: Iterable[float] | None = None,
    ) -> IKResult:
        target_pos = np.asarray(target_pos, dtype=np.float64)
        if target_pos.shape != (3,):
            raise ValueError(f"target_pos must have shape (3,), got {target_pos.shape}")

        if target_R is None:
            target_R = self.env.get_ee_rotation_matrix()
        else:
            target_R = np.asarray(target_R, dtype=np.float64)
            if target_R.shape != (3, 3):
                raise ValueError(f"target_R must have shape (3, 3), got {target_R.shape}")

        if q_init is None:
            q = self.env.get_qpos().copy()
        else:
            q = np.asarray(list(q_init), dtype=np.float64)
            if q.shape != (self.env.dof,):
                raise ValueError(f"q_init must have shape {(self.env.dof,)}, got {q.shape}")

        cfg = self.config
        last_pos_err = np.inf
        last_rot_err = np.inf

        for it in range(cfg.max_iters):
            self.env.set_qpos(q, zero_velocity=True, forward=True)

            cur_pos, cur_R = self.env.get_ee_pose()
            e_pos, e_rot, pos_err, rot_err = pose_error(
                current_pos=cur_pos,
                current_R=cur_R,
                target_pos=target_pos,
                target_R=target_R,
            )

            last_pos_err = pos_err
            last_rot_err = rot_err

            if pos_err <= cfg.pos_tol_m and (not cfg.use_orientation or rot_err <= cfg.rot_tol_rad):
                return IKResult(
                    success=True,
                    q=q.copy(),
                    final_pos_error_m=pos_err,
                    final_rot_error_rad=rot_err,
                    num_iters=it,
                    message="converged",
                )

            jacp = np.zeros((3, self.env.model.nv), dtype=np.float64)
            jacr = np.zeros((3, self.env.model.nv), dtype=np.float64)

            mujoco.mj_jacSite(
                self.env.model,
                self.env.data,
                jacp,
                jacr,
                self.env.ee_site_id,
            )

            J_pos = jacp[:, self.env.qvel_addrs]

            if cfg.use_orientation:
                J_rot = jacr[:, self.env.qvel_addrs]
                J = np.vstack(
                    [
                        cfg.position_weight * J_pos,
                        cfg.orientation_weight * J_rot,
                    ]
                )
                err = np.concatenate(
                    [
                        cfg.position_weight * e_pos,
                        cfg.orientation_weight * e_rot,
                    ]
                )
            else:
                J = cfg.position_weight * J_pos
                err = cfg.position_weight * e_pos

            # Optional nominal posture regularization:
            # Solve augmented least squares:
            #   [J              ] dq = [err]
            #   [sqrt(w) I      ]      [sqrt(w)(q_nom - q)]
            if cfg.nominal_weight > 0.0:
                w = float(np.sqrt(cfg.nominal_weight))
                J = np.vstack([J, w * np.eye(self.env.dof)])
                err = np.concatenate([err, w * (self.q_nominal - q)])

            dq = self._damped_least_squares(J, err, cfg.damping)
            dq = cfg.step_size * dq

            dq_norm = float(np.linalg.norm(dq))
            if dq_norm > cfg.max_delta_q_norm:
                dq = dq * (cfg.max_delta_q_norm / max(dq_norm, 1e-12))

            q = q + dq
            q = self._clip_to_joint_limits(q)

            if cfg.verbose and (it % 20 == 0 or it == cfg.max_iters - 1):
                print(
                    f"[IK] iter={it:03d} "
                    f"pos_err={pos_err:.5f} "
                    f"rot_err={rot_err:.5f} "
                    f"|dq|={dq_norm:.5f}"
                )

        self.env.set_qpos(q, zero_velocity=True, forward=True)

        return IKResult(
            success=False,
            q=q.copy(),
            final_pos_error_m=float(last_pos_err),
            final_rot_error_rad=float(last_rot_err),
            num_iters=cfg.max_iters,
            message="max_iters_reached",
        )

    def solve_pose_sequence(
        self,
        poses: np.ndarray,
        *,
        q_seed: Iterable[float] | None = None,
        stop_on_failure: bool = False,
    ) -> IKSequenceResult:
        """
        Solve IK for a sequence of target poses.

        poses:
            shape (N, 4, 4)

        Warm-start:
            waypoint i starts from solution of waypoint i-1.
        """
        poses = np.asarray(poses, dtype=np.float64)

        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise ValueError(f"poses must have shape (N, 4, 4), got {poses.shape}")

        if q_seed is None:
            q_current = self.env.get_qpos().copy()
        else:
            q_current = np.asarray(list(q_seed), dtype=np.float64)
            if q_current.shape != (self.env.dof,):
                raise ValueError(f"q_seed must have shape {(self.env.dof,)}, got {q_current.shape}")

        results: list[IKResult] = []
        q_list: list[np.ndarray] = []
        failed_indices: list[int] = []

        for i, T in enumerate(poses):
            target_pos = T[:3, 3]
            target_R = T[:3, :3]

            result = self.solve_pose(
                target_pos=target_pos,
                target_R=target_R,
                q_init=q_current,
            )

            results.append(result)
            q_list.append(result.q.copy())

            if result.success:
                q_current = result.q.copy()
            else:
                failed_indices.append(i)
                # Continue from best-effort result unless asked to stop.
                q_current = result.q.copy()

                if stop_on_failure:
                    break

            if self.config.verbose:
                status = "OK" if result.success else "FAIL"
                print(
                    f"[IK sequence] {i:03d}/{len(poses)} {status} "
                    f"pos_err={result.final_pos_error_m:.5f} "
                    f"rot_err={result.final_rot_error_rad:.5f} "
                    f"iters={result.num_iters}"
                )

        q_traj = np.asarray(q_list, dtype=np.float64)
        success_rate = 0.0 if len(results) == 0 else sum(r.success for r in results) / len(results)

        return IKSequenceResult(
            success=(len(failed_indices) == 0 and len(results) == len(poses)),
            q_traj=q_traj,
            waypoint_results=results,
            success_rate=float(success_rate),
            failed_indices=failed_indices,
        )

    @staticmethod
    def _damped_least_squares(J: np.ndarray, err: np.ndarray, damping: float) -> np.ndarray:
        """
        dq = J^T (J J^T + λ² I)^-1 err
        """
        J = np.asarray(J, dtype=np.float64)
        err = np.asarray(err, dtype=np.float64)

        A = J @ J.T + (float(damping) ** 2) * np.eye(J.shape[0])
        return J.T @ np.linalg.solve(A, err)

    def _clip_to_joint_limits(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).copy()

        for local_i, joint_id in enumerate(self.env.joint_ids):
            jid = int(joint_id)

            if not bool(self.env.model.jnt_limited[jid]):
                continue

            lo, hi = self.env.model.jnt_range[jid]
            lo = float(lo) + self.config.joint_limit_margin_rad
            hi = float(hi) - self.config.joint_limit_margin_rad

            q[local_i] = float(np.clip(q[local_i], lo, hi))

        return q