from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np

from semantic_safety.perception_2d3d.mujoco_camera import MujocoCamera


@dataclass
class UR5EnvConfig:
    xml_path: str
    ee_site_name: str = "attachment_site"
    camera_name: str = "main_cam"
    width: int = 640
    height: int = 480

    # If None, we auto-detect common UR5 joint names.
    joint_names: tuple[str, ...] | None = None

    # If the XML has a keyframe named "home", we can optionally use it later.
    keyframe_name: str | None = None

    settle_steps: int = 50


class MujocoUR5Env:
    """
    Minimal MuJoCo wrapper for the UR5 static-scene experiment.

    This class is intentionally small. It does NOT do:
      - risk field generation
      - A* planning
      - IK solving
      - PyRoki optimization
      - full collision checking

    It only provides:
      - model/data loading
      - UR5 joint qpos/qvel access
      - end-effector site pose query
      - camera RGB-D capture
      - simple stepping/resetting
    """

    def __init__(self, config: UR5EnvConfig):
        self.config = config
        self.xml_path = Path(config.xml_path)

        if not self.xml_path.exists():
            raise FileNotFoundError(f"MuJoCo XML not found: {self.xml_path}")

        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        self.ee_site_id = self._require_id(
            mujoco.mjtObj.mjOBJ_SITE,
            config.ee_site_name,
        )

        self.joint_names = (
            tuple(config.joint_names)
            if config.joint_names is not None
            else self._auto_detect_ur5_joints()
        )

        if len(self.joint_names) != 6:
            available = self.list_names(mujoco.mjtObj.mjOBJ_JOINT)
            raise ValueError(
                "Expected 6 UR5 joints, but detected "
                f"{len(self.joint_names)}: {self.joint_names}\n"
                f"Available joints: {available}\n"
                "Pass joint_names explicitly in UR5EnvConfig if needed."
            )

        self.joint_ids = np.array(
            [
                self._require_id(mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.joint_names
            ],
            dtype=np.int32,
        )

        self.qpos_addrs = np.array(
            [self.model.jnt_qposadr[jid] for jid in self.joint_ids],
            dtype=np.int32,
        )

        self.qvel_addrs = np.array(
            [self.model.jnt_dofadr[jid] for jid in self.joint_ids],
            dtype=np.int32,
        )

        self.camera: MujocoCamera | None = None
        self.camera_id: int | None = None

        if config.camera_name:
            self.camera_id = self._require_id(
                mujoco.mjtObj.mjOBJ_CAMERA,
                config.camera_name,
            )
            self.camera = MujocoCamera(
                self.model,
                self.data,
                cam_name=config.camera_name,
                width=config.width,
                height=config.height,
            )

        self.reset(settle_steps=config.settle_steps)

    # ------------------------------------------------------------------
    # Basic simulation control
    # ------------------------------------------------------------------

    def reset(self, settle_steps: int | None = None) -> None:
        mujoco.mj_resetData(self.model, self.data)

        if self.config.keyframe_name is not None:
            self.set_keyframe(self.config.keyframe_name)

        n_steps = self.config.settle_steps if settle_steps is None else settle_steps
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)

        mujoco.mj_forward(self.model, self.data)

    def forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)

    def step(self, n: int = 1) -> None:
        for _ in range(int(n)):
            mujoco.mj_step(self.model, self.data)

    def set_keyframe(self, keyframe_name: str) -> None:
        key_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_KEY,
            keyframe_name,
        )
        if key_id < 0:
            available = self.list_names(mujoco.mjtObj.mjOBJ_KEY)
            raise ValueError(
                f"Keyframe '{keyframe_name}' not found. "
                f"Available keyframes: {available}"
            )

        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------
    # Joint access
    # ------------------------------------------------------------------

    @property
    def dof(self) -> int:
        return len(self.joint_names)

    def get_qpos(self) -> np.ndarray:
        return self.data.qpos[self.qpos_addrs].copy()

    def get_qvel(self) -> np.ndarray:
        return self.data.qvel[self.qvel_addrs].copy()

    def set_qpos(
        self,
        q: Iterable[float],
        *,
        zero_velocity: bool = True,
        forward: bool = True,
    ) -> None:
        q_arr = np.asarray(list(q), dtype=np.float64)

        if q_arr.shape != (self.dof,):
            raise ValueError(f"Expected q shape {(self.dof,)}, got {q_arr.shape}")

        self.data.qpos[self.qpos_addrs] = q_arr

        if zero_velocity:
            self.data.qvel[self.qvel_addrs] = 0.0

        if forward:
            mujoco.mj_forward(self.model, self.data)

    def set_qpos_by_dict(
        self,
        q_dict: dict[str, float],
        *,
        zero_velocity: bool = True,
        forward: bool = True,
    ) -> None:
        q = self.get_qpos()

        name_to_local_idx = {
            name: i for i, name in enumerate(self.joint_names)
        }

        for name, value in q_dict.items():
            if name not in name_to_local_idx:
                raise ValueError(
                    f"Joint '{name}' is not one of UR5 joints: {self.joint_names}"
                )
            q[name_to_local_idx[name]] = float(value)

        self.set_qpos(q, zero_velocity=zero_velocity, forward=forward)

    # ------------------------------------------------------------------
    # End-effector pose
    # ------------------------------------------------------------------

    def get_ee_position(self) -> np.ndarray:
        return self.data.site_xpos[self.ee_site_id].copy()

    def get_ee_rotation_matrix(self) -> np.ndarray:
        return self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()

    def get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.get_ee_position(), self.get_ee_rotation_matrix()

    def get_ee_pose_matrix(self) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        pos, rot = self.get_ee_pose()
        T[:3, :3] = rot
        T[:3, 3] = pos
        return T

    # ------------------------------------------------------------------
    # Object / camera helpers
    # ------------------------------------------------------------------

    def get_body_position(self, body_name: str) -> np.ndarray:
        body_id = self._require_id(mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self.data.xpos[body_id].copy()

    def get_geom_position(self, geom_name: str) -> np.ndarray:
        geom_id = self._require_id(mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        return self.data.geom_xpos[geom_id].copy()

    def get_camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        if self.camera_id is None:
            raise RuntimeError("No camera configured.")

        pos = self.data.cam_xpos[self.camera_id].copy()
        rot = self.data.cam_xmat[self.camera_id].reshape(3, 3).copy()
        return pos, rot

    def capture_rgbd(self) -> tuple[np.ndarray, np.ndarray, dict]:
        if self.camera is None:
            raise RuntimeError("No camera configured for this environment.")

        rgb, depth, intrinsics = self.camera.get_frames()
        return rgb, depth, intrinsics

    # ------------------------------------------------------------------
    # Replay helpers
    # ------------------------------------------------------------------

    def replay_joint_trajectory(
        self,
        q_traj: np.ndarray,
        *,
        steps_per_waypoint: int = 1,
        render: bool = False,
    ) -> list[np.ndarray]:
        """
        Simple position replay by directly setting qpos.

        This is not a controller. It is only for debugging/replay.
        """
        q_traj = np.asarray(q_traj, dtype=np.float64)

        if q_traj.ndim != 2 or q_traj.shape[1] != self.dof:
            raise ValueError(
                f"q_traj must have shape (N, {self.dof}), got {q_traj.shape}"
            )

        frames: list[np.ndarray] = []

        for q in q_traj:
            self.set_qpos(q, zero_velocity=True, forward=True)

            for _ in range(max(1, int(steps_per_waypoint))):
                self.step(1)

            if render:
                rgb, _, _ = self.capture_rgbd()
                frames.append(rgb.copy())

        return frames

    # ------------------------------------------------------------------
    # Debug / names
    # ------------------------------------------------------------------

    def print_debug_summary(self) -> None:
        ee_pos, ee_rot = self.get_ee_pose()

        print("\n=== MujocoUR5Env Debug Summary ===")
        print(f"XML path          : {self.xml_path}")
        print(f"nq / nv           : {self.model.nq} / {self.model.nv}")
        print(f"UR5 dof           : {self.dof}")
        print(f"EE site           : {self.config.ee_site_name}")

        print("\nUR5 joints:")
        for local_i, (name, jid, qadr, dadr) in enumerate(
            zip(self.joint_names, self.joint_ids, self.qpos_addrs, self.qvel_addrs)
        ):
            print(
                f"  [{local_i}] {name:<24s} "
                f"joint_id={int(jid):2d} "
                f"qpos_addr={int(qadr):2d} "
                f"qvel_addr={int(dadr):2d} "
                f"q={self.data.qpos[qadr]: .4f}"
            )

        print("\nEE pose:")
        print(f"  position: {np.round(ee_pos, 5)}")
        print("  rotation:")
        print(np.round(ee_rot, 5))

        if self.camera is not None:
            cam_pos, cam_rot = self.get_camera_pose()
            print("\nCamera:")
            print(f"  name      : {self.config.camera_name}")
            print(f"  image size: {self.config.width} x {self.config.height}")
            print(f"  intrinsics: {self.camera.intrinsics}")
            print(f"  position  : {np.round(cam_pos, 5)}")
            print("  rotation:")
            print(np.round(cam_rot, 5))

        print("\nAvailable sites:")
        for i, name in enumerate(self.list_names(mujoco.mjtObj.mjOBJ_SITE)):
            print(f"  [{i}] {name}")

        print("\nAvailable cameras:")
        for i, name in enumerate(self.list_names(mujoco.mjtObj.mjOBJ_CAMERA)):
            print(f"  [{i}] {name}")

    def list_names(self, obj_type: mujoco.mjtObj) -> list[str]:
        if obj_type == mujoco.mjtObj.mjOBJ_JOINT:
            n = self.model.njnt
        elif obj_type == mujoco.mjtObj.mjOBJ_SITE:
            n = self.model.nsite
        elif obj_type == mujoco.mjtObj.mjOBJ_BODY:
            n = self.model.nbody
        elif obj_type == mujoco.mjtObj.mjOBJ_GEOM:
            n = self.model.ngeom
        elif obj_type == mujoco.mjtObj.mjOBJ_CAMERA:
            n = self.model.ncam
        elif obj_type == mujoco.mjtObj.mjOBJ_KEY:
            n = self.model.nkey
        else:
            raise ValueError(f"Unsupported object type for listing: {obj_type}")

        out: list[str] = []
        for i in range(n):
            name = mujoco.mj_id2name(self.model, obj_type, i)
            if name is not None:
                out.append(name)
        return out

    def _require_id(self, obj_type: mujoco.mjtObj, name: str) -> int:
        obj_id = mujoco.mj_name2id(self.model, obj_type, name)
        if obj_id < 0:
            available = self.list_names(obj_type)
            raise ValueError(
                f"Name '{name}' not found for object type {obj_type}.\n"
                f"Available names: {available}"
            )
        return int(obj_id)

    def _auto_detect_ur5_joints(self) -> tuple[str, ...]:
        available = self.list_names(mujoco.mjtObj.mjOBJ_JOINT)

        preferred_order = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]

        detected: list[str] = []

        for target in preferred_order:
            exact = [name for name in available if name == target]
            if exact:
                detected.append(exact[0])
                continue

            fuzzy = [name for name in available if target in name]
            if fuzzy:
                detected.append(fuzzy[0])
                continue

        return tuple(detected)