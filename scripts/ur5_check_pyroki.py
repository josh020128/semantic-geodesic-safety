from __future__ import annotations

import argparse
import importlib
import inspect
import os
import pkgutil
import sys
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MUJOCO_UR5_JOINT_ORDER = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


COMMON_URDF_CANDIDATES = [
    "data/assets/ur5e/ur5e.urdf",
    "data/assets/universal_robots_ur5e/ur5e.urdf",
    "data/assets/universal_robot/ur_description/urdf/ur5e.urdf",
    "data/assets/ur_description/urdf/ur5e.urdf",
    "external/universal_robot/ur_description/urdf/ur5e.urdf",
    "external/Universal_Robots_ROS2_Description/urdf/ur.urdf",
]

COMMON_PYROKI_EXAMPLES_PATHS = [
    "/home/gl34/research/pyroki/examples",
    str(PROJECT_ROOT.parent / "pyroki" / "examples"),
    str(PROJECT_ROOT / "external" / "pyroki" / "examples"),
]


def add_pyroki_examples_to_syspath(user_path: str | None = None) -> None:
    """
    Add pyroki/examples to sys.path so that `import pyroki_snippets` works.

    PyRoki examples often use:
        import pyroki_snippets as pks

    but pyroki_snippets may live under the cloned repo's examples directory.
    """
    candidates = []

    if user_path:
        candidates.append(user_path)

    env_path = os.environ.get("PYROKI_EXAMPLES_PATH", "")
    if env_path:
        candidates.append(env_path)

    candidates.extend(COMMON_PYROKI_EXAMPLES_PATHS)

    for path_str in candidates:
        path = Path(path_str).expanduser().resolve()
        if path.exists() and path.is_dir():
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
                print(f"[OK] added pyroki examples to sys.path: {path}")
            return

    print("[WARN] Could not find pyroki examples path for pyroki_snippets.")
    print("       You can pass --pyroki-examples-path /home/gl34/research/pyroki/examples")

def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def import_optional(module_name: str):
    try:
        mod = importlib.import_module(module_name)
        print(f"[OK] import {module_name}")
        version = getattr(mod, "__version__", None)
        file = getattr(mod, "__file__", None)
        if version is not None:
            print(f"     version: {version}")
        if file is not None:
            print(f"     file   : {file}")
        return mod
    except Exception as e:
        print(f"[FAIL] import {module_name}: {type(e).__name__}: {e}")
        return None


def find_existing_urdf(user_path: str | None) -> Path | None:
    if user_path:
        path = Path(user_path).expanduser()
        if path.exists():
            return path.resolve()
        print(f"[WARN] user-provided URDF does not exist: {path}")
        return None

    env_path = os.environ.get("UR5_URDF_PATH", "")
    if env_path:
        path = Path(env_path).expanduser()
        if path.exists():
            return path.resolve()
        print(f"[WARN] UR5_URDF_PATH set but file does not exist: {path}")

    for rel in COMMON_URDF_CANDIDATES:
        path = (PROJECT_ROOT / rel).resolve()
        if path.exists():
            return path

    return None


def parse_urdf_basic(urdf_path: Path) -> dict[str, Any]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    if root.tag != "robot":
        raise ValueError(f"Expected root tag <robot>, got <{root.tag}>")

    robot_name = root.attrib.get("name", "unknown")

    links = []
    joints = []
    movable_joints = []
    fixed_joints = []

    for link in root.findall("link"):
        name = link.attrib.get("name")
        if name:
            links.append(name)

    for joint in root.findall("joint"):
        name = joint.attrib.get("name")
        jtype = joint.attrib.get("type", "unknown")

        parent_el = joint.find("parent")
        child_el = joint.find("child")
        parent = parent_el.attrib.get("link") if parent_el is not None else None
        child = child_el.attrib.get("link") if child_el is not None else None

        info = {
            "name": name,
            "type": jtype,
            "parent": parent,
            "child": child,
        }
        joints.append(info)

        if jtype == "fixed":
            fixed_joints.append(info)
        else:
            movable_joints.append(info)

    return {
        "robot_name": robot_name,
        "links": links,
        "joints": joints,
        "movable_joints": movable_joints,
        "fixed_joints": fixed_joints,
    }


def print_urdf_summary(info: dict[str, Any]) -> None:
    print(f"robot name        : {info['robot_name']}")
    print(f"num links         : {len(info['links'])}")
    print(f"num joints        : {len(info['joints'])}")
    print(f"num movable joints: {len(info['movable_joints'])}")
    print(f"num fixed joints  : {len(info['fixed_joints'])}")

    print("\nMovable joints:")
    for i, j in enumerate(info["movable_joints"]):
        print(
            f"  [{i}] {j['name']:<28s} "
            f"type={j['type']:<10s} "
            f"parent={j['parent']} child={j['child']}"
        )

    print("\nLikely EE / tool links:")
    likely_terms = ["tool", "ee", "flange", "wrist_3", "wrist3", "tcp"]
    candidates = [
        link for link in info["links"]
        if any(term in link.lower() for term in likely_terms)
    ]
    for link in candidates:
        print(f"  - {link}")

    print("\nJoint order check against MuJoCo UR5 order:")
    urdf_movable_names = [j["name"] for j in info["movable_joints"]]
    for name in MUJOCO_UR5_JOINT_ORDER:
        if name in urdf_movable_names:
            print(f"  [OK] {name}")
        else:
            print(f"  [MISSING] {name}")

    if urdf_movable_names[:6] == MUJOCO_UR5_JOINT_ORDER:
        print("\n[OK] First 6 URDF movable joints exactly match MuJoCo UR5 joint order.")
    else:
        print("\n[WARN] URDF movable joint order may differ from MuJoCo order.")
        print("       MuJoCo order:", MUJOCO_UR5_JOINT_ORDER)
        print("       URDF order  :", urdf_movable_names[:10])


def try_robot_descriptions_load(description_name: str | None):
    if not description_name:
        return None

    print_header(f"Trying robot_descriptions loader: {description_name}")

    try:
        from robot_descriptions.loaders.yourdfpy import load_robot_description
    except Exception as e:
        print(f"[FAIL] could not import robot_descriptions loader: {e}")
        return None

    try:
        urdf = load_robot_description(description_name)
        print(f"[OK] loaded robot description: {description_name}")
        print(f"     type: {type(urdf)}")
        return urdf
    except Exception as e:
        print(f"[FAIL] load_robot_description({description_name!r}): {type(e).__name__}: {e}")
        return None


def try_load_urdf_with_yourdfpy(urdf_path: Path | None):
    if urdf_path is None:
        return None

    print_header("Trying yourdfpy / urdfpy URDF load")

    for module_name, class_name in [
        ("yourdfpy", "URDF"),
        ("urdfpy", "URDF"),
    ]:
        try:
            mod = importlib.import_module(module_name)
            URDF = getattr(mod, class_name)
        except Exception as e:
            print(f"[SKIP] {module_name}.{class_name} not available: {e}")
            continue

        try:
            urdf = URDF.load(str(urdf_path))
            print(f"[OK] loaded with {module_name}.{class_name}.load")
            print(f"     type: {type(urdf)}")
            return urdf
        except Exception as e:
            print(f"[FAIL] {module_name}.{class_name}.load: {type(e).__name__}: {e}")

    return None


def try_pyroki_robot_from_urdf(pyroki_mod, urdf_obj):
    print_header("Trying pk.Robot.from_urdf(...)")

    if pyroki_mod is None:
        print("[SKIP] pyroki is not importable.")
        return None

    Robot = getattr(pyroki_mod, "Robot", None)
    if Robot is None:
        print("[FAIL] pyroki.Robot does not exist.")
        print("       Top-level pyroki names:", [x for x in dir(pyroki_mod) if not x.startswith("_")][:80])
        return None

    print(f"[OK] found pyroki.Robot: {Robot}")

    if urdf_obj is None:
        print("[SKIP] no URDF object loaded.")
        return None

    for method_name in ["from_urdf", "from_urdf_path", "from_urdf_file"]:
        method = getattr(Robot, method_name, None)
        if method is None:
            print(f"[SKIP] Robot.{method_name} not found.")
            continue

        try:
            print(f"Trying Robot.{method_name}(...), signature:")
            try:
                print(f"  {inspect.signature(method)}")
            except Exception:
                print("  <signature unavailable>")

            if method_name == "from_urdf":
                robot = method(urdf_obj)
            else:
                # If caller passed URDF object, this may fail.
                # The script reports the failure without crashing.
                robot = method(str(urdf_obj))

            print(f"[OK] Robot.{method_name} succeeded.")
            print(f"     robot type: {type(robot)}")
            print("     robot attrs sample:", [x for x in dir(robot) if not x.startswith("_")][:80])
            return robot
        except Exception as e:
            print(f"[FAIL] Robot.{method_name}: {type(e).__name__}: {e}")
            if method_name == "from_urdf":
                traceback.print_exc(limit=1)

    return None

def _to_list_if_sequence(value: Any) -> list[Any] | None:
    """
    Convert common sequence-like objects to Python list.
    Return None if it does not look like a sequence.
    """
    if value is None:
        return None

    if isinstance(value, str):
        return [value]

    if isinstance(value, dict):
        return list(value.keys())

    if isinstance(value, (list, tuple)):
        return list(value)

    # numpy / jax arrays or similar
    if hasattr(value, "tolist"):
        try:
            out = value.tolist()
            if isinstance(out, list):
                return out
            return [out]
        except Exception:
            pass

    return None


def _safe_name(obj: Any) -> str:
    """
    Extract a readable name from PyRoki joint/link objects.
    """
    if isinstance(obj, str):
        return obj

    for attr in ["name", "joint_name", "link_name"]:
        if hasattr(obj, attr):
            try:
                value = getattr(obj, attr)
                if value is not None:
                    return str(value)
            except Exception:
                pass

    if hasattr(obj, "_asdict"):
        try:
            d = obj._asdict()
            for key in ["name", "joint_name", "link_name"]:
                if key in d:
                    return str(d[key])
        except Exception:
            pass

    return str(obj)


def _extract_names_from_container(
    container: Any,
    *,
    preferred_attrs: list[str],
    label: str,
) -> list[str]:
    """
    PyRoki robot.joints / robot.links may be:
      - list/tuple of objects
      - a JointInfo / LinkInfo object with name arrays inside
      - a dataclass-like object

    This function tries several common attribute names.
    """
    if container is None:
        return []

    # Case 1: directly iterable sequence.
    direct_list = _to_list_if_sequence(container)
    if direct_list is not None:
        return [_safe_name(x) for x in direct_list]

    # Case 2: object has a useful names-like attribute.
    for attr in preferred_attrs:
        if hasattr(container, attr):
            try:
                value = getattr(container, attr)
                seq = _to_list_if_sequence(value)
                if seq is not None:
                    return [_safe_name(x) for x in seq]
            except Exception:
                pass

    # Case 3: search all public attributes containing "name".
    candidate_attrs = []
    for attr in dir(container):
        if attr.startswith("_"):
            continue
        if "name" not in attr.lower():
            continue

        try:
            value = getattr(container, attr)
        except Exception:
            continue

        if callable(value):
            continue

        seq = _to_list_if_sequence(value)
        if seq is not None:
            candidate_attrs.append((attr, [_safe_name(x) for x in seq]))

    if candidate_attrs:
        print(f"\n[INFO] Found possible {label} name attributes:")
        for attr, names in candidate_attrs:
            print(f"  {attr}: {names}")
        # Use the first candidate by default.
        return candidate_attrs[0][1]

    # Case 4: give up but print object summary.
    print(f"\n[WARN] Could not extract {label} names from container.")
    print(f"container type: {type(container)}")
    print(f"container repr: {repr(container)}")
    print("public attrs:")
    for attr in dir(container):
        if not attr.startswith("_"):
            print(f"  - {attr}")

    return []

def get_pyroki_joint_names(robot: Any) -> list[str]:
    joints = getattr(robot, "joints", None)
    return _extract_names_from_container(
        joints,
        preferred_attrs=[
            # Important:
            # PyRoki FK expects cfg with shape (*batch, actuated_count),
            # so we should use actuated joint names, not all joint names.
            "actuated_names",
            "actuated_joint_names",
            "active_joint_names",
            "joint_names",
            "names",
            "name",
        ],
        label="actuated joint",
    )

def get_pyroki_link_names(robot: Any) -> list[str]:
    links = getattr(robot, "links", None)
    return _extract_names_from_container(
        links,
        preferred_attrs=[
            "names",
            "link_names",
            "frame_names",
            "name",
        ],
        label="link",
    )

def _print_object_summary(obj: Any, prefix: str = "  ") -> None:
    if obj is None:
        print(f"{prefix}<None>")
        return

    print(f"{prefix}type: {type(obj)}")
    print(f"{prefix}repr: {repr(obj)}")

    attrs = []
    for name in dir(obj):
        if name.startswith("_"):
            continue

        try:
            value = getattr(obj, name)
        except Exception:
            continue

        if callable(value):
            continue

        attrs.append(name)

    if attrs:
        print(f"{prefix}public non-callable attrs:")
        for attr in attrs[:50]:
            try:
                value = getattr(obj, attr)
                shape = getattr(value, "shape", None)
                if shape is not None:
                    print(f"{prefix}  - {attr}: type={type(value)}, shape={shape}")
                else:
                    print(f"{prefix}  - {attr}: type={type(value)}, value={repr(value)[:120]}")
            except Exception:
                print(f"{prefix}  - {attr}: <unreadable>")

def inspect_pyroki_robot(robot: Any) -> dict[str, Any]:
    """
    Print PyRoki robot joints/links and compare joint order with MuJoCo UR5 order.
    """
    print_header("Deep inspecting PyRoki Robot")

    joint_container = getattr(robot, "joints", None)
    link_container = getattr(robot, "links", None)

    print("\nrobot.joints container:")
    _print_object_summary(joint_container, prefix="  ")

    print("\nrobot.links container:")
    _print_object_summary(link_container, prefix="  ")

    joint_names = get_pyroki_joint_names(robot)
    link_names = get_pyroki_link_names(robot)

    print("\nExtracted PyRoki actuated joint names:")
    for i, name in enumerate(joint_names):
        print(f"  [{i}] {name}")

    print("\nExtracted PyRoki link names:")
    for i, name in enumerate(link_names):
        print(f"  [{i}] {name}")

    print("\nLikely EE / tool links:")
    likely_terms = ["tool", "ee", "tcp", "flange", "wrist_3", "wrist3"]
    ee_candidates = [
        name for name in link_names
        if any(term in name.lower() for term in likely_terms)
    ]

    if not ee_candidates:
        print("  [WARN] no obvious EE link candidates found.")
    else:
        for name in ee_candidates:
            print(f"  - {name}")

    print("\nActuated joint order comparison:")
    print("  MuJoCo order:")
    for i, name in enumerate(MUJOCO_UR5_JOINT_ORDER):
        print(f"    [{i}] {name}")

    print("  PyRoki order:")
    for i, name in enumerate(joint_names):
        print(f"    [{i}] {name}")

    missing = [name for name in MUJOCO_UR5_JOINT_ORDER if name not in joint_names]

    if missing:
        print("\n[WARN] Some MuJoCo UR5 joints are missing in PyRoki robot.joints:")
        for name in missing:
            print(f"  - {name}")
        mujoco_to_pyroki = None
    else:
        mujoco_to_pyroki = [
            joint_names.index(name)
            for name in MUJOCO_UR5_JOINT_ORDER
        ]

        print("\n[OK] All MuJoCo UR5 joint names found in PyRoki joints.")
        print("MuJoCo index -> PyRoki index:")
        for mj_i, pk_i in enumerate(mujoco_to_pyroki):
            print(f"  MuJoCo[{mj_i}] {MUJOCO_UR5_JOINT_ORDER[mj_i]} -> PyRoki[{pk_i}]")

        if mujoco_to_pyroki == list(range(6)):
            print("[OK] First 6 PyRoki joints match MuJoCo order exactly.")
        else:
            print("[WARN] PyRoki joint order differs from MuJoCo order.")
            print("       Need reorder mapping when passing q between PyRoki and MuJoCo.")

    print("\nRobot methods / attributes:")
    names = [x for x in dir(robot) if not x.startswith("_")]
    for name in names[:100]:
        print(f"  - {name}")

    if hasattr(robot, "forward_kinematics"):
        print("\nforward_kinematics signature:")
        try:
            print(" ", inspect.signature(robot.forward_kinematics))
        except Exception as e:
            print("  signature unavailable:", e)

    return {
        "joint_names": joint_names,
        "link_names": link_names,
        "ee_candidates": ee_candidates,
        "missing_mujoco_joints": missing,
        "mujoco_to_pyroki": mujoco_to_pyroki,
    }

def convert_mujoco_q_to_pyroki_order(
    q_mujoco: np.ndarray,
    joint_names_pyroki: list[str],
) -> np.ndarray:
    """
    Convert q from MuJoCo UR5 order to PyRoki robot.joints order.

    MuJoCo order:
      shoulder_pan_joint, shoulder_lift_joint, elbow_joint,
      wrist_1_joint, wrist_2_joint, wrist_3_joint
    """
    q_mujoco = np.asarray(q_mujoco, dtype=np.float64)
    if q_mujoco.shape != (6,):
        raise ValueError(f"q_mujoco must have shape (6,), got {q_mujoco.shape}")

    q_pyroki = np.zeros(len(joint_names_pyroki), dtype=np.float64)

    for mj_i, joint_name in enumerate(MUJOCO_UR5_JOINT_ORDER):
        if joint_name not in joint_names_pyroki:
            raise ValueError(f"Joint {joint_name} not found in PyRoki joints.")
        pk_i = joint_names_pyroki.index(joint_name)
        q_pyroki[pk_i] = q_mujoco[mj_i]

    return q_pyroki


def _summarize_fk_output(obj: Any, indent: str = "  ", max_depth: int = 2) -> None:
    """
    Print a compact summary of FK output without assuming exact PyRoki return type.
    """
    if max_depth < 0:
        print(indent + "<max depth>")
        return

    print(indent + f"type: {type(obj)}")

    if isinstance(obj, dict):
        print(indent + f"dict keys: {list(obj.keys())[:30]}")
        for k, v in list(obj.items())[:5]:
            print(indent + f"key={k!r}:")
            _summarize_fk_output(v, indent + "  ", max_depth - 1)
        return

    if isinstance(obj, (list, tuple)):
        print(indent + f"len: {len(obj)}")
        for i, v in enumerate(obj[:5]):
            print(indent + f"[{i}]:")
            _summarize_fk_output(v, indent + "  ", max_depth - 1)
        return

    shape = getattr(obj, "shape", None)
    dtype = getattr(obj, "dtype", None)
    if shape is not None:
        print(indent + f"shape: {shape}, dtype: {dtype}")
        try:
            arr = np.asarray(obj)
            flat = arr.reshape(-1)
            print(indent + f"sample: {flat[:8]}")
        except Exception:
            pass
        return

    attrs = [x for x in dir(obj) if not x.startswith("_")]
    print(indent + f"attrs sample: {attrs[:30]}")
    print(indent + f"repr: {repr(obj)[:300]}")


def try_pyroki_forward_kinematics(
    robot: Any,
    q_seed_mujoco: np.ndarray | None = None,
) -> None:
    """
    Try robot.forward_kinematics with zero q and optional q_seed.
    PyRoki FK expects actuated joint cfg with shape:
        (num_actuated_joints,) or (batch, num_actuated_joints)
    """
    print_header("Trying PyRoki forward_kinematics(...)")

    if not hasattr(robot, "forward_kinematics"):
        print("[FAIL] robot has no forward_kinematics method.")
        return

    fk = robot.forward_kinematics

    try:
        print("forward_kinematics signature:")
        print(" ", inspect.signature(fk))
    except Exception as e:
        print("signature unavailable:", e)

    joint_names = get_pyroki_joint_names(robot)

    # Prefer PyRoki's explicit num_actuated_joints if available.
    joints_obj = getattr(robot, "joints", None)
    if joints_obj is not None and hasattr(joints_obj, "num_actuated_joints"):
        n = int(getattr(joints_obj, "num_actuated_joints"))
    else:
        n = len(joint_names)

    print(f"\nUsing actuated_count={n}")
    print("Actuated joint names:")
    for i, name in enumerate(joint_names):
        print(f"  [{i}] {name}")

    q_tests: list[tuple[str, np.ndarray]] = [
        ("zero_actuated_order", np.zeros(n, dtype=np.float64)),
    ]

    if q_seed_mujoco is not None:
        try:
            q_seed_pyroki = convert_mujoco_q_to_pyroki_order(
                q_seed_mujoco,
                joint_names_pyroki=joint_names,
            )
            q_tests.append(("q_seed_converted_to_pyroki_actuated_order", q_seed_pyroki))
        except Exception as e:
            print(f"[WARN] could not convert q_seed to PyRoki order: {e}")

    try:
        import jax.numpy as jnp
    except Exception:
        jnp = None

    link_names = get_pyroki_link_names(robot)

    for label, q in q_tests:
        print(f"\n--- FK test: {label} ---")
        print("q:", np.round(q, 5))

        attempts = [
            ("numpy_unbatched", q),
            ("numpy_batched", q[None, :]),
        ]

        if jnp is not None:
            attempts.extend(
                [
                    ("jax_unbatched", jnp.asarray(q)),
                    ("jax_batched", jnp.asarray(q[None, :])),
                ]
            )

        for attempt_name, q_in in attempts:
            print(f"\nAttempt: {attempt_name}, input shape={getattr(q_in, 'shape', None)}")
            try:
                out = fk(q_in)
                print("[OK] FK succeeded.")
                _summarize_fk_output(out, indent="  ", max_depth=3)

                # Try to print tool0 / flange / wrist_3_link pose if available.
                try:
                    out_np = np.asarray(out)
                    if out_np.ndim == 2:
                        # shape: (link_count, 7)
                        fk_arr = out_np
                    elif out_np.ndim == 3:
                        # shape: (batch, link_count, 7)
                        fk_arr = out_np[0]
                    else:
                        fk_arr = None

                    if fk_arr is not None:
                        print("\nSelected FK link poses:")
                        for link_name in ["tool0", "flange", "wrist_3_link"]:
                            if link_name in link_names:
                                idx = link_names.index(link_name)
                                pose7 = fk_arr[idx]
                                print(f"  {link_name:<12s} idx={idx} pose7={np.round(pose7, 5)}")
                except Exception as e:
                    print(f"[WARN] could not print selected FK link poses: {e}")

                break
            except Exception as e:
                print(f"[FAIL] FK failed: {type(e).__name__}: {e}")

def inspect_pyroki_snippets_functions() -> None:
    """
    Print solve/IK/traj-related functions from pyroki_snippets.
    """
    print_header("Inspecting pyroki_snippets solve/IK/trajectory functions")

    try:
        import pyroki_snippets as pks
    except Exception as e:
        print(f"[FAIL] cannot import pyroki_snippets: {e}")
        return

    names = [name for name in dir(pks) if not name.startswith("_")]

    keywords = ["solve", "ik", "traj", "motion", "opt", "coll"]
    likely = [
        name for name in names
        if any(k in name.lower() for k in keywords)
    ]

    print("All public pyroki_snippets names:")
    for name in names:
        print(f"  - {name}")

    print("\nLikely solver-related names:")
    for name in likely:
        obj = getattr(pks, name)
        print(f"\n{name}: {obj}")
        try:
            print("  signature:", inspect.signature(obj))
        except Exception as e:
            print("  signature unavailable:", e)

        doc = getattr(obj, "__doc__", None)
        if doc:
            first_lines = "\n".join(doc.strip().splitlines()[:8])
            print("  doc:")
            print("  " + first_lines.replace("\n", "\n  "))

def inspect_pyroki_package(pyroki_mod) -> None:
    print_header("Inspecting pyroki package")

    if pyroki_mod is None:
        print("[SKIP] pyroki is not importable.")
        return

    print("Top-level names:")
    names = [x for x in dir(pyroki_mod) if not x.startswith("_")]
    for name in names[:120]:
        print(f"  - {name}")

    pkg_path = getattr(pyroki_mod, "__path__", None)
    if pkg_path is None:
        print("pyroki has no __path__; cannot walk submodules.")
        return

    print("\nSubmodules:")
    submodules = []
    for m in pkgutil.walk_packages(pyroki_mod.__path__, prefix="pyroki."):
        submodules.append(m.name)

    for name in submodules[:120]:
        print(f"  - {name}")

    print("\nLikely URDF / robot APIs:")
    keywords = ["robot", "urdf", "kinemat", "ik", "traj"]
    likely = [m for m in submodules if any(k in m.lower() for k in keywords)]
    for name in likely[:80]:
        print(f"  - {name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check PyRoki installation and UR5/UR5e URDF compatibility."
    )

    parser.add_argument(
        "--urdf-path",
        type=str,
        default=None,
        help="Path to UR5/UR5e URDF. If omitted, tries UR5_URDF_PATH and common local paths.",
    )
    parser.add_argument(
        "--robot-description",
        type=str,
        default=None,
        help=(
            "Optional robot_descriptions name, e.g. a package available through "
            "robot_descriptions. Leave empty for local URDF path check."
        ),
    )
    parser.add_argument(
        "--require-urdf",
        action="store_true",
        help="Exit with error if no URDF is found.",
    )
    parser.add_argument(
        "--inspect-package",
        action="store_true",
        help="Print pyroki top-level names and submodules.",
    )
    parser.add_argument(
        "--try-pyroki-load",
        action="store_true",
        help="Try pk.Robot.from_urdf(...) if URDF object can be loaded.",
    )
    parser.add_argument(
        "--pyroki-examples-path",
        type=str,
        default=None,
        help="Path to pyroki/examples so that `import pyroki_snippets` works.",
    )
    parser.add_argument(
        "--deep-inspect-robot",
        action="store_true",
        help="Print PyRoki robot joints, links, joint order mapping, and FK API details.",
    )
    parser.add_argument(
        "--try-fk",
        action="store_true",
        help="Try robot.forward_kinematics with zero q and optional q-seed.",
    )
    parser.add_argument(
        "--q-seed",
        type=float,
        nargs=6,
        default=[3.1416, -1.2, 1.6, -1.9, -1.5708, 0.0],
        help="MuJoCo-order q seed used for FK sanity check.",
    )
    parser.add_argument(
        "--inspect-snippets",
        action="store_true",
        help="Print pyroki_snippets solve/IK/trajectory function signatures.",
    )

    args = parser.parse_args()
    add_pyroki_examples_to_syspath(args.pyroki_examples_path)

    print_header("Python environment")
    print("python:", sys.version.replace("\n", " "))
    print("executable:", sys.executable)
    print("project root:", PROJECT_ROOT)

    print_header("Import checks")
    jax_mod = import_optional("jax")
    import_optional("jaxlib")
    pyroki_mod = import_optional("pyroki")
    import_optional("pyroki_snippets")
    import_optional("viser")
    import_optional("robot_descriptions")
    import_optional("yourdfpy")

    if jax_mod is not None:
        try:
            print("\nJAX devices:")
            for dev in jax_mod.devices():
                print(f"  - {dev}")
        except Exception as e:
            print(f"[WARN] could not query jax devices: {e}")

    if args.inspect_package:
        inspect_pyroki_package(pyroki_mod)

    print_header("URDF discovery")
    urdf_path = find_existing_urdf(args.urdf_path)

    if urdf_path is None:
        print("[WARN] No local URDF found.")
        print("\nCommon ways to provide one:")
        print("  1) pass --urdf-path /path/to/ur5e.urdf")
        print("  2) set export UR5_URDF_PATH=/path/to/ur5e.urdf")
        print("  3) install/download a Universal Robots URDF package")
        if args.require_urdf:
            raise FileNotFoundError("No URDF found and --require-urdf was set.")
    else:
        print(f"[OK] URDF path: {urdf_path}")

        try:
            info = parse_urdf_basic(urdf_path)
            print_urdf_summary(info)
        except Exception as e:
            print(f"[FAIL] could not parse URDF XML: {type(e).__name__}: {e}")

    urdf_obj = None

    if args.robot_description:
        urdf_obj = try_robot_descriptions_load(args.robot_description)

    if urdf_obj is None and urdf_path is not None:
        urdf_obj = try_load_urdf_with_yourdfpy(urdf_path)

    robot = None
    robot_inspection = None

    if args.try_pyroki_load:
        robot = try_pyroki_robot_from_urdf(pyroki_mod, urdf_obj)

    if args.inspect_snippets:
        inspect_pyroki_snippets_functions()

    if robot is not None and args.deep_inspect_robot:
        robot_inspection = inspect_pyroki_robot(robot)

    if robot is not None and args.try_fk:
        try_pyroki_forward_kinematics(
            robot,
            q_seed_mujoco=np.asarray(args.q_seed, dtype=np.float64),
        )

    print_header("Next step guidance")
    print("If all imports work and URDF joint order looks correct, next file can be:")
    print("  semantic_safety/ur5_experiment/pyroki_solver.py")
    print("  scripts/ur5_solve_pyroki.py")
    print("\nFor MuJoCo replay compatibility, make sure PyRoki q order matches:")
    print("  " + ", ".join(MUJOCO_UR5_JOINT_ORDER))


if __name__ == "__main__":
    main()