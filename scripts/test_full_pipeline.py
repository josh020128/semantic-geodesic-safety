import argparse
import json

import cv2
import re
import mujoco
import numpy as np
from pathlib import Path
from scipy import ndimage

from semantic_safety.perception_2d3d.mujoco_camera import MujocoCamera
from semantic_safety.perception_2d3d.lang_sam_wrapper import SemanticPerception
from semantic_safety.perception_2d3d.transform import WorldTransform
from semantic_safety.semantic_router.router import SemanticRouter
from semantic_safety.metric_propagation.fmm_distance import WorkspaceGrid
from semantic_safety.risk_field.superposition import (
    shielding_ratio,
    compute_logsumexp_superposition,
    compute_hybrid_superposition,
    compute_sum_superposition,
)
from semantic_safety.risk_field.templates import build_risk_field_from_params
from semantic_safety.phase0_dataset.prompts import SYSTEM_INSTRUCTION

PRIOR_JSON_DEFAULT = "/home/gl34/research/semantic-geodesic-safety/data/semantic_risk_demo_claude.json"

# ----------------------------------------------------------------------
# Optional Claude hook
# ----------------------------------------------------------------------

try:
    from semantic_safety.semantic_router.claude_callbacks import claude_batch_callback
except Exception:
    claude_batch_callback = None

# ----------------------------------------------------------------------
# JSON Object Loader Helpers
# ----------------------------------------------------------------------
def load_scene_candidates_from_prior(json_path: str, manipulated_obj: str) -> list[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    manipulated_norm = manipulated_obj.strip().lower()
    labels = []

    for row in rows:
        if not isinstance(row, dict):
            continue
        manip = str(row.get("manipulated", "")).strip().lower()
        scene = str(row.get("scene", "")).strip()
        if manip == manipulated_norm and scene:
            labels.append(scene)

    # preserve order + dedupe
    seen = set()
    out = []
    for x in labels:
        key = x.lower()
        if key not in seen:
            seen.add(key)
            out.append(x)
    return out

def build_candidate_labels(
    manipulated_obj: str,
    target_label: str,
    user_candidate_labels: list[str] | None,
    prior_json_path: str,
) -> tuple[list[str], list[str]]:
    prior_scenes = load_scene_candidates_from_prior(prior_json_path, manipulated_obj)

    pass1 = []
    if target_label.lower() != "auto":
        pass1.append(target_label)
    pass1.extend(prior_scenes)

    # 중요: user가 직접 준 labels도 pass1에 포함
    if user_candidate_labels:
        pass1.extend(user_candidate_labels)

    seen = set()
    pass1 = [x for x in pass1 if not (x.lower() in seen or seen.add(x.lower()))]

    pass2 = list(pass1)
    return pass1, pass2

def get_object_max_risk_score(risk_params: dict) -> float:
    weights = risk_params.get("weights", {})
    if not weights:
        return 1.0
    return float(np.clip(max(float(v) for v in weights.values()), 0.0, 1.0))


def normalize_weights_for_shape(weights: dict) -> dict:
    if not weights:
        return weights
    m = max(float(v) for v in weights.values())
    if m <= 1e-8:
        return weights
    return {k: float(v) / m for k, v in weights.items()}

# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------

def get_mujoco_table_top_z(model, data, geom_name="table") -> float:
    """Return top surface z of a MuJoCo box geom."""
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if geom_id < 0:
        raise ValueError(f"Could not find geom '{geom_name}' in MuJoCo model.")

    geom_center = data.geom_xpos[geom_id]
    geom_half_size = model.geom_size[geom_id]
    table_top_z = float(geom_center[2] + geom_half_size[2])
    return table_top_z

def get_optional_mujoco_table_top_z(model, data, geom_name="table") -> float | None:
    """
    Return top surface z of a MuJoCo box geom if it exists.
    If the geom is missing or is not a box, return None instead of crashing.
    """
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if geom_id < 0:
        print(f"Warning: table geom '{geom_name}' not found. table_top_z will be disabled.")
        return None

    if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_BOX:
        print(f"Warning: geom '{geom_name}' exists but is not a box. table_top_z will be disabled.")
        return None

    geom_center = data.geom_xpos[geom_id]
    geom_half_size = model.geom_size[geom_id]
    return float(geom_center[2] + geom_half_size[2])


def try_geom_box_to_voxel_mask(
    model,
    data,
    grid: WorkspaceGrid,
    geom_name: str,
    pad: float = 0.0,
) -> tuple[np.ndarray | None, str | None]:
    """
    Safe version of GT blocker voxelization.

    Returns:
      (mask, None) if success
      (None, reason) if skipped
    """
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if geom_id < 0:
        return None, "missing_geom"

    if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_BOX:
        return None, "not_box_geom"

    center = data.geom_xpos[geom_id].copy()
    rot = data.geom_xmat[geom_id].reshape(3, 3).copy()
    half = model.geom_size[geom_id].copy() + pad

    pts = np.stack(
        [
            grid.X - center[0],
            grid.Y - center[1],
            grid.Z - center[2],
        ],
        axis=-1,
    )

    local = pts @ rot

    mask = (
        (np.abs(local[..., 0]) <= half[0]) &
        (np.abs(local[..., 1]) <= half[1]) &
        (np.abs(local[..., 2]) <= half[2])
    )
    return mask, None


def collect_debug_gt_blocker_masks(
    model,
    data,
    grid: WorkspaceGrid,
    geom_names: list[str],
    pad: float = 0.0,
) -> tuple[list[np.ndarray], list[dict]]:
    """
    Optional debug/oracle hook.
    Missing geoms are skipped rather than raising exceptions.
    """
    masks: list[np.ndarray] = []
    info: list[dict] = []

    if model is None or data is None:
        print("Skipping GT blocker injection: available only in MuJoCo mode.")
        return masks, info

    if not geom_names:
        print("GT blocker mode enabled, but no geom names were provided.")
        return masks, info

    print("\n--- DEBUG ORACLE BLOCKER INJECTION ---")
    for geom_name in geom_names:
        mask, reason = try_geom_box_to_voxel_mask(
            model=model,
            data=data,
            grid=grid,
            geom_name=geom_name,
            pad=pad,
        )

        if mask is None:
            print(f"Skipping GT blocker '{geom_name}': {reason}")
            continue

        voxels = int(mask.sum())
        if voxels == 0:
            print(f"Skipping GT blocker '{geom_name}': empty voxelization")
            continue

        print(f"Injected GT blocker '{geom_name}': voxels={voxels}")
        masks.append(mask)
        info.append(
            {
                "geom_name": geom_name,
                "voxels": voxels,
                "pad": float(pad),
            }
        )

    return masks, info


def build_global_workspace_bounds(
    world_pts: np.ndarray,
    grid_margin_xy: float = 0.30,
    grid_margin_z_up: float = 0.55,
    z_margin_below: float = 0.08,
    max_xy_span: float = 1.2,
    max_z_span: float = 0.9,
):
    """
    Build one global workspace that covers all detected objects,
    with safety clamps against corrupted world points.
    """
    mn = world_pts.min(axis=0)
    mx = world_pts.max(axis=0)

    center = 0.5 * (mn + mx)
    span = mx - mn

    span[0] = min(span[0], max_xy_span)
    span[1] = min(span[1], max_xy_span)
    span[2] = min(span[2], max_z_span)

    xmin = float(center[0] - span[0] / 2.0 - grid_margin_xy)
    xmax = float(center[0] + span[0] / 2.0 + grid_margin_xy)
    ymin = float(center[1] - span[1] / 2.0 - grid_margin_xy)
    ymax = float(center[1] + span[1] / 2.0 + grid_margin_xy)
    zmin = float(max(0.0, center[2] - span[2] / 2.0 - z_margin_below))
    zmax = float(center[2] + span[2] / 2.0 + grid_margin_z_up)

    return (xmin, xmax, ymin, ymax, zmin, zmax)

def build_blocker_solid_mask(
    occupancy_surface_mask: np.ndarray,
    occupancy_bbox_3d: tuple[float, float, float, float, float, float],
    grid,
    pad_xy_voxels: int = 2,
) -> np.ndarray:
    """
    Build a conservative blocker mask from the observed surface footprint.
    No label hard-coding.

    - use observed x-y footprint from occupancy_surface_mask
    - close tiny holes in 2D
    - dilate only in x-y
    - extrude through the object's z-range
    """
    if not np.any(occupancy_surface_mask):
        return np.zeros_like(occupancy_surface_mask, dtype=bool)

    # 2D footprint from observed surface
    footprint_xy = np.any(occupancy_surface_mask, axis=2)

    structure2d = ndimage.generate_binary_structure(2, 1)
    footprint_xy = ndimage.binary_closing(footprint_xy, structure=structure2d, iterations=1)

    if pad_xy_voxels > 0:
        footprint_xy = ndimage.binary_dilation(
            footprint_xy,
            structure=structure2d,
            iterations=pad_xy_voxels,
        )

    xmin, xmax, ymin, ymax, zmin, zmax = occupancy_bbox_3d

    blocker_solid_mask = (
        footprint_xy[:, :, None]
        & (grid.Z >= zmin)
        & (grid.Z <= zmax)
    )

    return blocker_solid_mask

def segmentation_mask_to_world_points(
    mask_2d: np.ndarray,
    depth_metric: np.ndarray,
    intrinsics: dict,
    world_engine: WorldTransform,
    label: str | None = None,
    table_top_z: float | None = None,
    depth_band_m: float = 0.12,
    table_z_band_m: float = 0.03,
    use_table_top_filter: bool = False,
) -> np.ndarray:
    """
    Convert a 2D segmentation mask into 3D world points using depth,
    with robust depth filtering to reject background leakage.

    Strategy:
      1) valid depth only
      2) keep points near mask median depth
      3) optional table-top plane filter for special debugging only
    """
    ys, xs = np.where(mask_2d > 0)
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    h, w = depth_metric.shape
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = w / 2.0, h / 2.0

    z_all = depth_metric[ys, xs].astype(np.float64)
    valid = z_all > 0.0
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float64)

    xs = xs[valid].astype(np.float64)
    ys = ys[valid].astype(np.float64)
    z_all = z_all[valid]

    z_med = float(np.median(z_all))
    keep = np.abs(z_all - z_med) <= depth_band_m

    if not np.any(keep):
        return np.zeros((0, 3), dtype=np.float64)

    xs = xs[keep]
    ys = ys[keep]
    z = z_all[keep]

    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    cam_pts = np.stack([x, y, z], axis=1)

    pts_world = []
    label_norm = (label or "").strip().lower()

    for cam_pt in cam_pts:
        world_pt = world_engine.to_world(cam_pt)

        if use_table_top_filter and label_norm == "table" and table_top_z is not None:
            if abs(float(world_pt[2]) - float(table_top_z)) > table_z_band_m:
                continue

        pts_world.append(world_pt)

    if not pts_world:
        return np.zeros((0, 3), dtype=np.float64)

    return np.asarray(pts_world, dtype=np.float64)


def world_points_to_voxel_mask(
    grid: WorkspaceGrid,
    world_points: np.ndarray,
    dilation_iters: int = 1,
) -> np.ndarray:
    """
    Voxelize 3D world points into a boolean occupancy mask on the WorkspaceGrid.
    """
    mask = np.zeros(grid.shape, dtype=bool)
    if world_points is None or len(world_points) == 0:
        return mask

    xmin, xmax, ymin, ymax, zmin, zmax = grid.bounds
    res = grid.res

    ix = np.floor((world_points[:, 0] - xmin) / res).astype(int)
    iy = np.floor((world_points[:, 1] - ymin) / res).astype(int)
    iz = np.floor((world_points[:, 2] - zmin) / res).astype(int)

    valid = (
        (ix >= 0) & (ix < grid.shape[0]) &
        (iy >= 0) & (iy < grid.shape[1]) &
        (iz >= 0) & (iz < grid.shape[2])
    )

    ix, iy, iz = ix[valid], iy[valid], iz[valid]
    if len(ix) == 0:
        return mask

    mask[ix, iy, iz] = True

    if dilation_iters > 0:
        struct = ndimage.generate_binary_structure(3, 1)
        mask = ndimage.binary_dilation(mask, structure=struct, iterations=dilation_iters)

    return mask

def keep_largest_connected_component(mask: np.ndarray, connectivity: int = 1) -> np.ndarray:
    """
    Keep only the largest connected component of a boolean mask.
    Works for both 2D and 3D masks.
    """
    if not np.any(mask):
        return mask

    structure = ndimage.generate_binary_structure(mask.ndim, connectivity)
    labels, num = ndimage.label(mask, structure=structure)
    if num <= 1:
        return mask

    counts = np.bincount(labels.ravel())
    counts[0] = 0
    largest = int(np.argmax(counts))
    return labels == largest


def build_precise_source_mask(
    grid: WorkspaceGrid,
    world_points: np.ndarray,
    min_component_voxels: int = 30,
) -> np.ndarray:
    """
    Build a tighter source mask for risk emission.
    Compared with occupancy masks, this avoids overly large hazard footprints.
    """
    raw = world_points_to_voxel_mask(
        grid=grid,
        world_points=world_points,
        dilation_iters=0,
    )

    raw = keep_largest_connected_component(raw, connectivity=1)

    # Fallback if the mask becomes too tiny
    if int(raw.sum()) < min_component_voxels:
        raw = world_points_to_voxel_mask(
            grid=grid,
            world_points=world_points,
            dilation_iters=1,
        )
        raw = keep_largest_connected_component(raw, connectivity=1)

    return raw


def clean_source_surface_mask(surface_mask: np.ndarray) -> np.ndarray:
    """
    Clean a top-surface mask by denoising its footprint in 2D,
    then restoring only the top voxel band from the original surface.
    """
    if not np.any(surface_mask):
        return surface_mask

    footprint = np.any(surface_mask, axis=2)
    footprint = keep_largest_connected_component(footprint, connectivity=1)

    structure2d = ndimage.generate_binary_structure(2, 1)
    cleaned_fp = ndimage.binary_opening(footprint, structure=structure2d, iterations=1)
    cleaned_fp = ndimage.binary_closing(cleaned_fp, structure=structure2d, iterations=1)

    if not np.any(cleaned_fp):
        cleaned_fp = footprint

    cleaned = np.zeros_like(surface_mask, dtype=bool)
    nx, ny, _ = surface_mask.shape

    for i in range(nx):
        for j in range(ny):
            if not cleaned_fp[i, j]:
                continue
            z_idx = np.where(surface_mask[i, j, :])[0]
            if len(z_idx) == 0:
                continue
            z_top = int(z_idx.max())
            cleaned[i, j, z_top] = True

    return cleaned

def extract_top_surface_mask(
    object_mask: np.ndarray,
    band_voxels: int = 1,
) -> np.ndarray:
    """
    Keep only the topmost occupied voxel band at each (x, y).
    Useful for planar/support objects like tables.
    """
    top_mask = np.zeros_like(object_mask, dtype=bool)

    nx, ny, nz = object_mask.shape
    for i in range(nx):
        for j in range(ny):
            z_idx = np.where(object_mask[i, j, :])[0]
            if len(z_idx) == 0:
                continue
            z_top = int(z_idx.max())
            z_start = max(0, z_top - band_voxels + 1)
            top_mask[i, j, z_start:z_top + 1] = True

    return top_mask


def voxel_mask_to_bbox(
    grid: WorkspaceGrid,
    object_mask: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    """
    Compute an axis-aligned bbox from a voxel mask.
    """
    idx = np.argwhere(object_mask)
    if len(idx) == 0:
        raise RuntimeError("voxel_mask_to_bbox got an empty object mask.")

    xs = grid.x[idx[:, 0]]
    ys = grid.y[idx[:, 1]]
    zs = grid.z[idx[:, 2]]

    pad = grid.res
    return (
        float(xs.min() - pad),
        float(xs.max() + pad),
        float(ys.min() - pad),
        float(ys.max() + pad),
        float(zs.min() - pad),
        float(zs.max() + pad),
    )

def bbox_to_voxel_mask(
    grid: WorkspaceGrid,
    bbox: tuple[float, float, float, float, float, float],
    pad_xy_voxels: int = 1,
    pad_z_voxels: int = 0,
) -> np.ndarray:
    """
    Fill an axis-aligned bbox as a solid voxel mask.

    pad_xy_voxels:
        expand conservatively in x/y so perception shrink does not leave holes.
    pad_z_voxels:
        usually keep 0 for shelves/tables so we do not thicken vertically too much.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = bbox
    pad_xy = pad_xy_voxels * grid.res
    pad_z = pad_z_voxels * grid.res

    return (
        (grid.X >= (xmin - pad_xy)) & (grid.X <= (xmax + pad_xy)) &
        (grid.Y >= (ymin - pad_xy)) & (grid.Y <= (ymax + pad_xy)) &
        (grid.Z >= (zmin - pad_z))  & (grid.Z <= (zmax + pad_z))
    )

def build_multi_object_occupancy(
    grid: WorkspaceGrid,
    object_masks: list[np.ndarray],
    table_top_z: float | None,
):
    """
    Build one global occupancy grid:
      True  = free
      False = obstacle
    Obstacles:
      1) anything below the table top plane
      2) every detected object mask
    """
    free = np.ones(grid.shape, dtype=bool)

    if table_top_z is not None:
        free[grid.Z < table_top_z] = False

    for mask in object_masks:
        free[mask] = False

    return free

def build_conservative_blocker_mask(
    grid: WorkspaceGrid,
    occupancy_mask: np.ndarray,
    occupancy_surface_mask: np.ndarray,
    occupancy_bbox_3d: tuple[float, float, float, float, float, float],
    pad_xy_voxels: int = 6,
) -> np.ndarray:
    """
    Build a generic conservative blocker mask:
    - keep the raw occupied voxels
    - take the observed x-y footprint from the object's top surface
    - close small holes in 2D
    - dilate in x-y only
    - extrude through the object's z-range
    """
    if not np.any(occupancy_mask):
        return occupancy_mask

    footprint = np.any(occupancy_surface_mask, axis=2)

    structure2d = ndimage.generate_binary_structure(2, 1)
    footprint = ndimage.binary_closing(footprint, structure=structure2d, iterations=1)
    footprint = ndimage.binary_dilation(footprint, structure=structure2d, iterations=pad_xy_voxels)

    zmin, zmax = occupancy_bbox_3d[4], occupancy_bbox_3d[5]
    extruded = footprint[:, :, None] & (grid.Z >= zmin) & (grid.Z <= zmax)

    blocker_mask = np.logical_or(extruded, occupancy_mask)
    return blocker_mask

def dilate_mask_xy(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    """
    Dilate only in the x-y plane, not along z.
    Useful for conservative blocker occupancy without making shelves thicker vertically.
    """
    if not np.any(mask):
        return mask

    struct_xy = np.zeros((3, 3, 3), dtype=bool)
    struct_xy[:, :, 1] = ndimage.generate_binary_structure(2, 1)

    out = ndimage.binary_closing(mask, structure=struct_xy, iterations=1)
    out = ndimage.binary_dilation(out, structure=struct_xy, iterations=iterations)
    return out

def gravity_column_from_surface_mask(
    grid: WorkspaceGrid,
    top_surface_mask: np.ndarray,
    A_field: np.ndarray,
    base_risk: float,
    w_plus_z: float,
    lateral_decay: str = "moderate",
):
    """
    Gravity-aware upward field from the full top surface footprint.
    This avoids center-only +z risk for wide planar/support objects.
    """
    footprint = np.any(top_surface_mask, axis=2)
    if not np.any(footprint):
        return np.zeros(grid.shape, dtype=np.float64)

    d_xy = ndimage.distance_transform_edt(~footprint) * grid.res

    z_top_field = np.full(grid.shape[:2], -np.inf, dtype=np.float64)
    nx, ny, _ = top_surface_mask.shape
    for i in range(nx):
        for j in range(ny):
            z_idx = np.where(top_surface_mask[i, j, :])[0]
            if len(z_idx) > 0:
                z_top_field[i, j] = grid.z[int(z_idx.max())]

    valid_cols = footprint
    z_top_3d = z_top_field[:, :, None]
    above_mask = valid_cols[:, :, None] & (grid.Z >= z_top_3d)

    lateral_alpha_map = {
        "narrow": 18.0,
        "moderate": 10.0,
        "wide": 5.0,
    }
    alpha_xy = lateral_alpha_map.get(lateral_decay, 10.0)

    d_xy_3d = d_xy[:, :, None]
    V = base_risk * float(w_plus_z) * A_field * np.exp(-alpha_xy * d_xy_3d)
    V *= above_mask.astype(np.float64)
    V[np.isnan(V)] = 0.0
    return V


def canonicalize_label_for_filter(router: SemanticRouter, label: str) -> str:
    return router._canonicalize_label(label)

def normalize_text_label(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def token_jaccard(a: str, b: str) -> float:
    sa = set(normalize_text_label(a).split())
    sb = set(normalize_text_label(b).split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def representative_priority(
    obj: dict,
    router: SemanticRouter,
    target_label: str | None = None,
) -> float:
    """
    Higher is better.

    Generic rule:
    - start with detector confidence
    - if a target label is specified, prefer candidates whose raw label
      matches that target more closely
    - among same-target duplicates, mildly prefer tighter boxes
    """
    score = 1000.0 * float(obj.get("score", 0.0))

    if target_label is None or target_label.lower() == "auto":
        return score

    obj_raw = normalize_text_label(obj["label"])
    tgt_raw = normalize_text_label(target_label)

    obj_canon = router._canonicalize_label(obj["label"])
    tgt_canon = router._canonicalize_label(target_label)

    if obj_canon == tgt_canon:
        # Strong bonus for exact raw label match to the requested target.
        if obj_raw == tgt_raw:
            score += 10000.0

        # Bonus for lexical overlap with requested target.
        score += 1000.0 * token_jaccard(obj_raw, tgt_raw)

        # Mild bonus for more specific labels (more words).
        score += 100.0 * min(len(obj_raw.split()), 4)

        # Mild penalty for overly large boxes among same canonical target.
        box = obj["box_2d"]
        area = max(1, (box["xmax"] - box["xmin"]) * (box["ymax"] - box["ymin"]))
        score -= 0.001 * float(area)

    return score

def dedupe_object_infos_by_canonical_label_and_geometry(
    object_infos: list[dict],
    router: SemanticRouter,
    point_dist_thresh: float = 0.03,
    bbox_iou_thresh: float = 0.25,
    target_label: str | None = None,
) -> list[dict]:
    """
    Dedupe objects after 3D localization.

    Keep one representative among entries that:
    - share the same canonical label
    - and are geometrically almost the same object

    Representative choice is generic:
    - detector score
    - optional target-label match quality
    - mild preference for tighter boxes among same target duplicates
    """
    if not object_infos:
        return []

    def bbox_iou_2d(a, b):
        ax1, ay1, ax2, ay2 = a["xmin"], a["ymin"], a["xmax"], a["ymax"]
        bx1, by1, bx2, by2 = b["xmin"], b["ymin"], b["xmax"], b["ymax"]

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih

        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        union = area_a + area_b - inter
        return inter / union

    def containment_ratio(inner, outer):
        ix1 = max(inner["xmin"], outer["xmin"])
        iy1 = max(inner["ymin"], outer["ymin"])
        ix2 = min(inner["xmax"], outer["xmax"])
        iy2 = min(inner["ymax"], outer["ymax"])

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih

        inner_area = max(1, (inner["xmax"] - inner["xmin"]) * (inner["ymax"] - inner["ymin"]))
        return inter / inner_area

    # IMPORTANT: sort by representative priority, not raw score only
    object_infos = sorted(
        object_infos,
        key=lambda o: representative_priority(o, router, target_label=target_label),
        reverse=True,
    )

    kept = []

    for obj in object_infos:
        obj_canon = router._canonicalize_label(obj["label"])
        obj_pt = np.asarray(obj["world_pt"], dtype=np.float64)
        obj_box = obj["box_2d"]

        duplicate = False
        for prev in kept:
            prev_canon = router._canonicalize_label(prev["label"])
            if obj_canon != prev_canon:
                continue

            prev_pt = np.asarray(prev["world_pt"], dtype=np.float64)
            prev_box = prev["box_2d"]

            dist = float(np.linalg.norm(obj_pt - prev_pt))
            iou = bbox_iou_2d(obj_box, prev_box)
            contain_a = containment_ratio(obj_box, prev_box)
            contain_b = containment_ratio(prev_box, obj_box)

            same_geom = (
                dist <= point_dist_thresh
                or iou >= bbox_iou_thresh
                or contain_a >= 0.85
                or contain_b >= 0.85
            )

            if same_geom:
                duplicate = True
                break

        if not duplicate:
            kept.append(obj)

    return kept

def suppress_contained_duplicate_objects(
    object_infos: list[dict],
    router: SemanticRouter,
    containment_threshold: float = 0.85,
    area_ratio_threshold: float = 0.45,
    center_dist_threshold: float = 0.08,
    depth_diff_threshold: float = 0.08,
) -> list[dict]:
    if not object_infos:
        return object_infos

    def box_area(box):
        return max(1, (box["xmax"] - box["xmin"]) * (box["ymax"] - box["ymin"]))

    def containment_ratio(inner, outer):
        ix1 = max(inner["xmin"], outer["xmin"])
        iy1 = max(inner["ymin"], outer["ymin"])
        ix2 = min(inner["xmax"], outer["xmax"])
        iy2 = min(inner["ymax"], outer["ymax"])
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih
        return inter / max(1, box_area(inner))

    ordered = sorted(
        object_infos,
        key=lambda o: representative_priority(o, router, target_label=None),
        reverse=True,
    )

    kept = []
    for obj in ordered:
        obj_box = obj["box_2d"]
        obj_area = box_area(obj_box)
        obj_center = np.asarray(obj["world_pt"], dtype=np.float64)

        obj_depth = float(obj_center[2])

        drop = False
        for parent in kept:
            parent_box = parent["box_2d"]
            parent_area = box_area(parent_box)
            parent_center = np.asarray(parent["world_pt"], dtype=np.float64)
            parent_depth = float(parent_center[2])

            contain = containment_ratio(obj_box, parent_box)
            area_ratio = obj_area / max(1.0, parent_area)
            center_dist = float(np.linalg.norm(obj_center - parent_center))
            depth_diff = abs(obj_depth - parent_depth)

            if (
                contain >= containment_threshold
                and area_ratio <= area_ratio_threshold
                and center_dist <= center_dist_threshold
                and depth_diff <= depth_diff_threshold
            ):
                drop = True
                break

        if not drop:
            kept.append(obj)

    return kept

def compute_axis_occlusion_caps(
    grid: WorkspaceGrid,
    source_mask: np.ndarray,
    occupancy_free: np.ndarray,
    source_surface_mask: np.ndarray | None = None,
    gap_fill_voxels: int = 2,
) -> tuple[dict[str, np.ndarray], dict]:
    """
    Compute hard axis-aligned occlusion caps for all 6 directions:
      w_+x, w_-x, w_+y, w_-y, w_+z, w_-z

    For each axis ray starting from the source extent, find the first obstacle.
    Everything beyond that first obstacle along that axis is masked out.

    Returns:
      dict[str, bool array] with same shape as grid.shape
    """
    nx, ny, nz = source_mask.shape
    caps = {}

    # Source footprint for debugging: this is the actual emission footprint we care about.
    source_xy = None
    if source_surface_mask is not None and np.any(source_surface_mask):
        source_xy = np.any(source_surface_mask, axis=2)

    occupancy_xy = np.any(source_mask, axis=2)

    # ------------------------------------------------------------
    # +X / -X  (support plane: y,z)
    # ------------------------------------------------------------
    support_yz = np.any(source_mask, axis=0)  # (ny, nz)
    max_x_idx = np.full((ny, nz), -1, dtype=int)
    min_x_idx = np.full((ny, nz), nx, dtype=int)

    for j in range(ny):
        for k in range(nz):
            xs = np.where(source_mask[:, j, k])[0]
            if len(xs) > 0:
                min_x_idx[j, k] = int(xs.min())
                max_x_idx[j, k] = int(xs.max())

    plus_x_block = np.full((ny, nz), np.inf, dtype=np.float64)
    minus_x_block = np.full((ny, nz), -np.inf, dtype=np.float64)

    for j in range(ny):
        for k in range(nz):
            if not support_yz[j, k]:
                continue

            # +x
            start = max_x_idx[j, k] + 1
            if start < nx:
                blocked = np.where(~occupancy_free[start:, j, k])[0]
                if len(blocked) > 0:
                    first_idx = start + int(blocked[0])
                    plus_x_block[j, k] = float(grid.x[first_idx])

            # -x
            stop = min_x_idx[j, k] - 1
            if stop >= 0:
                blocked = np.where(~occupancy_free[:min_x_idx[j, k], j, k])[0]
                if len(blocked) > 0:
                    first_idx = int(blocked[-1])  # closest blocker on negative side
                    minus_x_block[j, k] = float(grid.x[first_idx])

    if np.any(support_yz):
        _, nearest_idx = ndimage.distance_transform_edt(~support_yz, return_indices=True)
        nearest_j = nearest_idx[0]
        nearest_k = nearest_idx[1]

        plus_x_inherited = plus_x_block[nearest_j, nearest_k]       # (ny, nz)
        minus_x_inherited = minus_x_block[nearest_j, nearest_k]    # (ny, nz)

        caps["w_+x"] = grid.X < plus_x_inherited[None, :, :]
        caps["w_-x"] = grid.X > minus_x_inherited[None, :, :]
    else:
        caps["w_+x"] = np.ones(grid.shape, dtype=bool)
        caps["w_-x"] = np.ones(grid.shape, dtype=bool)

    # ------------------------------------------------------------
    # +Y / -Y  (support plane: x,z)
    # ------------------------------------------------------------
    support_xz = np.any(source_mask, axis=1)  # (nx, nz)
    max_y_idx = np.full((nx, nz), -1, dtype=int)
    min_y_idx = np.full((nx, nz), ny, dtype=int)

    for i in range(nx):
        for k in range(nz):
            ys = np.where(source_mask[i, :, k])[0]
            if len(ys) > 0:
                min_y_idx[i, k] = int(ys.min())
                max_y_idx[i, k] = int(ys.max())

    plus_y_block = np.full((nx, nz), np.inf, dtype=np.float64)
    minus_y_block = np.full((nx, nz), -np.inf, dtype=np.float64)

    for i in range(nx):
        for k in range(nz):
            if not support_xz[i, k]:
                continue

            # +y
            start = max_y_idx[i, k] + 1
            if start < ny:
                blocked = np.where(~occupancy_free[i, start:, k])[0]
                if len(blocked) > 0:
                    first_idx = start + int(blocked[0])
                    plus_y_block[i, k] = float(grid.y[first_idx])

            # -y
            if min_y_idx[i, k] > 0:
                blocked = np.where(~occupancy_free[i, :min_y_idx[i, k], k])[0]
                if len(blocked) > 0:
                    first_idx = int(blocked[-1])
                    minus_y_block[i, k] = float(grid.y[first_idx])

    if np.any(support_xz):
        _, nearest_idx = ndimage.distance_transform_edt(~support_xz, return_indices=True)
        nearest_i = nearest_idx[0]
        nearest_k = nearest_idx[1]

        plus_y_inherited = plus_y_block[nearest_i, nearest_k]       # (nx, nz)
        minus_y_inherited = minus_y_block[nearest_i, nearest_k]     # (nx, nz)

        caps["w_+y"] = grid.Y < plus_y_inherited[:, None, :]
        caps["w_-y"] = grid.Y > minus_y_inherited[:, None, :]
    else:
        caps["w_+y"] = np.ones(grid.shape, dtype=bool)
        caps["w_-y"] = np.ones(grid.shape, dtype=bool)

    # ------------------------------------------------------------
    # +Z / -Z  (support plane: x,y)
    # IMPORTANT:
    # - support_xy should follow the true emission footprint
    # - but ray start/end indices must come from the FULL source object,
    #   otherwise the object can block itself.
    # ------------------------------------------------------------
    if source_surface_mask is not None and np.any(source_surface_mask):
        support_xy = np.any(source_surface_mask, axis=2)  # true emission footprint
    else:
        support_xy = np.any(source_mask, axis=2)

    max_z_idx = np.full((nx, ny), -1, dtype=int)
    min_z_idx = np.full((nx, ny), nz, dtype=int)

    for i in range(nx):
        for j in range(ny):
            if not support_xy[i, j]:
                continue

            # IMPORTANT: use FULL source geometry here, not source_surface_mask
            zs_full = np.where(source_mask[i, j, :])[0]
            if len(zs_full) > 0:
                min_z_idx[i, j] = int(zs_full.min())
                max_z_idx[i, j] = int(zs_full.max())

    plus_z_block = np.full((nx, ny), np.inf, dtype=np.float64)
    minus_z_block = np.full((nx, ny), -np.inf, dtype=np.float64)

    for i in range(nx):
        for j in range(ny):
            if not support_xy[i, j]:
                continue

            # +z
            start = max_z_idx[i, j] + 1
            if start < nz:
                blocked = np.where(~occupancy_free[i, j, start:])[0]
                if len(blocked) > 0:
                    first_idx = start + int(blocked[0])
                    plus_z_block[i, j] = float(grid.z[first_idx])

            # -z
            if min_z_idx[i, j] > 0:
                blocked = np.where(~occupancy_free[i, j, :min_z_idx[i, j]])[0]
                if len(blocked) > 0:
                    first_idx = int(blocked[-1])
                    minus_z_block[i, j] = float(grid.z[first_idx])

    # ------------------------------------------------------------
    # Residual-strip cleanup inside source footprint
    # Fill only small open gaps using nearest valid blocker column.
    # ------------------------------------------------------------
    z_gap_fill_voxels = 2  # start with 2; if still leaking, try 3

    plus_z_block = fill_small_open_strips_2d(
        plus_z_block,
        support_xy,
        max_gap_voxels=z_gap_fill_voxels,
        invalid_fill_value=None,   # valid = finite
    )

    minus_z_block = fill_small_open_strips_2d(
        minus_z_block,
        support_xy,
        max_gap_voxels=z_gap_fill_voxels,
        invalid_fill_value=-np.inf,
    )

    if np.any(support_xy):
        _, nearest_idx = ndimage.distance_transform_edt(~support_xy, return_indices=True)
        nearest_i = nearest_idx[0]
        nearest_j = nearest_idx[1]

        plus_z_inherited = plus_z_block[nearest_i, nearest_j]       # (nx, ny)
        minus_z_inherited = minus_z_block[nearest_i, nearest_j]     # (nx, ny)

        caps["w_+z"] = grid.Z < plus_z_inherited[:, :, None]
        caps["w_-z"] = grid.Z > minus_z_inherited[:, :, None]
    else:
        caps["w_+z"] = np.ones(grid.shape, dtype=bool)
        caps["w_-z"] = np.ones(grid.shape, dtype=bool)

    finite_plus = np.isfinite(plus_z_block)
    finite_minus = np.isfinite(minus_z_block)

    support_cols = int(support_xy.sum())
    plus_blocked_cols = int((finite_plus & support_xy).sum())
    minus_blocked_cols = int((finite_minus & support_xy).sum())

    finite_plus_after = np.isfinite(plus_z_block)
    open_after = int((support_xy & (~finite_plus_after)).sum())
    print(f"open source cols(+z) after fill = {open_after}")
    print(f"z_gap_fill_voxels               = {z_gap_fill_voxels}")

    print("\n--- PLUS/MINUS Z BLOCKER COLUMN DEBUG ---")
    print(f"occupancy_xy columns         = {int(occupancy_xy.sum())}")
    print(f"support_xy columns           = {support_cols}")

    print(f"+z blocked columns(support)  = {plus_blocked_cols}")
    print(f"-z blocked columns(support)  = {minus_blocked_cols}")

    if support_cols > 0:
        print(f"+z blocked fraction(support) = {plus_blocked_cols / support_cols:.4f}")
        print(f"-z blocked fraction(support) = {minus_blocked_cols / support_cols:.4f}")

    if source_xy is not None:
        source_cols = int(source_xy.sum())
        plus_blocked_source = int((finite_plus & source_xy).sum())
        minus_blocked_source = int((finite_minus & source_xy).sum())

        print(f"source_xy columns            = {source_cols}")
        print(f"+z blocked columns(source)   = {plus_blocked_source}")
        print(f"-z blocked columns(source)   = {minus_blocked_source}")

        if source_cols > 0:
            print(f"+z blocked fraction(source)  = {plus_blocked_source / source_cols:.4f}")
            print(f"-z blocked fraction(source)  = {minus_blocked_source / source_cols:.4f}")

        source_not_in_support = int((source_xy & (~support_xy)).sum())
        support_not_in_source = int((support_xy & (~source_xy)).sum())

        print(f"source_not_in_support cols   = {source_not_in_support}")
        print(f"support_not_in_source cols   = {support_not_in_source}")

        if source_not_in_support > 0:
            print("WARNING: some source footprint columns are outside cap support footprint")

        # Extra debug: where are the open source columns?
        open_source_cols = source_xy & (~finite_plus)
        open_count = int(open_source_cols.sum())
        print(f"open source cols(+z)         = {open_count}")
        if open_count > 0:
            ii, jj = np.where(open_source_cols)
            print(
                f"open source x range(+z)      = "
                f"{grid.x[ii].min():.4f} .. {grid.x[ii].max():.4f}"
            )
            print(
                f"open source y range(+z)      = "
                f"{grid.y[jj].min():.4f} .. {grid.y[jj].max():.4f}"
            )
    else:
        print("source_xy columns            = none")

    if plus_blocked_cols > 0:
        vals = plus_z_block[finite_plus & support_xy]
        print(f"+z blocker z range(support)  = {vals.min():.4f} .. {vals.max():.4f}")
    else:
        print("+z blocker z range(support)  = none")

    if source_xy is not None and np.any(finite_plus & source_xy):
        vals_src = plus_z_block[finite_plus & source_xy]
        print(f"+z blocker z range(source)   = {vals_src.min():.4f} .. {vals_src.max():.4f}")
    else:
        print("+z blocker z range(source)   = none")

    debug_info = {
        "support_yz": support_yz,
        "support_xz": support_xz,
        "support_xy": support_xy,
        "plus_x_block": plus_x_block,
        "minus_x_block": minus_x_block,
        "plus_y_block": plus_y_block,
        "minus_y_block": minus_y_block,
        "plus_z_block": plus_z_block,
        "minus_z_block": minus_z_block,
    }

    return caps, debug_info

def select_axis_occlusion_cap(
    axis_caps: dict[str, np.ndarray],
    risk_params: dict,
) -> np.ndarray:
    """
    Select or combine axis-aligned occlusion caps based on topology + weights.

    Strategy:
    - upward_vertical_cone: use +z cap
    - forward_directional_cone: use dominant horizontal axis cap
    - planar_half_space: use dominant axis cap
    - isotropic_sphere: do not hard-cap by default (keep soft geodesic attenuation only)
    """
    topology = risk_params.get("topology_template", "isotropic_sphere")
    weights = risk_params.get("weights", {})

    if topology == "upward_vertical_cone":
        return axis_caps["w_+z"]

    if topology == "forward_directional_cone":
        horiz_keys = ["w_+x", "w_-x", "w_+y", "w_-y"]
        dominant = max(horiz_keys, key=lambda k: float(weights.get(k, 0.0)))
        return axis_caps[dominant]

    if topology == "planar_half_space":
        all_keys = ["w_+x", "w_-x", "w_+y", "w_-y", "w_+z", "w_-z"]
        dominant = max(all_keys, key=lambda k: float(weights.get(k, 0.0)))
        return axis_caps[dominant]

    # isotropic_sphere: keep hard cap disabled by default
    return np.ones_like(next(iter(axis_caps.values())), dtype=bool)

def compose_axis_occlusion_cap(
    axis_caps: dict[str, np.ndarray],
    risk_params: dict,
    weight_eps: float = 1e-8,
) -> tuple[np.ndarray, list[str]]:
    """
    Compose hard occlusion caps over all ACTIVE directions.

    Principle:
    - If a direction has nonzero semantic weight, risk can propagate there.
    - Therefore, if an obstacle exists along that direction, we hard-block it.
    - Topology still shapes the soft field itself, but hard blocking is applied
      to every active axis direction.

    Returns:
        hard_axis_cap: bool array, True = allowed, False = blocked
        active_cap_keys: list of axis keys that participated in the composition
    """
    weights = risk_params.get("weights", {})

    ordered_keys = ["w_+x", "w_-x", "w_+y", "w_-y", "w_+z", "w_-z"]
    active_cap_keys = [
        k for k in ordered_keys
        if float(weights.get(k, 0.0)) > weight_eps and k in axis_caps
    ]

    if not active_cap_keys:
        return np.ones_like(next(iter(axis_caps.values())), dtype=bool), []

    hard_axis_cap = np.ones_like(next(iter(axis_caps.values())), dtype=bool)

    for k in active_cap_keys:
        hard_axis_cap &= axis_caps[k]

    return hard_axis_cap, active_cap_keys

def fill_small_open_strips_2d(
    block_values_2d: np.ndarray,
    support_mask_2d: np.ndarray,
    max_gap_voxels: int = 2,
    invalid_fill_value: float | None = None,
) -> np.ndarray:
    """
    Fill small uncovered gaps inside a 2D support footprint by copying the
    nearest valid blocker value.

    block_values_2d:
        2D array of blocker coordinates (e.g. plus_z_block).
        Valid entries are finite (or custom-valid if invalid_fill_value used).
    support_mask_2d:
        Where source support exists.
    max_gap_voxels:
        Only fill gaps whose nearest valid blocker column is within this radius.
    invalid_fill_value:
        Optional sentinel like -np.inf for minus-direction arrays.
    """
    out = block_values_2d.copy()

    if invalid_fill_value is None:
        valid = np.isfinite(out)
    else:
        valid = out != invalid_fill_value

    valid &= support_mask_2d
    if not np.any(valid):
        return out

    # distance to nearest valid blocker column
    dist, nearest_idx = ndimage.distance_transform_edt(
        ~valid,
        return_indices=True,
    )

    nearest_i = nearest_idx[0]
    nearest_j = nearest_idx[1]

    fillable = support_mask_2d & (~valid) & (dist <= max_gap_voxels)
    out[fillable] = out[nearest_i[fillable], nearest_j[fillable]]
    return out

# ----------------------------------------------------------------------
# Visualization helpers
# ----------------------------------------------------------------------

def draw_segmentation_by_attenuation(
    bgr_image: np.ndarray,
    object_infos: list[dict],
    alpha_min: float = 0.20,
    alpha_max: float = 0.65,
    output_path: str = "segmentation_attenuation_overlay.png",
):
    """
    Color each object's 2D segmentation by receptacle_attenuation.
    """
    canvas = bgr_image.copy()

    for obj in object_infos:
        mask = obj.get("mask_2d", None)
        risk_params = obj.get("risk_params", None)

        if mask is None or risk_params is None:
            continue

        att = float(risk_params.get("receptacle_attenuation", 1.0))
        att = float(np.clip(att, 0.1, 1.0))
        t = (att - 0.1) / 0.9

        color = np.array([
            int(255 * (1.0 - t)),
            int(180 * (1.0 - t)),
            int(255 * t),
        ], dtype=np.uint8)

        alpha = alpha_min + (alpha_max - alpha_min) * t

        overlay = canvas.copy()
        overlay[mask] = color
        canvas = cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0)

        box = obj["box_2d"]
        label = obj["label"]
        cv2.rectangle(canvas, (box["xmin"], box["ymin"]), (box["xmax"], box["ymax"]), (0, 0, 255), 2)
        cv2.putText(
            canvas,
            f"{label} | att={att:.2f}",
            (box["xmin"], max(20, box["ymin"] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
        )

    cv2.imwrite(output_path, canvas)
    print(f"Saved segmentation attenuation overlay to '{output_path}'.")


def project_and_render_overlay(
    bgr_image,
    depth_metric,
    intrinsics,
    world_pos,
    world_mat,
    X,
    Y,
    Z,
    V_final,
    detections,
    res,
    output_path="continuous_risk_overlay.png",
):
    print("\n--- VECTORIZED VOLUMETRIC SPLATTING ---")

    h, w, _ = bgr_image.shape
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = w / 2.0, h / 2.0

    valid_mask = V_final > 2.0
    pts_world = np.vstack((X[valid_mask], Y[valid_mask], Z[valid_mask]))
    risks = V_final[valid_mask]

    if pts_world.shape[1] == 0:
        print("No high-risk voxels to draw.")
        canvas = bgr_image.copy()
        for det in detections:
            box = det["box"]
            label = det["label"]
            cv2.rectangle(canvas, (box["xmin"], box["ymin"]), (box["xmax"], box["ymax"]), (0, 0, 255), 2)
            cv2.putText(
                canvas,
                label.upper(),
                (box["xmin"], max(20, box["ymin"] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
        cv2.imwrite(output_path, canvas)
        return

    world_mat_3x3 = world_mat.reshape(3, 3)
    pts_mj = world_mat_3x3.T @ (pts_world - world_pos.reshape(3, 1))

    cv2mj_rot = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ])
    pts_cam = cv2mj_rot.T @ pts_mj

    front_mask = pts_cam[2, :] > 0.01
    pts_cam = pts_cam[:, front_mask]
    risks = risks[front_mask]

    if pts_cam.shape[1] == 0:
        print("No front-facing high-risk voxels to draw.")
        cv2.imwrite(output_path, bgr_image)
        return

    u = (fx * pts_cam[0, :] / pts_cam[2, :] + cx).astype(int)
    v = (fy * pts_cam[1, :] / pts_cam[2, :] + cy).astype(int)
    z_dist = pts_cam[2, :]

    sort_idx = np.argsort(risks)
    u, v, risks, z_dist = u[sort_idx], v[sort_idx], risks[sort_idx], z_dist[sort_idx]

    risk_canvas = np.zeros((h, w), dtype=np.float32)
    alpha_canvas = np.zeros((h, w), dtype=np.float32)

    drawn_voxels = 0
    for px, py, r, z in zip(u, v, risks, z_dist):
        if 0 <= px < w and 0 <= py < h:
            scene_z = depth_metric[py, px]

            if z < scene_z + 0.02 or scene_z <= 0.01:
                radius = max(2, int(fx * res * 1.0 / z))
                cv2.circle(risk_canvas, (px, py), radius, float(r), -1)

                alpha_val = min(0.7, max(0.0, (r - 5.0) / 85.0))
                cv2.circle(alpha_canvas, (px, py), radius, float(alpha_val), -1)
                drawn_voxels += 1

    print(f"Successfully painted {drawn_voxels} high-risk voxels onto screen.")

    # ------------------------------------------------------------------
    # DEBUG: save raw pre-blur canvases
    # ------------------------------------------------------------------
    raw_positive = risk_canvas[risk_canvas > 0]
    if raw_positive.size > 0:
        raw_vis = risk_canvas.copy()
        raw_vis = raw_vis / max(1e-6, float(raw_vis.max()))
        raw_vis = (255.0 * raw_vis).astype(np.uint8)
        cv2.imwrite("debug_risk_canvas_raw.png", raw_vis)

    raw_alpha_vis = np.clip(alpha_canvas, 0.0, 1.0)
    raw_alpha_vis = (255.0 * raw_alpha_vis).astype(np.uint8)
    cv2.imwrite("debug_alpha_canvas_raw.png", raw_alpha_vis)

    print("Saved pre-blur debug images: debug_risk_canvas_raw.png, debug_alpha_canvas_raw.png")

    risk_canvas = cv2.GaussianBlur(risk_canvas, (21, 21), 0)
    alpha_canvas = cv2.GaussianBlur(alpha_canvas, (21, 21), 0)

    positive = risk_canvas[risk_canvas > 0]
    scene_peak = float(positive.max()) if positive.size > 0 else 0.0

    unit_anchor = 100.0
    vis_max = unit_anchor if scene_peak <= unit_anchor else scene_peak

    norm_abs = np.clip(risk_canvas, 0, vis_max) / vis_max

    # low-risk visibility boost for display only
    display_gamma = 1.0
    norm_disp = np.power(norm_abs, display_gamma)

    # alpha도 조금 올려서 너무 흐리지 않게
    alpha_floor = 0.0
    alpha_canvas = np.where(
        norm_abs > 0,
        alpha_floor + (1.0 - alpha_floor) * norm_disp,
        0.0,
    )

    col_idx = ((1.0 - norm_disp) * 255).astype(np.uint8)
    color_heatmap = cv2.applyColorMap(col_idx, cv2.COLORMAP_RAINBOW)

    alpha_3d = np.clip(alpha_canvas, 0.0, 0.85)[..., np.newaxis]
    blended = (color_heatmap * alpha_3d + bgr_image * (1.0 - alpha_3d)).astype(np.uint8)

    for det in detections:
        box = det["box"]
        label = det["label"]
        cv2.rectangle(blended, (box["xmin"], box["ymin"]), (box["xmax"], box["ymax"]), (0, 0, 255), 2)
        cv2.putText(
            blended,
            label.upper(),
            (box["xmin"], max(20, box["ymin"] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    cv2.imwrite(output_path, blended)
    print(f"Saved overlay to '{output_path}'.")


def geom_box_to_voxel_mask(
    model,
    data,
    grid: WorkspaceGrid,
    geom_name: str,
    pad: float = 0.0,
) -> np.ndarray:
    """
    Convert a MuJoCo box geom into a voxel mask on the current WorkspaceGrid.

    Works for arbitrarily rotated box geoms using geom_xpos / geom_xmat.
    """
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if geom_id < 0:
        raise ValueError(f"Could not find geom '{geom_name}'.")

    geom_type = model.geom_type[geom_id]
    if geom_type != mujoco.mjtGeom.mjGEOM_BOX:
        raise ValueError(f"geom '{geom_name}' is not a box geom.")

    center = data.geom_xpos[geom_id].copy()                  # (3,)
    rot = data.geom_xmat[geom_id].reshape(3, 3).copy()      # local -> world
    half = model.geom_size[geom_id].copy() + pad            # (3,)

    pts = np.stack(
        [
            grid.X - center[0],
            grid.Y - center[1],
            grid.Z - center[2],
        ],
        axis=-1,
    )  # (..., 3)

    # world delta -> local delta
    local = pts @ rot

    mask = (
        (np.abs(local[..., 0]) <= half[0]) &
        (np.abs(local[..., 1]) <= half[1]) &
        (np.abs(local[..., 2]) <= half[2])
    )
    return mask

def print_final_risk_profiles(
    grid: WorkspaceGrid,
    d_euc: np.ndarray,
    d_geo: np.ndarray,
    A_field: np.ndarray,
    hard_axis_cap: np.ndarray,
    V_object: np.ndarray,
    occupancy_free: np.ndarray,
    center_world_pt: np.ndarray,
    offsets=(0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28),
    title: str = "FINAL RISK PROFILE",
):
    """
    Print axis-aligned probe profiles that show:
      - Euclidean distance
      - Geodesic distance
      - soft shielding A_field
      - hard occlusion cap
      - final capped risk V_object
      - whether the voxel itself is free/occupied

    This is the most useful end-to-end debug print for understanding
    why risk survives or disappears behind blockers.
    """
    def nearest_idx(arr, value):
        return int(np.argmin(np.abs(arr - value)))

    cx, cy, cz = center_world_pt
    axes = {
        "+x": np.array([1.0, 0.0, 0.0]),
        "-x": np.array([-1.0, 0.0, 0.0]),
        "+y": np.array([0.0, 1.0, 0.0]),
        "-y": np.array([0.0, -1.0, 0.0]),
        "+z": np.array([0.0, 0.0, 1.0]),
        "-z": np.array([0.0, 0.0, -1.0]),
    }

    print(f"\n--- {title} ---")
    for axis_name, direction in axes.items():
        print(f"[direction {axis_name}]")
        for off in offsets:
            pt = np.array([cx, cy, cz], dtype=np.float64) + off * direction

            ix = nearest_idx(grid.x, pt[0])
            iy = nearest_idx(grid.y, pt[1])
            iz = nearest_idx(grid.z, pt[2])

            de = float(d_euc[ix, iy, iz])
            dg = float(d_geo[ix, iy, iz])
            af = float(A_field[ix, iy, iz])
            cap = int(hard_axis_cap[ix, iy, iz])
            vfinal = float(V_object[ix, iy, iz])
            free = int(occupancy_free[ix, iy, iz])

            print(
                f"  off={off:.3f} "
                f"pt={np.round(pt, 4)} "
                f"free={free} "
                f"d_euc={de:.4f} "
                f"d_geo={dg:.4f} "
                f"A={af:.4f} "
                f"cap={cap} "
                f"V={vfinal:.4f}"
            )

def save_depth_debug_image(
    depth_metric: np.ndarray,
    output_path: str = "test_depth_debug.png",
):
    """
    Save a normalized depth visualization for quick sanity checking.
    """
    valid = np.isfinite(depth_metric) & (depth_metric > 0)
    if not np.any(valid):
        print("Depth debug: no valid depth values.")
        return

    vals = depth_metric[valid]
    d_min = float(vals.min())
    d_max = float(vals.max())
    d_med = float(np.median(vals))
    finite_ratio = float(valid.mean())

    print(
        f"Depth debug: min={d_min:.4f} max={d_max:.4f} "
        f"median={d_med:.4f} finite_ratio={finite_ratio:.4f}"
    )

    lo = float(np.percentile(vals, 2))
    hi = float(np.percentile(vals, 98))
    if hi <= lo:
        hi = lo + 1e-6

    clipped = np.clip(depth_metric, lo, hi)
    vis = ((clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)
    vis[~valid] = 0
    cv2.imwrite(output_path, vis)
    print(f"Saved depth debug image to '{output_path}'.")


def print_mask_depth_stats(
    mask_2d: np.ndarray,
    depth_metric: np.ndarray,
    label: str,
):
    """
    Print quick depth stats inside one detection mask.
    Useful for checking whether a detected object has coherent depth.
    """
    z = depth_metric[mask_2d > 0]
    z = z[np.isfinite(z) & (z > 0)]

    if z.size == 0:
        print(f"[DEPTH] {label}: no valid depth")
        return

    print(
        f"[DEPTH] {label}: "
        f"n={z.size} "
        f"min={float(z.min()):.4f} "
        f"p10={float(np.percentile(z, 10)):.4f} "
        f"med={float(np.median(z)):.4f} "
        f"p90={float(np.percentile(z, 90)):.4f} "
        f"max={float(z.max()):.4f}"
    )

def compute_mask_centroid_px(mask_2d: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask_2d > 0)
    if len(xs) == 0:
        return None
    return np.array([float(xs.mean()), float(ys.mean())], dtype=np.float64)


def compute_surface_center_world(
    grid: WorkspaceGrid,
    surface_mask: np.ndarray,
) -> np.ndarray | None:
    idx = np.argwhere(surface_mask)
    if len(idx) == 0:
        return None

    center = np.array(
        [
            float(grid.x[idx[:, 0]].mean()),
            float(grid.y[idx[:, 1]].mean()),
            float(grid.z[idx[:, 2]].mean()),
        ],
        dtype=np.float64,
    )
    return center


def project_world_point_to_image(
    world_pt: np.ndarray,
    intrinsics: dict,
    world_pos: np.ndarray,
    world_mat: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray | None:
    if world_pt is None:
        return None

    h, w = image_shape
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = w / 2.0, h / 2.0

    world_mat_3x3 = world_mat.reshape(3, 3)
    pt_world = np.asarray(world_pt, dtype=np.float64).reshape(3, 1)

    pts_mj = world_mat_3x3.T @ (pt_world - world_pos.reshape(3, 1))

    cv2mj_rot = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ])
    pt_cam = cv2mj_rot.T @ pts_mj

    z = float(pt_cam[2, 0])
    if z <= 0.01:
        return None

    u = float(fx * pt_cam[0, 0] / z + cx)
    v = float(fy * pt_cam[1, 0] / z + cy)
    return np.array([u, v], dtype=np.float64)


def fmt_vec(vec, digits: int = 4) -> str:
    if vec is None:
        return "None"
    return str(np.round(np.asarray(vec, dtype=np.float64), digits))

def choose_adaptive_cap_margin_xy_voxels(
    *,
    support_mask_2d: np.ndarray,
    plus_block_2d: np.ndarray,
    grid_res: float,
    max_margin_voxels: int = 4,
) -> int:
    """
    Choose an adaptive x-y cap margin based on the residual uncovered fraction
    in the support plane after gap filling.

    Logic:
    - no residual open columns -> 0
    - tiny residual -> 1
    - small residual -> 2
    - moderate residual -> ~1.5 cm in voxels
    - large residual -> clamp to max_margin_voxels
    """
    total_cols = int(support_mask_2d.sum())
    if total_cols == 0:
        return 0

    valid_plus = np.isfinite(plus_block_2d)
    open_cols = int((support_mask_2d & (~valid_plus)).sum())
    open_frac = open_cols / max(total_cols, 1)

    if open_cols == 0:
        return 0
    if open_frac <= 0.005:
        return 1
    if open_frac <= 0.02:
        return 2
    if open_frac <= 0.05:
        return min(max_margin_voxels, max(2, int(round(0.015 / grid_res))))
    return max_margin_voxels

def apply_cap_margin_xy(
    hard_axis_cap: np.ndarray,
    margin_xy_voxels: int,
) -> np.ndarray:
    """
    Dilate the blocked region only in x-y, not in z.
    True = allowed, False = blocked
    """
    if margin_xy_voxels <= 0:
        return hard_axis_cap

    blocked = ~hard_axis_cap

    structure = np.ones(
        (
            2 * margin_xy_voxels + 1,
            2 * margin_xy_voxels + 1,
            1,
        ),
        dtype=bool,
    )

    blocked = ndimage.binary_dilation(
        blocked,
        structure=structure,
        iterations=1,
    )
    return ~blocked

# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------

def run_pipeline(
    manipulated_obj: str,
    camera_type: str,
    target_label: str = "auto",
    candidate_labels: list[str] | None = None,
    xml_path: str = "tabletop.xml",
    use_gt_blockers: bool = False,
    gt_blocker_geoms: list[str] | None = None,
    use_table_top_filter: bool = False,
    prior_json_path: str = PRIOR_JSON_DEFAULT,
):
    if gt_blocker_geoms is None:
        gt_blocker_geoms = []

    model = None
    data = None

    print(f"--- INITIALIZING PIPELINE ---")
    print(f"Hardware: [{camera_type.upper()}]")
    print(f"Robot holding: [{manipulated_obj.upper()}]")
    print(f"Scene target mode: [{target_label.upper()}]")
    print(f"XML path: [{xml_path}]")
    print(f"GT blocker debug mode: [{'ON' if use_gt_blockers else 'OFF'}]\n")
    print(f"Table-top filter: [{'ON' if use_table_top_filter else 'OFF'}]\n")
    print(f"Prior JSON path: [{prior_json_path}]")

    table_top_z = None

    if camera_type == "Mujoco":
        xml_file = Path(xml_path)
        if not xml_file.exists():
            raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")

        model = mujoco.MjModel.from_xml_path(str(xml_file))
        data = mujoco.MjData(model)
        for _ in range(50):
            mujoco.mj_step(model, data)

        camera = MujocoCamera(model, data, cam_name="main_cam", width=640, height=480)
        color_image, depth_metric, intrinsics = camera.get_frames()
        bgr_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "main_cam")
        world_pos = data.cam_xpos[cam_id].copy()
        world_mat = data.cam_xmat[cam_id].copy()

        table_top_z = get_optional_mujoco_table_top_z(model, data, geom_name="table")
        if table_top_z is not None:
            print(f"MuJoCo table top z = {table_top_z:.4f} m")

    elif camera_type == "RealSense":
        from semantic_safety.perception_2d3d.realsense import RealSenseCamera

        camera = RealSenseCamera(width=640, height=480, fps=30)
        for _ in range(15):
            camera.get_frames()

        color_image, depth_image, _ = camera.get_frames()
        bgr_image = color_image
        depth_metric = depth_image * camera.get_depth_scale()
        intrinsics = {"fx": camera.intrinsics.fx, "fy": camera.intrinsics.fy}
        world_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        world_mat = np.eye(3, dtype=np.float64).flatten()
        camera.stop()

    else:
        raise ValueError(f"Unsupported camera_type: {camera_type}")

    cv2.imwrite("test_rgb.png", bgr_image)
    save_depth_debug_image(depth_metric, output_path="test_depth_debug.png")

    # ------------------------------------------------------------------
    # 1) Router init first + prior-driven candidate filtering
    # ------------------------------------------------------------------
    if not Path(prior_json_path).exists():
        raise FileNotFoundError(f"Prior JSON not found: {prior_json_path}")

    router = SemanticRouter(
        json_path=prior_json_path,
        system_instruction=SYSTEM_INSTRUCTION,
        llm_batch_callback=None,
        # llm_batch_callback=claude_batch_callback,
        persist_updates=False,
        # persist_updates=True,
        nearest_threshold=0.68,
        batch_window_s=0.15,
        max_batch_size=8,
        max_workers=2,
    )

    try:
        pass1_candidates, pass2_candidates = build_candidate_labels(
            manipulated_obj=manipulated_obj,
            target_label=target_label,
            user_candidate_labels=candidate_labels,
            prior_json_path=prior_json_path,
        )

        # Fallback if prior JSON has no entry for this manipulated object
        if not pass1_candidates:
            if target_label.lower() != "auto":
                pass1_candidates = [target_label]
            elif candidate_labels:
                pass1_candidates = list(candidate_labels)
            else:
                raise ValueError(
                    "No detector candidates available from prior JSON or user candidate labels."
                )

        if not pass2_candidates:
            pass2_candidates = list(pass1_candidates)

        print("\n--- PRIOR-DRIVEN CANDIDATE FILTERING ---")
        print("pass1_candidates:", pass1_candidates)
        print("pass2_candidates:", pass2_candidates)

        ai_detector = SemanticPerception()

        def _run_detector_pass(candidate_list: list[str], debug_dir: str):
            print(f"\n--- DETECTOR PASS | {debug_dir} ---")
            print("candidate count:", len(candidate_list))
            return ai_detector.detect_scene_objects(
                image_path="test_rgb.png",
                candidate_labels=candidate_list,
                save_debug=True,
                debug_dir=debug_dir,
            )

        def _target_hit(ds: list[dict]) -> bool:
            if target_label.lower() == "auto":
                return True

            target_canon = canonicalize_label_for_filter(router, target_label)
            for det in ds:
                if canonicalize_label_for_filter(router, det["label"]) == target_canon:
                    return True
            return False

        detections = _run_detector_pass(pass1_candidates, "perception_debug/pass1")
        detection_pass_used = "pass1"
        fallback_used = False

        if (not detections) or (target_label.lower() != "auto" and not _target_hit(detections)):
            if pass2_candidates != pass1_candidates:
                print("\nPass 1 missed target or returned no detections. Running fallback pass.")
                detections = _run_detector_pass(pass2_candidates, "perception_debug/pass2")
                detection_pass_used = "pass2"
                fallback_used = True

        if not detections:
            print("No scene detections found after fallback.")
            return

        # IMPORTANT:
        # Keep all detections for occupancy / blockers.
        occupancy_detections = list(detections)

        # Only store the target canonical label for later hazard-source filtering.
        hazard_target_canon = None
        if target_label.lower() != "auto":
            hazard_target_canon = canonicalize_label_for_filter(router, target_label)

        print("\n--- DETECTED SCENE CANDIDATES (ALL FOR OCCUPANCY) ---")
        for det in occupancy_detections:
            print(f"- {det['label']} ({det['score']:.3f}) box={det['box']}")

        print("\n--- DETECTION FILTER DEBUG ---")
        print("target_label:", target_label)
        print("hazard_target_canon:", hazard_target_canon)
        print("detection_pass_used:", detection_pass_used)
        print("fallback_used:", fallback_used)
        print("all detected labels:", [det["label"] for det in occupancy_detections])

        router.prefetch_scene_pairs(
            manipulated_label=manipulated_obj,
            scene_labels=[det["label"] for det in occupancy_detections],
        )

        # ------------------------------------------------------------------
        # 3) 2D -> 3D localization for all occupancy detections
        # ------------------------------------------------------------------
        world_engine = WorldTransform(world_pos, world_mat)

        object_infos = []
        for det in occupancy_detections:
            box = det["box"]
            mask_2d = det.get("mask", None)

            if mask_2d is None:
                h, w = depth_metric.shape
                rect_mask = np.zeros((h, w), dtype=bool)
                rect_mask[box["ymin"]:box["ymax"], box["xmin"]:box["xmax"]] = True
                mask_2d = rect_mask

            print_mask_depth_stats(mask_2d, depth_metric, det["label"])

            world_points = segmentation_mask_to_world_points(
                mask_2d=mask_2d,
                depth_metric=depth_metric,
                intrinsics=intrinsics,
                world_engine=world_engine,
                label=det["label"],
                table_top_z=table_top_z,
                depth_band_m=0.12,
                table_z_band_m=0.03,
                use_table_top_filter=use_table_top_filter,
            )

            if len(world_points) == 0:
                print(f"Skipping '{det['label']}' because segmentation produced no valid 3D points.")
                continue

            world_pt = np.mean(world_points, axis=0)

            span = world_points.max(axis=0) - world_points.min(axis=0)
            if span[0] > 2.0 or span[1] > 2.0 or span[2] > 1.0:
                print(
                    f"Skipping '{det['label']}' because voxelization span is implausibly large: "
                    f"{np.round(span, 3)}"
                )
                continue

            object_infos.append(
                {
                    "label": det["label"],
                    "score": float(det["score"]),
                    "box_2d": box,
                    "mask_2d": mask_2d,
                    "seg_centroid_px": compute_mask_centroid_px(mask_2d),
                    "world_points": world_points,
                    "world_pt": world_pt,
                    "source": det.get("source", "proposal"),
                    "top_k": det.get("top_k", []),
                }
            )

        if not object_infos:
            print("No valid 3D-localized objects remained.")
            return

        print("\n--- 3D LOCALIZATION ---")
        for obj in object_infos:
            print(
                f"- {obj['label']}: "
                f"num_pts={len(obj['world_points'])}, "
                f"world_pt={np.round(obj['world_pt'], 4)}"
            )

        object_infos = dedupe_object_infos_by_canonical_label_and_geometry(
            object_infos=object_infos,
            router=router,
            point_dist_thresh=0.03,
            bbox_iou_thresh=0.25,
            target_label=(None if target_label.lower() == "auto" else target_label),
        )

        object_infos = suppress_contained_duplicate_objects(
            object_infos=object_infos,
            router=router,
            containment_threshold=0.80,
            area_ratio_threshold=0.45,
            center_dist_threshold=0.15,
            depth_diff_threshold=0.08,
        )

        print("\n--- AFTER 3D CANONICAL DEDUPE ---")
        for obj in object_infos:
            rep_score = representative_priority(
                obj,
                router,
                target_label=(None if target_label.lower() == "auto" else target_label),
            )
            print(
                f"- {obj['label']}: "
                f"world_pt={np.round(obj['world_pt'], 4)}, "
                f"score={obj['score']:.3f}, "
                f"rep_priority={rep_score:.2f}"
            )

        # ------------------------------------------------------------
        # Split:
        # - object_infos: all objects for occupancy / blockers
        # - hazard_object_infos: only objects that should emit risk fields
        # ------------------------------------------------------------
        if hazard_target_canon is None:
            hazard_object_infos = list(object_infos)
        else:
            hazard_object_infos = [
                obj for obj in object_infos
                if canonicalize_label_for_filter(router, obj["label"]) == hazard_target_canon
            ]

        print("\n--- HAZARD SOURCE OBJECTS ---")
        for obj in hazard_object_infos:
            print(f"- {obj['label']}: world_pt={np.round(obj['world_pt'], 4)}, score={obj['score']:.3f}")

        if not hazard_object_infos:
            print("Warning: no hazard source objects matched target label. Falling back to all objects.")
            hazard_object_infos = list(object_infos)

        # ------------------------------------------------------------------
        # 4) Global voxel workspace
        # ------------------------------------------------------------------
        world_pts = np.concatenate([obj["world_points"] for obj in object_infos], axis=0)
        res = 0.004
        bounds = build_global_workspace_bounds(
            world_pts,
            grid_margin_xy=0.30,
            grid_margin_z_up=0.55,
            z_margin_below=0.08,
        )
        grid = WorkspaceGrid(bounds=bounds, resolution=res)

        for obj in object_infos:
            # Conservative geometry for topology / occupancy / FMM
            # ============================================================
            # 2) geometry loop: keep raw source mask separate from blocker
            # ============================================================
            source_raw_mask = world_points_to_voxel_mask(
                grid=grid,
                world_points=obj["world_points"],
                dilation_iters=1,
            )
            source_raw_mask = keep_largest_connected_component(source_raw_mask, connectivity=1)

            if not np.any(source_raw_mask):
                print(f"Skipping '{obj['label']}' because source_raw_mask is empty.")
                continue

            occupancy_surface_mask = extract_top_surface_mask(source_raw_mask, band_voxels=1)
            occupancy_bbox_3d = voxel_mask_to_bbox(grid, source_raw_mask)
            source_surface_mask = clean_source_surface_mask(occupancy_surface_mask)

            blocker_solid_mask = build_blocker_solid_mask(
                occupancy_surface_mask=occupancy_surface_mask,
                occupancy_bbox_3d=occupancy_bbox_3d,
                grid=grid,
                pad_xy_voxels=2,   # start small
            )

            obj["source_raw_mask"] = source_raw_mask
            obj["occupancy_mask"] = source_raw_mask          # keep old key for compatibility if needed
            obj["occupancy_surface_mask"] = occupancy_surface_mask
            obj["source_surface_mask"] = source_surface_mask
            obj["occupancy_bbox_3d"] = occupancy_bbox_3d
            obj["blocker_solid_mask"] = blocker_solid_mask

            occ_center_world = compute_surface_center_world(grid, occupancy_surface_mask)
            src_center_world = compute_surface_center_world(grid, source_surface_mask)

            occ_center_px = project_world_point_to_image(
                occ_center_world,
                intrinsics=intrinsics,
                world_pos=world_pos,
                world_mat=world_mat,
                image_shape=depth_metric.shape,
            )
            src_center_px = project_world_point_to_image(
                src_center_world,
                intrinsics=intrinsics,
                world_pos=world_pos,
                world_mat=world_mat,
                image_shape=depth_metric.shape,
            )

            obj["occupancy_surface_center_world"] = occ_center_world
            obj["source_surface_center_world"] = src_center_world
            obj["occupancy_surface_center_px"] = occ_center_px
            obj["source_surface_center_px"] = src_center_px

        object_infos = [
            obj for obj in object_infos
            if "occupancy_mask" in obj and "source_surface_mask" in obj
        ]
        hazard_object_infos = [
            obj for obj in hazard_object_infos
            if "occupancy_mask" in obj and "source_surface_mask" in obj
        ]

        print("\n--- OBJECT GEOMETRY DEBUG ---")
        for obj in object_infos:
            print(f"[{obj['label']}]")
            print(f"  occupancy_bbox_3d        = {tuple(round(v, 4) for v in obj['occupancy_bbox_3d'])}")
            print(f"  num_world_points         = {len(obj['world_points'])}")
            print(f"  occupancy_voxels         = {int(obj['occupancy_mask'].sum())}")
            print(f"  occupancy_surface_voxels = {int(obj['occupancy_surface_mask'].sum())}")
            print(f"  source_surface_voxels    = {int(obj['source_surface_mask'].sum())}")
            print(f"  source_footprint_voxels  = {int(np.any(obj['source_surface_mask'], axis=2).sum())}")

            if canonicalize_label_for_filter(router, obj["label"]) == "power drill":
                z_occ = grid.Z[obj["occupancy_surface_mask"]]
                z_src = grid.Z[obj["source_surface_mask"]]
                if z_occ.size > 0:
                    print(f"  occupancy surface z range = {z_occ.min():.4f} .. {z_occ.max():.4f}")
                if z_src.size > 0:
                    print(f"  source surface z range    = {z_src.min():.4f} .. {z_src.max():.4f}")

            print(f"  seg_centroid_px          = {fmt_vec(obj.get('seg_centroid_px'), 2)}")
            print(f"  seg_center_world         = {fmt_vec(obj.get('world_pt'), 4)}")
            print(f"  occ_surface_center_px    = {fmt_vec(obj.get('occupancy_surface_center_px'), 2)}")
            print(f"  src_surface_center_px    = {fmt_vec(obj.get('source_surface_center_px'), 2)}")
            print(f"  occ_surface_center_world = {fmt_vec(obj.get('occupancy_surface_center_world'), 4)}")
            print(f"  src_surface_center_world = {fmt_vec(obj.get('source_surface_center_world'), 4)}")

            seg_px = obj.get("seg_centroid_px", None)
            occ_px = obj.get("occupancy_surface_center_px", None)
            src_px = obj.get("source_surface_center_px", None)

            if seg_px is not None and occ_px is not None:
                print(f"  delta_occ_minus_seg_px   = {fmt_vec(occ_px - seg_px, 2)}")
            if seg_px is not None and src_px is not None:
                print(f"  delta_src_minus_seg_px   = {fmt_vec(src_px - seg_px, 2)}")

            if obj.get("occupancy_surface_center_world") is not None:
                print(
                    f"  delta_occ_minus_seg_world= "
                    f"{fmt_vec(obj['occupancy_surface_center_world'] - obj['world_pt'], 4)}"
                )
            if obj.get("source_surface_center_world") is not None:
                print(
                    f"  delta_src_minus_seg_world= "
                    f"{fmt_vec(obj['source_surface_center_world'] - obj['world_pt'], 4)}"
                )
        print("--- END DEBUG ---")

        if not object_infos:
            print("No valid voxelized objects remained.")
            return

        occupancy_masks = [obj["occupancy_mask"] for obj in object_infos]
        gt_blocker_info = []

        if use_gt_blockers:
            gt_masks, gt_blocker_info = collect_debug_gt_blocker_masks(
                model=model,
                data=data,
                grid=grid,
                geom_names=gt_blocker_geoms,
                pad=0.0,
            )
            occupancy_masks.extend(gt_masks)

        occupancy_free = build_multi_object_occupancy(
            grid=grid,
            object_masks=occupancy_masks,
            table_top_z=table_top_z,
        )

        print("\n--- OCCUPANCY DEBUG ---")
        print("occupancy_free fraction:", float(occupancy_free.mean()))

        print("occupancy objects:")
        for obj in object_infos:
            print(
                f"  - {obj['label']}: "
                f"occupancy_voxels={int(obj['occupancy_mask'].sum())}, "
                f"occupancy_surface_voxels={int(obj['occupancy_surface_mask'].sum())}"
            )

        print("hazard source objects:")
        for obj in hazard_object_infos:
            print(
                f"  - {obj['label']}: "
                f"occupancy_voxels={int(obj['occupancy_mask'].sum())}, "
                f"source_surface_voxels={int(obj['source_surface_mask'].sum())}, "
                f"source_footprint_voxels={int(np.any(obj['source_surface_mask'], axis=2).sum())}"
            )

        if use_gt_blockers:
            print("debug oracle blocker geoms:")
            for row in gt_blocker_info:
                print(f"  - {row['geom_name']}: obstacle_voxels={row['voxels']}")

        # ------------------------------------------------------------------
        # 5) Per-object risk field generation
        # ------------------------------------------------------------------
        hazard_fields = []
        per_object_debug = []

        print("\n--- RISK FIELD DEBUG ---")

        def nearest_idx(arr, value):
            return int(np.argmin(np.abs(arr - value)))

        for obj in hazard_object_infos:
            risk_params = router.get_risk_parameters(manipulated_obj, obj["label"])
            obj["risk_params"] = risk_params

            scene_role = risk_params.get("scene_role", "hazard_target")
            obj["scene_role"] = scene_role

            # ------------------------------------------------------------
            # Build source-specific occupancy:
            # - current source object uses raw occupancy_mask
            # - all other objects use conservative blocker_mask if available
            # ------------------------------------------------------------
            # ============================================================
            # 3) local occupancy: source uses raw, others use blocker_solid
            # ============================================================
            source_specific_masks = []
            for other in object_infos:
                if other is obj:
                    source_specific_masks.append(other["source_raw_mask"])
                else:
                    source_specific_masks.append(other["blocker_solid_mask"])

            occupancy_free_local = build_multi_object_occupancy(
                grid=grid,
                object_masks=source_specific_masks,
                table_top_z=table_top_z,
            )

            print("[ROUTER PRIOR]", risk_params)

            d_euc, d_geo, seed_mask = grid.compute_boundary_seeded_distances(
                object_mask=obj["source_raw_mask"],
                occupancy_grid=occupancy_free_local,
                connectivity=1,
            )
            A_field = shielding_ratio(d_geo, d_euc)

            axis_caps, axis_debug = compute_axis_occlusion_caps(
                grid=grid,
                source_mask=obj["source_raw_mask"],
                occupancy_free=occupancy_free_local,
                source_surface_mask=obj.get("source_surface_mask", None),
                gap_fill_voxels=2,
            )

            hard_axis_cap, active_cap_keys = compose_axis_occlusion_cap(
                axis_caps=axis_caps,
                risk_params=risk_params,
            )

            cap_margin_xy_voxels = choose_adaptive_cap_margin_xy_voxels(
                support_mask_2d=axis_debug["support_xy"],
                plus_block_2d=axis_debug["plus_z_block"],
                grid_res=grid.res,
                max_margin_voxels=4,
            )

            hard_axis_cap = apply_cap_margin_xy(
                hard_axis_cap,
                margin_xy_voxels=cap_margin_xy_voxels,
            )

            valid_plus = np.isfinite(axis_debug["plus_z_block"])
            support_xy = axis_debug["support_xy"]
            open_cols_after = int((support_xy & (~valid_plus)).sum())

            print("\n--- AXIS OCCLUSION CAP DEBUG ---")
            print("active_cap_keys:", active_cap_keys)
            print("adaptive cap_margin_xy_voxels:", cap_margin_xy_voxels)
            print("open source cols(+z) after fill:", open_cols_after)
            print("allowed fraction:", float(hard_axis_cap.mean()))
            print("blocked fraction:", 1.0 - float(hard_axis_cap.mean()))

            if canonicalize_label_for_filter(router, obj["label"]) == "power drill":
                cx, cy, cz = obj["world_pt"]

                probes = [
                    ("drill_above", np.array([cx, cy, cz + 0.22], dtype=np.float64)),
                    ("right_open",  np.array([cx + 0.18, cy, cz + 0.22], dtype=np.float64)),
                    ("left_open",   np.array([cx - 0.18, cy, cz + 0.22], dtype=np.float64)),
                ]

                print("\n--- SHIELDING PROBE DEBUG ---")
                for name, pt in probes:
                    ix = nearest_idx(grid.x, pt[0])
                    iy = nearest_idx(grid.y, pt[1])
                    iz = nearest_idx(grid.z, pt[2])

                    de = float(d_euc[ix, iy, iz])
                    dg = float(d_geo[ix, iy, iz])
                    af = float(A_field[ix, iy, iz])
                    free = int(occupancy_free_local[ix, iy, iz])
                    cap_here = int(hard_axis_cap[ix, iy, iz])

                    print(
                        f"[{name}] pt={np.round(pt, 4)} "
                        f"idx=({ix},{iy},{iz}) "
                        f"free={free} "
                        f"d_euc={de:.4f} d_geo={dg:.4f} A={af:.4f} "
                        f"cap={cap_here}"
                    )

                print("\n--- PROBE VS BLOCKER BBOX DEBUG ---")
                for name, pt in probes:
                    print(f"[{name}] pt={np.round(pt, 4)}")

                    for other in object_infos:
                        if other is obj:
                            continue

                        bb = other["occupancy_bbox_3d"]
                        inside = (
                            (bb[0] <= pt[0] <= bb[1]) and
                            (bb[2] <= pt[1] <= bb[3]) and
                            (bb[4] <= pt[2] <= bb[5])
                        )

                        print(
                            f"    blocker={other['label']:<12} "
                            f"inside_bbox={inside} "
                            f"bbox={tuple(round(v, 4) for v in bb)}"
                        )

            attenuation = float(risk_params.get("receptacle_attenuation", 1.0))
            object_max_risk_score = get_object_max_risk_score(risk_params)

            base_risk = 100.0 * attenuation * object_max_risk_score

            risk_params_for_field = dict(risk_params)
            risk_params_for_field["weights"] = normalize_weights_for_shape(
                risk_params.get("weights", {})
            )

            print(f"[RISK PARAMS FIELD] keys={sorted(risk_params_for_field.keys())}")
            print(f"[RISK PARAMS FIELD] radius_m={risk_params_for_field.get('radius_m', None)}")

            print(f"[BASE_RISK DEBUG] {obj['label']} base_risk={base_risk:.4f}")
            print(f"[WEIGHT DEBUG] max_score={get_object_max_risk_score(obj):.4f}")

            print(f"[RISK PARAMS RAW] keys={sorted(risk_params.keys())}")
            print(f"[RISK PARAMS RAW] radius_m={risk_params.get('radius_m', None)}")

            V_object = build_risk_field_from_params(
                grid=grid,
                risk_params=risk_params_for_field,
                d_geo=d_geo,
                A_field=A_field,
                base_risk=base_risk,
                bbox=obj["occupancy_bbox_3d"],
                object_mask=obj["occupancy_mask"],
                surface_mask=obj["occupancy_surface_mask"],
                footprint_mask=np.any(obj["occupancy_surface_mask"], axis=2),
            )

            print(f"[RISK STATS] {obj['label']}")
            print("  max =", float(np.max(V_object)))
            print("  p99 =", float(np.percentile(V_object, 99)))
            print("  p95 =", float(np.percentile(V_object, 95)))
            print("  p90 =", float(np.percentile(V_object, 90)))
            print("  voxels > 0.01 =", int(np.sum(V_object > 0.01)))
            print("  voxels > 0.05 =", int(np.sum(V_object > 0.05)))
            print("  voxels > 0.10 =", int(np.sum(V_object > 0.10)))

            # hard directional shadow cap
            V_object = V_object * hard_axis_cap.astype(np.float64)

            if risk_params.get("vertical_rule", "standard_decay") == "gravity_column":
                V_surface = gravity_column_from_surface_mask(
                    grid=grid,
                    top_surface_mask=obj["source_surface_mask"],
                    A_field=A_field,
                    base_risk=base_risk,
                    w_plus_z=float(risk_params.get("weights", {}).get("w_+z", 0.0)),
                    lateral_decay=risk_params.get("lateral_decay", "moderate"),
                )

                V_surface = V_surface * hard_axis_cap.astype(np.float64)
                V_object = np.maximum(V_object, V_surface)

            obj["d_euc"] = d_euc
            obj["d_geo"] = d_geo
            obj["A_field"] = A_field
            obj["seed_mask"] = seed_mask
            obj["V_object"] = V_object

            print(f"\n[{obj['label']}]")
            print(f"  topology            = {risk_params.get('topology_template')}")
            print(f"  vertical_rule       = {risk_params.get('vertical_rule')}")
            print(f"  scene_role          = {scene_role}")
            print(f"  attenuation         = {risk_params.get('receptacle_attenuation')}")
            print(f"  weights             = {risk_params.get('weights')}")
            print(f"  occupancy_voxels       = {int(obj['occupancy_mask'].sum())}")
            #print(f"  source_voxels           = {int(obj['source_mask'].sum())}")
            print(f"  source_surface_voxels   = {int(obj['source_surface_mask'].sum())}")
            print(f"  source_footprint_voxels = {int(np.any(obj['source_surface_mask'], axis=2).sum())}")
            print(f"  frac > 5            = {float((V_object > 5.0).mean()):.4f}")
            print(f"  frac > 10           = {float((V_object > 10.0).mean()):.4f}")

            if risk_params.get("vertical_rule", "standard_decay") == "gravity_column":
                z_vals = grid.Z[obj["source_surface_mask"]]
                if z_vals.size > 0:
                    print(f"  top surface z range = {z_vals.min():.4f} .. {z_vals.max():.4f}")

            if scene_role == "hazard_target":
                hazard_fields.append(V_object)
            else:
                print(f"  -> not added to hazard superposition (scene_role={scene_role})")

            selected_cap_direction = "none"
            topology = risk_params.get("topology_template", "isotropic_sphere")
            weights = risk_params.get("weights", {})

            if topology == "upward_vertical_cone":
                selected_cap_direction = "w_+z"
            elif topology == "forward_directional_cone":
                horiz_keys = ["w_+x", "w_-x", "w_+y", "w_-y"]
                selected_cap_direction = max(horiz_keys, key=lambda k: float(weights.get(k, 0.0)))
            elif topology == "planar_half_space":
                all_keys = ["w_+x", "w_-x", "w_+y", "w_-y", "w_+z", "w_-z"]
                selected_cap_direction = max(all_keys, key=lambda k: float(weights.get(k, 0.0)))

            print("\n--- AXIS OCCLUSION CAP DEBUG ---")
            print("active_cap_keys:", active_cap_keys)
            print("allowed fraction:", float(hard_axis_cap.mean()))
            print("blocked fraction:", 1.0 - float(hard_axis_cap.mean()))

            print_final_risk_profiles(
                grid=grid,
                d_euc=d_euc,
                d_geo=d_geo,
                A_field=A_field,
                hard_axis_cap=hard_axis_cap,
                V_object=V_object,
                occupancy_free=occupancy_free,
                center_world_pt=obj.get("occupancy_surface_center_world", obj["world_pt"]),
                offsets=(0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32),
                title=f"FINAL RISK PROFILE | {obj['label']}",
            )


            per_object_debug.append(
                {
                    "label": obj["label"],
                    "scene_role": scene_role,
                    "world_pt": obj["world_pt"].tolist(),
                    "bbox_3d": list(obj["occupancy_bbox_3d"]),
                    "num_world_points": int(len(obj["world_points"])),
                    "risk_params": risk_params,
                    "max_risk": float(np.max(V_object)),
                    "mean_risk": float(np.mean(V_object)),
                }
            )

        if not hazard_fields:
            print("No scene_role='hazard_target' objects were found. Final risk field will be zeros.")
            V_final = np.zeros(grid.shape, dtype=np.float32)
        else:
            V_final = compute_sum_superposition(
                hazard_fields,
                v_max=300.0,
            )
            
        scene_max_risk_score = float(V_final.max())
        print(f"Scene max risk value: {scene_max_risk_score:.4f}")
        print(f"Scene max risk score: {scene_max_risk_score / 100.0:.4f}")

        # ------------------------------------------------------------------
        # 6) Save Loop 1 outputs
        # ------------------------------------------------------------------
        metadata_json = json.dumps(
            {
                "manipulated_obj": manipulated_obj,
                "camera_type": camera_type,
                "xml_path": xml_path,
                "target_label_mode": target_label,
                "debug_oracle_blockers_enabled": bool(use_gt_blockers),
                "debug_oracle_blockers": gt_blocker_info,
                "detected_objects": per_object_debug,
                "prior_json_path": prior_json_path,
                "pass1_candidates": pass1_candidates,
                "pass2_candidates": pass2_candidates,
                "detection_pass_used": detection_pass_used,
                "fallback_used": fallback_used,
            },
            ensure_ascii=False,
        )

        np.savez_compressed(
            "loop1_risk_field.npz",
            risk_field=V_final.astype(np.float32),
            x=grid.x.astype(np.float32),
            y=grid.y.astype(np.float32),
            z=grid.z.astype(np.float32),
            occupancy_free=occupancy_free.astype(np.uint8),
            table_top_z=np.array([-1.0 if table_top_z is None else table_top_z], dtype=np.float32),
            metadata_json=np.array(metadata_json),
        )
        print("\nSaved core Loop 1 output to 'loop1_risk_field.npz'.")

        with open("loop1_scene_objects.json", "w", encoding="utf-8") as f:
            json.dump(per_object_debug, f, indent=2, ensure_ascii=False)
        print("Saved object/risk debug info to 'loop1_scene_objects.json'.")

        # ------------------------------------------------------------------
        # 7) 2D segmentation attenuation overlay
        # ------------------------------------------------------------------
        draw_segmentation_by_attenuation(
            bgr_image=bgr_image,
            object_infos=hazard_object_infos,
            output_path="segmentation_attenuation_overlay.png",
        )

        # ------------------------------------------------------------------
        # 8) 3D risk visualization
        # ------------------------------------------------------------------
        vis_detections = [
            {
                "label": obj["label"],
                "box": obj["box_2d"],
                "score": obj["score"],
            }
            for obj in object_infos
        ]

        project_and_render_overlay(
            bgr_image=bgr_image,
            depth_metric=depth_metric,
            intrinsics=intrinsics,
            world_pos=world_pos,
            world_mat=world_mat,
            X=grid.X,
            Y=grid.Y,
            Z=grid.Z,
            V_final=V_final,
            detections=vis_detections,
            res=res,
            output_path="continuous_risk_overlay.png",
        )

        print("\nExecution complete.")

    finally:
        router.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manipulated", type=str, default="cup of water")
    parser.add_argument("--camera", type=str, default="Mujoco")
    parser.add_argument(
        "--xml-path",
        type=str,
        default="tabletop.xml",
        help="MuJoCo XML path, e.g. tabletop.xml or tabletop_shelf.xml",
    )
    parser.add_argument(
        "--scene-label",
        type=str,
        default="auto",
        help="Use 'auto' to process all detected scene objects, or specify one canonical target label.",
    )
    parser.add_argument(
        "--candidate-labels",
        nargs="*",
        default=None,
        help="Optional candidate label vocabulary for open-vocabulary detection.",
    )
    parser.add_argument(
        "--use-gt-blockers",
        action="store_true",
        help="Optional debug/oracle mode only. Inject specific MuJoCo geoms into occupancy.",
    )
    parser.add_argument(
        "--gt-blocker-geoms",
        nargs="*",
        default=[],
        help="Optional MuJoCo geom names for debug/oracle blocker injection. Missing geoms are skipped.",
    )
    parser.add_argument(
        "--use-table-top-filter",
        action="store_true",
        help="Optional debug filter only. Restrict detections labeled 'table' to the MuJoCo table-top plane.",
    )
    parser.add_argument(
        "--prior-json-path",
        type=str,
        default=PRIOR_JSON_DEFAULT,
        help="Path to semantic prior JSON used for prior-driven candidate filtering.",
    )
    args = parser.parse_args()

    run_pipeline(
        manipulated_obj=args.manipulated,
        camera_type=args.camera,
        target_label=args.scene_label,
        candidate_labels=args.candidate_labels,
        xml_path=args.xml_path,
        use_gt_blockers=args.use_gt_blockers,
        gt_blocker_geoms=args.gt_blocker_geoms,
        use_table_top_filter=args.use_table_top_filter,
        prior_json_path=args.prior_json_path,
    )