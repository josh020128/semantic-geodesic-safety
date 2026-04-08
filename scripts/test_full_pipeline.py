import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy import ndimage

from semantic_safety.perception_2d3d.mujoco_camera import MujocoCamera
from semantic_safety.perception_2d3d.lang_sam_wrapper import SemanticPerception
from semantic_safety.perception_2d3d.deproject_3d import CameraGeometry
from semantic_safety.perception_2d3d.transform import WorldTransform
from semantic_safety.semantic_router.router import SemanticRouter
from semantic_safety.metric_propagation.fmm_distance import WorkspaceGrid
from semantic_safety.risk_field.superposition import (
    shielding_ratio,
    compute_logsumexp_superposition,
)
from semantic_safety.risk_field.templates import build_risk_field_from_params
from semantic_safety.phase0_dataset.prompts import GEMINI_SYSTEM_INSTRUCTION


# ----------------------------------------------------------------------
# Optional Gemini hook
# ----------------------------------------------------------------------

try:
    from semantic_safety.semantic_router.gemini_callbacks import gemini_batch_callback
except Exception:
    gemini_batch_callback = None


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


def segmentation_mask_to_world_points(
    mask_2d: np.ndarray,
    depth_metric: np.ndarray,
    intrinsics: dict,
    world_engine: WorldTransform,
    label: str | None = None,
    table_top_z: float | None = None,
    depth_band_m: float = 0.12,
    table_z_band_m: float = 0.03,
) -> np.ndarray:
    """
    Convert a 2D segmentation mask into 3D world points using depth,
    with robust depth filtering to reject background leakage.

    Strategy:
      1) valid depth only
      2) keep points near mask median depth
      3) if label is table and table_top_z is known, keep only points near table top plane
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

    # Robust depth filtering inside the mask
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

        # Extra filtering for known planar support surface
        if label_norm == "table" and table_top_z is not None:
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

    # 2D xy distance to the nearest footprint voxel
    d_xy = ndimage.distance_transform_edt(~footprint) * grid.res

    # top z value for each (x, y) footprint column
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


def dedupe_object_infos_by_canonical_label_and_geometry(
    object_infos: list[dict],
    router: SemanticRouter,
    point_dist_thresh: float = 0.03,
    bbox_iou_thresh: float = 0.25,
) -> list[dict]:
    """
    Dedupe objects after 3D localization.

    Keep the highest-score object among entries that:
    - share the same canonical label
    - and are geometrically almost the same object

    This is better than raw 2D IoU because nested detections like
    'drill' and 'power drill' may have low IoU but identical 3D centers.
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

    object_infos = sorted(object_infos, key=lambda o: float(o["score"]), reverse=True)
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
    Lower attenuation -> cooler/less intense.
    Higher attenuation -> hotter/more intense.
    """
    canvas = bgr_image.copy()

    for obj in object_infos:
        mask = obj.get("mask_2d", None)
        risk_params = obj.get("risk_params", {})
        if mask is None or risk_params is None:
            continue

        att = float(risk_params.get("receptacle_attenuation", 1.0))
        att = float(np.clip(att, 0.1, 1.0))
        t = (att - 0.1) / 0.9

        # BGR: blue -> red
        color = np.array([
            int(255 * (1.0 - t)),   # B
            int(180 * (1.0 - t)),   # G
            int(255 * t),           # R
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

    risk_canvas = cv2.GaussianBlur(risk_canvas, (21, 21), 0)
    alpha_canvas = cv2.GaussianBlur(alpha_canvas, (21, 21), 0)

    norm_risk = np.clip(risk_canvas, 0, 100) / 100.0
    col_idx = ((1.0 - norm_risk) * 255).astype(np.uint8)
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


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------

def run_pipeline(
    manipulated_obj: str,
    camera_type: str,
    target_label: str = "auto",
    candidate_labels: list[str] | None = None,
):
    print(f"--- INITIALIZING PIPELINE ---")
    print(f"Hardware: [{camera_type.upper()}]")
    print(f"Robot holding: [{manipulated_obj.upper()}]")
    print(f"Scene target mode: [{target_label.upper()}]\n")

    table_top_z = None

    if camera_type == "Mujoco":
        model = mujoco.MjModel.from_xml_path("tabletop.xml")
        data = mujoco.MjData(model)
        for _ in range(50):
            mujoco.mj_step(model, data)

        camera = MujocoCamera(model, data, cam_name="main_cam", width=640, height=480)
        color_image, depth_metric, intrinsics = camera.get_frames()
        bgr_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "main_cam")
        world_pos = data.cam_xpos[cam_id].copy()
        world_mat = data.cam_xmat[cam_id].copy()

        table_top_z = get_mujoco_table_top_z(model, data, geom_name="table")
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

    # ------------------------------------------------------------------
    # 1) Scene perception
    # ------------------------------------------------------------------
    ai_detector = SemanticPerception()

    if candidate_labels is None or len(candidate_labels) == 0:
        candidate_labels = [
            "power drill", "drill", "bowl", "ceramic bowl", "mixing bowl",
            "cup", "mug", "glass", "container", "tray", "sink",
            "laptop", "computer", "monitor", "phone", "tablet",
            "table", "shelf", "wall", "box",
        ]

    detections = ai_detector.detect_scene_objects(
        image_path="test_rgb.png",
        candidate_labels=candidate_labels,
        save_debug=True,
        debug_dir="perception_debug",
    )

    if not detections:
        print("No scene detections found.")
        return

    # ------------------------------------------------------------------
    # 2) Router init + optional target filtering
    # ------------------------------------------------------------------
    router = SemanticRouter(
        json_path="data/semantic_risk_demo.json",
        system_instruction=GEMINI_SYSTEM_INSTRUCTION,
        llm_batch_callback=None,
        #llm_batch_callback=gemini_batch_callback,
        persist_updates=False,
        #persist_updates=True,
        nearest_threshold=0.68,
        batch_window_s=0.15,
        max_batch_size=8,
        max_workers=2,
    )

    try:
        if target_label.lower() != "auto":
            target_canon = canonicalize_label_for_filter(router, target_label)
            filtered = [
                det for det in detections
                if canonicalize_label_for_filter(router, det["label"]) == target_canon
            ]
            if filtered:
                detections = filtered
            else:
                print(f"Warning: target_label='{target_label}' not found after canonicalization. Using all detections.")

        print("\n--- DETECTED SCENE CANDIDATES ---")
        for det in detections:
            print(f"- {det['label']} ({det['score']:.3f}) box={det['box']}")

        # Warm the cache for unknown labels
        router.prefetch_scene_pairs(
            manipulated_label=manipulated_obj,
            scene_labels=[det["label"] for det in detections],
        )

        # ------------------------------------------------------------------
        # 3) 2D -> 3D localization for all detections using segmentation masks
        # ------------------------------------------------------------------
        world_engine = WorldTransform(world_pos, world_mat)

        object_infos = []
        for det in detections:
            box = det["box"]
            mask_2d = det.get("mask", None)

            if mask_2d is None:
                # fallback: use 2D box rectangle if mask is unavailable
                h, w = depth_metric.shape
                rect_mask = np.zeros((h, w), dtype=bool)
                rect_mask[box["ymin"]:box["ymax"], box["xmin"]:box["xmax"]] = True
                mask_2d = rect_mask

            world_points = segmentation_mask_to_world_points(
                mask_2d=mask_2d,
                depth_metric=depth_metric,
                intrinsics=intrinsics,
                world_engine=world_engine,
                label=det["label"],
                table_top_z=table_top_z,
                depth_band_m=0.12,
                table_z_band_m=0.03,
            )

            if len(world_points) == 0:
                print(f"Skipping '{det['label']}' because segmentation produced no valid 3D points.")
                continue

            world_pt = np.mean(world_points, axis=0)

            if len(world_points) > 0:
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
        )

        print("\n--- AFTER 3D CANONICAL DEDUPE ---")
        for obj in object_infos:
            print(f"- {obj['label']}: world_pt={np.round(obj['world_pt'], 4)}, score={obj['score']:.3f}")

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

        object_masks = []
        for obj in object_infos:
            object_mask = world_points_to_voxel_mask(
                grid=grid,
                world_points=obj["world_points"],
                dilation_iters=1,
            )

            if not np.any(object_mask):
                print(f"Skipping '{obj['label']}' because voxelized object mask is empty.")
                continue

            obj["object_mask"] = object_mask
            obj["surface_mask"] = extract_top_surface_mask(object_mask, band_voxels=1)
            obj["bbox_3d"] = voxel_mask_to_bbox(grid, object_mask)

            object_masks.append(object_mask)

        print("\n--- OBJECT GEOMETRY DEBUG ---")
        for obj in object_infos:
            print(f"[{obj['label']}]")
            print(f"  bbox_3d             = {tuple(round(v, 4) for v in obj['bbox_3d'])}")
            print(f"  num_world_points    = {len(obj['world_points'])}")
            print(f"  object_voxels       = {int(obj['object_mask'].sum())}")
            print(f"  surface_voxels      = {int(obj['surface_mask'].sum())}")
            print(f"  footprint_voxels    = {int(np.any(obj['surface_mask'], axis=2).sum())}")
        print("--- END DEBUG ---")

        object_infos = [obj for obj in object_infos if "object_mask" in obj]
        if not object_infos:
            print("No valid voxelized objects remained.")
            return

        occupancy_free = build_multi_object_occupancy(
            grid=grid,
            object_masks=[obj["object_mask"] for obj in object_infos],
            table_top_z=table_top_z,
        )

        # ------------------------------------------------------------------
        # 5) Per-object risk field generation
        # ------------------------------------------------------------------
        hazard_fields = []
        per_object_debug = []

        print("\n--- RISK FIELD DEBUG ---")

        for obj in object_infos:
            risk_params = router.get_risk_parameters(manipulated_obj, obj["label"])
            obj["risk_params"] = risk_params

            scene_role = risk_params.get("scene_role", "hazard_target")
            obj["scene_role"] = scene_role

            d_euc, d_geo, seed_mask = grid.compute_boundary_seeded_distances(
                object_mask=obj["object_mask"],
                occupancy_grid=occupancy_free,
                connectivity=1,
            )
            A_field = shielding_ratio(d_geo, d_euc)

            gamma = float(risk_params.get("receptacle_attenuation", 1.0))
            base_risk = 100.0 * gamma

            V_object = build_risk_field_from_params(
                grid=grid,
                risk_params=risk_params,
                d_geo=d_geo,
                A_field=A_field,
                base_risk=base_risk,
                bbox=obj["bbox_3d"],
                object_mask=obj["object_mask"],
                surface_mask=obj["surface_mask"],
                footprint_mask=np.any(obj["surface_mask"], axis=2),
            )

            if risk_params.get("vertical_rule", "standard_decay") == "gravity_column":
                weights = risk_params.get("weights", {})
                V_surface = gravity_column_from_surface_mask(
                    grid=grid,
                    top_surface_mask=obj["surface_mask"],
                    A_field=A_field,
                    base_risk=base_risk,
                    w_plus_z=float(weights.get("w_+z", 0.0)),
                    lateral_decay=risk_params.get("lateral_decay", "moderate"),
                )
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
            print(f"  object_voxels       = {int(obj['object_mask'].sum())}")
            print(f"  surface_voxels      = {int(obj['surface_mask'].sum())}")
            print(f"  footprint_voxels    = {int(np.any(obj['surface_mask'], axis=2).sum())}")
            print(f"  frac > 5            = {float((V_object > 5.0).mean()):.4f}")
            print(f"  frac > 10           = {float((V_object > 10.0).mean()):.4f}")

            if risk_params.get("vertical_rule", "standard_decay") == "gravity_column":
                z_vals = grid.Z[obj["surface_mask"]]
                if z_vals.size > 0:
                    print(f"  top surface z range = {z_vals.min():.4f} .. {z_vals.max():.4f}")

            if scene_role == "hazard_target":
                hazard_fields.append(V_object)
            else:
                print(f"  -> not added to hazard superposition (scene_role={scene_role})")

            per_object_debug.append(
                {
                    "label": obj["label"],
                    "scene_role": scene_role,
                    "world_pt": obj["world_pt"].tolist(),
                    "bbox_3d": list(obj["bbox_3d"]),
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
            V_final = compute_logsumexp_superposition(
                hazard_fields,
                beta=10.0,
                v_max=100.0,
            )

        # ------------------------------------------------------------------
        # 6) Save Loop 1 outputs
        # ------------------------------------------------------------------
        metadata_json = json.dumps(
            {
                "manipulated_obj": manipulated_obj,
                "camera_type": camera_type,
                "target_label_mode": target_label,
                "detected_objects": per_object_debug,
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
            object_infos=object_infos,
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
    args = parser.parse_args()

    run_pipeline(
        manipulated_obj=args.manipulated,
        camera_type=args.camera,
        target_label=args.scene_label,
        candidate_labels=args.candidate_labels,
    )