import argparse
import time
import json

import cv2
import mujoco
import mujoco.viewer
import numpy as np

from semantic_safety.perception_2d3d.mujoco_camera import MujocoCamera
from semantic_safety.perception_2d3d.lang_sam_wrapper import SemanticPerception
from semantic_safety.perception_2d3d.transform import WorldTransform
from semantic_safety.semantic_router.router import SemanticRouter
from semantic_safety.metric_propagation.fmm_distance import WorkspaceGrid
from semantic_safety.risk_field.superposition import (
    shielding_ratio,
    compute_logsumexp_superposition,
)
from semantic_safety.risk_field.templates import build_risk_field_from_params
from semantic_safety.phase0_dataset.prompts import GEMINI_SYSTEM_INSTRUCTION

# Reuse helpers from your current script in the same scripts/ directory.
from test_full_pipeline import (
    get_mujoco_table_top_z,
    build_global_workspace_bounds,
    segmentation_mask_to_world_points,
    world_points_to_voxel_mask,
    extract_top_surface_mask,
    voxel_mask_to_bbox,
    build_multi_object_occupancy,
    gravity_column_from_surface_mask,
    canonicalize_label_for_filter,
    dedupe_object_infos_by_canonical_label_and_geometry,
)


def role_rgba(scene_role: str) -> np.ndarray:
    if scene_role == "hazard_target":
        return np.array([1.0, 0.1, 0.1, 0.9], dtype=np.float32)
    if scene_role == "safe_receptacle":
        return np.array([0.1, 0.4, 1.0, 0.8], dtype=np.float32)
    if scene_role == "support_context":
        return np.array([0.6, 0.6, 0.6, 0.6], dtype=np.float32)
    return np.array([1.0, 1.0, 1.0, 0.7], dtype=np.float32)


def risk_rgba(value: float, vmin: float, vmax: float) -> np.ndarray:
    if vmax <= vmin:
        t = 1.0
    else:
        t = float(np.clip((value - vmin) / (vmax - vmin), 0.0, 1.0))

    # blue -> green -> yellow -> red
    if t < 0.33:
        u = t / 0.33
        rgba = np.array([0.0, u, 1.0 - u, 0.35 + 0.25 * t], dtype=np.float32)
    elif t < 0.66:
        u = (t - 0.33) / 0.33
        rgba = np.array([u, 1.0, 0.0, 0.45 + 0.25 * t], dtype=np.float32)
    else:
        u = (t - 0.66) / 0.34
        rgba = np.array([1.0, 1.0 - u, 0.0, 0.55 + 0.30 * t], dtype=np.float32)

    rgba[3] = float(np.clip(rgba[3], 0.2, 0.9))
    return rgba


def sample_risk_points(
    grid: WorkspaceGrid,
    V_final: np.ndarray,
    threshold: float = 5.0,
    max_points: int = 1200,
) -> tuple[np.ndarray, np.ndarray]:
    mask = V_final > threshold
    idx = np.argwhere(mask)
    if len(idx) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    vals = V_final[mask]
    if len(vals) > max_points:
        top_idx = np.argpartition(vals, -max_points)[-max_points:]
        idx = idx[top_idx]
        vals = vals[top_idx]

    pts = np.stack(
        [
            grid.x[idx[:, 0]],
            grid.y[idx[:, 1]],
            grid.z[idx[:, 2]],
        ],
        axis=1,
    )
    return pts, vals


def add_sphere_geom(scene, geom_index: int, pos, radius: float, rgba: np.ndarray) -> int:
    mujoco.mjv_initGeom(
        scene.geoms[geom_index],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([radius, 0.0, 0.0], dtype=np.float64),
        pos=np.asarray(pos, dtype=np.float64),
        mat=np.eye(3).reshape(-1),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    return geom_index + 1


def render_user_scene(
    viewer,
    grid: WorkspaceGrid,
    V_final: np.ndarray,
    object_infos: list[dict],
    risk_threshold: float,
    max_geoms: int,
    voxel_radius: float,
    show_object_markers: bool = True,
):
    viewer.user_scn.ngeom = 0
    i = 0
    geom_cap = getattr(viewer.user_scn, "maxgeom", len(viewer.user_scn.geoms))

    # 1) object centroids / roles
    if show_object_markers:
        for obj in object_infos:
            if i >= geom_cap:
                break
            pos = obj["world_pt"]
            rgba = role_rgba(obj.get("scene_role", "neutral_context"))
            i = add_sphere_geom(viewer.user_scn, i, pos, 0.02, rgba)

    # 2) risk field points
    risk_pts, risk_vals = sample_risk_points(
        grid=grid,
        V_final=V_final,
        threshold=risk_threshold,
        max_points=max(0, min(max_geoms, geom_cap - i)),
    )

    if len(risk_vals) > 0:
        vmax = float(np.max(risk_vals))
        vmin = float(np.min(risk_vals))
        for pos, val in zip(risk_pts, risk_vals):
            if i >= geom_cap:
                break
            rgba = risk_rgba(float(val), vmin=vmin, vmax=vmax)
            i = add_sphere_geom(viewer.user_scn, i, pos, voxel_radius, rgba)

    viewer.user_scn.ngeom = i


def compute_loop1_state(
    manipulated_obj: str,
    target_label: str,
    candidate_labels: list[str] | None,
):
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

    cv2.imwrite("test_rgb.png", bgr_image)

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
        raise RuntimeError("No scene detections found.")

    router = SemanticRouter(
        json_path="data/semantic_risk_demo.json",
        system_instruction=GEMINI_SYSTEM_INSTRUCTION,
        llm_batch_callback=None,
        persist_updates=False,
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

        print("\n--- DETECTED SCENE CANDIDATES ---")
        for det in detections:
            print(f"- {det['label']} ({det['score']:.3f}) box={det['box']}")

        world_engine = WorldTransform(world_pos, world_mat)
        object_infos = []

        for det in detections:
            box = det["box"]
            mask_2d = det.get("mask", None)

            if mask_2d is None:
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
                continue

            span = world_points.max(axis=0) - world_points.min(axis=0)
            if span[0] > 2.0 or span[1] > 2.0 or span[2] > 1.0:
                continue

            object_infos.append(
                {
                    "label": det["label"],
                    "score": float(det["score"]),
                    "box_2d": box,
                    "mask_2d": mask_2d,
                    "world_points": world_points,
                    "world_pt": np.mean(world_points, axis=0),
                    "source": det.get("source", "proposal"),
                    "top_k": det.get("top_k", []),
                }
            )

        if not object_infos:
            raise RuntimeError("No valid 3D-localized objects remained.")

        print("\n--- 3D LOCALIZATION ---")
        for obj in object_infos:
            print(
                f"- {obj['label']}: num_pts={len(obj['world_points'])}, "
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
                continue

            obj["object_mask"] = object_mask
            obj["surface_mask"] = extract_top_surface_mask(object_mask, band_voxels=1)
            obj["bbox_3d"] = voxel_mask_to_bbox(grid, object_mask)
            object_masks.append(object_mask)

        object_infos = [obj for obj in object_infos if "object_mask" in obj]
        if not object_infos:
            raise RuntimeError("No valid voxelized objects remained.")

        occupancy_free = build_multi_object_occupancy(
            grid=grid,
            object_masks=[obj["object_mask"] for obj in object_infos],
            table_top_z=table_top_z,
        )

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

            obj["V_object"] = V_object
            obj["d_euc"] = d_euc
            obj["d_geo"] = d_geo
            obj["A_field"] = A_field
            obj["seed_mask"] = seed_mask

            print(f"\n[{obj['label']}]")
            print(f"  topology            = {risk_params.get('topology_template')}")
            print(f"  vertical_rule       = {risk_params.get('vertical_rule')}")
            print(f"  scene_role          = {scene_role}")
            print(f"  attenuation         = {risk_params.get('receptacle_attenuation')}")
            print(f"  weights             = {risk_params.get('weights')}")

            if scene_role == "hazard_target":
                hazard_fields.append(V_object)

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
            V_final = np.zeros(grid.shape, dtype=np.float32)
        else:
            V_final = compute_logsumexp_superposition(
                hazard_fields,
                beta=10.0,
                v_max=100.0,
            )

        metadata_json = json.dumps(
            {
                "manipulated_obj": manipulated_obj,
                "camera_type": "Mujoco",
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
            table_top_z=np.array([table_top_z], dtype=np.float32),
            metadata_json=np.array(metadata_json),
        )
        print("\nSaved core Loop 1 output to 'loop1_risk_field.npz'.")

        return {
            "model": model,
            "data": data,
            "grid": grid,
            "V_final": V_final,
            "object_infos": object_infos,
            "table_top_z": table_top_z,
        }

    finally:
        router.close()


def launch_viewer_with_risk(
    state: dict,
    risk_threshold: float = 5.0,
    max_geoms: int = 1200,
    voxel_radius: float = 0.009,
    fixed_cam_name: str | None = None,
):
    model = state["model"]
    data = state["data"]
    grid = state["grid"]
    V_final = state["V_final"]
    object_infos = state["object_infos"]

    with mujoco.viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=True) as viewer:
        with viewer.lock():
            if fixed_cam_name is not None:
                cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, fixed_cam_name)
                if cam_id >= 0:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    viewer.cam.fixedcamid = cam_id
            else:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.08])
                viewer.cam.distance = 1.1
                viewer.cam.azimuth = 90.0
                viewer.cam.elevation = -25.0

        print("\nViewer controls:")
        print("  - Close window to exit")
        print("  - Use mouse to orbit / zoom")
        print("  - Risk voxels are decorative spheres in user_scn")
        print("  - Small colored spheres at object centers show scene_role")

        while viewer.is_running():
            with viewer.lock():
                render_user_scene(
                    viewer=viewer,
                    grid=grid,
                    V_final=V_final,
                    object_infos=object_infos,
                    risk_threshold=risk_threshold,
                    max_geoms=max_geoms,
                    voxel_radius=voxel_radius,
                    show_object_markers=True,
                )
            viewer.sync()
            time.sleep(0.03)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manipulated", type=str, default="cup of water")
    parser.add_argument("--scene-label", type=str, default="auto")
    parser.add_argument("--candidate-labels", nargs="*", default=None)
    parser.add_argument("--risk-threshold", type=float, default=5.0)
    parser.add_argument("--max-geoms", type=int, default=1200)
    parser.add_argument("--voxel-radius", type=float, default=0.009)
    parser.add_argument(
        "--fixedcam",
        type=str,
        default=None,
        help="Optional MuJoCo camera name, e.g. main_cam",
    )
    args = parser.parse_args()

    state = compute_loop1_state(
        manipulated_obj=args.manipulated,
        target_label=args.scene_label,
        candidate_labels=args.candidate_labels,
    )

    launch_viewer_with_risk(
        state=state,
        risk_threshold=args.risk_threshold,
        max_geoms=args.max_geoms,
        voxel_radius=args.voxel_radius,
        fixed_cam_name=args.fixedcam,
    )


if __name__ == "__main__":
    main()