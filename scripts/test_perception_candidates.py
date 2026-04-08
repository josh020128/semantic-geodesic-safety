import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2
import mujoco
import numpy as np

from semantic_safety.perception_2d3d.mujoco_camera import MujocoCamera
from semantic_safety.perception_2d3d.lang_sam_wrapper import SemanticPerception


DEFAULT_SCENE_CANDIDATES = [
    # tools
    "power drill",
    "drill",
    "screwdriver",
    "hammer",
    "wrench",
    "pliers",
    "scissors",
    "knife",
    "soldering iron",
    # vessels / containers
    "bowl",
    "ceramic bowl",
    "mixing bowl",
    "cup",
    "mug",
    "glass",
    "bottle",
    "plate",
    "tray",
    "sink",
    "bucket",
    "container",
    # electronics
    "laptop",
    "computer",
    "keyboard",
    "monitor",
    "phone",
    "tablet",
    "power strip",
    "charger",
    "cable",
    # scene structure
    "table",
    "shelf",
    "wall",
    "box",
    # extra fragile / household
    "wine glass",
    "vase",
    "pot",
    "pan",
    "kettle",
]


DRAW_COLORS = [
    (0, 0, 255),      # red
    (255, 0, 0),      # blue
    (0, 255, 0),      # green
    (0, 255, 255),    # yellow
    (255, 0, 255),    # magenta
    (255, 255, 0),    # cyan
    (128, 0, 255),
    (255, 128, 0),
    (0, 128, 255),
    (128, 255, 0),
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sanitize_filename(text: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in text)


def chunk_list(items: Sequence[str], chunk_size: int) -> List[List[str]]:
    return [list(items[i:i + chunk_size]) for i in range(0, len(items), chunk_size)]


def render_rgb_from_mujoco(
    xml_path: str,
    camera_name: str = "main_cam",
    width: int = 640,
    height: int = 480,
) -> np.ndarray:
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    for _ in range(50):
        mujoco.mj_step(model, data)

    camera = MujocoCamera(model, data, cam_name=camera_name, width=width, height=height)
    color_image, _, _ = camera.get_frames()
    bgr_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    return bgr_image


def save_image(path: str, bgr_image: np.ndarray) -> None:
    cv2.imwrite(path, bgr_image)


def draw_candidate_overlay(
    image_bgr: np.ndarray,
    detections: List[Dict[str, Any]],
    outpath: str,
) -> None:
    canvas = image_bgr.copy()

    for idx, det in enumerate(detections):
        color = DRAW_COLORS[idx % len(DRAW_COLORS)]

        label = det.get("label", "unknown")
        score = float(det.get("score", 0.0))
        box = det.get("box", {})

        xmin = int(box.get("xmin", 0))
        ymin = int(box.get("ymin", 0))
        xmax = int(box.get("xmax", 0))
        ymax = int(box.get("ymax", 0))

        cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), color, 2)

        text = f"{label}: {score:.2f}"
        y_text = max(20, ymin - 8)
        cv2.putText(
            canvas,
            text,
            (xmin, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            lineType=cv2.LINE_AA,
        )

    cv2.imwrite(outpath, canvas)


def dedupe_by_label_keep_best(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for det in detections:
        label = str(det.get("label", "")).strip().lower()
        prev = best.get(label)
        if prev is None or float(det.get("score", 0.0)) > float(prev.get("score", 0.0)):
            best[label] = det
    return sorted(best.values(), key=lambda d: float(d.get("score", 0.0)), reverse=True)


def run_detection_with_fallback(
    detector: SemanticPerception,
    image_path: str,
    candidate_labels: Sequence[str],
    threshold: float,
    chunk_size: int,
) -> List[Dict[str, Any]]:
    """
    Works with:
    1) new wrapper API: detect_scene_objects(...)
    2) old wrapper API only: detect_objects(image_path, labels)
    """
    if hasattr(detector, "detect_scene_objects"):
        try:
            results = detector.detect_scene_objects(
                image_path=image_path,
                candidate_labels=candidate_labels,
                max_results=50,
                save_debug=True,
                debug_dir="perception_debug",
            )
            return results
        except TypeError as e:
            print(f"detect_scene_objects raised a TypeError during execution: {e}")

    print("Wrapper does not expose compatible detect_scene_objects(...). Falling back to detect_objects(...) batches.")
    all_results: List[Dict[str, Any]] = []
    for group in chunk_list(candidate_labels, chunk_size):
        batch_results = detector.detect_objects(image_path, group)
        all_results.extend(batch_results)
    return all_results


def print_summary(detections: List[Dict[str, Any]], title: str) -> None:
    print(f"\n=== {title} ===")
    if not detections:
        print("(no detections)")
        return

    for i, det in enumerate(detections, start=1):
        label = det.get("label", "unknown")
        score = float(det.get("score", 0.0))
        box = det.get("box", {})
        print(
            f"[{i:02d}] label={label:<18} "
            f"score={score:.3f} "
            f"box=({box.get('xmin')}, {box.get('ymin')}, {box.get('xmax')}, {box.get('ymax')})"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, default=None, help="MuJoCo XML path to render from")
    parser.add_argument("--image", type=str, default=None, help="Existing RGB image path")
    parser.add_argument("--camera-name", type=str, default="main_cam")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--threshold", type=float, default=0.12)
    parser.add_argument("--chunk-size", type=int, default=12)
    parser.add_argument("--outdir", type=str, default="perception_debug")
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional custom candidate labels. If omitted, uses DEFAULT_SCENE_CANDIDATES.",
    )
    args = parser.parse_args()

    ensure_dir(args.outdir)

    if args.image is None and args.xml is None:
        raise ValueError("Provide either --image or --xml.")

    if args.image is not None:
        image_path = args.image
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise ValueError(f"Failed to read image at: {image_path}")
    else:
        image_bgr = render_rgb_from_mujoco(
            xml_path=args.xml,
            camera_name=args.camera_name,
            width=args.width,
            height=args.height,
        )
        image_path = os.path.join(args.outdir, "rendered_scene.png")
        save_image(image_path, image_bgr)

    candidate_labels = list(args.labels) if args.labels else list(DEFAULT_SCENE_CANDIDATES)

    print("\n--- TEST PERCEPTION CANDIDATES ---")
    print(f"Image path: {image_path}")
    print(f"Candidate label count: {len(candidate_labels)}")
    print(f"Threshold: {args.threshold}")
    print(f"Chunk size: {args.chunk_size}")

    detector = SemanticPerception(
        grounding_box_threshold=args.threshold,
        grounding_text_threshold=0.05,
    )

    print(f"Grounding box threshold: {args.threshold}")
    print("Grounding text threshold: 0.05")

    raw_results = run_detection_with_fallback(
        detector=detector,
        image_path=image_path,
        candidate_labels=candidate_labels,
        threshold=args.threshold,
        chunk_size=args.chunk_size,
    )

    raw_results = sorted(raw_results, key=lambda d: float(d.get("score", 0.0)), reverse=True)
    deduped_results = dedupe_by_label_keep_best(raw_results)

    print_summary(raw_results, "RAW DETECTIONS")
    print_summary(deduped_results, "DEDUPED DETECTIONS")

    raw_json_path = os.path.join(args.outdir, "candidate_detections_raw.json")
    deduped_json_path = os.path.join(args.outdir, "candidate_detections_deduped.json")
    overlay_raw_path = os.path.join(args.outdir, "candidate_overlay_raw.png")
    overlay_deduped_path = os.path.join(args.outdir, "candidate_overlay_deduped.png")

    with open(raw_json_path, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=2, ensure_ascii=False)

    with open(deduped_json_path, "w", encoding="utf-8") as f:
        json.dump(deduped_results, f, indent=2, ensure_ascii=False)

    draw_candidate_overlay(image_bgr, raw_results, overlay_raw_path)
    draw_candidate_overlay(image_bgr, deduped_results, overlay_deduped_path)

    print("\nSaved outputs:")
    print(f"  raw json     : {raw_json_path}")
    print(f"  deduped json : {deduped_json_path}")
    print(f"  raw overlay  : {overlay_raw_path}")
    print(f"  deduped overlay: {overlay_deduped_path}")

    # Quick targeted report for your current scene
    target_names = {"power drill", "drill", "bowl", "ceramic bowl", "mixing bowl"}
    matched_targets = [d for d in deduped_results if str(d.get("label", "")).strip().lower() in target_names]

    print_summary(matched_targets, "TARGETED CHECK (DRILL / BOWL FAMILY)")


if __name__ == "__main__":
    main()