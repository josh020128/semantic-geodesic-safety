#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

try:
    import torch
except Exception as e:  # pragma: no cover
    torch = None  # type: ignore
    _TORCH_IMPORT_ERROR = e
else:
    _TORCH_IMPORT_ERROR = None

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from semantic_safety.perception_2d3d.mobilesamv2_wrapper_v2 import MobileSAMV2WrapperV2

def _load_rgb(path: str) -> np.ndarray:
    rgb = np.array(Image.open(path).convert("RGB"))
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image at {path}, got shape {rgb.shape}")
    return rgb


def _overlay_masks(rgb: np.ndarray, masks: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    import colorsys

    out = rgb.astype(np.float32).copy()
    if masks.size == 0:
        return rgb.copy()

    n = masks.shape[0]
    colors = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        h = (i * 0.61803398875) % 1.0
        s, v = 0.65, 1.0
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors[i] = (255.0 * r, 255.0 * g, 255.0 * b)

    for i in range(n):
        m = masks[i].astype(bool)
        if not m.any():
            continue
        out[m] = out[m] * (1.0 - alpha) + colors[i] * alpha

    return np.clip(out, 0, 255).astype(np.uint8)


def _draw_boxes(rgb: np.ndarray, boxes: np.ndarray, scores: np.ndarray | None = None) -> np.ndarray:
    img = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(img)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
        draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=3)
        label = f"obj_{i:02d}"
        if scores is not None and i < len(scores):
            label += f" {float(scores[i]):.2f}"
        tx = max(0, x1)
        ty = max(0, y1 - 14)
        draw.text((tx, ty), label, fill=(0, 255, 0))

    return np.array(img)


def main() -> None:
    if torch is None:
        raise SystemExit(
            "PyTorch is required to run test_instance_frontend_v2.py. "
            f"Original import error: {_TORCH_IMPORT_ERROR}"
        )

    ap = argparse.ArgumentParser(description="Phase-1 MobileSAMV2 instance-mask frontend test.")
    ap.add_argument("--image", required=True, help="Path to RGB image (jpg/png)")
    ap.add_argument("--out_dir", required=True, help="Directory to save outputs")
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    ap.add_argument("--sam_size", type=int, default=1024, help="Long edge resize for MobileSAMV2")
    ap.add_argument("--yolo_conf", type=float, default=0.40, help="ObjAwareModel confidence threshold")
    ap.add_argument("--yolo_iou", type=float, default=0.90, help="ObjAwareModel IoU threshold")
    ap.add_argument("--min_mask_size", type=int, default=1000, help="Minimum kept mask area in pixels")
    ap.add_argument("--max_batch_size", type=int, default=320, help="Max batched prompt decoding size")
    ap.add_argument("--overlay_alpha", type=float, default=0.45, help="Mask overlay alpha")
    ap.add_argument("--save_masks", action="store_true", help="Save each mask as a PNG")
    ap.add_argument(
        "--mps_fallback",
        action="store_true",
        help="Enable PYTORCH_ENABLE_MPS_FALLBACK=1 when device is mps.",
    )
    ap.add_argument("--enable_post_merge", action="store_true", help="Enable conservative post-merge of near-duplicate masks")
    ap.add_argument("--merge_box_iou_thresh", type=float, default=0.85, help="Box IoU threshold for optional post-merge")
    ap.add_argument("--merge_mask_iou_thresh", type=float, default=0.80, help="Mask IoU threshold for optional post-merge")
    ap.add_argument("--merge_containment_thresh", type=float, default=0.90, help="Containment threshold for optional post-merge")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "mps" and args.mps_fallback:
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    rgb = _load_rgb(args.image)

    wrapper = MobileSAMV2WrapperV2(
        sam_size=args.sam_size,
        yolo_conf=args.yolo_conf,
        yolo_iou=args.yolo_iou,
        min_mask_size=args.min_mask_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        enable_post_merge=args.enable_post_merge,
        merge_box_iou_thresh=args.merge_box_iou_thresh,
        merge_mask_iou_thresh=args.merge_mask_iou_thresh,
        merge_containment_thresh=args.merge_containment_thresh,
    )

    results = wrapper.process_image(rgb, verbose=True)

    masks_np = results.sam_masks.detach().cpu().to(torch.uint8).numpy()
    boxes_np = results.input_boxes.detach().cpu().to(torch.float32).numpy()
    scores_np = None
    if results.box_scores is not None:
        scores_np = results.box_scores.detach().cpu().to(torch.float32).numpy()

    np.save(out_dir / "sam_masks.npy", masks_np)
    np.save(out_dir / "bboxes.npy", boxes_np)
    if scores_np is not None:
        np.save(out_dir / "box_scores.npy", scores_np)

    overlay = _overlay_masks(rgb, masks_np, alpha=float(args.overlay_alpha))
    Image.fromarray(overlay).save(out_dir / "mask_overlay.png")

    bbox_overlay = _draw_boxes(rgb, boxes_np, scores_np)
    Image.fromarray(bbox_overlay).save(out_dir / "bbox_overlay.png")

    if args.save_masks:
        mask_dir = out_dir / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        for i in range(masks_np.shape[0]):
            Image.fromarray((255 * masks_np[i]).astype(np.uint8)).save(mask_dir / f"obj_{i:03d}.png")

    meta = {
        "image_path": str(args.image),
        "num_masks": int(masks_np.shape[0]),
        "device": wrapper.device,
        "sam_size": int(args.sam_size),
        "yolo_conf": float(args.yolo_conf),
        "yolo_iou": float(args.yolo_iou),
        "min_mask_size": int(args.min_mask_size),
        "processed_shape": list(results.processed_shape) if results.processed_shape is not None else None,
        "original_shape": list(results.original_shape) if results.original_shape is not None else None,
        "timings": {k: float(v) for k, v in results.timings.items()},
        "enable_post_merge": bool(args.enable_post_merge),
        "merge_box_iou_thresh": float(args.merge_box_iou_thresh),
        "merge_mask_iou_thresh": float(args.merge_mask_iou_thresh),
        "merge_containment_thresh": float(args.merge_containment_thresh),
    }
    with open(out_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {masks_np.shape[0]} masks to {out_dir}")
    if masks_np.shape[0] > 0:
        print("\nPer-instance summary:")
        print("  idx | box_score | box_xyxy                  | mask_area_px")
        print("  ----+----------+----------------------------+-------------")
        areas_px = masks_np.reshape(masks_np.shape[0], -1).sum(axis=1).astype(int)
        for i in range(masks_np.shape[0]):
            box_str = "[missing]"
            if i < boxes_np.shape[0]:
                x1, y1, x2, y2 = [float(v) for v in boxes_np[i].tolist()]
                box_str = f"[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]"

            score_str = "n/a"
            if scores_np is not None and i < len(scores_np):
                score_str = f"{float(scores_np[i]):.3f}"

            print(f"  {i:3d} | {score_str:8s} | {box_str:26s} | {int(areas_px[i]):11d}")


if __name__ == "__main__":
    main()
