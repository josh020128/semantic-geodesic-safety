from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ---------------------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from semantic_safety.perception_2d3d.siglip2_wrapper_v2 import SigLIP2WrapperV2
from semantic_safety.perception_2d3d.lvis_bank_v2 import LVISBankV2, load_aliases_from_json, load_custom_labels_from_txt
from semantic_safety.perception_2d3d.siglip2_labeler_v2 import SigLIP2LabelerV2


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_image_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _load_masks(path: str) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)

    # torch-saved style fallback is unlikely here, but keep it robust
    if hasattr(arr, "numpy"):
        arr = arr.numpy()

    arr = np.asarray(arr)

    if arr.ndim != 3:
        raise ValueError(f"Expected masks with shape [N,H,W], got {arr.shape}")

    return arr.astype(bool)


def _load_boxes(path: str) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)

    if hasattr(arr, "numpy"):
        arr = arr.numpy()

    arr = np.asarray(arr)

    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"Expected boxes with shape [N,4], got {arr.shape}")

    return arr.astype(np.float32)


def _draw_overlay(
    image_rgb: np.ndarray,
    masks: np.ndarray,
    results: list[dict[str, Any]],
    alpha: float = 0.28,
) -> np.ndarray:
    """
    Draw colored masks + bbox + canonical label.
    Saves as BGR-friendly image for cv2.imwrite.
    """
    rgb = image_rgb.copy()
    overlay = rgb.copy()

    # deterministic-ish palette
    palette = [
        (255, 0, 255),   # magenta
        (0, 255, 255),   # cyan
        (0, 255, 0),     # green
        (255, 255, 0),   # yellow
        (255, 128, 0),   # orange-ish
        (128, 255, 0),
        (0, 128, 255),
        (255, 0, 128),
    ]

    for rank, r in enumerate(results):
        idx = int(r["instance_index"])
        if idx < 0 or idx >= masks.shape[0]:
            continue

        mask = masks[idx].astype(bool)
        if mask.sum() == 0:
            continue

        color_rgb = np.array(palette[rank % len(palette)], dtype=np.uint8)
        overlay[mask] = (
            (1.0 - alpha) * overlay[mask].astype(np.float32)
            + alpha * color_rgb.astype(np.float32)
        ).astype(np.uint8)

    rgb = overlay

    for rank, r in enumerate(results):
        idx = int(r["instance_index"])
        if idx < 0 or idx >= masks.shape[0]:
            continue

        mask = (masks[idx].astype(np.uint8) * 255)
        if mask.sum() == 0:
            continue

        color_rgb = tuple(int(x) for x in palette[rank % len(palette)])

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, contours, -1, color_rgb, 2)

        x1, y1, x2, y2 = [int(v) for v in r["crop_box_xyxy"]]
        cv2.rectangle(rgb, (x1, y1), (x2, y2), color_rgb, 2)

        label = str(r["canonical_label"])
        score = float(r["score"])
        text = f"{label} ({score:.3f})"
        cv2.putText(
            rgb,
            text,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color_rgb,
            2,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _results_to_jsonable(results: list[Any]) -> list[dict[str, Any]]:
    out = []
    for r in results:
        if hasattr(r, "to_dict"):
            out.append(r.to_dict())
        elif isinstance(r, dict):
            out.append(r)
        else:
            raise TypeError(f"Unsupported result type: {type(r)}")
    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Test SigLIP2 + LVIS/custom labeler on instance masks.")
    p.add_argument("--image", required=True, help="Path to RGB image.")
    p.add_argument("--masks", required=True, help="Path to masks .npy [N,H,W].")
    p.add_argument("--bboxes", required=True, help="Path to bboxes .npy [N,4].")
    p.add_argument("--out_dir", required=True, help="Output directory.")

    p.add_argument(
        "--model_name",
        default="google/siglip2-base-patch16-224",
        help="Hugging Face SigLIP2 model name.",
    )
    p.add_argument("--device", default=None, help="cuda / cpu / mps. Default = auto.")
    p.add_argument(
        "--lvis_json",
        default=None,
        help="Optional path to LVIS annotation json. If omitted, uses custom bank only.",
    )
    p.add_argument(
        "--custom_labels_txt",
        default=None,
        help="Optional txt file with one custom label per line.",
    )
    p.add_argument(
        "--custom_aliases_json",
        default=None,
        help="Optional json dict mapping alias -> canonical label.",
    )
    p.add_argument(
        "--custom_label",
        action="append",
        default=[],
        help="Extra custom label. Can be repeated.",
    )
    p.add_argument(
        "--include_synonyms",
        action="store_true",
        help="Include LVIS synonyms when LVIS json is provided.",
    )
    p.add_argument(
        "--candidate_label",
        action="append",
        default=[],
        help="Optional candidate label hint. Can be repeated.",
    )
    p.add_argument(
        "--candidate_subset_enabled",
        action="store_true",
        help="If set, use candidate_label subset instead of the full bank.",
    )
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--crop_pad_frac", type=float, default=0.08)
    p.add_argument("--min_mask_area_px", type=int, default=64)
    p.add_argument(
        "--use_masked_crop",
        action="store_true",
        help="Use masked crop instead of raw bbox crop.",
    )
    p.add_argument(
        "--square_crop",
        action="store_true",
        help="Force square crop around each instance.",
    )
    p.add_argument(
        "--canonical_score_reduce",
        default="max",
        choices=["max", "mean"],
        help="How to aggregate alias scores to canonical label scores.",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    _ensure_dir(args.out_dir)

    image_rgb = _load_image_rgb(args.image)
    masks = _load_masks(args.masks)
    boxes = _load_boxes(args.bboxes)

    if masks.shape[0] != boxes.shape[0]:
        raise ValueError(
            f"Mask/box count mismatch: masks={masks.shape[0]}, boxes={boxes.shape[0]}"
        )

    custom_labels = list(args.custom_label)

    if args.custom_labels_txt:
        custom_labels.extend(load_custom_labels_from_txt(args.custom_labels_txt))

    custom_aliases = {}
    if args.custom_aliases_json:
        custom_aliases.update(load_aliases_from_json(args.custom_aliases_json))

    bank = LVISBankV2.build_default(
        lvis_json_path=args.lvis_json,
        include_synonyms=bool(args.include_synonyms),
        custom_labels=custom_labels,
        custom_aliases=custom_aliases,
    )

    wrapper = SigLIP2WrapperV2(
        model_name=args.model_name,
        device=args.device,
    )

    labeler = SigLIP2LabelerV2(
        wrapper=wrapper,
        label_bank=bank,
        top_k=args.top_k,
        crop_pad_frac=args.crop_pad_frac,
        use_masked_crop=bool(args.use_masked_crop),
        square_crop=bool(args.square_crop),
        min_mask_area_px=args.min_mask_area_px,
        candidate_subset_enabled=bool(args.candidate_subset_enabled),
        canonical_score_reduce=args.canonical_score_reduce,
    )

    out = labeler.label_instances(
        image_rgb=image_rgb,
        masks=masks,
        boxes_xyxy=boxes,
        candidate_labels=args.candidate_label if len(args.candidate_label) > 0 else None,
        top_k=args.top_k,
        return_debug_tensors=True,
    )

    results_json = _results_to_jsonable(out["results"])

    # Save JSON
    json_path = os.path.join(args.out_dir, "siglip2_label_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": args.model_name,
                "bank_info": bank.describe(),
                "text_bank_labels_used": out["text_bank_labels_used"],
                "canonical_labels_used": out["canonical_labels_used"],
                "results": results_json,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Save debug arrays
    np.save(os.path.join(args.out_dir, "raw_similarity.npy"), out["raw_similarity"])
    np.save(os.path.join(args.out_dir, "canonical_similarity.npy"), out["canonical_similarity"])
    np.save(os.path.join(args.out_dir, "image_embeddings.npy"), out["image_embeddings"])

    # Save bank debug
    bank.save_debug_json(os.path.join(args.out_dir, "label_bank_debug.json"))
    bank.save_text_bank_txt(os.path.join(args.out_dir, "text_bank_labels.txt"), canonical_only=False)
    bank.save_text_bank_txt(os.path.join(args.out_dir, "canonical_labels.txt"), canonical_only=True)

    # Save overlay
    overlay_bgr = _draw_overlay(image_rgb, masks, results_json, alpha=0.28)
    overlay_path = os.path.join(args.out_dir, "siglip2_label_overlay.png")
    cv2.imwrite(overlay_path, overlay_bgr)

    # Console summary
    print(f"Saved SigLIP2 labeling outputs to {args.out_dir}")
    print(f" - siglip2_label_results.json")
    print(f" - raw_similarity.npy: {out['raw_similarity'].shape}")
    print(f" - canonical_similarity.npy: {out['canonical_similarity'].shape}")
    print(f" - image_embeddings.npy: {out['image_embeddings'].shape}")
    print(f" - siglip2_label_overlay.png")
    print()

    print("Per-instance summary:")
    print("  idx | mask_area | canonical_label                 | score   | margin  | top3")
    print("  ----+-----------+---------------------------------+---------+---------+------------------------------")
    for r in results_json:
        idx = int(r["instance_index"])
        area = int(r["mask_area_px"])
        label = str(r["canonical_label"])[:31].ljust(31)
        score = float(r["score"])
        margin = float(r["score_margin"])
        top3 = ", ".join([f"{lbl}:{s:.3f}" for lbl, s in r["topk_canonical"][:3]])
        print(
            f"  {idx:>3d} | {area:>9d} | {label} | "
            f"{score:>7.3f} | {margin:>7.3f} | {top3}"
        )


if __name__ == "__main__":
    main()