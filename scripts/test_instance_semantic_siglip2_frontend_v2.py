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

from semantic_safety.perception_2d3d.mobilesamv2_wrapper_v2 import MobileSAMV2WrapperV2
from semantic_safety.perception_2d3d.siglip2_wrapper_v2 import SigLIP2WrapperV2
from semantic_safety.perception_2d3d.lvis_bank_v2 import (
    LVISBankV2,
    load_aliases_from_json,
    load_custom_labels_from_txt,
)
from semantic_safety.perception_2d3d.siglip2_labeler_v2 import SigLIP2LabelerV2
from semantic_safety.perception_2d3d.instance_semantic_siglip2_frontend_v2 import (
    InstanceSemanticSigLIP2FrontendV2,
)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_image_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Test MobileSAMV2 + SigLIP2 instance semantic frontend."
    )

    # Core IO
    p.add_argument("--image", required=True, help="Path to RGB image.")
    p.add_argument("--out_dir", required=True, help="Output directory.")

    # SigLIP2
    p.add_argument(
        "--siglip_model_name",
        default="google/siglip2-base-patch16-224",
        help="Hugging Face SigLIP2 model name.",
    )
    p.add_argument(
        "--device",
        default=None,
        help="cuda / cpu / mps. Default = auto.",
    )

    # LVIS/custom bank
    p.add_argument(
        "--lvis_json",
        default=None,
        help="Optional path to LVIS annotation json.",
    )
    p.add_argument(
        "--include_synonyms",
        action="store_true",
        help="Include LVIS synonyms when LVIS json is provided.",
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

    # Optional candidate hints
    p.add_argument(
        "--candidate_label",
        action="append",
        default=[],
        help="Optional label hint. Can be repeated.",
    )
    p.add_argument(
        "--candidate_subset_enabled",
        action="store_true",
        help="Restrict SigLIP2 scoring to candidate labels only.",
    )

    # Labeler crop / confidence behavior
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
    p.add_argument(
        "--low_confidence_mode",
        default="drop",
        choices=["unknown", "drop", "keep"],
        help="What to do with low-confidence labels.",
    )
    p.add_argument("--unknown_score_thresh", type=float, default=0.07)
    p.add_argument("--unknown_margin_thresh", type=float, default=0.00)
    p.add_argument("--unknown_label_prefix", type=str, default="unknown_object")

    # Proposal model knobs
    p.add_argument("--sam_size", type=int, default=1024)
    p.add_argument("--yolo_conf", type=float, default=0.40)
    p.add_argument("--yolo_iou", type=float, default=0.90)
    p.add_argument("--min_proposal_mask_size", type=int, default=1000)
    p.add_argument("--max_batch_size", type=int, default=320)

    p.add_argument("--enable_post_merge", action="store_true")
    p.add_argument("--merge_box_iou_thresh", type=float, default=0.85)
    p.add_argument("--merge_mask_iou_thresh", type=float, default=0.80)
    p.add_argument("--merge_containment_thresh", type=float, default=0.90)

    p.add_argument("--enable_geometry_cleanup", action="store_true")
    p.add_argument("--enable_container_suppression", action="store_true")
    p.add_argument("--postprocess_debug", action="store_true")

    p.add_argument("--enable_post_box_score_filter", action="store_true")
    p.add_argument("--post_box_score_thresh", type=float, default=0.80)

    # Frontend-level filtering
    p.add_argument("--min_semantic_score_keep", type=float, default=0.0)
    p.add_argument("--drop_unknown_instances", action="store_true")
    p.add_argument("--enable_same_canonical_dedupe", action="store_true")
    p.add_argument(
        "--dedupe_keep_mode",
        default="semantic_score",
        choices=["semantic_score", "box_score"],
    )
    p.add_argument("--overlay_alpha", type=float, default=0.28)

    # Misc
    p.add_argument("--verbose", action="store_true")

    return p


def _write_summary_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = build_argparser().parse_args()
    _ensure_dir(args.out_dir)

    image_rgb = _load_image_rgb(args.image)

    custom_labels = list(args.custom_label)
    if args.custom_labels_txt:
        custom_labels.extend(load_custom_labels_from_txt(args.custom_labels_txt))

    custom_aliases: dict[str, str] = {}
    if args.custom_aliases_json:
        custom_aliases.update(load_aliases_from_json(args.custom_aliases_json))

    # -----------------------------------------------------------------
    # Build components
    # -----------------------------------------------------------------
    proposal_model = MobileSAMV2WrapperV2(
        sam_size=args.sam_size,
        yolo_conf=args.yolo_conf,
        yolo_iou=args.yolo_iou,
        min_mask_size=args.min_proposal_mask_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        enable_post_merge=bool(args.enable_post_merge),
        merge_box_iou_thresh=args.merge_box_iou_thresh,
        merge_mask_iou_thresh=args.merge_mask_iou_thresh,
        merge_containment_thresh=args.merge_containment_thresh,
        enable_geometry_cleanup=bool(args.enable_geometry_cleanup),
        enable_container_suppression=bool(args.enable_container_suppression),
        postprocess_debug=bool(args.postprocess_debug),
        enable_post_box_score_filter=bool(args.enable_post_box_score_filter),
        post_box_score_thresh=args.post_box_score_thresh,
    )

    siglip = SigLIP2WrapperV2(
        model_name=args.siglip_model_name,
        device=args.device,
    )

    bank = LVISBankV2.build_default(
        lvis_json_path=args.lvis_json,
        include_synonyms=bool(args.include_synonyms),
        custom_labels=custom_labels,
        custom_aliases=custom_aliases,
    )

    labeler = SigLIP2LabelerV2(
        wrapper=siglip,
        label_bank=bank,
        top_k=args.top_k,
        crop_pad_frac=args.crop_pad_frac,
        use_masked_crop=bool(args.use_masked_crop),
        square_crop=bool(args.square_crop),
        min_mask_area_px=args.min_mask_area_px,
        candidate_subset_enabled=bool(args.candidate_subset_enabled),
        canonical_score_reduce=args.canonical_score_reduce,
        low_confidence_mode=args.low_confidence_mode,
        unknown_score_thresh=args.unknown_score_thresh,
        unknown_margin_thresh=args.unknown_margin_thresh,
        unknown_label_prefix=args.unknown_label_prefix,
    )

    frontend = InstanceSemanticSigLIP2FrontendV2(
        proposal_model=proposal_model,
        labeler=labeler,
        min_semantic_score_keep=args.min_semantic_score_keep,
        drop_unknown_instances=bool(args.drop_unknown_instances),
        enable_same_canonical_dedupe=bool(args.enable_same_canonical_dedupe),
        dedupe_keep_mode=args.dedupe_keep_mode,
        overlay_alpha=args.overlay_alpha,
    )

    # -----------------------------------------------------------------
    # Run frontend
    # -----------------------------------------------------------------
    out = frontend.process_image(
        image_rgb=image_rgb,
        candidate_labels=args.candidate_label if len(args.candidate_label) > 0 else None,
        out_dir=args.out_dir,
        save_debug=True,
        top_k=args.top_k,
        verbose=args.verbose,
    )

    object_infos = out["object_infos"]
    timings = out["timings"]

    # Additional top-level summary for convenience
    summary_payload = {
        "image_path": args.image,
        "num_proposals_after_frontend": len(object_infos),
        "timings": timings,
        "bank_info": bank.describe(),
        "label_bank_texts_count": len(out.get("label_bank_texts", [])),
        "canonical_labels_used_count": len(out.get("canonical_labels_used", [])),
        "object_infos": [
            {
                k: (None if k == "mask_2d" else v)
                for k, v in obj.items()
            }
            for obj in object_infos
        ],
    }
    _write_summary_json(
        os.path.join(args.out_dir, "frontend_siglip2_summary.json"),
        summary_payload,
    )

    # Console summary
    print(f"Saved frontend outputs to {args.out_dir}")
    print("Generated debug files:")
    print(" - sam_masks.npy")
    print(" - bboxes_xyxy.npy")
    print(" - box_scores.npy")
    if out.get("raw_similarity", None) is not None:
        print(" - raw_similarity.npy")
    if out.get("canonical_similarity", None) is not None:
        print(" - canonical_similarity.npy")
    if out.get("image_embeddings", None) is not None:
        print(" - image_embeddings.npy")
    print(" - object_infos.json")
    print(" - frontend_siglip2_overlay.png")
    print(" - frontend_siglip2_summary.json")
    print()

    print("Per-instance summary:")
    print(
        "  idx | label                           | sem_score | margin   | box_score | mask_area | bbox_xyxy"
    )
    print(
        "  ----+---------------------------------+-----------+----------+-----------+-----------+---------------------------"
    )
    for obj in object_infos:
        idx = int(obj["instance_index"])
        label = str(obj["label"])[:31].ljust(31)
        sem_score = float(obj["score"])
        margin = float(obj["score_margin"])
        box_score = obj.get("box_score", None)
        box_score_str = "nan" if box_score is None else f"{float(box_score):.3f}"
        area = int(obj["mask_area_px"])
        bbox = obj["bbox_xyxy"]
        print(
            f"  {idx:>3d} | {label} | "
            f"{sem_score:>9.3f} | {margin:>8.3f} | {box_score_str:>9} | "
            f"{area:>9d} | {bbox}"
        )

    print()
    print("Execution complete.")


if __name__ == "__main__":
    main()