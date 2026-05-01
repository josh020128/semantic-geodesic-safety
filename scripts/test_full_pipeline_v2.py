from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from semantic_safety.perception_2d3d.instance_semantic_frontend_v2 import (
    InstanceSemanticFrontendV2,
)
from semantic_safety.perception_2d3d.lang_sam_wrapper import SemanticPerception


# ----------------------------------------------------------------------
# Global V2 config
# ----------------------------------------------------------------------

V2_CONFIG: dict[str, Any] = {}


# ----------------------------------------------------------------------
# Optional open-vocabulary fallback:
# crop the unknown instance and score candidate labels with Grounding DINO
# ----------------------------------------------------------------------

class CropOpenVocabFallback:
    def __init__(self, crop_padding_px: int = 8):
        self.crop_padding_px = int(crop_padding_px)
        self.detector = SemanticPerception()

    def __call__(
        self,
        rgb_np: np.ndarray,
        det_stub: dict[str, Any],
        candidate_labels: Sequence[str],
    ) -> dict[str, Any] | None:
        if not candidate_labels:
            return None

        mask = det_stub.get("mask", None)
        if mask is None:
            return None

        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None

        h, w = rgb_np.shape[:2]
        xmin = max(0, int(xs.min()) - self.crop_padding_px)
        ymin = max(0, int(ys.min()) - self.crop_padding_px)
        xmax = min(w - 1, int(xs.max()) + self.crop_padding_px)
        ymax = min(h - 1, int(ys.max()) + self.crop_padding_px)

        if xmax <= xmin or ymax <= ymin:
            return None

        crop = rgb_np[ymin:ymax + 1, xmin:xmax + 1].copy()
        crop_pil = Image.fromarray(crop)

        # We deliberately reuse the existing label scorer as a conservative fallback.
        scores = self.detector._score_candidate_labels_on_crop(crop_pil, candidate_labels)
        if not scores:
            return None

        top_k = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:5]
        best_label, best_score = top_k[0]

        return {
            "label": str(best_label),
            "score": float(best_score),
            "source": "open_vocab_crop_fallback",
            "top_k": [[str(k), float(v)] for k, v in top_k],
        }


# ----------------------------------------------------------------------
# Adapter class:
# match the interface expected by the current test_full_pipeline.py
# ----------------------------------------------------------------------

class SemanticPerceptionV2Adapter:
    """
    Drop-in replacement for SemanticPerception used by the current pipeline.
    """

    def __init__(self):
        open_vocab_callback = None
        if not bool(V2_CONFIG.get("disable_open_vocab_fallback", False)):
            open_vocab_callback = CropOpenVocabFallback(
                crop_padding_px=int(V2_CONFIG.get("open_vocab_crop_padding_px", 8))
            )

        self.frontend = InstanceSemanticFrontendV2(
            prior_json_path=V2_CONFIG.get("prior_json_path"),
            prototype_bank_path=V2_CONFIG.get("prototype_bank_path"),
            open_vocab_callback=open_vocab_callback,
            device=V2_CONFIG.get("device"),
            sam_size=int(V2_CONFIG.get("sam_size", 1024)),
            yolo_conf=float(V2_CONFIG.get("yolo_conf", 0.40)),
            yolo_iou=float(V2_CONFIG.get("yolo_iou", 0.90)),
            min_mask_size=int(V2_CONFIG.get("min_mask_size", 1000)),
            max_batch_size=int(V2_CONFIG.get("max_batch_size", 320)),
            enable_post_merge=bool(V2_CONFIG.get("enable_post_merge", False)),
            closed_set_similarity_threshold=float(
                V2_CONFIG.get("closed_set_similarity_threshold", 0.35)
            ),
            dino_target_h=int(V2_CONFIG.get("dino_target_h", 500)),
            unknown_prefix=str(V2_CONFIG.get("unknown_prefix", "unknown_object")),
        )

    def detect_scene_objects(
        self,
        image_path: str,
        candidate_labels: Sequence[str],
        max_proposals: int = 25,
        max_results: int = 30,
        save_debug: bool = False,
        debug_dir: str = "perception_debug_v2",
        include_unknown: bool = True,
    ):
        return self.frontend.detect_scene_objects(
            image_path=image_path,
            candidate_labels=candidate_labels,
            max_proposals=max_proposals,
            max_results=max_results,
            save_debug=save_debug,
            debug_dir=debug_dir,
            include_unknown=include_unknown,
        )


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

def load_base_pipeline_module():
    """
    Load the user's current scripts/test_full_pipeline.py as a module so we can
    monkey-patch SemanticPerception and reuse the rest unchanged.
    """
    path = REPO_ROOT / "scripts" / "test_full_pipeline.py"
    if not path.exists():
        raise FileNotFoundError(f"Could not find base pipeline at: {path}")

    spec = importlib.util.spec_from_file_location("test_full_pipeline_base", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to construct import spec for base pipeline.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    ap = argparse.ArgumentParser()

    # Keep the same primary CLI as the current pipeline
    ap.add_argument("--manipulated", type=str, default="cup of water")
    ap.add_argument("--camera", type=str, default="Mujoco")
    ap.add_argument("--xml-path", type=str, default="tabletop.xml")
    ap.add_argument(
        "--scene-label",
        type=str,
        default="auto",
        help="Use 'auto' to process all detected scene objects, or specify one canonical target label.",
    )
    ap.add_argument(
        "--candidate-labels",
        nargs="*",
        default=None,
        help="Optional candidate label vocabulary for open-vocabulary detection.",
    )
    ap.add_argument(
        "--use-gt-blockers",
        action="store_true",
        help="Optional debug/oracle mode only. Inject specific MuJoCo geoms into occupancy.",
    )
    ap.add_argument(
        "--gt-blocker-geoms",
        nargs="*",
        default=[],
        help="Optional MuJoCo geom names for debug/oracle blocker injection. Missing geoms are skipped.",
    )
    ap.add_argument(
        "--use-table-top-filter",
        action="store_true",
        help="Optional debug filter only. Restrict detections labeled 'table' to the MuJoCo table-top plane.",
    )
    ap.add_argument(
        "--prior-json-path",
        type=str,
        default="/home/gl34/research/semantic-geodesic-safety/data/semantic_risk_demo_claude.json",
        help="Path to semantic prior JSON used for prior-driven candidate filtering.",
    )

    # V2 frontend options
    ap.add_argument("--prototype-bank-path", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--sam-size", type=int, default=1024)
    ap.add_argument("--yolo-conf", type=float, default=0.40)
    ap.add_argument("--yolo-iou", type=float, default=0.90)
    ap.add_argument("--min-mask-size", type=int, default=1000)
    ap.add_argument("--max-batch-size", type=int, default=320)
    ap.add_argument("--enable-post-merge", action="store_true")
    ap.add_argument("--closed-set-similarity-threshold", type=float, default=0.35)
    ap.add_argument("--dino-target-h", type=int, default=500)
    ap.add_argument("--unknown-prefix", type=str, default="unknown_object")
    ap.add_argument("--disable-open-vocab-fallback", action="store_true")
    ap.add_argument("--open-vocab-crop-padding-px", type=int, default=8)

    return ap.parse_args()


def main():
    args = parse_args()

    V2_CONFIG.update(
        {
            "prior_json_path": args.prior_json_path,
            "prototype_bank_path": args.prototype_bank_path,
            "device": args.device,
            "sam_size": args.sam_size,
            "yolo_conf": args.yolo_conf,
            "yolo_iou": args.yolo_iou,
            "min_mask_size": args.min_mask_size,
            "max_batch_size": args.max_batch_size,
            "enable_post_merge": args.enable_post_merge,
            "closed_set_similarity_threshold": args.closed_set_similarity_threshold,
            "dino_target_h": args.dino_target_h,
            "unknown_prefix": args.unknown_prefix,
            "disable_open_vocab_fallback": args.disable_open_vocab_fallback,
            "open_vocab_crop_padding_px": args.open_vocab_crop_padding_px,
        }
    )

    base = load_base_pipeline_module()

    # Monkey-patch only the perception frontend; keep the rest of the current pipeline.
    base.SemanticPerception = SemanticPerceptionV2Adapter

    print("--- RUNNING V2 FRONTEND ON CURRENT PIPELINE BACKEND ---")
    print("Base pipeline module:", str(REPO_ROOT / "scripts" / "test_full_pipeline.py"))
    print("Perception frontend:", "InstanceSemanticFrontendV2")
    print("Prototype bank:", args.prototype_bank_path)
    print("Open-vocab fallback:", "OFF" if args.disable_open_vocab_fallback else "ON")

    base.run_pipeline(
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


if __name__ == "__main__":
    main()
