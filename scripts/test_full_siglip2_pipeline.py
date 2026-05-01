from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


_THIS_FILE = Path(__file__).resolve()
# This file lives in <repo>/scripts/, so repo root is one level up.
_PROJECT_ROOT = _THIS_FILE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


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
from semantic_safety.perception_2d3d.mobilesamv2_wrapper_v2 import MobileSAMV2WrapperV2


def _load_base_pipeline_module():
    """
    Load the original scripts/test_full_pipeline.py exactly as-is.

    We keep all risk-field construction code in the original file untouched and
    only replace the frontend detector class via monkey patching.
    """
    base_path = _PROJECT_ROOT / "scripts" / "test_full_pipeline.py"
    if not base_path.exists():
        raise FileNotFoundError(f"Could not find base pipeline: {base_path}")

    spec = importlib.util.spec_from_file_location("base_test_full_pipeline", str(base_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for {base_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["base_test_full_pipeline"] = module
    spec.loader.exec_module(module)
    return module


def _load_image_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class SemanticPerceptionSigLIP2Adapter:
    """
    Drop-in replacement for lang_sam_wrapper.SemanticPerception.

    Base pipeline expects:
        detect_scene_objects(image_path, candidate_labels, save_debug, debug_dir)

    This adapter preserves that interface and internally runs:
        MobileSAMV2 proposals -> SigLIP2 labeler -> object detections
    """

    def __init__(
        self,
        frontend: InstanceSemanticSigLIP2FrontendV2,
        verbose: bool = False,
    ) -> None:
        self.frontend = frontend
        self.verbose = bool(verbose)

    def detect_scene_objects(
        self,
        image_path: str,
        candidate_labels: Optional[list[str]] = None,
        save_debug: bool = True,
        debug_dir: str = "perception_debug",
    ) -> list[dict[str, Any]]:
        image_rgb = _load_image_rgb(image_path)

        out = self.frontend.process_image(
            image_rgb=image_rgb,
            candidate_labels=candidate_labels,
            out_dir=debug_dir if save_debug else None,
            save_debug=save_debug,
            verbose=self.verbose,
        )

        detections: list[dict[str, Any]] = []
        for obj in out["object_infos"]:
            detections.append(
                {
                    "label": obj["label"],
                    "score": float(obj["score"]),
                    "box": obj["box_2d"],
                    "mask": obj["mask_2d"],
                    "source": "siglip2_frontend",
                    "top_k": obj.get("topk_canonical", []),
                    "raw_top_k": obj.get("topk_raw", []),
                    "score_margin": float(obj.get("score_margin", 0.0)),
                    "box_score": obj.get("box_score", None),
                }
            )

        if self.verbose:
            print("\n[SemanticPerceptionSigLIP2Adapter] detections:")
            for det in detections:
                print(
                    f"  - {det['label']} "
                    f"(sem={det['score']:.3f}, "
                    f"margin={det.get('score_margin', 0.0):.3f}, "
                    f"box_score={det.get('box_score', None)})"
                )

        return detections


class _InjectedSemanticPerception:
    """
    Class object installed into the base pipeline module namespace.

    The base run_pipeline() constructs SemanticPerception() with no arguments,
    so we store the actual frontend on class variables before monkey patching.
    """

    _frontend: InstanceSemanticSigLIP2FrontendV2 | None = None
    _verbose: bool = False

    def __init__(self, *args, **kwargs):
        if self.__class__._frontend is None:
            raise RuntimeError("SigLIP2 frontend was not initialized before detector injection.")
        self._impl = SemanticPerceptionSigLIP2Adapter(
            frontend=self.__class__._frontend,
            verbose=self.__class__._verbose,
        )

    def detect_scene_objects(self, image_path, candidate_labels, save_debug=True, debug_dir="perception_debug"):
        return self._impl.detect_scene_objects(
            image_path=image_path,
            candidate_labels=candidate_labels,
            save_debug=save_debug,
            debug_dir=debug_dir,
        )


def build_siglip2_frontend(args) -> InstanceSemanticSigLIP2FrontendV2:
    custom_labels = list(args.custom_label)
    if args.custom_labels_txt:
        custom_labels.extend(load_custom_labels_from_txt(args.custom_labels_txt))

    custom_aliases = {}
    if args.custom_aliases_json:
        custom_aliases.update(load_aliases_from_json(args.custom_aliases_json))

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

    return InstanceSemanticSigLIP2FrontendV2(
        proposal_model=proposal_model,
        labeler=labeler,
        min_semantic_score_keep=args.min_semantic_score_keep,
        drop_unknown_instances=bool(args.drop_unknown_instances),
        enable_same_canonical_dedupe=bool(args.enable_same_canonical_dedupe),
        dedupe_keep_mode=args.dedupe_keep_mode,
        overlay_alpha=args.overlay_alpha,
    )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run the original test_full_pipeline.py risk-field backend while replacing "
            "only the frontend detector with MobileSAMV2 + SigLIP2 + LVIS/custom labeling."
        )
    )

    # ----------------------------------------------------------
    # Original base pipeline args (kept identical to the pasted run_pipeline)
    # ----------------------------------------------------------
    p.add_argument("--manipulated", type=str, default="cup of water")
    p.add_argument("--camera", type=str, default="Mujoco")
    p.add_argument("--xml-path", type=str, default="tabletop.xml")
    p.add_argument("--scene-label", type=str, default="auto")
    p.add_argument("--candidate-labels", nargs="*", default=None)
    p.add_argument("--use-gt-blockers", action="store_true")
    p.add_argument("--gt-blocker-geoms", nargs="*", default=[])
    p.add_argument("--use-table-top-filter", action="store_true")
    p.add_argument("--prior-json-path", type=str, default=None)

    # ----------------------------------------------------------
    # SigLIP2 frontend args
    # ----------------------------------------------------------
    p.add_argument("--siglip-model-name", default="google/siglip2-base-patch16-224")
    p.add_argument("--device", default=None)
    p.add_argument("--lvis-json", default=None)
    p.add_argument("--include-synonyms", action="store_true")
    p.add_argument("--custom-labels-txt", default=None)
    p.add_argument("--custom-aliases-json", default=None)
    p.add_argument("--custom-label", action="append", default=["power drill", "table", "shelf", "bowl", "mug", "hot soldering iron"])

    p.add_argument("--candidate-subset-enabled", action="store_true")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--crop-pad-frac", type=float, default=0.08)
    p.add_argument("--min-mask-area-px", type=int, default=64)
    p.add_argument("--use-masked-crop", action="store_true")
    p.add_argument("--square-crop", action="store_true")
    p.add_argument("--canonical-score-reduce", default="max", choices=["max", "mean"])

    p.add_argument("--low-confidence-mode", default="drop", choices=["unknown", "drop", "keep"])
    p.add_argument("--unknown-score-thresh", type=float, default=0.07)
    p.add_argument("--unknown-margin-thresh", type=float, default=0.000)
    p.add_argument("--unknown-label-prefix", type=str, default="unknown_object")

    p.add_argument("--sam-size", type=int, default=1024)
    p.add_argument("--yolo-conf", type=float, default=0.40)
    p.add_argument("--yolo-iou", type=float, default=0.90)
    p.add_argument("--min-proposal-mask-size", type=int, default=1000)
    p.add_argument("--max-batch-size", type=int, default=320)

    p.add_argument("--enable-post-merge", action="store_true")
    p.add_argument("--merge-box-iou-thresh", type=float, default=0.85)
    p.add_argument("--merge-mask-iou-thresh", type=float, default=0.80)
    p.add_argument("--merge-containment-thresh", type=float, default=0.90)

    p.add_argument("--enable-geometry-cleanup", action="store_true")
    p.add_argument("--enable-container-suppression", action="store_true")
    p.add_argument("--postprocess-debug", action="store_true")

    p.add_argument("--enable-post-box-score-filter", action="store_true")
    p.add_argument("--post-box-score-thresh", type=float, default=0.80)

    p.add_argument("--min-semantic-score-keep", type=float, default=0.0)
    p.add_argument("--drop-unknown-instances", action="store_true")
    p.add_argument("--enable-same-canonical-dedupe", action="store_true")
    p.add_argument("--dedupe-keep-mode", default="semantic_score", choices=["semantic_score", "box_score"])
    p.add_argument("--overlay-alpha", type=float, default=0.28)

    p.add_argument("--verbose", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    base = _load_base_pipeline_module()
    frontend = build_siglip2_frontend(args)

    # install frontend into the exact global symbol name used by base.run_pipeline()
    _InjectedSemanticPerception._frontend = frontend
    _InjectedSemanticPerception._verbose = bool(args.verbose)
    base.SemanticPerception = _InjectedSemanticPerception

    prior_json_path = args.prior_json_path
    if prior_json_path is None:
        prior_json_path = getattr(base, "PRIOR_JSON_DEFAULT")

    print("--- RUNNING BASE PIPELINE WITH SIGLIP2 FRONTEND ---")
    print(f"Base file      : {getattr(base, '__file__', 'unknown')}")
    print(f"Injected class : {_InjectedSemanticPerception.__name__}")
    print(f"Frontend       : {frontend.__class__.__name__}")
    print("Risk backend   : unchanged (delegated to original test_full_pipeline.py)")

    base.run_pipeline(
        manipulated_obj=args.manipulated,
        camera_type=args.camera,
        target_label=args.scene_label,
        candidate_labels=args.candidate_labels,
        xml_path=args.xml_path,
        use_gt_blockers=bool(args.use_gt_blockers),
        gt_blocker_geoms=args.gt_blocker_geoms,
        use_table_top_filter=bool(args.use_table_top_filter),
        prior_json_path=prior_json_path,
    )


if __name__ == "__main__":
    main()
