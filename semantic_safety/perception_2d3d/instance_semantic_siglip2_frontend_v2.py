from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence

import cv2
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from semantic_safety.perception_2d3d.mobilesamv2_wrapper_v2 import MobileSAMV2WrapperV2
from semantic_safety.perception_2d3d.siglip2_labeler_v2 import SigLIP2LabelerV2

@dataclass
class FrontendInstanceV2:
    instance_index: int
    label: str
    canonical_label: str
    score: float
    score_margin: float
    box_score: float | None
    box_2d: dict[str, int]
    bbox_xyxy: list[int]
    crop_box_xyxy: list[int]
    mask_area_px: int
    topk_canonical: list[tuple[str, float]]
    topk_raw: list[tuple[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class InstanceSemanticSigLIP2FrontendV2:
    """
    Instance frontend:
      RGB image
        -> MobileSAMV2 proposals (masks + boxes)
        -> SigLIP2 label assignment over LVIS/custom vocabulary
        -> object_infos compatible with downstream 3D/risk pipeline

    Design goals
    ------------
    - Keep proposal generation unchanged from the current MobileSAMV2 frontend
    - Reduce dependence on prior-JSON candidate labels
    - Preserve enough debug info to inspect segmentation vs labeling failures
    """

    def __init__(
        self,
        proposal_model: MobileSAMV2WrapperV2,
        labeler: SigLIP2LabelerV2,
        *,
        min_semantic_score_keep: float = 0.0,
        drop_unknown_instances: bool = False,
        enable_same_canonical_dedupe: bool = False,
        dedupe_keep_mode: str = "semantic_score",  # semantic_score | box_score
        overlay_alpha: float = 0.28,
    ) -> None:
        self.proposal_model = proposal_model
        self.labeler = labeler
        self.min_semantic_score_keep = float(min_semantic_score_keep)
        self.drop_unknown_instances = bool(drop_unknown_instances)
        self.enable_same_canonical_dedupe = bool(enable_same_canonical_dedupe)
        self.dedupe_keep_mode = str(dedupe_keep_mode).lower()
        self.overlay_alpha = float(overlay_alpha)

        if self.dedupe_keep_mode not in {"semantic_score", "box_score"}:
            raise ValueError("dedupe_keep_mode must be one of {'semantic_score', 'box_score'}")

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def process_image(
        self,
        image_rgb: np.ndarray,
        *,
        candidate_labels: Optional[Sequence[str]] = None,
        out_dir: Optional[str] = None,
        save_debug: bool = True,
        top_k: Optional[int] = None,
        verbose: bool = False,
    ) -> dict[str, Any]:
        t_total = time.time()

        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB image [H,W,3], got {image_rgb.shape}")

        # --------------------------------------------------------------
        # 1) Proposal generation
        # --------------------------------------------------------------
        t0 = time.time()
        proposal_out: MobileSAMV2Results = self.proposal_model.process_image(image_rgb, verbose=verbose)
        proposal_time = time.time() - t0

        sam_masks = self._to_numpy_masks(proposal_out.sam_masks)
        bboxes_xyxy = self._to_numpy_boxes(proposal_out.input_boxes)
        box_scores = self._to_numpy_scores(proposal_out.box_scores, n_expected=sam_masks.shape[0])

        # --------------------------------------------------------------
        # 2) SigLIP2 labeling
        # --------------------------------------------------------------
        t0 = time.time()
        label_out = self.labeler.label_instances(
            image_rgb=image_rgb,
            masks=sam_masks,
            boxes_xyxy=bboxes_xyxy,
            candidate_labels=candidate_labels,
            top_k=top_k,
            return_debug_tensors=True,
        )
        labeling_time = time.time() - t0

        label_results = self._normalize_label_results(label_out["results"])

        # --------------------------------------------------------------
        # 3) Join proposal + labels -> object_infos
        # --------------------------------------------------------------
        object_infos = self._build_object_infos(
            sam_masks=sam_masks,
            bboxes_xyxy=bboxes_xyxy,
            box_scores=box_scores,
            label_results=label_results,
        )

        object_infos = self._apply_instance_filters(object_infos)

        if self.enable_same_canonical_dedupe:
            object_infos = self._dedupe_same_canonical(object_infos)

        timings = {
            "proposal_time": float(proposal_time),
            "labeling_time": float(labeling_time),
            "total_time": float(time.time() - t_total),
        }

        result = {
            "object_infos": object_infos,
            "sam_masks": sam_masks,
            "bboxes_xyxy": bboxes_xyxy,
            "box_scores": box_scores,
            "label_results": label_results,
            "timings": timings,
            "label_bank_texts": label_out.get("text_bank_labels_used", []),
            "canonical_labels_used": label_out.get("canonical_labels_used", []),
            "raw_similarity": label_out.get("raw_similarity", None),
            "canonical_similarity": label_out.get("canonical_similarity", None),
            "image_embeddings": label_out.get("image_embeddings", None),
        }

        if out_dir is not None and save_debug:
            self._save_debug_artifacts(
                out_dir=out_dir,
                image_rgb=image_rgb,
                result=result,
            )

        if verbose:
            print("\n[InstanceSemanticSigLIP2FrontendV2]")
            print(f"  proposals kept : {sam_masks.shape[0]}")
            print(f"  labeled objs   : {len(object_infos)}")
            print(f"  proposal time  : {proposal_time:.3f}s")
            print(f"  labeling time  : {labeling_time:.3f}s")
            print(f"  total time     : {timings['total_time']:.3f}s")

        return result

    __call__ = process_image

    # ------------------------------------------------------------------
    # Internal conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_numpy_masks(masks: Any) -> np.ndarray:
        if torch is not None and isinstance(masks, torch.Tensor):
            masks = masks.detach().cpu().numpy()
        masks = np.asarray(masks)
        if masks.ndim != 3:
            raise ValueError(f"Expected masks [N,H,W], got {masks.shape}")
        return masks.astype(bool)

    @staticmethod
    def _to_numpy_boxes(boxes: Any) -> np.ndarray:
        if torch is not None and isinstance(boxes, torch.Tensor):
            boxes = boxes.detach().cpu().numpy()
        boxes = np.asarray(boxes)
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError(f"Expected boxes [N,4], got {boxes.shape}")
        return boxes.astype(np.float32)

    @staticmethod
    def _to_numpy_scores(scores: Any, n_expected: int) -> np.ndarray:
        if scores is None:
            return np.full((n_expected,), np.nan, dtype=np.float32)
        if torch is not None and isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        scores = np.asarray(scores).reshape(-1)
        if scores.shape[0] != n_expected:
            raise ValueError(
                f"Score count mismatch: got {scores.shape[0]} scores for {n_expected} proposals"
            )
        return scores.astype(np.float32)

    @staticmethod
    def _normalize_label_results(results: list[Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for r in results:
            if hasattr(r, "to_dict"):
                out.append(r.to_dict())
            elif isinstance(r, dict):
                out.append(r)
            else:
                raise TypeError(f"Unsupported label result type: {type(r)}")
        return out

    @staticmethod
    def _xyxy_to_box_dict(box: Sequence[float]) -> dict[str, int]:
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        return {
            "xmin": x1,
            "ymin": y1,
            "xmax": x2,
            "ymax": y2,
        }

    # ------------------------------------------------------------------
    # Object-info construction
    # ------------------------------------------------------------------

    def _build_object_infos(
        self,
        *,
        sam_masks: np.ndarray,
        bboxes_xyxy: np.ndarray,
        box_scores: np.ndarray,
        label_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        label_by_index = {int(r["instance_index"]): r for r in label_results}

        object_infos: list[dict[str, Any]] = []
        for i in range(sam_masks.shape[0]):
            if i not in label_by_index:
                # Unlabeled small/filtered instance: skip
                continue

            r = label_by_index[i]
            bbox_xyxy = [int(round(float(v))) for v in bboxes_xyxy[i].tolist()]
            mask = sam_masks[i].astype(bool)

            label = str(r["canonical_label"])
            score = float(r["score"])
            margin = float(r["score_margin"])
            box_score = None if np.isnan(box_scores[i]) else float(box_scores[i])

            object_infos.append(
                {
                    "instance_index": int(i),
                    "label": label,
                    "canonical_label": str(r["canonical_label"]),
                    "raw_label": str(r["label"]),
                    "score": score,
                    "score_margin": margin,
                    "box_score": box_score,
                    "box_2d": self._xyxy_to_box_dict(bbox_xyxy),
                    "bbox_xyxy": bbox_xyxy,
                    "crop_box_xyxy": [int(v) for v in r["crop_box_xyxy"]],
                    "mask_2d": mask,
                    "mask_area_px": int(r["mask_area_px"]),
                    "topk_canonical": list(r.get("topk_canonical", [])),
                    "topk_raw": list(r.get("topk_raw", [])),
                }
            )

        return object_infos

    def _apply_instance_filters(self, object_infos: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []

        for obj in object_infos:
            score = float(obj["score"])
            label = str(obj["label"])

            if score < self.min_semantic_score_keep:
                continue

            if self.drop_unknown_instances and label.startswith("unknown_object"):
                continue

            out.append(obj)

        return out

    def _dedupe_same_canonical(self, object_infos: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Keep only one instance per canonical label.

        Useful as an optional debugging mode, but disabled by default because
        downstream 3D canonical dedupe already exists in the main pipeline.
        """
        grouped: dict[str, list[dict[str, Any]]] = {}
        for obj in object_infos:
            canon = str(obj["canonical_label"])
            grouped.setdefault(canon, []).append(obj)

        kept: list[dict[str, Any]] = []
        for canon, group in grouped.items():
            if len(group) == 1:
                kept.append(group[0])
                continue

            if self.dedupe_keep_mode == "box_score":
                def priority(x: dict[str, Any]) -> float:
                    bs = x.get("box_score", None)
                    if bs is None or (isinstance(bs, float) and np.isnan(bs)):
                        return -1e9
                    return float(bs)
            else:
                def priority(x: dict[str, Any]) -> float:
                    return float(x["score"])

            best = max(group, key=priority)
            kept.append(best)

        return kept

    # ------------------------------------------------------------------
    # Debug artifact saving
    # ------------------------------------------------------------------

    def _save_debug_artifacts(
        self,
        *,
        out_dir: str,
        image_rgb: np.ndarray,
        result: dict[str, Any],
    ) -> None:
        os.makedirs(out_dir, exist_ok=True)

        # Save arrays
        np.save(os.path.join(out_dir, "sam_masks.npy"), result["sam_masks"])
        np.save(os.path.join(out_dir, "bboxes_xyxy.npy"), result["bboxes_xyxy"])
        np.save(os.path.join(out_dir, "box_scores.npy"), result["box_scores"])

        if result.get("raw_similarity", None) is not None:
            np.save(os.path.join(out_dir, "raw_similarity.npy"), result["raw_similarity"])
        if result.get("canonical_similarity", None) is not None:
            np.save(os.path.join(out_dir, "canonical_similarity.npy"), result["canonical_similarity"])
        if result.get("image_embeddings", None) is not None:
            np.save(os.path.join(out_dir, "image_embeddings.npy"), result["image_embeddings"])

        # Save JSON summaries
        serializable_objects = []
        for obj in result["object_infos"]:
            obj_copy = dict(obj)
            if "mask_2d" in obj_copy:
                obj_copy["mask_2d"] = None  # keep JSON compact
            serializable_objects.append(obj_copy)

        with open(os.path.join(out_dir, "object_infos.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "object_infos": serializable_objects,
                    "timings": result["timings"],
                    "label_bank_texts": result.get("label_bank_texts", []),
                    "canonical_labels_used": result.get("canonical_labels_used", []),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        # Save overlay
        overlay_bgr = self._draw_overlay(
            image_rgb=image_rgb,
            object_infos=result["object_infos"],
            alpha=self.overlay_alpha,
        )
        cv2.imwrite(os.path.join(out_dir, "frontend_siglip2_overlay.png"), overlay_bgr)

    @staticmethod
    def _draw_overlay(
        *,
        image_rgb: np.ndarray,
        object_infos: list[dict[str, Any]],
        alpha: float = 0.28,
    ) -> np.ndarray:
        rgb = image_rgb.copy()
        overlay = rgb.copy()

        palette = [
            (255, 0, 255),   # magenta
            (0, 255, 255),   # cyan
            (0, 255, 0),     # green
            (255, 255, 0),   # yellow
            (255, 128, 0),
            (128, 255, 0),
            (0, 128, 255),
            (255, 0, 128),
        ]

        for rank, obj in enumerate(object_infos):
            mask = np.asarray(obj["mask_2d"]).astype(bool)
            if mask.sum() == 0:
                continue

            color_rgb = np.array(palette[rank % len(palette)], dtype=np.uint8)
            overlay[mask] = (
                (1.0 - alpha) * overlay[mask].astype(np.float32)
                + alpha * color_rgb.astype(np.float32)
            ).astype(np.uint8)

        rgb = overlay

        for rank, obj in enumerate(object_infos):
            mask = (np.asarray(obj["mask_2d"]).astype(np.uint8) * 255)
            if mask.sum() == 0:
                continue

            color_rgb = tuple(int(x) for x in palette[rank % len(palette)])
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb, contours, -1, color_rgb, 2)

            box = obj["box_2d"]
            x1, y1, x2, y2 = int(box["xmin"]), int(box["ymin"]), int(box["xmax"]), int(box["ymax"])
            cv2.rectangle(rgb, (x1, y1), (x2, y2), color_rgb, 2)

            label = str(obj["label"])
            score = float(obj["score"])
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