from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    pipeline,
)
from transformers.utils import logging

logging.set_verbosity_error()


class SemanticPerception:
    """
    Generic proposal-first perception frontend.

    Responsibilities:
      1) SAM automatic mask proposals
      2) geometry-based proposal filtering
      3) Grounding DINO label-by-label scoring on each proposal crop
      4) generic geometric merge / dedupe
      5) whole-image rescue branch
      6) return raw scene candidates

    IMPORTANT:
      - This class does NOT perform canonicalization.
      - This class does NOT perform risk retrieval.
      - Canonicalization (alias + embedding NN + family mapping) belongs in router.py.
      - Risk retrieval (exact -> nearest -> family -> LLM -> conservative fallback) belongs in router.py.
    """

    DRAW_COLORS = [
        (0, 0, 255),
        (255, 0, 0),
        (0, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
        (128, 0, 255),
        (255, 128, 0),
        (0, 128, 255),
        (128, 255, 0),
    ]

    GENERIC_LABELS = {
        "container",
        "device",
        "object",
        "tool",
        "item",
    }

    SUPPORT_LABELS = {
        "table",
        "wall",
        "floor",
        "shelf",
        "counter",
        "desk",
    }

    def __init__(
        self,
        grounding_model: str = "IDEA-Research/grounding-dino-tiny",
        sam_model: str = "facebook/sam-vit-base",
        grounding_box_threshold: float = 0.12,
        grounding_text_threshold: float = 0.05,
        label_accept_threshold: float = 0.05,
        proposal_min_area_ratio: float = 0.003,
        proposal_max_area_ratio: float = 0.35,
        proposal_min_bbox_fill_ratio: float = 0.08,
        max_border_touches_for_small_object: int = 2,
        crop_padding_px: int = 8,
        use_masked_crop_first: bool = True,
        unknown_label: str = "unknown_object",
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.grounding_box_threshold = float(grounding_box_threshold)
        self.grounding_text_threshold = float(grounding_text_threshold)
        self.label_accept_threshold = float(label_accept_threshold)

        self.proposal_min_area_ratio = float(proposal_min_area_ratio)
        self.proposal_max_area_ratio = float(proposal_max_area_ratio)
        self.proposal_min_bbox_fill_ratio = float(proposal_min_bbox_fill_ratio)
        self.max_border_touches_for_small_object = int(max_border_touches_for_small_object)

        self.crop_padding_px = int(crop_padding_px)
        self.use_masked_crop_first = bool(use_masked_crop_first)
        self.unknown_label = str(unknown_label)

        print("Loading Grounding DINO...")
        self.grounding_processor = AutoProcessor.from_pretrained(grounding_model)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            grounding_model
        ).to(self.device)
        self.grounding_model.eval()
        print("Grounding DINO loaded successfully.")

        self.mask_generator = None
        try:
            print("Loading SAM mask generator...")
            self.mask_generator = pipeline(
                task="mask-generation",
                model=sam_model,
                device=0 if self.device == "cuda" else -1,
            )
            print("SAM loaded successfully.")
        except Exception as e:
            print(
                "Warning: SAM could not be loaded. "
                f"Falling back to whole-image Grounding DINO detection. Error: {e}"
            )

    # ------------------------------------------------------------------
    # Public APIs
    # ------------------------------------------------------------------

    def detect_objects(
        self,
        image_path: str,
        candidate_labels: Sequence[str],
        max_results: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Whole-image rescue detection.

        Scores each candidate label independently on the full image.
        """
        if not candidate_labels:
            raise ValueError("candidate_labels must be non-empty.")

        image = Image.open(image_path).convert("RGB")

        detections: List[Dict[str, Any]] = []
        for label in candidate_labels:
            detections.extend(
                self._grounding_dino_detect_single_label(
                    image=image,
                    label=label,
                    box_threshold=self.grounding_box_threshold,
                    text_threshold=self.grounding_text_threshold,
                )
            )

        detections = self._sort_and_clean_detections(detections)
        detections = self._global_nms_detections(detections, iou_threshold=0.5)
        detections = self._dedupe_same_label_same_box(detections)
        return detections[:max_results]

    def detect_scene_objects(
        self,
        image_path: str,
        candidate_labels: Sequence[str],
        max_proposals: int = 25,
        max_results: int = 30,
        save_debug: bool = False,
        debug_dir: str = "perception_debug",
        include_unknown: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Proposal-first scene detection.

        Returns raw scene candidates. These are NOT canonicalized.
        """
        if not candidate_labels:
            raise ValueError("candidate_labels must be non-empty.")

        image_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(image_pil)

        if self.mask_generator is None:
            print("SAM unavailable. Using whole-image Grounding DINO fallback.")
            detections = self.detect_objects(
                image_path=image_path,
                candidate_labels=candidate_labels,
                max_results=max_results,
            )
            if save_debug:
                os.makedirs(debug_dir, exist_ok=True)
                self.save_labeled_overlay(
                    image_np=image_np,
                    detections=detections,
                    out_path=os.path.join(debug_dir, "labeled_overlay.png"),
                )
            return detections

        proposals = self.generate_mask_proposals(
            image_path=image_path,
            max_proposals=max_proposals,
        )

        if save_debug:
            os.makedirs(debug_dir, exist_ok=True)
            self.save_proposal_overlay(
                image_np=image_np,
                proposals=proposals,
                out_path=os.path.join(debug_dir, "proposal_overlay.png"),
            )

        proposal_detections = self.label_mask_proposals(
            image_path=image_path,
            proposals=proposals,
            candidate_labels=candidate_labels,
            top_k=5,
            save_topk_debug=save_debug,
            debug_dir=debug_dir,
            save_debug_crops=save_debug,
            include_unknown=include_unknown,
        )

        proposal_detections = self._merge_geometrically_similar_detections(
            detections=proposal_detections,
            iou_threshold=0.55,
            gap_threshold_px=28.0,
        )

        rescue_detections = self.detect_objects(
            image_path=image_path,
            candidate_labels=candidate_labels,
            max_results=max_results,
        )

        detections = self._merge_with_whole_image_rescue(
            proposal_detections=proposal_detections,
            rescue_detections=rescue_detections,
        )

        detections = self._attach_best_proposal_masks(
            detections=detections,
            proposals=proposals,
            min_iou=0.15,
        )

        detections = self._global_nms_detections(detections, iou_threshold=0.5)
        detections = self._dedupe_same_label_same_box(detections)

        detections = self._postprocess_scene_detections(
            detections=detections,
            image_shape=image_np.shape[:2],
            min_scene_score=0.30,
            containment_threshold=0.85,
            small_inside_large_ratio=0.55,
            drop_support_labels=False,
        )

        detections = detections[:max_results]

        if save_debug:
            self.save_labeled_overlay(
                image_np=image_np,
                detections=detections,
                out_path=os.path.join(debug_dir, "labeled_overlay.png"),
            )
            serializable_detections = []
            for det in detections:
                row = {k: v for k, v in det.items() if k != "mask"}
                serializable_detections.append(row)

            with open(os.path.join(debug_dir, "scene_detections.json"), "w", encoding="utf-8") as f:
                json.dump(serializable_detections, f, indent=2, ensure_ascii=False)

        return detections

    # ------------------------------------------------------------------
    # Stage 1: SAM proposals
    # ------------------------------------------------------------------

    def generate_mask_proposals(
        self,
        image_path: str,
        max_proposals: int = 25,
    ) -> List[Dict[str, Any]]:
        """
        Generate object-like proposals from SAM automatic mask generation.

        Filtering is geometry-based only.
        """
        image_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(image_pil)
        h, w = image_np.shape[:2]
        image_area = h * w

        raw = self.mask_generator(image_pil)
        masks = self._extract_masks_from_sam_output(raw)

        proposals: List[Dict[str, Any]] = []
        for idx, mask in enumerate(masks):
            mask_bool = self._to_bool_mask(mask)
            if mask_bool is None:
                continue

            ys, xs = np.where(mask_bool)
            if len(xs) == 0 or len(ys) == 0:
                continue

            xmin, xmax = int(xs.min()), int(xs.max())
            ymin, ymax = int(ys.min()), int(ys.max())
            area = int(mask_bool.sum())

            bbox_w = max(1, xmax - xmin)
            bbox_h = max(1, ymax - ymin)
            bbox_area = bbox_w * bbox_h
            area_ratio = area / float(image_area)
            bbox_fill_ratio = area / float(bbox_area)

            proposal = {
                "proposal_id": idx,
                "bbox": {
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                },
                "area": area,
                "area_ratio": area_ratio,
                "bbox_fill_ratio": bbox_fill_ratio,
                "mask": mask_bool,
            }

            if not self._passes_geometry_filter(proposal, image_shape=(h, w)):
                continue

            proposals.append(proposal)

        proposals = self._nms_proposals_by_bbox(proposals, iou_threshold=0.85)
        proposals = sorted(proposals, key=lambda p: p["area"], reverse=True)[:max_proposals]
        return proposals

    def _passes_geometry_filter(
        self,
        proposal: Dict[str, Any],
        image_shape: Tuple[int, int],
    ) -> bool:
        """
        Geometry-only proposal filtering.
        No object-specific rules here.
        """
        h, w = image_shape
        box = proposal["bbox"]
        area_ratio = float(proposal["area_ratio"])
        bbox_fill_ratio = float(proposal["bbox_fill_ratio"])

        if area_ratio < self.proposal_min_area_ratio:
            return False
        if area_ratio > self.proposal_max_area_ratio:
            return False
        if bbox_fill_ratio < self.proposal_min_bbox_fill_ratio:
            return False

        touches = 0
        if box["xmin"] <= 1:
            touches += 1
        if box["xmax"] >= w - 2:
            touches += 1
        if box["ymin"] <= 1:
            touches += 1
        if box["ymax"] >= h - 2:
            touches += 1

        # Highly border-touching proposals are often large background/support masks.
        if touches > self.max_border_touches_for_small_object and area_ratio > 0.08:
            return False

        return True

    def _attach_best_proposal_masks(
        self,
        detections: list[dict[str, Any]],
        proposals: list[dict[str, Any]],
        min_iou: float = 0.15,
    ) -> list[dict[str, Any]]:
        """
        If a detection does not already have a mask, attach the best-overlapping proposal mask.
        Useful for whole-image rescue detections.
        """
        if not detections or not proposals:
            return detections

        enriched: list[dict[str, Any]] = []

        for det in detections:
            if det.get("mask", None) is not None:
                enriched.append(det)
                continue

            best_iou = 0.0
            best_mask = None

            for prop in proposals:
                prop_box = prop["bbox"]
                iou = self._iou(det["box"], prop_box)
                if iou > best_iou:
                    best_iou = iou
                    best_mask = prop.get("mask", None)

            out = dict(det)
            if best_mask is not None and best_iou >= min_iou:
                out["mask"] = best_mask.copy()
                out["mask_source"] = "attached_from_proposal"
            else:
                out["mask"] = None
                out["mask_source"] = out.get("mask_source", "whole_image")

            enriched.append(out)

        return enriched

    # ------------------------------------------------------------------
    # Stage 2: Grounding DINO label-by-label crop scoring
    # ------------------------------------------------------------------

    def label_mask_proposals(
        self,
        image_path: str,
        proposals: List[Dict[str, Any]],
        candidate_labels: Sequence[str],
        top_k: int = 5,
        save_topk_debug: bool = False,
        debug_dir: str = "perception_debug",
        save_debug_crops: bool = False,
        include_unknown: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Label each SAM proposal using Grounding DINO, scoring candidate labels one-by-one.

        Final localization box is the proposal box.
        """
        image_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(image_pil)

        detections: List[Dict[str, Any]] = []
        debug_records: List[Dict[str, Any]] = []

        for proposal in proposals:
            proposal_id = int(proposal["proposal_id"])
            proposal_box = proposal["bbox"]

            masked_crop_pil = None
            bbox_crop_pil = None

            masked_crop_result = self._make_masked_crop(
                image_np=image_np,
                proposal=proposal,
                dilate_iters=1,
                background_mode="black",
            )
            if masked_crop_result is not None:
                masked_crop_pil = masked_crop_result[0]

            bbox_crop_result = self._make_bbox_crop(
                image_np=image_np,
                proposal=proposal,
            )
            if bbox_crop_result is not None:
                bbox_crop_pil = bbox_crop_result[0]

            if masked_crop_pil is None and bbox_crop_pil is None:
                continue

            if save_debug_crops:
                if masked_crop_pil is not None:
                    self._save_debug_crop(masked_crop_pil, proposal_id, debug_dir, suffix="masked")
                if bbox_crop_pil is not None:
                    self._save_debug_crop(bbox_crop_pil, proposal_id, debug_dir, suffix="bbox")

            masked_scores: Dict[str, float] = {}
            bbox_scores: Dict[str, float] = {}

            if self.use_masked_crop_first and masked_crop_pil is not None:
                masked_scores = self._score_candidate_labels_on_crop(
                    crop_pil=masked_crop_pil,
                    candidate_labels=candidate_labels,
                )

            if bbox_crop_pil is not None:
                bbox_scores = self._score_candidate_labels_on_crop(
                    crop_pil=bbox_crop_pil,
                    candidate_labels=candidate_labels,
                )

            chosen_mode = "bbox"
            chosen_scores = bbox_scores

            if masked_scores:
                best_masked_score = max(masked_scores.values()) if masked_scores else 0.0
                best_bbox_score = max(bbox_scores.values()) if bbox_scores else 0.0
                if best_masked_score >= best_bbox_score:
                    chosen_mode = "masked"
                    chosen_scores = masked_scores

            if not chosen_scores:
                continue

            # best prediction from chosen_scores
            sorted_topk = sorted(
                chosen_scores.items(),
                key=lambda kv: kv[1],
                reverse=True,
            )[:top_k]

            if len(sorted_topk) == 0:
                continue

            best_label, best_score = sorted_topk[0]

            # IMPORTANT: always initialize these first
            final_label = best_label
            final_score = float(best_score)

            # threshold handling
            if final_score < self.label_accept_threshold:
                if not include_unknown:
                    continue
                final_label = self.unknown_label

            debug_records.append(
                {
                    "proposal_id": proposal_id,
                    "proposal_box": proposal_box,
                    "proposal_area": int(proposal["area"]),
                    "proposal_area_ratio": float(proposal["area_ratio"]),
                    "proposal_bbox_fill_ratio": float(proposal["bbox_fill_ratio"]),
                    "mode_used": chosen_mode,
                    "top_k": [[label, float(score)] for label, score in sorted_topk],
                }
            )

            detections.append(
                {
                    "label": str(final_label),
                    "score": final_score,
                    "box": {
                        "xmin": int(proposal_box["xmin"]),
                        "ymin": int(proposal_box["ymin"]),
                        "xmax": int(proposal_box["xmax"]),
                        "ymax": int(proposal_box["ymax"]),
                    },
                    "proposal_id": proposal_id,
                    "proposal_box": proposal_box,
                    "proposal_area": int(proposal["area"]),
                    "area_ratio": float(proposal["area_ratio"]),
                    "bbox_fill_ratio": float(proposal["bbox_fill_ratio"]),
                    "mode_used": chosen_mode,
                    "label_scores": {label: float(score) for label, score in chosen_scores.items()},
                    "top_k": [[label, float(score)] for label, score in sorted_topk],
                    "mask": proposal["mask"].copy(),
                    "mask_source": "proposal",
                    "source": "proposal",
                }
            )

        if save_topk_debug:
            self._save_topk_debug_json(debug_records, debug_dir)

        return self._sort_and_clean_detections(detections)

    def _normalize_label(self, label: str) -> str:
        return str(label).strip().lower()


    def _box_area(self, box: dict[str, int]) -> float:
        return max(0, box["xmax"] - box["xmin"]) * max(0, box["ymax"] - box["ymin"])


    def _intersection_area(self, a: dict[str, int], b: dict[str, int]) -> float:
        inter_x1 = max(a["xmin"], b["xmin"])
        inter_y1 = max(a["ymin"], b["ymin"])
        inter_x2 = min(a["xmax"], b["xmax"])
        inter_y2 = min(a["ymax"], b["ymax"])

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        return inter_w * inter_h


    def _containment_ratio(self, inner: dict[str, int], outer: dict[str, int]) -> float:
        """
        Fraction of `inner` covered by `outer`.
        1.0 means inner is fully inside outer.
        """
        inner_area = self._box_area(inner)
        if inner_area <= 0:
            return 0.0
        inter = self._intersection_area(inner, outer)
        return inter / inner_area


    def _is_generic_label(self, label: str) -> bool:
        return self._normalize_label(label) in self.GENERIC_LABELS


    def _is_support_label(self, label: str) -> bool:
        return self._normalize_label(label) in self.SUPPORT_LABELS


    def _semantic_priority(self, label: str) -> int:
        """
        Higher = keep earlier.
        specific object > generic object > support/background
        """
        label = self._normalize_label(label)

        if label in self.SUPPORT_LABELS:
            return 0
        if label in self.GENERIC_LABELS:
            return 1

        # More specific labels get slightly higher priority
        return 2 + min(len(label.split()), 3)

    # ------------------------------------------------------------------
    # Grounding DINO backend
    # ------------------------------------------------------------------

    def _grounding_dino_detect_single_label(
        self,
        image: Image.Image,
        label: str,
        box_threshold: float,
        text_threshold: float,
    ) -> list[dict[str, Any]]:
        image = image.convert("RGB")
        prompt = self._format_prompt(label)

        inputs = self.grounding_processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.grounding_model(**inputs)

        results = self.grounding_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )

        parsed: list[dict[str, Any]] = []
        result = results[0]

        for box, score in zip(result["boxes"], result["scores"]):
            xmin, ymin, xmax, ymax = [int(round(v)) for v in box.tolist()]
            parsed.append(
                {
                    "label": str(label).strip().lower(),
                    "score": float(score.item()),
                    "box": {
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax,
                    },
                    "mask": None,                 # whole-image detection은 기본적으로 mask 없음
                    "mask_source": "whole_image",
                    "source": "whole_image",
                }
            )

        return self._sort_and_clean_detections(parsed)

    def _score_candidate_labels_on_crop(
        self,
        crop_pil: Image.Image,
        candidate_labels: Sequence[str],
    ) -> Dict[str, float]:
        scores: Dict[str, float] = {}

        for label in candidate_labels:
            results = self._grounding_dino_detect_single_label(
                image=crop_pil,
                label=label,
                box_threshold=self.grounding_box_threshold,
                text_threshold=self.grounding_text_threshold,
            )

            if not results:
                scores[label] = 0.0
                continue

            best_score = max(float(r["score"]) for r in results)
            scores[label] = best_score

        return scores

    def _format_prompt(self, label: str) -> str:
        label = str(label).strip().lower()
        if not label:
            return label
        if label.endswith("."):
            return label
        return f"{label}."

    # ------------------------------------------------------------------
    # Crop helpers
    # ------------------------------------------------------------------

    def _make_masked_crop(
        self,
        image_np: np.ndarray,
        proposal: Dict[str, Any],
        dilate_iters: int = 1,
        background_mode: str = "black",
    ):
        h, w = image_np.shape[:2]
        pb = proposal["bbox"]
        xmin, ymin, xmax, ymax = pb["xmin"], pb["ymin"], pb["xmax"], pb["ymax"]

        crop_xmin = max(0, xmin - self.crop_padding_px)
        crop_ymin = max(0, ymin - self.crop_padding_px)
        crop_xmax = min(w - 1, xmax + self.crop_padding_px)
        crop_ymax = min(h - 1, ymax + self.crop_padding_px)

        if crop_xmax <= crop_xmin or crop_ymax <= crop_ymin:
            return None, None, None

        crop_img = image_np[crop_ymin:crop_ymax, crop_xmin:crop_xmax].copy()
        full_mask = proposal["mask"]
        crop_mask = full_mask[crop_ymin:crop_ymax, crop_xmin:crop_xmax].copy()

        if dilate_iters > 0:
            kernel = np.ones((3, 3), np.uint8)
            crop_mask = cv2.dilate(crop_mask.astype(np.uint8), kernel, iterations=dilate_iters) > 0

        if background_mode == "black":
            masked_crop = np.zeros_like(crop_img)
            masked_crop[crop_mask] = crop_img[crop_mask]
        elif background_mode == "mean":
            if np.any(crop_mask):
                mean_color = crop_img[crop_mask].mean(axis=0)
            else:
                mean_color = np.array([0, 0, 0], dtype=np.float32)
            masked_crop = np.zeros_like(crop_img)
            masked_crop[:] = mean_color.astype(np.uint8)
            masked_crop[crop_mask] = crop_img[crop_mask]
        else:
            masked_crop = crop_img

        crop_pil = Image.fromarray(masked_crop)
        return crop_pil, (crop_xmin, crop_ymin, crop_xmax, crop_ymax), crop_mask

    def _make_bbox_crop(
        self,
        image_np: np.ndarray,
        proposal: Dict[str, Any],
    ):
        pb = proposal["bbox"]
        xmin, ymin, xmax, ymax = pb["xmin"], pb["ymin"], pb["xmax"], pb["ymax"]

        h, w = image_np.shape[:2]
        crop_xmin = max(0, xmin - self.crop_padding_px)
        crop_ymin = max(0, ymin - self.crop_padding_px)
        crop_xmax = min(w - 1, xmax + self.crop_padding_px)
        crop_ymax = min(h - 1, ymax + self.crop_padding_px)

        if crop_xmax <= crop_xmin or crop_ymax <= crop_ymin:
            return None, None

        crop_img = image_np[crop_ymin:crop_ymax, crop_xmin:crop_xmax].copy()
        crop_pil = Image.fromarray(crop_img)
        return crop_pil, (crop_xmin, crop_ymin, crop_xmax, crop_ymax)

    # ------------------------------------------------------------------
    # Generic merge / rescue logic
    # ------------------------------------------------------------------

    def _merge_geometrically_similar_detections(
        self,
        detections: list[dict[str, Any]],
        iou_threshold: float = 0.55,
        gap_threshold_px: float = 28.0,
    ) -> list[dict[str, Any]]:
        if not detections:
            return []

        detections = sorted(detections, key=lambda d: float(d["score"]), reverse=True)
        used = [False] * len(detections)
        merged: list[dict[str, Any]] = []

        for i, det_i in enumerate(detections):
            if used[i]:
                continue

            used[i] = True
            cluster = [det_i]
            box_union = dict(det_i["box"])

            cluster_mask = None
            if det_i.get("mask", None) is not None:
                cluster_mask = det_i["mask"].copy()

            for j in range(i + 1, len(detections)):
                if used[j]:
                    continue

                det_j = detections[j]
                if det_i["label"] != det_j["label"]:
                    continue

                iou = self._iou(box_union, det_j["box"])
                gap = self._bbox_gap(box_union, det_j["box"])

                if iou >= iou_threshold or gap <= gap_threshold_px:
                    cluster.append(det_j)
                    used[j] = True
                    box_union = self._bbox_union(box_union, det_j["box"])

                    if det_j.get("mask", None) is not None:
                        if cluster_mask is None:
                            cluster_mask = det_j["mask"].copy()
                        else:
                            cluster_mask = np.logical_or(cluster_mask, det_j["mask"])

            best = max(cluster, key=lambda d: float(d["score"]))
            out = dict(best)
            out["box"] = box_union
            out["cluster_size"] = len(cluster)
            out["mask"] = cluster_mask
            out["mask_source"] = "merged_proposals" if cluster_mask is not None else out.get("mask_source", None)
            merged.append(out)

        return self._sort_and_clean_detections(merged)

    def _merge_with_whole_image_rescue(
        self,
        proposal_detections: List[Dict[str, Any]],
        rescue_detections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Generic rescue merge.

        Add whole-image detections only if proposal detections do not already
        cover the same label geometrically.
        """
        merged = list(proposal_detections)

        for rescue in rescue_detections:
            rescue_label = rescue["label"]
            rescue_box = rescue["box"]

            covered = False
            for det in proposal_detections:
                if det["label"] != rescue_label:
                    continue

                iou = self._iou(det["box"], rescue_box)
                gap = self._bbox_gap(det["box"], rescue_box)

                if iou > 0.2 or gap < 25.0:
                    covered = True
                    break

            if not covered:
                out = dict(rescue)
                out["source"] = "whole_image_rescue"
                merged.append(out)

        return self._sort_and_clean_detections(merged)

    def _bbox_union(self, a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
        return {
            "xmin": min(a["xmin"], b["xmin"]),
            "ymin": min(a["ymin"], b["ymin"]),
            "xmax": max(a["xmax"], b["xmax"]),
            "ymax": max(a["ymax"], b["ymax"]),
        }

    def _bbox_gap(self, a: Dict[str, int], b: Dict[str, int]) -> float:
        dx = max(a["xmin"] - b["xmax"], b["xmin"] - a["xmax"], 0)
        dy = max(a["ymin"] - b["ymax"], b["ymin"] - a["ymax"], 0)
        return float(np.sqrt(dx * dx + dy * dy))

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    def save_proposal_overlay(
        self,
        image_np: np.ndarray,
        proposals: List[Dict[str, Any]],
        out_path: str,
    ) -> None:
        canvas = image_np.copy()
        if canvas.dtype != np.uint8:
            canvas = np.clip(canvas, 0, 255).astype(np.uint8)

        if canvas.shape[-1] == 3:
            canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        else:
            canvas_bgr = canvas

        colored = canvas_bgr.copy()
        for i, proposal in enumerate(proposals):
            color = self.DRAW_COLORS[i % len(self.DRAW_COLORS)]
            mask = proposal["mask"]

            overlay = colored.copy()
            overlay[mask] = (0.6 * np.array(color) + 0.4 * overlay[mask]).astype(np.uint8)
            colored = overlay

            box = proposal["bbox"]
            cv2.rectangle(
                colored,
                (box["xmin"], box["ymin"]),
                (box["xmax"], box["ymax"]),
                color,
                2,
            )
            cv2.putText(
                colored,
                f"P{i}",
                (box["xmin"], max(18, box["ymin"] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                lineType=cv2.LINE_AA,
            )

        cv2.imwrite(out_path, colored)

    def save_labeled_overlay(
        self,
        image_np: np.ndarray,
        detections: List[Dict[str, Any]],
        out_path: str,
    ) -> None:
        canvas = image_np.copy()
        if canvas.dtype != np.uint8:
            canvas = np.clip(canvas, 0, 255).astype(np.uint8)

        if canvas.shape[-1] == 3:
            canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        else:
            canvas_bgr = canvas

        for i, det in enumerate(detections):
            color = self.DRAW_COLORS[i % len(self.DRAW_COLORS)]
            box = det["box"]
            label = det["label"]
            score = det["score"]

            cv2.rectangle(
                canvas_bgr,
                (box["xmin"], box["ymin"]),
                (box["xmax"], box["ymax"]),
                color,
                2,
            )
            cv2.putText(
                canvas_bgr,
                f"{label}: {score:.2f}",
                (box["xmin"], max(18, box["ymin"] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                lineType=cv2.LINE_AA,
            )

        cv2.imwrite(out_path, canvas_bgr)

    def _save_topk_debug_json(self, debug_records: List[Dict[str, Any]], debug_dir: str) -> None:
        os.makedirs(debug_dir, exist_ok=True)
        with open(os.path.join(debug_dir, "proposal_topk_debug.json"), "w", encoding="utf-8") as f:
            json.dump(debug_records, f, indent=2, ensure_ascii=False)

    def _save_debug_crop(self, crop_pil: Image.Image, proposal_id: int, debug_dir: str, suffix: str) -> None:
        os.makedirs(debug_dir, exist_ok=True)
        crop_path = os.path.join(debug_dir, f"proposal_{proposal_id}_{suffix}.png")
        crop_pil.save(crop_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_masks_from_sam_output(self, raw_output: Any) -> List[Any]:
        if raw_output is None:
            return []

        if isinstance(raw_output, dict):
            if "masks" in raw_output:
                return list(raw_output["masks"])
            return []

        if isinstance(raw_output, list):
            if len(raw_output) == 0:
                return []
            if isinstance(raw_output[0], dict) and "mask" in raw_output[0]:
                return [x["mask"] for x in raw_output if "mask" in x]
            return raw_output

        return []

    def _to_bool_mask(self, mask: Any) -> Optional[np.ndarray]:
        if mask is None:
            return None

        if isinstance(mask, Image.Image):
            arr = np.array(mask)
        elif torch.is_tensor(mask):
            arr = mask.detach().cpu().numpy()
        else:
            arr = np.asarray(mask)

        if arr.ndim == 3:
            arr = arr[..., 0]

        if arr.ndim != 2:
            return None

        return arr > 0

    def _sort_and_clean_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cleaned: List[Dict[str, Any]] = []
        for det in detections:
            box = det["box"]
            cleaned.append(
                {
                    **det,
                    "box": {
                        "xmin": int(box["xmin"]),
                        "ymin": int(box["ymin"]),
                        "xmax": int(box["xmax"]),
                        "ymax": int(box["ymax"]),
                    },
                    "score": float(det["score"]),
                }
            )
        return sorted(cleaned, key=lambda d: float(d["score"]), reverse=True)

    def _nms_proposals_by_bbox(
        self,
        proposals: List[Dict[str, Any]],
        iou_threshold: float = 0.85,
    ) -> List[Dict[str, Any]]:
        proposals = sorted(proposals, key=lambda p: p["area"], reverse=True)
        kept: List[Dict[str, Any]] = []

        while proposals:
            current = proposals.pop(0)
            kept.append(current)

            remaining = []
            for p in proposals:
                if self._iou(current["bbox"], p["bbox"]) < iou_threshold:
                    remaining.append(p)
            proposals = remaining

        return kept

    def _global_nms_detections(
        self,
        detections: List[Dict[str, Any]],
        iou_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        detections = sorted(detections, key=lambda d: float(d["score"]), reverse=True)
        kept: List[Dict[str, Any]] = []

        while detections:
            current = detections.pop(0)
            kept.append(current)

            remaining = []
            for d in detections:
                if self._iou(current["box"], d["box"]) < iou_threshold:
                    remaining.append(d)
            detections = remaining

        return kept

    def _dedupe_same_label_same_box(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        detections = sorted(detections, key=lambda d: float(d["score"]), reverse=True)
        kept: List[Dict[str, Any]] = []

        for det in detections:
            duplicate = False
            for prev in kept:
                if det["label"] == prev["label"] and self._iou(det["box"], prev["box"]) > 0.9:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(det)

        return kept

    def _iou(self, a: Dict[str, int], b: Dict[str, int]) -> float:
        ax1, ay1, ax2, ay2 = a["xmin"], a["ymin"], a["xmax"], a["ymax"]
        bx1, by1, bx2, by2 = b["xmin"], b["ymin"], b["xmax"], b["ymax"]

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union

    def _postprocess_scene_detections(
        self,
        detections: list[dict[str, Any]],
        image_shape: tuple[int, int],
        min_scene_score: float = 0.25,
        containment_threshold: float = 0.85,
        small_inside_large_ratio: float = 0.55,
        drop_support_labels: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Scene candidate post-processing.

        Rules:
        1) confidence threshold
        2) contained-box suppression
        3) generic-vs-specific suppression
        4) optional support/background filtering

        This is intentionally generic:
        - geometry-based
        - semantic-specificity based
        - not object-pair hardcoded
        """
        if not detections:
            return []

        h, w = image_shape

        # 1) drop very low-confidence detections
        filtered = [
            d for d in detections
            if float(d.get("score", 0.0)) >= min_scene_score
        ]

        # optional: remove support/background labels here
        if drop_support_labels:
            filtered = [
                d for d in filtered
                if not self._is_support_label(d["label"])
            ]

        # 2) sort by semantic priority first, then confidence, then larger area
        filtered = sorted(
            filtered,
            key=lambda d: (
                self._semantic_priority(d["label"]),
                float(d.get("score", 0.0)),
                self._box_area(d["box"]),
            ),
            reverse=True,
        )

        kept: list[dict[str, Any]] = []

        for det in filtered:
            label = self._normalize_label(det["label"])
            box = det["box"]
            area = self._box_area(box)

            should_drop = False

            for prev in kept:
                prev_label = self._normalize_label(prev["label"])
                prev_box = prev["box"]
                prev_area = self._box_area(prev_box)

                # Standard near-duplicate suppression
                iou = self._iou(box, prev_box)
                if label == prev_label and iou > 0.75:
                    should_drop = True
                    break

                contain_in_prev = self._containment_ratio(box, prev_box)
                prev_in_current = self._containment_ratio(prev_box, box)

                # If current box is mostly contained in a much larger kept box
                if contain_in_prev >= containment_threshold and area < small_inside_large_ratio * prev_area:
                    # Drop current if it is generic and prev is more specific
                    if self._is_generic_label(label) and not self._is_generic_label(prev_label):
                        should_drop = True
                        break

                    # Drop current if it is support/background and prev is object-like
                    if self._is_support_label(label) and not self._is_support_label(prev_label):
                        should_drop = True
                        break

                    # Drop low-priority inner box when labels differ and prev is more semantically specific
                    if self._semantic_priority(label) < self._semantic_priority(prev_label):
                        should_drop = True
                        break

                # If previous kept box is generic but current is more specific and overlaps heavily,
                # replace the generic one later by skipping current=False and removing prev
                if prev_in_current >= containment_threshold and prev_area < small_inside_large_ratio * area:
                    if self._is_generic_label(prev_label) and not self._is_generic_label(label):
                        prev["_marked_for_removal"] = True

            if not should_drop:
                kept.append(det)

        kept = [d for d in kept if not d.get("_marked_for_removal", False)]

        # final stable ordering for display/use
        kept = sorted(
            kept,
            key=lambda d: float(d.get("score", 0.0)),
            reverse=True,
        )

        return kept


def run_perception_demo():
    image_path = "test_rgb.png"

    candidate_labels = [
        "power drill",
        "drill",
        "bowl",
        "ceramic bowl",
        "mixing bowl",
        "table",
    ]

    detector = SemanticPerception(
        grounding_box_threshold=0.12,
        grounding_text_threshold=0.05,
        label_accept_threshold=0.05,
    )

    print("\nRunning generic proposal-first scene detection...")
    results = detector.detect_scene_objects(
        image_path=image_path,
        candidate_labels=candidate_labels,
        save_debug=True,
        debug_dir="perception_debug",
        include_unknown=True,
    )

    if len(results) == 0:
        print("No scene objects detected.")
        return

    print(f"Detected {len(results)} raw scene candidates:")
    for result in results:
        print(f" - {result['label']} ({result['score']:.3f}) box={result['box']} source={result.get('source', 'proposal')}")

    print("\nSaved debug outputs to 'perception_debug/'.")


if __name__ == "__main__":
    run_perception_demo()